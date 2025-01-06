import aiohttp
import json
import logging
import requests
import sys
import time
from typing import Iterable, Iterator
from urllib.parse import urlsplit

from ping3 import ping

from benchmark.messagegeneration import (
    BaseMessagesGenerator,
    RandomMessagesGenerator,
    ReplayMessagesGenerator,
)
from .asynchttpexecuter import AsyncHTTPExecuter
from .oairequester import OAIRequester
from .ratelimiting import NoRateLimiter, RateLimiter
from .statsaggregator import _StatsAggregator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # Send logs to stdout for Docker
)


class _RequestBuilder:
    """
    Wrapper iterator class to build request payloads.
    """

    def __init__(
        self,
        messages_generator: BaseMessagesGenerator,
        max_tokens=None,
        completions=None,
        frequency_penalty=None,
        presence_penalty=None,
        temperature=None,
        top_p=None,
        model=None,
    ):
        self.messages_generator = messages_generator
        self.max_tokens = max_tokens
        self.completions = completions
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.model = model

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> (dict, int):
        messages, messages_tokens = self.messages_generator.generate_messages()
        body = {"messages": messages}
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.completions is not None:
            body["n"] = self.completions
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.top_p is not None:
            body["top_p"] = self.top_p
        # model param is only for openai.com endpoints
        if self.model is not None:
            body["model"] = self.model
        return body, messages_tokens


def load(args: dict):
    """
    Main entry point for loading with a dictionary-based `args`.
    """
    try:
        _validate(args)
    except ValueError as e:
        logging.error(f"invalid argument(s): {e}")
        sys.exit(1)
    
    logging.info("Args validated")

    # Log the run arguments for debugging
    run_args = {
        "api_base_endpoint": args.get("api_base_endpoint"),
        "deployment": args.get("deployment"),
        "clients": args.get("clients"),
        "requests": args.get("requests"),
        "duration": args.get("duration"),
        "run_end_condition_mode": args.get("run_end_condition_mode"),
        "rate": args.get("rate"),
        "aggregation_window": args.get("aggregation_window"),
        "context_generation_method": args.get("context_generation_method"),
        "replay_path": args.get("replay_path"),
        "shape_profile": args.get("shape_profile"),
        "context_tokens": args.get("context_tokens"),
        "max_tokens": args.get("max_tokens"),
        "prevent_server_caching": args.get("prevent_server_caching"),
        "completions": args.get("completions"),
        "retry": args.get("retry"),
        "api_version": args.get("api_version"),
        "frequency_penalty": args.get("frequency_penalty"),
        "presence_penalty": args.get("presence_penalty"),
        "temperature": args.get("temperature"),
        "top_p": args.get("top_p"),
        "adjust_for_network_latency": args.get("adjust_for_network_latency"),
        "output_format": args.get("output_format"),
        "log_request_content": args.get("log_request_content"),
        "api_key": args.get("api_key"),  # ensure we see the key as well
    }
    logging.debug("Load test args: " + json.dumps(run_args))

    if not args.get("api_key"):
        raise ValueError("API key is not provided in the request. Please provide an API key.")

    # Check if endpoint is openai.com, otherwise assume it is Azure OpenAI
    if not args.get("api_base_endpoint"):
        raise ValueError("api_base_endpoint is missing or empty.")
    
    is_openai_com_endpoint = "openai.com" in args.get("api_base_endpoint")
    if is_openai_com_endpoint:
        url = args.get("api_base_endpoint")
    else:
        url = args.get("api_base_endpoint") + f"/openai/deployments/{args.get('deployment')}/chat/completions"
        url += f"?api-version={args.get('api_version')}"

    logging.debug("Using endpoint: " + url)

    # Create or skip a rate limiter
    rate_limiter = NoRateLimiter()
    if args.get("rate") is not None and args.get("rate") > 0:
        rate_limiter = RateLimiter(args.get("rate"), 60)

    # Determine model name for token estimation
    if is_openai_com_endpoint:
        model = args.get("deployment")
    else:
        model_check_headers = {
            "api-key": args.get("api_key"),
            "Content-Type": "application/json",
        }
        model_check_body = {"messages": [{"content": "What is 1+1?", "role": "user"}]}
        response = requests.post(url, headers=model_check_headers, json=model_check_body)
        if response.status_code != 200:
            raise ValueError(
                f"Deployment check failed with status code {response.status_code}. "
                f"Reason: {response.reason}. Data: {response.text}"
            )
        model = response.json()["model"]
    logging.info(f"model detected: {model}")

    # Optional network latency adjustment
    if args.get("adjust_for_network_latency"):
        logging.info("checking ping to endpoint...")
        network_latency_adjustment = measure_avg_ping(url)
        logging.info(
            f"average ping to endpoint: {int(network_latency_adjustment*1000)}ms. "
            "this will be subtracted from all aggregate latency metrics."
        )
    else:
        network_latency_adjustment = 0

    # Adjust shape profile if context_generation_method == "generate"
    max_tokens = args.get("max_tokens")
    context_tokens = args.get("context_tokens")
    if args.get("context_generation_method") == "generate":
        if args.get("shape_profile") == "balanced":
            context_tokens, max_tokens = 500, 500
        elif args.get("shape_profile") == "context":
            context_tokens, max_tokens = 2000, 200
        elif args.get("shape_profile") == "generation":
            context_tokens, max_tokens = 500, 1000

        logging.info(
            f"using random messages generation with shape profile {args.get('shape_profile')}: "
            f"context tokens: {context_tokens}, max tokens: {max_tokens}"
        )
        messages_generator = RandomMessagesGenerator(
            model=model,
            prevent_server_caching=args.get("prevent_server_caching"),
            tokens=context_tokens,
            max_tokens=max_tokens,
        )
    else:
        # context_generation_method == "replay"
        logging.info(f"Replaying messages from {args.get('replay_path')}")
        messages_generator = ReplayMessagesGenerator(
            model=model,
            prevent_server_caching=args.get("prevent_server_caching"),
            path=args.get("replay_path"),
        )

    if args.get("run_end_condition_mode") == "and":
        logging.info(
            "run-end-condition-mode='and': "
            "run will not end until BOTH the `requests` and `duration` limits are reached"
        )
    else:
        logging.info(
            "run-end-condition-mode='or': "
            "run will end when EITHER the `requests` or `duration` limit is reached"
        )

    request_builder = _RequestBuilder(
        messages_generator=messages_generator,
        max_tokens=max_tokens,
        completions=args.get("completions"),
        frequency_penalty=args.get("frequency_penalty"),
        presence_penalty=args.get("presence_penalty"),
        temperature=args.get("temperature"),
        top_p=args.get("top_p"),
        model=args.get("deployment") if is_openai_com_endpoint else None,
    )

    logging.info("Starting load...")

    _run_load(
        request_builder=request_builder,
        max_concurrency=args.get("clients"),
        api_key=args.get("api_key"),
        url=url,
        rate_limiter=rate_limiter,
        backoff=(args.get("retry") == "exponential"),
        request_count=args.get("requests"),
        duration=args.get("duration"),
        aggregation_duration=args.get("aggregation_window", 60),
        run_end_condition_mode=args.get("run_end_condition_mode", "or"),
        json_output=(args.get("output_format") == "jsonl"),
        log_request_content=args.get("log_request_content", False),
        network_latency_adjustment=network_latency_adjustment,
    )


def _run_load(
    request_builder: Iterable[dict],
    max_concurrency: int,
    api_key: str,
    url: str,
    rate_limiter=None,
    backoff=False,
    duration=None,
    aggregation_duration=60,
    request_count=None,
    run_end_condition_mode="or",
    json_output=False,
    log_request_content=False,
    network_latency_adjustment=0,
):
    aggregator = _StatsAggregator(
        window_duration=aggregation_duration,
        dump_duration=1,
        expected_gen_tokens=request_builder.max_tokens,
        clients=max_concurrency,
        json_output=json_output,
        log_request_content=log_request_content,
        network_latency_adjustment=network_latency_adjustment,
    )
    requester = OAIRequester(api_key, url, backoff=backoff)

    async def request_func(session: aiohttp.ClientSession):
        
        nonlocal aggregator
        nonlocal requester
        request_body, messages_tokens = request_builder.__next__()
        aggregator.record_new_request()

        stats = await requester.call(session, request_body)
        stats.context_tokens = messages_tokens
        try:
            aggregator.aggregate_request(stats)
        except Exception as e:
            logging.error(f"Error in {e}")

    def finish_run_func():
        """Function to run when run is finished."""
        nonlocal aggregator
        aggregator.dump_raw_call_stats()

    executer = AsyncHTTPExecuter(
        request_func,
        rate_limiter=rate_limiter,
        max_concurrency=max_concurrency,
        finish_run_func=finish_run_func,
    )
    aggregator.start()

    logging.info("Started aggregator")

    executer.run(
        call_count=request_count,
        duration=duration,
        run_end_condition_mode=run_end_condition_mode,
    )

    logging.info("Called executor")

    aggregator.stop()

    logging.info("finished load test")


def _validate(args: dict):
    if not args.get("api_version"):
        raise ValueError("api-version is required")
    if not args.get("api_key"):
        raise ValueError("api-key is required")
    if args.get("clients", 0) < 1:
        raise ValueError("clients must be > 0")
    if args.get("requests") is not None and args.get("requests") < 0:
        raise ValueError("requests must be > 0")
    if (
        args.get("duration") is not None
        and args.get("duration") != 0
        and args.get("duration") < 30
    ):
        raise ValueError("duration must be > 30")
    if args.get("run_end_condition_mode") not in ("and", "or"):
        raise ValueError("run-end-condition-mode must be one of: ['and', 'or']")
    if args.get("rate") is not None and args.get("rate") < 0:
        raise ValueError("rate must be > 0")
    if args.get("context_generation_method") == "replay":
        if not args.get("replay_path"):
            raise ValueError(
                "replay-path is required when context-generation-method=replay"
            )
    if args.get("context_generation_method") == "generate":
        if (
            args.get("shape_profile") == "custom"
            and args.get("context_tokens", 0) < 1
        ):
            raise ValueError("context-tokens must be specified with shape=custom")
        if args.get("shape_profile") == "custom":
            if args.get("context_tokens", 0) < 1:
                raise ValueError("context-tokens must be specified with shape=custom")
    if args.get("max_tokens") is not None and args.get("max_tokens") < 0:
        raise ValueError("max-tokens must be > 0")
    if args.get("completions", 0) < 1:
        raise ValueError("completions must be > 0")
    if args.get("frequency_penalty") is not None and (
        args.get("frequency_penalty") < -2 or args.get("frequency_penalty") > 2
    ):
        raise ValueError("frequency-penalty must be between -2.0 and 2.0")
    if args.get("presence_penalty") is not None and (
        args.get("presence_penalty") < -2 or args.get("presence_penalty") > 2
    ):
        raise ValueError("presence-penalty must be between -2.0 and 2.0")
    if args.get("temperature") is not None and (
        args.get("temperature") < 0 or args.get("temperature") > 2
    ):
        raise ValueError("temperature must be between 0 and 2.0")


def measure_avg_ping(url: str, num_requests: int = 5, max_time: int = 5):
    """Measures average network latency for a given URL by sending multiple ping requests."""
    ping_url = urlsplit(url).netloc
    latencies = []
    latency_test_start_time = time.time()
    while len(latencies) < num_requests and time.time() < latency_test_start_time + max_time:
        delay = ping(ping_url, timeout=5)
        latencies.append(delay)
        if delay < 0.5:  # Ensure at least 0.5 seconds between requests
            time.sleep(0.5 - delay)
    # exclude first request if needed, but here we just average all
    avg_latency = round(sum(latencies) / len(latencies), 2)
    return avg_latency
