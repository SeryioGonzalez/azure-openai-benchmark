import datetime
import json
import logging
import numpy as np
import threading
import time
import traceback

from prometheus_client import start_http_server, Gauge
from typing import Optional

from .oairequester import RequestStats

logger = logging.getLogger()

class _Samples:
    def __init__(self):
        # [0] timestamp, [1] value
        self.samples: [(float, float)] = []

    def _trim_oldest(self, duration: float):
        while len(self.samples) > 0 and (time.time() - self.samples[0][0]) > duration:
            self.samples.pop(0)

    def _append(self, timestamp: float, value: float):
        self.samples.append((timestamp, value))

    def _values(self) -> [float]:
        return [entry[1] for entry in self.samples]

    def _len(self) -> int:
        return len(self.samples)


class _StatsAggregator(threading.Thread):
    """
    A thread-safe request stats aggregator that can periodically emit statistics
    and expose them via Prometheus metrics without dumping text output.
    """

    lock = threading.Lock()
    terminate: threading.Event

    start_time: float = 0
    processing_requests_count: int = 0
    total_requests_count: int = 0
    total_failed_count: int = 0
    throttled_count: int = 0

    request_timestamps = _Samples()
    request_latency = _Samples()
    call_tries = _Samples()
    response_latencies = _Samples()
    first_token_latencies = _Samples()
    token_latencies = _Samples()
    context_tokens = _Samples()
    generated_tokens = _Samples()
    utilizations = _Samples()

    raw_stat_dicts = list()

    def __init__(
        self,
        clients: int,
        dump_duration: float = 5,
        window_duration: float = 60,
        expected_gen_tokens: Optional[int] = None,
        json_output: bool = False,
        log_request_content: bool = False,
        network_latency_adjustment: float = 0,
        prometheus_output: bool = True,
        metrics_port: int = 8000,  # <--- configurable metrics port
        *args,
        **kwargs
    ):
        """
        :param clients: number of clients being used in testing.
        :param dump_duration: interval (seconds) between metric updates.
        :param window_duration: duration (seconds) of sliding window for stats.
        :param expected_gen_tokens: number of tokens expected in each response.
        :param json_output: log stats in JSON.
        :param log_request_content: include request data in raw stats.
        :param network_latency_adjustment: subtracted from latency metrics.
        :param prometheus_output: whether to expose metrics via Prometheus.
        :param metrics_port: the port on which Prometheus will scrape metrics (default 8000).
        """
        super(_StatsAggregator, self).__init__(*args, **kwargs)

        self.clients = clients
        self.dump_duration = dump_duration
        self.window_duration = window_duration
        self.expected_gen_tokens = expected_gen_tokens
        self.json_output = json_output
        self.log_request_content = log_request_content
        self.network_latency_adjustment = network_latency_adjustment
        self.prometheus_output = prometheus_output
        self.metrics_port = metrics_port

        # ---------------------------
        # PROMETHEUS GAUGES
        # ---------------------------
        self.g_requests_total = Gauge(
            "azure_openai_requests_total",
            "Number of completed requests"
        )
        self.g_requests_failed = Gauge(
            "azure_openai_requests_failed_total",
            "Number of failed requests"
        )
        self.g_requests_throttled = Gauge(
            "azure_openai_requests_throttled_total",
            "Number of throttled requests (HTTP 429)"
        )
        self.g_requests_per_minute = Gauge(
            "azure_openai_requests_per_minute",
            "Requests-per-minute in the sliding window"
        )
        self.g_e2e_latency_avg = Gauge(
            "azure_openai_e2e_latency_avg",
            "Average E2E latency in the sliding window"
        )
        self.g_e2e_latency_95th = Gauge(
            "azure_openai_e2e_latency_95th",
            "95th percentile E2E latency"
        )
        self.g_processing_requests_count = Gauge(
            "azure_openai_processing_requests_count",
            "Number of requests currently being processed"
        )
        self.g_context_per_minute = Gauge(
            "azure_openai_context_per_minute",
            "Context tokens used per minute in the window"
        )
        self.g_gen_per_minute = Gauge(
            "azure_openai_gen_per_minute",
            "Generated tokens per minute in the window"
        )
        self.g_tokens_per_minute = Gauge(
            "azure_openai_tokens_per_minute",
            "Total tokens (context + generated) per minute in the window"
        )
        self.g_context_tpr_avg = Gauge(
            "azure_openai_context_tpr_avg",
            "Average context tokens per response"
        )
        self.g_gen_tpr_avg = Gauge(
            "azure_openai_gen_tpr_avg",
            "Average generated tokens per response"
        )
        self.g_gen_tpr_10th = Gauge(
            "azure_openai_gen_tpr_10th",
            "10th percentile of generated tokens per response"
        )
        self.g_gen_tpr_90th = Gauge(
            "azure_openai_gen_tpr_90th",
            "90th percentile of generated tokens per response"
        )
        self.g_ttft_avg = Gauge(
            "azure_openai_ttft_avg",
            "Average time-to-first-token"
        )
        self.g_ttft_95th = Gauge(
            "azure_openai_ttft_95th",
            "95th percentile time-to-first-token"
        )
        self.g_tbt_avg = Gauge(
            "azure_openai_tbt_avg",
            "Average time-between-tokens (post first token)"
        )
        self.g_tbt_95th = Gauge(
            "azure_openai_tbt_95th",
            "95th percentile time-between-tokens"
        )
        self.g_util_avg = Gauge(
            "azure_openai_util_avg",
            "Average utilization"
        )
        self.g_util_95th = Gauge(
            "azure_openai_util_95th",
            "95th percentile utilization"
        )

    def dump_raw_call_stats(self):
        """Dumps raw stats for each individual call within the aggregation window."""
        logger.info(f"Raw call stats: {json.dumps(self.raw_stat_dicts)}")

    def run(self):
        """
        Start the periodic aggregator thread. If prometheus_output is True,
        we start the Prometheus HTTP server so it can scrape these Gauges.
        """
        if self.prometheus_output:
            logger.info(f"Starting Prometheus metrics server on port {self.metrics_port}")
            start_http_server(self.metrics_port)

        self.start_time = time.time()
        self.terminate = threading.Event()
        while not self.terminate.wait(self.dump_duration):
            self._dump()
            self._slide_window()

    def stop(self):
        self.terminate.set()
        # Dump one more time to capture final stats
        self._dump()

    def record_new_request(self):
        """
        Increments the number of currently processing requests.
        """
        with self.lock:
            self.processing_requests_count += 1

    def aggregate_request(self, stats: RequestStats):
        """
        Aggregates stats for a completed request in the sliding window.
        """
        with self.lock:
            try:
                self.processing_requests_count -= 1
                self.total_requests_count += 1
                self.call_tries._append(stats.request_start_time, stats.calls)

                if stats.response_status_code != 200:
                    self.total_failed_count += 1
                    if stats.response_status_code == 429:
                        self.throttled_count += 1
                else:
                    request_latency = (
                        stats.response_end_time
                        - stats.request_start_time
                        - self.network_latency_adjustment
                    )
                    self.request_latency._append(stats.request_start_time, request_latency)

                    # Warn if request_latency >> window
                    if request_latency > self.window_duration:
                        logging.warning(
                            f"request completed in {round(request_latency, 2)}s, "
                            f"exceeds window {round(self.window_duration, 2)}s; "
                            "consider increasing aggregation-window."
                        )

                    # Add other latencies
                    self.request_timestamps._append(stats.request_start_time, stats.request_start_time)
                    self.response_latencies._append(
                        stats.request_start_time,
                        stats.response_time - stats.request_start_time - self.network_latency_adjustment
                    )
                    self.first_token_latencies._append(
                        stats.request_start_time,
                        stats.first_token_time - stats.request_start_time - self.network_latency_adjustment
                    )
                    self.token_latencies._append(
                        stats.request_start_time,
                        (stats.response_end_time - stats.first_token_time - self.network_latency_adjustment)
                        / stats.generated_tokens
                    )
                    self.context_tokens._append(stats.request_start_time, stats.context_tokens)
                    self.generated_tokens._append(stats.request_start_time, stats.generated_tokens)

                if stats.deployment_utilization is not None:
                    self.utilizations._append(stats.request_start_time, stats.deployment_utilization)

            except Exception as e:
                exc_str = '\n'.join(traceback.format_exc().splitlines()[-3:])
                logging.error(f"error while aggregating request stats: {exc_str}")

            # Save raw stat for the call
            self.raw_stat_dicts.append(stats.as_dict(include_request_content=self.log_request_content))

    def _dump(self):
        with self.lock:
            run_seconds = round(time.time() - self.start_time)
            dynamic_window = min(run_seconds, self.window_duration)

            # Basic metrics
            e2e_latency_avg = (
                round(np.average(self.request_latency._values()), 3)
                if self.request_latency._len() > 0 else "n/a"
            )
            e2e_latency_95th = (
                round(np.percentile(self.request_latency._values(), 95), 3)
                if self.request_latency._len() > 1 else "n/a"
            )
            context_per_minute = (
                round(60.0 * np.sum(self.context_tokens._values()) / dynamic_window, 0)
                if self.context_tokens._len() > 0 else "n/a"
            )
            gen_per_minute = (
                round(60.0 * np.sum(self.generated_tokens._values()) / dynamic_window, 0)
                if self.generated_tokens._len() > 0 else "n/a"
            )

            tokens_per_minute = 0
            if context_per_minute != "n/a":
                tokens_per_minute += context_per_minute
            if gen_per_minute != "n/a":
                tokens_per_minute += gen_per_minute

            context_tpr_avg = (
                int(np.sum(self.context_tokens._values()) / self.context_tokens._len())
                if self.context_tokens._len() > 0 else "n/a"
            )
            gen_tpr_avg = (
                int(np.sum(self.generated_tokens._values()) / self.generated_tokens._len())
                if self.generated_tokens._len() > 0 else "n/a"
            )
            gen_tpr_10th = (
                int(np.percentile(self.generated_tokens._values(), 10))
                if self.generated_tokens._len() > 1 else "n/a"
            )
            gen_tpr_90th = (
                int(np.percentile(self.generated_tokens._values(), 90))
                if self.generated_tokens._len() > 1 else "n/a"
            )
            ttft_avg = (
                round(np.average(self.first_token_latencies._values()), 3)
                if self.first_token_latencies._len() > 0 else "n/a"
            )
            ttft_95th = (
                round(np.percentile(self.first_token_latencies._values(), 95), 3)
                if self.first_token_latencies._len() > 1 else "n/a"
            )
            tbt_avg = (
                round(np.average(self.token_latencies._values()), 3)
                if self.token_latencies._len() > 0 else "n/a"
            )
            tbt_95th = (
                round(np.percentile(self.token_latencies._values(), 95), 3)
                if self.token_latencies._len() > 1 else "n/a"
            )
            util_avg = (
                f"{round(np.average(self.utilizations._values()), 1)}%"
                if self.utilizations._len() > 0 else "n/a"
            )
            util_95th = (
                f"{round(np.percentile(self.utilizations._values(), 95), 1)}%"
                if self.utilizations._len() > 1 else "n/a"
            )
            rpm = (
                round(60.0 * self.request_timestamps._len() / dynamic_window, 1)
                if self.request_timestamps._len() > 0 else "n/a"
            )

            # Periodic warning if generated tokens are too low
            warning_period_secs = 10
            if all((
                run_seconds % warning_period_secs == 0,
                self.expected_gen_tokens is not None,
                isinstance(gen_tpr_avg, int)
            )) and gen_tpr_avg < 0.9 * self.expected_gen_tokens:
                logging.warning(
                    f"average tokens per response is {gen_tpr_avg}, "
                    f"compared to requested max_tokens of {self.expected_gen_tokens}. "
                    "This may mean measured RPM is higher/faster than real-world workloads."
                )

            processing_requests_count = min(self.clients, self.processing_requests_count)

            # ---------------------------
            # LOG (JSON or human-readable)
            # ---------------------------
            if self.json_output:
                j = {
                    "run_seconds": run_seconds,
                    "rpm": rpm,
                    "processing": processing_requests_count,
                    "completed": self.total_requests_count,
                    "failures": self.total_failed_count,
                    "throttled": self.throttled_count,
                    "requests": self.total_requests_count,
                    "tpm": {
                        "context": context_per_minute,
                        "gen": gen_per_minute,
                        "total": tokens_per_minute,
                    },
                    "e2e": {
                        "avg": e2e_latency_avg,
                        "95th": e2e_latency_95th,
                    },
                    "ttft": {
                        "avg": ttft_avg,
                        "95th": ttft_95th,
                    },
                    "tbt": {
                        "avg": tbt_avg,
                        "95th": tbt_95th,
                    },
                    "context_tpr_avg": context_tpr_avg,
                    "gen_tpr": {
                        "10th": gen_tpr_10th,
                        "avg": gen_tpr_avg,
                        "90th": gen_tpr_90th,
                    },
                    "util": {
                        "avg": util_avg,
                        "95th": util_95th,
                    },
                }
                logger.info(json.dumps(j))
            else:
                logger.info(
                    f"rpm: {rpm:<5} processing: {processing_requests_count:<4} "
                    f"completed: {self.total_requests_count:<5} failures: {self.total_failed_count:<4} "
                    f"throttled: {self.throttled_count:<4} requests: {self.total_requests_count:<5} "
                    f"tpm: {tokens_per_minute:<6} ttft_avg: {ttft_avg:<6} ttft_95th: {ttft_95th:<6} "
                    f"tbt_avg: {tbt_avg:<6} tbt_95th: {tbt_95th:<6} e2e_avg: {e2e_latency_avg:<6} "
                    f"e2e_95th: {e2e_latency_95th:<6} context_tpr_avg {context_tpr_avg:<4} "
                    f"gen_tpr_10th {gen_tpr_10th:<4} gen_tpr_avg {gen_tpr_avg:<4} "
                    f"gen_tpr_90th {gen_tpr_90th:<4} util_avg: {util_avg:<6} "
                    f"util_95th: {util_95th:<6}"
                )

            # ---------------------------
            # SET PROMETHEUS GAUGE VALUES
            # ---------------------------
            if self.prometheus_output:
                # Basic counters
                self.g_requests_total.set(self.total_requests_count)
                self.g_requests_failed.set(self.total_failed_count)
                self.g_requests_throttled.set(self.throttled_count)

                # Additional concurrency & tokens metrics
                self.g_processing_requests_count.set(processing_requests_count)

                rpm_val = rpm if rpm != "n/a" else 0
                self.g_requests_per_minute.set(rpm_val)

                e2e_avg_val = e2e_latency_avg if e2e_latency_avg != "n/a" else 0
                self.g_e2e_latency_avg.set(e2e_avg_val)

                e2e_95th_val = e2e_latency_95th if e2e_latency_95th != "n/a" else 0
                self.g_e2e_latency_95th.set(e2e_95th_val)

                cpm_val = context_per_minute if context_per_minute != "n/a" else 0
                self.g_context_per_minute.set(cpm_val)

                gpm_val = gen_per_minute if gen_per_minute != "n/a" else 0
                self.g_gen_per_minute.set(gpm_val)

                tpm_val = tokens_per_minute if tokens_per_minute != 0 else 0
                self.g_tokens_per_minute.set(tpm_val)

                ctx_tpr_val = context_tpr_avg if context_tpr_avg != "n/a" else 0
                self.g_context_tpr_avg.set(ctx_tpr_val)

                gen_tpr_val = gen_tpr_avg if gen_tpr_avg != "n/a" else 0
                self.g_gen_tpr_avg.set(gen_tpr_val)

                gen_tpr_10 = gen_tpr_10th if gen_tpr_10th != "n/a" else 0
                self.g_gen_tpr_10th.set(gen_tpr_10)

                gen_tpr_90 = gen_tpr_90th if gen_tpr_90th != "n/a" else 0
                self.g_gen_tpr_90th.set(gen_tpr_90)

                ttft_val = ttft_avg if ttft_avg != "n/a" else 0
                self.g_ttft_avg.set(ttft_val)

                ttft95_val = ttft_95th if ttft_95th != "n/a" else 0
                self.g_ttft_95th.set(ttft95_val)

                tbt_val = tbt_avg if tbt_avg != "n/a" else 0
                self.g_tbt_avg.set(tbt_val)

                tbt95_val = tbt_95th if tbt_95th != "n/a" else 0
                self.g_tbt_95th.set(tbt95_val)

                # Util is stored as a string with a '%' if it's not 'n/a'
                # e.g., "15.3%"
                # We'll strip the '%' and convert to float. If it's "n/a", use 0.
                if util_avg != "n/a":
                    avg_num = float(util_avg.replace("%", ""))
                    self.g_util_avg.set(avg_num)
                else:
                    self.g_util_avg.set(0)

                if util_95th != "n/a":
                    p95_num = float(util_95th.replace("%", ""))
                    self.g_util_95th.set(p95_num)
                else:
                    self.g_util_95th.set(0)

    def _slide_window(self):
        with self.lock:
            self.call_tries._trim_oldest(self.window_duration)
            self.request_timestamps._trim_oldest(self.window_duration)
            self.response_latencies._trim_oldest(self.window_duration)
            self.first_token_latencies._trim_oldest(self.window_duration)
            self.token_latencies._trim_oldest(self.window_duration)
            self.context_tokens._trim_oldest(self.window_duration)
            self.generated_tokens._trim_oldest(self.window_duration)
            self.utilizations._trim_oldest(self.window_duration)
