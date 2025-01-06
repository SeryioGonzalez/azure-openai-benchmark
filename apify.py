import logging
import os
import sys
import threading

from datetime import datetime
from flask import Flask, request, jsonify
from prometheus_client import start_http_server

from benchmark.loadcmd import load

API_PORT = 8000
PROMETHEUS_PORT = 8001

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout  # Send logs to stdout for Docker
)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global references to the active load process and aggregator
load_process = None
current_aggregator = None

def run_load_async(args):
    """
    Runs the load function asynchronously in a separate thread.
    """
    global load_status
    load_status = {"running": True, "message": "Load test running"}
    
    try:
        load(args)
        load_status = {"running": False, "message": "Load test completed"}
    except Exception as e:
        logger.error(f"Error in load: {e}")
        load_status = {"running": False, "message": f"Load test failed: {e}"}


@app.route("/status", methods=["GET"])
def check_status():
    logger.info("Status check")
    return "OK"

@app.route("/load", methods=["POST"])
def run_load():
    global load_process, current_aggregator  # Ensure we can modify the global variables

    # 1) If there's a running load process, terminate it
    if load_process and load_process.poll() is None:
        logger.info("Terminating previous load process...")
        load_process.terminate()
        load_process = None

    # 2) If there's a running aggregator, stop it
    if current_aggregator is not None:
        logger.info("Stopping previous aggregator...")
        current_aggregator.stop()
        current_aggregator = None

    args = request.json
    logger.info("Received args: %s", args)

    # Validate required arguments
    required_fields = ["api_base_endpoint", "deployment", "api_key"]
    for field in required_fields:
        if field not in args:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Set default values for optional arguments
    args.setdefault("api_version", "2023-05-15")
    args.setdefault("clients", 20)
    args.setdefault("run_end_condition_mode", "or")
    args.setdefault("context_generation_method", "generate")
    args.setdefault("shape_profile", "balanced")
    args.setdefault("completions", 1)
    args.setdefault("prevent_server_caching", True)
    args.setdefault("retry", "none")
    args.setdefault("output_format", "jsonl")
    args.setdefault("log_request_content", False)

    logger.info("Defaulted args: %s", args)

    # Handle logging directory if provided
    if args.get("log_save_dir"):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if args["context_generation_method"] == "generate":
            if args["shape_profile"] == "custom":
                token_config_str = (f"shape={args['shape_profile']}_"
                                    f"context-tokens={args.get('context_tokens')}_"
                                    f"max-tokens={args.get('max_tokens')}")
            else:
                token_config_str = f"shape={args['shape_profile']}"
        else:
            replay_base = os.path.basename(args['replay_path']).split('.')[0]
            token_config_str = f"replay-basename={replay_base}_max-tokens={args.get('max_tokens')}"

        rate_str = str(int(args.get("rate", 0))) if args.get("rate") else 'none'
        output_path = os.path.join(args["log_save_dir"],
                                   f"{now}_{args['deployment']}_{token_config_str}_"
                                   f"clients={int(args['clients'])}_rate={rate_str}.log")
        os.makedirs(args["log_save_dir"], exist_ok=True)
        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass

    # 3) Start a new load
    try:
        # Start the load in a separate thread
        load_thread = threading.Thread(target=run_load_async, args=(args,), daemon=True)
        load_thread.start()

        return jsonify({"status": "Load generation started"}), 202

    except Exception as e:
        logger.error("Error while starting load: %s", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting prometheus on port %s", PROMETHEUS_PORT)
    start_http_server(PROMETHEUS_PORT)

    logger.info("Starting API on port %s", API_PORT)
    app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
    
