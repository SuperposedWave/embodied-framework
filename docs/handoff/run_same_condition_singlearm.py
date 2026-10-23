"""Separate fresh framework/native servers on identical GPUs and dependencies.

Only owned diagnostic subprocesses are supervised; no Runner concurrency,
retry, recovery or public protocol changes. Native logs remain in memory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_mn_trials import server_spec, stop_owned


def client_child(args):
    import random
    import numpy as np
    from embodied_eval.config import load_config
    from tests.legacy_support.frozen.mn.runner import run
    config = load_config(args.child)
    if config.benchmark.adapter == "vlabench":
        sys.path.insert(0, config.benchmark.options["repo"])
        os.environ["VLABENCH_ROOT"] = str(Path(config.benchmark.options["repo"]) / "VLABench")
        import VLABench.robots
        import VLABench.tasks
        import VLABench.envs
    random.seed(args.setup_seed); np.random.seed(args.setup_seed)
    try:
        return run(config, args.output).exit_code
    except Exception as error:
        write(args.output.parent / "framework-client-exception.json", {"exception_chain": public_exception(error)})
        return 1


def execute(args):
    from dataclasses import replace
    from embodied_eval.codec import to_data
    from embodied_eval.config import load_config
    from tests.legacy_support.frozen.mn.preflight import check
    config = load_config(args.config)
    if config.benchmark.adapter not in ("libero", "vlabench", "robotwin"):
        raise ValueError("reviewed_native_simulator_required")
    config = replace(config, run_id=config.run_id + "-same-condition-20261009",
        model=replace(config.model, options=dict(config.model.options, port=args.port)))
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "config.json"
    write(path, to_data(config))
    structural = check(config)
    write(args.output / "preflight.json", structural)
    if not structural["structurally_compatible"]:
        raise ValueError("component_wiring_rejected")
    command, cwd, client_python = server_spec(config)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        QWEN3VL_PATH=str(WORKSPACE.parent.parent / "Model/Qwen3-VL-4B-Instruct-processor"))
    # Only the launched children bypass proxies for the loopback policy endpoint.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    server_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.model_gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.6")
    client_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.simulator_gpu), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    if config.benchmark.adapter == "robotwin":
        client_env.update(CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"))
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    receipt = {"kind": "paired_same_condition_fresh_framework_and_native_closedloops", "completed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": False,
        "model_gpu": args.model_gpu, "simulator_gpu": args.simulator_gpu, "current_gpu_hardware": hardware.stdout.strip().splitlines(),
        "model_launch": command, "model_source_cwd": str(cwd), "client_python_both_sides": str(client_python),
        "fresh_separate_policy_servers_same_command_and_child_environment": True,
        "setup_rng_seed_both_sides": args.setup_seed, "raw_native_logs_or_credentials_saved": False,
        "global_proxy_driver_or_occupancy_changed": False, "formal_acceptance": False, "sides": []}
    for side in ("framework", "native"):
        server = worker = None
        with ThreadPoolExecutor(max_workers=1) as pool:
            logs = None
            try:
                server = subprocess.Popen(command, cwd=cwd, env=server_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                logs = pool.submit(server.communicate)
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("owned_model_server_start_failed")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("diagnostic_server_start_bound")
                        time.sleep(1)
                if side == "framework":
                    invocation = [str(client_python), __file__, "--child", str(path), "--output", str(args.output / "framework"), "--setup-seed", str(args.setup_seed)]
                else:
                    script = "compare_native_robotwin_closedloop.py" if config.benchmark.adapter == "robotwin" else "compare_native_singlearm_closedloop.py"
                    invocation = [str(client_python), str(ROOT / "docs/handoff" / script), "--config", str(path),
                        "--framework", str(args.output / "framework"), "--output", str(args.output / "native"), "--setup-seed", str(args.setup_seed)]
                worker = subprocess.Popen(invocation, cwd=ROOT, env=client_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                out, err = worker.communicate(timeout=7800)
                receipt["sides"].append({"side": side, "exit_code": worker.returncode, "public_failure_categories": categories((out + err).decode(errors="replace"))})
                if worker.returncode != 0:
                    raise RuntimeError("diagnostic_side_incomplete")
                print(f"{side} closedloop complete: {args.output.name}", flush=True)
            except Exception as error:
                receipt["exception_chain"] = public_exception(error)
                break
            finally:
                stop_owned(worker); stop_owned(server)
                if logs is not None:
                    out, err = logs.result(timeout=15)
                    receipt.setdefault("server_public_failure_categories", []).append(categories((out + err).decode(errors="replace")))
    native_path = args.output / "native/native-result.json"
    if native_path.is_file():
        native = json.loads(native_path.read_text())
        receipt.update(completed=len(receipt["sides"]) == 2 and all(row["exit_code"] == 0 for row in receipt["sides"]) and native["completed"],
            strict_equal=native["strict_equal"], initial_observation_exact=native.get("initial_observation_exact"))
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("completed", "strict_equal", "initial_observation_exact")}), flush=True)
    return 0 if receipt["completed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-gpu", type=int, default=2)
    parser.add_argument("--simulator-gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=8950)
    parser.add_argument("--setup-seed", type=int, default=0)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if args.child:
        return client_child(args)
    if not args.config or args.model_gpu == args.simulator_gpu or not (0 <= args.model_gpu <= 7 and 0 <= args.simulator_gpu <= 7 and 1 <= args.port <= 65535):
        raise ValueError("reviewed_config_distinct_local_GPUs_and_port_required")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit("Retired historical execution tool. Use current 0.3 run/check or run_v03_bounded.py; historical references are test-only.")
