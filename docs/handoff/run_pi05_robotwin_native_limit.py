"""Owned ARX/RoboTwin full native action-limit diagnostic, not acceptance.

The source specifies 400 actions for adjust_bottle. One serial Runner is used
without new timeout/retry/recovery features. A separate policy process avoids
touching RNG/cache state of any other live trial. Native logs stay in memory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time

from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_cosmos_droid_bounded import stop_owned

sys.path.insert(0, str(ROOT))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-gpu", type=int, default=5)
    parser.add_argument("--simulator-gpu", type=int, default=7)
    parser.add_argument("--port", type=int, default=8947)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if args.child:
        from embodied_eval.config import load_config
        from embodied_eval.mn.runner import run
        from record_arx_policy_exchange import record_exchanges
        try:
            with record_exchanges(args.output.parent / "native-exchanges"):
                return run(load_config(args.child), args.output).exit_code
        except Exception as error:
            write(args.output.parent / "client-exception.json", {"exception_chain": public_exception(error)})
            return 1
    if not (0 <= args.server_gpu <= 7 and 0 <= args.simulator_gpu <= 7
            and args.server_gpu != args.simulator_gpu and 1 <= args.port <= 65535):
        raise ValueError("distinct_explicit_local_GPU_and_port_required")
    from prepare_mn_extension import configurations
    from embodied_eval.mn.preflight import check
    from embodied_eval.codec import to_data
    cfg = configurations()["pi05_arx_robotwin"]
    repo = Path(cfg.benchmark.options["repo"])
    # Registries verify source/assets again before constructing either component.
    # Here use the actual pinned task limit, not the example's generic 600 cap.
    source = repo / "task_config/_eval_step_limit.yml"
    content = source.read_text()
    match = re.search(r"^adjust_bottle:\s*(\d+)\s*$", content, re.MULTILINE)
    if not match or int(match.group(1)) != 400:
        raise ValueError("reviewed_native_adjust_bottle_limit_changed")
    cfg = replace(cfg, run_id="pi05-arx-robotwin-full-native-limit-diagnostic",
        model=replace(cfg.model, options=dict(cfg.model.options, port=args.port)),
        episodes=tuple(replace(ep, budgets=replace(ep.budgets, env_steps=400, model_calls=400,
            wall_time_s=7200.)) for ep in cfg.episodes))
    args.output.mkdir(parents=True, exist_ok=False)
    structural = check(cfg)
    write(args.output / "preflight.json", structural)
    if not structural["structurally_compatible"]:
        raise ValueError("independent_component_data_wiring_rejected")
    config_path = args.output / "config.json"
    write(config_path, to_data(cfg))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    model_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.server_gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.6")
    sim_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.simulator_gpu),
        CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"))
    egl = WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    if egl.is_file():
        sim_env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    model_command = [str(WORKSPACE / "upstream/envs/pi05-arx-mn-inference/bin/python"),
        str(ROOT / "docs/handoff/serve_pi05_arx_native.py"), "--port", str(args.port),
        "--checkpoint", cfg.model.options["checkpoint_dir"]]
    client_command = [str(WORKSPACE / "upstream/envs/robotwin-stage4/bin/python"), __file__,
        "--child", str(config_path), "--output", str(args.output / "framework")]
    receipt = {"kind": "source_limit_real_pi05_robotwin_mn_trial_not_formal_acceptance", "passed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": False,
        "task": "adjust_bottle", "task_config": "demo_clean", "seed_index": 0,
        "source_action_limit": 400, "action_budget": 400, "cooperative_wall_budget_s": 7200.,
        "native_limit_source_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "server_gpu": args.server_gpu, "simulator_gpu": args.simulator_gpu,
        "server_process_owned_and_not_reused": True, "other_trial_RNG_or_cache_touched": False,
        "native_logs_saved": False, "shared_driver_or_global_proxy_changed": False,
        "occupancy_process_changed": False, "formal_acceptance": False,
        "runner_hard_timeout_retry_recovery_or_concurrency_added": False}
    server = client = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        server_logs = None
        try:
            server = subprocess.Popen(model_command, cwd=ROOT, env=model_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            server_logs = pool.submit(server.communicate)
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("owned_native_server_start_failed")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("owned_diagnostic_server_start_bound")
                    time.sleep(1)
            receipt["owned_native_server_started"] = True
            client = subprocess.Popen(client_command, cwd=ROOT, env=sim_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            out, err = client.communicate(timeout=7800)
            receipt.update(client_exit_code=client.returncode,
                client_public_log_categories=categories((out + err).decode(errors="replace")))
            summary_path = args.output / "framework/summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                ep = args.output / "framework/episodes/ep-0"
                result = json.loads((ep / "result.json").read_text())
                receipt.update(framework_errors=summary["errors"], completed=summary["completed"],
                    execution_statistics=result["execution_statistics"], metrics=result["metrics"],
                    termination_reason=result["termination_reason"],
                    adapter_diagnostics=json.loads((ep / "diagnostics.internal.json").read_text()),
                    passed=client.returncode == 0 and summary["errors"] == 0 and summary["completed"] == 1)
        except Exception as error:
            receipt["exception_chain"] = public_exception(error)
        finally:
            stop_owned(client)
            stop_owned(server)
            if server_logs is not None:
                out, err = server_logs.result(timeout=15)
                receipt["server_public_log_categories"] = categories((out + err).decode(errors="replace"))
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("passed", "client_exit_code", "framework_errors", "execution_statistics", "metrics")}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
