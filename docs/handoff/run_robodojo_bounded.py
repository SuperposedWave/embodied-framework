"""Supervised independent M+N native RoboDojo trial, not formal acceptance.

Only this diagnostic's processes are supervised. Model and simulator select
their own environments; source-native controls/score and Runner stay unchanged.
Native logs are captured in memory, never saved or forwarded verbatim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
    parser.add_argument("--policy", choices=("openwam_arx", "pi05_arx"), required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--port", type=int, default=8946)
    parser.add_argument("--capture-native-exchanges", action="store_true")
    parser.add_argument("--wall-time-s", type=float, help="Explicit existing cooperative wall budget; no Runner timeout added")
    parser.add_argument("--render-gpu", type=int, default=0, help="Physical GPU for this Benchmark's application process only")
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        from embodied_eval.config import load_config
        from tests.legacy_support.frozen.mn.runner import run
        try:
            if args.capture_native_exchanges:
                from record_arx_policy_exchange import record_exchanges
                with record_exchanges(args.output.parent / "native-exchanges"):
                    return run(load_config(args.child), args.output).exit_code
            return run(load_config(args.child), args.output).exit_code
        except Exception as error:
            write(args.output.parent / "client-exception.json", {"exception_chain": public_exception(error)})
            return 1
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.accept_runtime_eula or not 1 <= args.steps <= 800 or not 1 <= args.port <= 65535:
        raise ValueError("project-only EULA approval and source-bounded action budget required")
    if not 0 <= args.render_gpu <= 7 or args.wall_time_s is not None and not 1 <= args.wall_time_s <= 14400:
        raise ValueError("explicit bounded diagnostic resource selection required")
    wall_budget = args.wall_time_s if args.wall_time_s is not None else (3600. if args.steps == 800 else 900.)
    if args.capture_native_exchanges and args.policy != "pi05_arx":
        raise ValueError("source-native socket exchange diagnostic applies to ARX joint policy only")
    preparation = args.isaac_libraries.resolve()
    preparation.relative_to(ROOT / "outputs")
    libraries = json.loads((preparation / "receipt.json").read_text())
    if not libraries["passed"]:
        raise ValueError("private shared-library preparation required")
    for row in libraries["libraries"]:
        if hashlib.sha256((preparation / row["file"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("private shared-library bytes changed")
    from prepare_mn_extension import configurations
    from embodied_eval.codec import to_data
    from tests.legacy_support.frozen.mn.preflight import check
    cfg = configurations()[args.policy + "_robodojo"]
    options = dict(cfg.model.options)
    if args.policy == "openwam_arx":
        options.update(runtime_python=str(WORKSPACE / "upstream/envs/openwam_arx-mn-native/bin/python"), runtime_gpu=4)
    else:
        options["port"] = args.port
    cfg = replace(cfg, run_id=args.policy.replace("_", "-") + "-robodojo-native-trial",
        model=replace(cfg.model, options=options),
        benchmark=replace(cfg.benchmark, options=dict(cfg.benchmark.options, device="cuda:0")),
        episodes=tuple(replace(ep, budgets=replace(ep.budgets, env_steps=args.steps, model_calls=args.steps,
            wall_time_s=wall_budget)) for ep in cfg.episodes))
    structural = check(cfg)
    write(args.output / "preflight.json", structural)
    if not structural["structurally_compatible"]:
        raise ValueError("independent component data wiring rejected")
    config_path = args.output / "config.json"
    write(config_path, to_data(cfg))
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    client_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.render_gpu), OMNI_KIT_ACCEPT_EULA="YES")
    directory = preparation / "library-root/usr/lib/x86_64-linux-gnu"
    client_env["LD_LIBRARY_PATH"] = str(directory) + (":" + client_env["LD_LIBRARY_PATH"] if client_env.get("LD_LIBRARY_PATH") else "")
    record = {"kind": "source_bounded_real_robodojo_mn_trial_not_formal_acceptance", "passed": False,
        "policy": args.policy, "native_task": "stack_bowls", "layout_collection_seed": 0, "layout_index": 0,
        "action_budget": args.steps, "full_native_action_limit_requested": args.steps == 800,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": False,
        "physics_device": "source-default-cpu", "render_gpu": args.render_gpu, "model_gpu": 4 if args.policy == "openwam_arx" else 2,
        "cooperative_wall_budget_s": wall_budget,
        "runtime_eula_project_process_only": True, "global_proxy_modified": False, "shared_driver_modified": False,
        "occupancy_process_changed": False, "raw_logs_saved": False, "formal_acceptance": False,
        "runner_hard_timeout_retry_recovery_or_concurrency_added": False}
    server = client = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        logs = None
        try:
            if args.policy == "pi05_arx":
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", args.port))
                model_env = dict(environment, CUDA_VISIBLE_DEVICES="2", XLA_PYTHON_CLIENT_MEM_FRACTION="0.6")
                command = [str(WORKSPACE / "upstream/envs/pi05-arx-mn-inference/bin/python"),
                    str(ROOT / "docs/handoff/serve_pi05_arx_native.py"), "--port", str(args.port), "--checkpoint", options["checkpoint_dir"]]
                server = subprocess.Popen(command, cwd=ROOT, env=model_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                logs = pool.submit(server.communicate)
                deadline = time.monotonic() + 600
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("native_policy_server_start_failed")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("diagnostic_native_server_start_deadline")
                        time.sleep(1)
                record["native_server_started"] = True
            command = [str(WORKSPACE / "upstream/envs/robodojo-mn-native/bin/python"), __file__, "--child", str(config_path),
                "--policy", args.policy, "--output", str(args.output / "framework"), "--isaac-libraries", str(preparation)]
            if args.capture_native_exchanges:
                command.append("--capture-native-exchanges")
            client = subprocess.Popen(command, cwd=ROOT, env=client_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            out, err = client.communicate(timeout=wall_budget + 600 if args.wall_time_s is not None else (3900 if args.steps == 800 else 1200))
            record.update(client_exit_code=client.returncode, client_public_log_categories=categories((out + err).decode(errors="replace")))
            summary = args.output / "framework/summary.json"
            if summary.is_file():
                value = json.loads(summary.read_text())
                record.update(framework_errors=value["errors"], completed=value["completed"], summary=str(summary.relative_to(ROOT)),
                    passed=client.returncode == 0 and value["errors"] == 0 and value["completed"] == 1)
                episode = args.output / "framework/episodes/ep-0"
                result = json.loads((episode / "result.json").read_text())
                diagnostic = json.loads((episode / "diagnostics.internal.json").read_text())
                record.update(execution_statistics=result["execution_statistics"], metrics=result["metrics"],
                    termination_reason=result["termination_reason"], adapter_diagnostics=diagnostic)
        except Exception as error:
            record["exception_chain"] = public_exception(error)
        finally:
            stop_owned(client)
            stop_owned(server)
            if logs is not None:
                out, err = logs.result(timeout=15)
                record["server_public_log_categories"] = categories((out + err).decode(errors="replace"))
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", record)
    print(json.dumps({key: record.get(key) for key in ("passed", "policy", "client_exit_code", "framework_errors", "completed", "execution_statistics", "metrics")}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
