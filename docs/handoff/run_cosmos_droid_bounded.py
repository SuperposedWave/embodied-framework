"""Owned four-action M+N runtime diagnostic; no retries or formal acceptance.

Native logs stay in memory. Model and simulator run in their own environments;
the existing Runner and independent adapters execute the real public pathway.
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
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
from prepare_mn_extension import configurations
from probe_mn_components import write, public_exception, categories


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--port", type=int, default=8940)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child:
        from embodied_eval.config import load_config
        from tests.legacy_support.frozen.mn.runner import run
        try:
            result = run(load_config(args.child), args.output)
            return result.exit_code
        except Exception as error:
            write(args.output.parent / "client-exception.json", {"exception_chain": public_exception(error)})
            return 1
    if not args.accept_runtime_eula or not 1 <= args.port <= 65535:
        raise ValueError("explicit project-only EULA approval and valid loopback port required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (args.vae, args.isaac_libraries):
        path.resolve().relative_to(ROOT / "outputs")
        if not json.loads((path / "receipt.json").read_text())["passed"]:
            raise ValueError("native asset/library preparation incomplete")
    library_receipt = json.loads((args.isaac_libraries / "receipt.json").read_text())
    for item in library_receipt["libraries"]:
        if hashlib.sha256((args.isaac_libraries / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("private library changed")
    cfg = configurations()["cosmos3_droid_sim"]
    cfg = replace(cfg, run_id="cosmos3-droid-four-action-diagnostic",
        model=replace(cfg.model, options=dict(cfg.model.options, port=args.port)),
        benchmark=replace(cfg.benchmark, options=dict(cfg.benchmark.options, device="cuda:0")),
        episodes=tuple(replace(ep, budgets=replace(ep.budgets, env_steps=4, model_calls=4, wall_time_s=600.)) for ep in cfg.episodes))
    from embodied_eval.codec import to_data
    from tests.legacy_support.frozen.mn.preflight import check
    structural = check(cfg)
    write(args.output / "preflight.json", structural)
    if not structural["structurally_compatible"]:
        raise ValueError("explicit data wiring rejected")
    config_path = args.output / "config.json"
    write(config_path, to_data(cfg))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    model_output = args.output / "native-server"
    model_output.mkdir()
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    libraries = WORKSPACE / "upstream/envs/cosmos3-mn-native/lib/python3.13/site-packages/nvidia"
    model_env = dict(environment, CUDA_VISIBLE_DEVICES="4", CUDA_HOME=str(libraries),
                     NVTE_CUDA_INCLUDE_DIR=str(libraries / "cuda_runtime/include"))
    sim_env = dict(environment, CUDA_VISIBLE_DEVICES="0", OMNI_KIT_ACCEPT_EULA="YES")
    private_lib = args.isaac_libraries.resolve() / "library-root/usr/lib/x86_64-linux-gnu"
    sim_env["LD_LIBRARY_PATH"] = str(private_lib) + (":" + sim_env["LD_LIBRARY_PATH"] if sim_env.get("LD_LIBRARY_PATH") else "")
    server_command = [str(WORKSPACE / "upstream/envs/cosmos3-mn-native/bin/python"),
        str(ROOT / "docs/handoff/probe_mn_components.py"), "--child", "--component", "cosmos3", "--mode", "cosmos-server",
        "--cosmos-native-vae", str(args.vae.resolve()), "--port", str(args.port), "--output", str(model_output)]
    client_command = [str(WORKSPACE / "upstream/envs/droid_sim-mn-native/bin/python"), __file__,
        "--child", str(config_path), "--output", str(args.output / "framework"),
        "--vae", str(args.vae.resolve()), "--isaac-libraries", str(args.isaac_libraries.resolve())]
    receipt = {"kind": "bounded_real_cosmos_droid_mn_trial_not_full_task_or_formal_acceptance",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False, "synthetic": False,
        "action_budget": 4, "server_gpu": 4, "simulator_gpu": 0, "official_task_success_scorer_available": False,
        "runtime_eula_project_process_only": True, "global_proxy_modified": False,
        "shared_driver_modified": False, "occupancy_process_changed": False, "raw_logs_persisted": False,
        "runner_recovery_or_concurrency_added": False, "formal_acceptance": False}
    server = client = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            server = subprocess.Popen(server_command, cwd=ROOT, env=model_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            logs = pool.submit(server.communicate)
            deadline = time.monotonic() + 600
            while not (model_output / "server-ready.json").is_file():
                if server.poll() is not None:
                    raise RuntimeError("native_server_start_failed")
                if time.monotonic() > deadline:
                    raise TimeoutError("bounded_server_start_deadline")
                time.sleep(1)
            # Ready receipt means weights loaded; additionally require the
            # source-native socket to be bound before starting the simulator.
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                        break
                except OSError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise TimeoutError("native_server_socket_deadline")
                    time.sleep(1)
            receipt["native_server_started"] = True
            client = subprocess.Popen(client_command, cwd=ROOT, env=sim_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            stdout, stderr = client.communicate(timeout=600)
            receipt.update(client_exit_code=client.returncode,
                client_public_log_categories=categories((stdout + stderr).decode("utf-8", errors="replace")))
            summary_path = args.output / "framework/summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                receipt.update(summary=str(summary_path.relative_to(ROOT)), framework_errors=summary["errors"],
                    completed=summary["completed"], passed=client.returncode == 0 and summary["errors"] == 0)
            else:
                receipt["failure_class"] = "no_completed_framework_summary"
        except Exception as error:
            receipt["exception_chain"] = public_exception(error)
        finally:
            stop_owned(client)
            stop_owned(server)
            if server is not None:
                stdout, stderr = logs.result(timeout=15)
                receipt["server_public_log_categories"] = categories((stdout + stderr).decode("utf-8", errors="replace"))
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
