"""Owned real DreamZero/DROID M+N diagnostic, not full-task acceptance.

Independent native policy server and simulator processes use their own pinned
environments. Existing Runner executes serially, without new retries, recovery,
concurrency or hard timeouts. Only this diagnostic's process groups are bounded.
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
import time
from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_cosmos_droid_bounded import stop_owned
from probe_dreamzero_request import verify_preparation


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--request-proof", type=Path, required=True)
    parser.add_argument("--cuda-toolchain", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--port", type=int, default=8941)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--model-gpus", default="1,6")
    parser.add_argument("--sim-gpu", type=int, default=7)
    parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if args.child:
        from embodied_eval.config import load_config
        from embodied_eval.mn.runner import run
        from record_dreamzero_policy_exchange import record_exchanges
        try:
            with record_exchanges(args.output.parent / "native-exchanges"):
                return run(load_config(args.child), args.output).exit_code
        except Exception as error:
            write(args.output.parent / "client-exception.json", {"exception_chain": public_exception(error)})
            return 1
    selected = args.model_gpus.split(",")
    if (not args.accept_runtime_eula or not 1 <= args.port <= 65535 or not 1 <= args.steps <= 32
            or len(selected) != 2 or any(not part.isdigit() or not 0 <= int(part) <= 7 for part in selected)
            or len(set(map(int, selected))) != 2 or not 0 <= args.sim_gpu <= 7 or args.sim_gpu in map(int, selected)):
        raise ValueError("project_only_EULA_two_distinct_model_GPUs_independent_sim_GPU_and_bounded_steps_required")
    for path in (args.preparation, args.request_proof, args.cuda_toolchain, args.isaac_libraries):
        path.resolve().relative_to(ROOT / "outputs")
    request_path = args.request_proof.resolve() / "receipt.json"
    request = json.loads(request_path.read_text())
    if not request["passed"] or request["ranks_completed"] != 2:
        raise ValueError("completed_native_two_rank_request_required_before_new_simulator_trial")
    for rank in (0, 1):
        evidence = json.loads((args.request_proof / f"rank-{rank}-result.json").read_text())
        if not evidence["passed"] or not evidence["model_loaded"] or not evidence["real_model_request"] or evidence["synthetic_input"]:
            raise ValueError("real_native_request_proof_required")
    args.output.mkdir(parents=True, exist_ok=False)
    directory, cache = verify_preparation(args.preparation)
    from prepare_cuda129_toolchain import private_toolchain
    toolkit = private_toolchain(args.cuda_toolchain)
    from embodied_eval.mn.additional_catalog import verify_policy
    from prepare_mn_extension import configurations
    cfg = configurations()["dreamzero_droid_sim"]
    write(args.output / "verified-model-assets.json", verify_policy("dreamzero", cfg.model.options))
    private = args.isaac_libraries.resolve()
    libraries = json.loads((private / "receipt.json").read_text())
    if not libraries["passed"] or any(hashlib.sha256((private / item["file"]).read_bytes()).hexdigest() != item["sha256"] for item in libraries["libraries"]):
        raise ValueError("verified_project_private_simulator_libraries_required")
    cfg = replace(cfg, run_id="dreamzero-droid-bounded-real-native-trial",
        model=replace(cfg.model, options=dict(cfg.model.options, port=args.port, request_timeout_s=1200.)),
        benchmark=replace(cfg.benchmark, options=dict(cfg.benchmark.options, device="cuda:0")),
        episodes=tuple(replace(ep, budgets=replace(ep.budgets, env_steps=args.steps, model_calls=args.steps,
            wall_time_s=2400.)) for ep in cfg.episodes))
    from embodied_eval.codec import to_data
    from embodied_eval.mn.preflight import check
    structural = check(cfg)
    write(args.output / "preflight.json", structural)
    if not structural["structurally_compatible"]:
        raise ValueError("independent_public_data_wiring_rejected")
    config_path = args.output / "config.json"
    write(config_path, to_data(cfg))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    model_output = args.output / "native-server"
    model_output.mkdir()
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    model_env = dict(environment, CUDA_VISIBLE_DEVICES=args.model_gpus, HF_HUB_CACHE=str(cache),
        CUDA_HOME=str(toolkit), PATH=str(toolkit / "bin") + os.pathsep + environment.get("PATH", ""))
    sim_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.sim_gpu), OMNI_KIT_ACCEPT_EULA="YES")
    library_dir = private / "library-root/usr/lib/x86_64-linux-gnu"
    sim_env["LD_LIBRARY_PATH"] = str(library_dir) + (":" + sim_env["LD_LIBRARY_PATH"] if sim_env.get("LD_LIBRARY_PATH") else "")
    server_command = [str(WORKSPACE / "upstream/envs/dreamzero-mn-native/bin/python"), "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=2", str(ROOT / "docs/handoff/serve_dreamzero_native.py"),
        "--port", str(args.port), "--tokenizer", str(directory / "google/umt5-xxl"), "--output", str(model_output)]
    client_command = [str(WORKSPACE / "upstream/envs/droid_sim-mn-native/bin/python"), __file__,
        "--child", str(config_path), "--output", str(args.output / "framework"),
        "--preparation", str(args.preparation.resolve()), "--request-proof", str(args.request_proof.resolve()),
        "--cuda-toolchain", str(args.cuda_toolchain.resolve()), "--isaac-libraries", str(private)]
    receipt = {"kind": "bounded_real_dreamzero_droid_mn_trial_not_full_task_or_formal_acceptance",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False, "synthetic": False,
        "action_budget": args.steps, "server_gpus": list(map(int, selected)), "simulator_gpu": args.sim_gpu,
        "official_task_success_scorer_available": False, "runtime_eula_project_process_only": True,
        "global_proxy_or_shared_driver_modified": False, "occupancy_process_changed": False, "raw_logs_saved": False,
        "runner_hard_timeout_retry_recovery_or_concurrency_added": False, "formal_acceptance": False,
        "prior_native_request_receipt_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest()}
    server = client = None
    logs = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            server = subprocess.Popen(server_command, cwd=ROOT, env=model_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            logs = pool.submit(server.communicate)
            deadline = time.monotonic() + 1800
            while not (model_output / "server-ready.json").is_file():
                if server.poll() is not None:
                    raise RuntimeError("owned_native_server_start_failed")
                if time.monotonic() > deadline:
                    raise TimeoutError("diagnostic_server_loading_deadline")
                time.sleep(1)
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                        break
                except OSError:
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise TimeoutError("diagnostic_native_socket_bind_deadline")
                    time.sleep(1)
            receipt["native_server_started"] = True
            client = subprocess.Popen(client_command, cwd=ROOT, env=sim_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            out, err = client.communicate(timeout=3000)
            receipt.update(client_exit_code=client.returncode, client_public_log_categories=categories((out + err).decode(errors="replace")))
            summary_path = args.output / "framework/summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                result = json.loads((args.output / "framework/episodes/ep-0/result.json").read_text())
                diagnostic = json.loads((args.output / "framework/episodes/ep-0/diagnostics.internal.json").read_text())
                receipt.update(summary=str(summary_path.relative_to(ROOT)), completed=summary["completed"],
                    framework_errors=summary["errors"], execution_statistics=result["execution_statistics"],
                    metrics=result["metrics"], termination_reason=result["termination_reason"], adapter_diagnostics=diagnostic,
                    passed=client.returncode == 0 and summary["errors"] == 0 and summary["completed"] == 1)
        except Exception as error:
            receipt["exception_chain"] = public_exception(error)
        finally:
            stop_owned(client)
            stop_owned(server)
            if logs is not None:
                out, err = logs.result(timeout=15)
                receipt["server_public_log_categories"] = categories((out + err).decode(errors="replace"))
                receipt["owned_server_exit_after_supervised_cleanup"] = server.returncode
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("passed", "client_exit_code", "framework_errors", "completed", "execution_statistics", "metrics")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
