"""Supervised fixed-server + real native-reset + ONE policy request per pair.

No simulator action or benchmark score; this is an integration gate, not a
closed-loop acceptance. Only this script's own server process groups are stopped.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def child(pair, output, closed_loop=False):
    from embodied_eval import contracts as c
    from embodied_eval.codec import to_data
    from embodied_eval.config import load_config, make_adapters, make_profile
    from embodied_eval.compatibility import check_compatibility
    from embodied_eval.validation import validate_observation, validate_prediction
    from embodied_eval.visibility import build_model_input, public_space
    config = load_config(ROOT / "examples" / {
        "openwam_vlabench": "stage4-openwam-vlabench-select-fruit.json",
        "openpi_vlabench": "stage4-pi05-vlabench-select-fruit.json",
        "openpi_libero": "stage4-pi05-libero-spatial.json"}[pair])
    if closed_loop:
        from embodied_eval.runner import run
        outcome = run(config, output / "framework")
        return {"pair": pair, "passed": outcome.exit_code == 0, "synthetic": False,
                "real_model_request": True, "closed_loop": True, "scoring_performed": True,
                "exit_code": outcome.exit_code, "results": [to_data(v) for v in outcome.results],
                "native_stepwise_comparison_performed": False, "formal_acceptance": False}
    model, benchmark = make_adapters(config, output)
    try:
        space, capability = benchmark.describe(), model.describe()
        profile = make_profile(config, space, capability)
        report = check_compatibility(space, capability, profile)
        if not report.compatible:
            raise RuntimeError("native_profile_not_compatible")
        task = benchmark.enumerate_tasks({"episodes": to_data(config.episodes)})[0]
        context = c.Context(run_id="native-one-request-probe", env_id="env0", episode_id=task.episode_id, message_id="begin")
        observation = benchmark.reset(task, context)
        validate_observation(observation, space)
        model.begin_episode(context, task.public, public_space(space, profile), profile)
        model.observe(build_model_input(task, observation, profile))
        request = replace(context, message_id="request1")
        prediction = model.predict(request)
        validate_prediction(prediction, observation, space, capability, profile, request_context=request)
        (output / "prediction.json").write_text(json.dumps(to_data(prediction), ensure_ascii=False, indent=2) + "\n")
        model.end_episode(c.ModelEndContext(context=context, reason="aborted"))
        return {"pair": pair, "passed": True, "synthetic": False, "real_model_request": True,
                "env_steps": 0, "native_resets": 1, "policy_requests": 1,
                "closed_loop": False, "scoring_performed": False, "server_identity_verified": False,
                "model_diagnostics": model.diagnostics(), "benchmark_diagnostics": benchmark.diagnostics()}
    finally:
        try:
            model.close()
        finally:
            benchmark.close()


def server_spec(pair):
    models = WORKSPACE.parent.parent / "Model"
    if pair == "openwam_vlabench":
        repo = WORKSPACE / "upstream/OpenWAM"
        return [str(WORKSPACE / "upstream/envs/openwam-py310/bin/python"), str(repo / "scripts/deploy.py"),
            "--ckpt-dir", str(models / "OpenWAM-Alpha-Sim-VLABench"), "--host", "127.0.0.1", "--port", "8922",
            "--device", "cuda:0", "--compile-enabled", "false", "--denoise-steps", "10",
            "--denoise-mode", "sync", "--inference-mode", "sync"], repo, 8922, "0", "vlabench-stage4"
    libero = pair == "openpi_libero"
    repo = WORKSPACE / "upstream" / ("openpi" if libero else "openpi-vlabench")
    return [str(WORKSPACE / "upstream/envs" / ("openpi-stage4" if libero else "openpi-vlabench-stage4") / "bin/python"),
        str(repo / "scripts/serve_policy.py"), "--port", "8921" if libero else "8942", "policy:checkpoint",
        "--policy.config", "pi05_libero" if libero else "pi05_ft_vlabench_primitive", "--policy.dir",
        str(models / ("pi05_libero" if libero else "pi05_vlabench_primitive10"))], repo, 8921 if libero else 8942, "2" if libero else "3", "libero-openpi-stage4" if libero else "vlabench-stage4"


def probe(pair, output, startup_timeout=600, closed_loop=False):
    from embodied_eval.config import load_config
    from embodied_eval.stage4_pairs import verify
    filename = {"openwam_vlabench": "stage4-openwam-vlabench-select-fruit.json",
                "openpi_libero": "stage4-pi05-libero-spatial.json",
                "openpi_vlabench": "stage4-pi05-vlabench-select-fruit.json"}[pair]
    # Verify dedicated assets/source before spawning a checkpoint loader.
    verify(load_config(ROOT / "examples" / filename))
    output.mkdir(parents=True, exist_ok=False)
    command, repo, port, gpu, environment_name = server_spec(pair)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", port))  # Refuse an occupied user/native port.
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                       HF_HUB_OFFLINE="1", XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    # Any small source-defined asset fetch is direct for THIS process only.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    (output / "launch.json").write_text(json.dumps({"pair": pair, "command": command,
        "source_cwd": str(repo), "gpu": gpu, "compile_disabled_for_openwam": pair == "openwam_vlabench",
        "kind": "real_single_trial_not_native_comparison" if closed_loop else "one_real_request_not_score",
        "global_proxy_modified": False}, indent=2) + "\n")
    server = None
    try:
        with (output / "server.txt").open("x") as log:
            server = subprocess.Popen(command, cwd=repo, env=environment, stdout=log, stderr=log, start_new_session=True)
            deadline = time.monotonic() + startup_timeout
            while True:
                if server.poll() is not None:
                    return {"pair": pair, "passed": False, "gate": "server_start", "exit_code": server.returncode, "real_model_request": False}
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        return {"pair": pair, "passed": False, "gate": "bounded_server_start_timeout", "real_model_request": False}
                    time.sleep(1)
            runner = WORKSPACE / "upstream/envs" / environment_name / "bin/python"
            child_command = [str(runner), __file__, "--child", pair, "--output", str(output)]
            if closed_loop:
                child_command.append("--closed-loop")
            result = subprocess.run(child_command, cwd=ROOT, capture_output=True, text=True,
                                    timeout=5400 if closed_loop else 180)
            (output / "client.txt").write_text(result.stdout + result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith("MODEL_PROBE ")]
            return json.loads(lines[-1].removeprefix("MODEL_PROBE ")) if lines else {"pair": pair, "passed": False, "gate": "client_process", "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"pair": pair, "passed": False, "gate": "bounded_client_timeout", "real_model_request": "unknown"}
    finally:
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=("openwam_vlabench", "openpi_libero", "openpi_vlabench"))
    parser.add_argument("--pairs", nargs="+", choices=("openwam_vlabench", "openpi_libero", "openpi_vlabench"),
                        default=["openwam_vlabench", "openpi_libero", "openpi_vlabench"])
    parser.add_argument("--startup-timeout-s", type=int, default=600)
    parser.add_argument("--closed-loop", action="store_true", help="Run the full reviewed single trial; no formal acceptance or native comparison")
    args = parser.parse_args()
    if args.child:
        try:
            value = child(args.child, args.output, args.closed_loop)
        except Exception as error:
            chain = []
            while error is not None and len(chain) < 6:
                chain.append(type(error).__name__)
                error = error.__cause__
            value = {"pair": args.child, "passed": False, "gate": "native_request", "error_chain": chain,
                     "closed_loop": False, "scoring_performed": False}
        print("MODEL_PROBE " + json.dumps(value), flush=True)
        return 0 if value["passed"] else 1
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for pair in args.pairs:
        value = probe(pair, args.output / pair, args.startup_timeout_s, args.closed_loop)
        results.append(value)
        print(json.dumps(value), flush=True)
    receipt = {"kind": "real_closed_loop_single_trial_not_native_comparison_or_acceptance" if args.closed_loop else "real_model_native_reset_one_request_not_closed_loop_or_score",
               "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(v["passed"] for v in results), "results": results}
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
