"""Supervised pinned native RoboTwin pair; single trial, never formal acceptance."""
import argparse
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
MODELS = WORKSPACE.parent.parent / "Model"
FILES = {"openwam_robotwin": "stage4-openwam-robotwin-adjust-bottle.json",
         "lingbot_robotwin": "stage4-lingbot-robotwin-adjust-bottle.json"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def child(pair, output, closed_loop):
    from embodied_eval.config import load_config, make_adapters, make_profile
    from embodied_eval.codec import to_data
    from embodied_eval import _legacy_contracts as c
    from embodied_eval._legacy_validation import validate_observation, validate_prediction
    from embodied_eval.visibility import build_model_input, public_space
    from embodied_eval.compatibility import check_compatibility
    config = load_config(ROOT / "examples" / FILES[pair])
    if closed_loop:
        from embodied_eval.runner import run
        result = run(config, output / "framework")
        return {"pair": pair, "passed": result.exit_code == 0, "exit_code": result.exit_code,
                "synthetic": False, "closed_loop": True, "scoring_performed": True,
                "formal_acceptance": False, "native_stepwise_comparison_performed": False,
                "results": [to_data(item) for item in result.results], "summary": result.summary}
    model, benchmark = make_adapters(config, output / "one-request-not-run")
    try:
        space, cap = benchmark.describe(), model.describe()
        profile = make_profile(config, space, cap)
        check_compatibility(space, cap, profile).require_compatible()
        task = benchmark.enumerate_tasks({"episodes": to_data(config.episodes)})[0]
        context = c.Context(run_id="robotwin-native-probe", env_id="env0", episode_id=task.episode_id, message_id="begin")
        observation = benchmark.reset(task, context)
        validate_observation(observation, space)
        model.begin_episode(context, task.public, public_space(space, profile), profile)
        model.observe(build_model_input(task, observation, profile))
        request = c.Context(run_id=context.run_id, env_id=context.env_id, episode_id=context.episode_id, message_id="request1")
        prediction = model.predict(request)
        validate_prediction(prediction, observation, space, cap, profile, request_context=request)
        (output / "prediction.json").write_text(json.dumps(to_data(prediction), indent=2) + "\n")
        model.end_episode(c.ModelEndContext(context=context, reason="aborted"))
        return {"pair": pair, "passed": True, "synthetic": False, "real_model_request": True,
                "closed_loop": False, "scoring_performed": False, "formal_acceptance": False,
                "benchmark": benchmark.diagnostics(), "model": model.diagnostics()}
    finally:
        try:
            model.close()
        finally:
            benchmark.close()


def probe(pair, output, closed_loop):
    from embodied_eval.config import load_config
    from embodied_eval.robotwin_pairs import verify
    verify(load_config(ROOT / "examples" / FILES[pair]))
    output.mkdir(parents=True, exist_ok=False)
    lingbot = pair == "lingbot_robotwin"
    port = 8925 if lingbot else 8924
    repo = WORKSPACE / "upstream" / ("lingbot-vla-v2" if lingbot else "OpenWAM")
    if lingbot:
        command = [str(WORKSPACE / "upstream/envs/lingbot-stage4/bin/python"), str(ROOT / "docs/handoff/serve_lingbot_native.py"),
            "--model-path", str(MODELS / "lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"), "--port", str(port)]
    else:
        command = [str(WORKSPACE / "upstream/envs/openwam-py310/bin/python"), str(repo / "scripts/deploy.py"),
            "--ckpt-dir", str(MODELS / "OpenWAM-Alpha-Sim-RoboTwin-Full"), "--host", "127.0.0.1", "--port", str(port),
            "--device", "cuda:0", "--denoise-steps", "10", "--denoise-mode", "sync", "--inference-mode", "sync"]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", port))
    environment = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    environment.update(CUDA_VISIBLE_DEVICES="6" if lingbot else "1", HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        QWEN3VL_PATH=str(MODELS / "Qwen3-VL-4B-Instruct-processor"))
    (output / "launch.json").write_text(json.dumps({"pair": pair, "source_cwd": str(repo), "command": command,
        "server_gpu": environment["CUDA_VISIBLE_DEVICES"], "simulator_gpu": "5" if lingbot else "7",
        "precision": "native published FP32" if lingbot else "native deployment defaults", "global_proxy_modified": False,
        "kind": "real_single_trial_not_native_comparison" if closed_loop else "one_real_request_not_score"}, indent=2) + "\n")
    server = None
    try:
        with (output / "server.txt").open("x") as log:
            server = subprocess.Popen(command, cwd=repo, env=environment, stdout=log, stderr=log, start_new_session=True)
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    return {"pair": pair, "passed": False, "gate": "server_start", "exit_code": server.returncode}
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        return {"pair": pair, "passed": False, "gate": "bounded_startup_timeout"}
                    time.sleep(1)
            runner_environment = environment.copy()
            runner_environment.update(CUDA_VISIBLE_DEVICES="5" if lingbot else "7",
                CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"))
            egl = WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
            if egl.is_file():
                runner_environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
            command = [str(WORKSPACE / "upstream/envs/robotwin-stage4/bin/python"), __file__, "--child", pair, "--output", str(output)]
            if closed_loop:
                command.append("--closed-loop")
            result = subprocess.run(command, cwd=ROOT, env=runner_environment, capture_output=True, text=True, timeout=5400 if closed_loop else 600)
            (output / "client.txt").write_text(result.stdout + result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith("ROBOTWIN_PROBE ")]
            return json.loads(lines[-1].removeprefix("ROBOTWIN_PROBE ")) if lines else {
                "pair": pair, "passed": False, "gate": "native_client", "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"pair": pair, "passed": False, "gate": "bounded_client_timeout", "formal_acceptance": False}
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
    parser.add_argument("--pairs", nargs="+", choices=tuple(FILES), default=list(FILES))
    parser.add_argument("--child", choices=tuple(FILES))
    parser.add_argument("--closed-loop", action="store_true")
    args = parser.parse_args()
    if args.child:
        try:
            record = child(args.child, args.output, args.closed_loop)
        except Exception as error:
            chain = []
            while error is not None and len(chain) < 6:
                chain.append({"type": type(error).__name__, "missing_module": getattr(error, "name", None),
                              "code": getattr(error, "code", None)})
                error = error.__cause__
            record = {"pair": args.child, "passed": False, "gate": "native_request", "error_chain": chain,
                      "closed_loop": False, "scoring_performed": False}
        print("ROBOTWIN_PROBE " + json.dumps(record), flush=True)
        return 0 if record["passed"] else 1
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for pair in args.pairs:
        try:
            record = probe(pair, args.output / pair, args.closed_loop)
        except Exception as error:
            record = {"pair": pair, "passed": False, "gate": "preflight", "error_type": type(error).__name__,
                      "error_code": getattr(error, "code", None), "model_started": False}
        results.append(record)
        print(json.dumps(record), flush=True)
    receipt = {"kind": "robotwin_real_single_trial_not_acceptance" if args.closed_loop else "robotwin_one_real_request_not_score",
               "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(r["passed"] for r in results), "results": results}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
