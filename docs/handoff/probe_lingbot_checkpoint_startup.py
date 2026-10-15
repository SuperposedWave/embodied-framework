"""Pinned checkpoint and real native FP32 server startup; no model request/score."""
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
sys.path.insert(0, str(ROOT))


def run(output):
    from embodied_eval.stage4_pairs import _revision, _hf_objects, _file
    from embodied_eval.adapters.robotwin_spec import VARIANTS
    source = WORKSPACE / "upstream/lingbot-vla-v2"
    model = WORKSPACE.parent.parent / "Model/lingbot-vla-v2-6b-robotwin"
    _revision(source, VARIANTS["lingbot_robotwin"][0])
    files = _hf_objects(model, ROOT / "embodied_eval/adapters/lingbot_robotwin_objects.json")
    files["native_robotwin_norm"] = _file(source / "assets/norm_stats/robotwin.json", VARIANTS["lingbot_robotwin"][5])
    files["native_robot_mapping"] = _file(source / "configs/robot_configs/robotwin.yaml",
        "2c2a26509fc9662a8ada64adaa7fa959627dd6dadc6b48b680c4285da2d8ffd8")
    port = 8935
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", port))
    environment = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environment.pop(key, None)
    environment.update(CUDA_VISIBLE_DEVICES="6", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        QWEN3VL_PATH=str(WORKSPACE.parent.parent / "Model/Qwen3-VL-4B-Instruct-processor"),
        TORCHINDUCTOR_CACHE_DIR=str(output.resolve() / "torch-cache"))
    command = [str(WORKSPACE / "upstream/envs/lingbot-stage4/bin/python"),
        str(ROOT / "docs/handoff/serve_lingbot_native.py"), "--model-path",
        str(model / "checkpoints/global_step_50000/hf_ckpt"), "--port", str(port)]
    launched = datetime.now(timezone.utc).isoformat()
    with (output / "server.txt").open("x") as log:
        server = subprocess.Popen(command, cwd=source, env=environment, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while server.poll() is None and time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.5):
                        return {"passed": True, "checkpoint_loaded": True, "native_server_ready": True,
                            "precision": "native release FP32", "compiled_native_api_enabled": True,
                            "model_forward_performed": False, "model_request_sent": False,
                            "verified_sha256": files, "launch_at_utc": launched, "owned_server_gpu": 6,
                            "server_command": command, "global_proxy_modified": False}
                except OSError:
                    time.sleep(1)
            return {"passed": False, "checkpoint_loaded": False, "native_server_ready": False,
                    "gate": "server_exited" if server.poll() is not None else "bounded_startup_timeout",
                    "exit_code": server.poll(), "verified_sha256": files, "launch_at_utc": launched}
        finally:
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.output)
    except Exception as error:
        report = {"passed": False, "gate": "preflight", "error_type": type(error).__name__,
                  "missing_module": getattr(error, "name", None), "error_code": getattr(error, "code", None)}
    report.update(kind="real_pinned_lingbot_server_startup_not_inference_or_benchmark",
        checked_at_utc=datetime.now(timezone.utc).isoformat(), scoring_performed=False, formal_acceptance=False)
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "verified_sha256"}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
