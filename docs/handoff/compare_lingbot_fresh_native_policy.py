"""Supervise only this diagnostic's native FP32 server and independent rollout."""
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


def stop(process):
    if process is None:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.framework, args.output = args.framework.resolve(), args.output.resolve()
    from embodied_eval.config import load_config
    from embodied_eval.robotwin_pairs import verify
    config = load_config(ROOT / "examples/stage4-lingbot-robotwin-adjust-bottle.json")
    verify(config)
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", config.model.port))
    models = WORKSPACE.parent.parent / "Model"
    command = [str(WORKSPACE / "upstream/envs/lingbot-stage4/bin/python"), str(ROOT / "docs/handoff/serve_lingbot_native.py"),
        "--model-path", config.model.checkpoint_dir + "/checkpoints/global_step_50000/hf_ckpt", "--port", str(config.model.port)]
    # Model config's root is the complete publisher snapshot, not its hf_ckpt child.
    if not Path(command[3]).is_dir():
        raise ValueError("matching_native_checkpoint_directory_missing")
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="6", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        QWEN3VL_PATH=str(models / "Qwen3-VL-4B-Instruct-processor"), OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    server = worker = None
    try:
        with (args.output / "server.txt").open("x") as log:
            server = subprocess.Popen(command, cwd=config.model.lingbot_repo, env=environment,
                                      stdout=log, stderr=log, start_new_session=True)
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("native_server_exit")
                try:
                    with socket.create_connection(("127.0.0.1", config.model.port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("external_start_timeout")
                    time.sleep(1)
            environment.update(CUDA_VISIBLE_DEVICES="5", CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"),
                __EGL_VENDOR_LIBRARY_FILENAMES=str(WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"))
            with (args.output / "client.txt").open("x") as log:
                worker = subprocess.Popen([str(WORKSPACE / "upstream/envs/robotwin-stage4/bin/python"),
                    str(ROOT / "docs/handoff/replay_robotwin_native_actions.py"), "--pair", "lingbot_robotwin",
                    "--fresh-native-lingbot", "--reproduce-seed-search", "--framework", str(args.framework),
                    "--output", str(args.output / "native")], cwd=ROOT, env=environment,
                    stdout=log, stderr=log, start_new_session=True)
                code = worker.wait(timeout=1800)
            receipt = args.output / "native/receipt.json"
            if not receipt.is_file():
                raise RuntimeError("native_child_missing_receipt")
            print(f"fresh native LingBot comparison exit={code} receipt={receipt}")
            return code
    except Exception as error:
        with (args.output / "supervision-error.json").open("x") as stream:
            json.dump({"kind": "native_comparison_supervision_error", "passed": False,
                "error_type": type(error).__name__, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "formal_acceptance": False}, stream)
        raise
    finally:
        stop(worker)
        stop(server)


if __name__ == "__main__":
    raise SystemExit(main())
