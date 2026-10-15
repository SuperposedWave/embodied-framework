"""Task-owned, bounded diagnostic process, NOT Runner recovery/hard timeout.

Survives native CUDA aborts and writes a receipt. Never stores arbitrary native
stdout/stderr or env values; only classify known failures. No GPU reset/flags.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    if not 0 <= args.gpu < 8 or args.timeout_s < 1:
        raise ValueError("bounded local GPU smoke required")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    child_output = args.output.resolve() / "model-result.json"
    python = ROOT.parent / "upstream/envs/pi05-arx-mn-inference/bin/python"
    command = [str(python), str(ROOT / "docs/handoff/smoke_pi05_arx_model.py"), "--output", str(child_output)]
    receipt = {"kind": "supervised_real_model_synthetic_observation_diagnostic_not_real_benchmark",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "gpu": args.gpu, "command": command,
        "timeout_s": args.timeout_s, "passed": False, "synthetic_observation": True,
        "new_real_benchmark_trials": 0, "official_score": None, "determinism_or_sampling_flags_changed": False,
        "global_proxy_or_driver_changed": False, "occupancy_process_modified": False}
    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = child.communicate(timeout=args.timeout_s)
        receipt["exit_code"] = child.returncode
        if b"cudaErrorIllegalAddress" in err or b"illegal memory access" in err:
            receipt["failure_class"] = "native_cuda_illegal_address"
        elif child.returncode != 0:
            receipt["failure_class"] = "native_child_failed_no_raw_log_saved"
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate(timeout=10)
        receipt["exit_code"] = child.returncode
        receipt["failure_class"] = "bounded_diagnostic_timeout_execution_state_unknown"
    if child_output.is_file():
        result = json.loads(child_output.read_text())
        receipt["passed"] = child.returncode == 0 and result["passed"]
        receipt["model_result"] = "model-result.json"
    progress = child_output.with_suffix(".progress.json")
    if progress.is_file():
        receipt["last_child_progress"] = json.loads(progress.read_text())
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt.get(key) for key in ("passed", "gpu", "exit_code", "failure_class", "last_child_progress")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
