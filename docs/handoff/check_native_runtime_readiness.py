"""Offline imports and optional small CUDA checks, without loading weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENVS = ROOT.parent / "upstream/envs"
PROBES = {
    "openpi-stage4": ("numpy", "torch", "torchvision", "jax", "flax", "transformers", "sentencepiece"),
    "openpi-vlabench-stage4": ("numpy", "torch", "torchvision", "jax", "flax", "transformers"),
    "lingbot-stage4": ("numpy", "torch", "torchvision", "transformers", "safetensors"),
    "openwam_arx-mn-native": ("numpy", "torch", "torchvision", "diffusers", "transformers", "jsonschema"),
    "pi05-arx-mn-inference": ("numpy", "torch", "torchvision", "jax", "flax", "jsonschema"),
    "droid_sim-mn-native": ("numpy", "torch", "isaaclab.app", "jsonschema"),
    "mn-usd-audit": ("pxr.Usd",),
    "libero": ("numpy", "torch", "mujoco", "robosuite", "PIL", "jsonschema"),
    "libero-openpi-stage4": ("numpy", "torch", "mujoco", "robosuite", "PIL", "jsonschema"),
    "vlabench-stage4": ("numpy", "mujoco", "dm_control", "jsonschema"),
    "robotwin-stage4": ("numpy", "torch", "sapien", "mplib", "curobo", "jsonschema"),
    "openwam-py310": ("numpy", "torch", "torchvision", "diffusers", "transformers"),
    "robodojo-mn-native": ("numpy", "torch", "isaaclab.app", "jsonschema"),
    "cosmos3-mn-native": ("numpy", "torch", "torchvision", "transformers", "jsonschema"),
}


def child(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from probe_mn_components import public_exception
    row = {"environment": args.child, "version": sys.version.split()[0],
           "prefix": sys.prefix, "modules": [], "passed": True,
           "weights_loaded": False, "simulator_started": False}
    for name in PROBES[args.child]:
        entry = {"module": name, "passed": False}
        try:
            module = importlib.import_module(name)
            entry.update(passed=True, version=str(getattr(module, "__version__", "not_exposed")))
        except Exception as error:
            entry["exception_chain"] = public_exception(error)
            row["passed"] = False
        row["modules"].append(entry)
    if args.cuda and "torch" in sys.modules and args.child not in ("libero", "libero-openpi-stage4"):
        try:
            import torch
            row["cuda"] = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
            x = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
            row["cuda"].update(matrix_sum=(x @ x.T).sum().item(), device=torch.cuda.get_device_name())
            torch.cuda.synchronize()
            row["cuda"]["passed"] = True
        except Exception as error:
            row["cuda"] = {"passed": False, "exception_chain": public_exception(error)}
            row["passed"] = False
    print("READINESS_JSON=" + json.dumps(row), flush=True)
    return 0 if row["passed"] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--environments", choices=PROBES, nargs="+", default=list(PROBES))
    p.add_argument("--child", choices=PROBES)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--import-timeout-s", type=int, default=600)
    args = p.parse_args()
    if args.child:
        return child(args)
    if not 1 <= args.import_timeout_s <= 1800:
        raise ValueError("bounded_import_timeout_required")
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=str(args.gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    # Isolate stale image Python/Torch libraries; driver lookup keeps system defaults.
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    def probe(name):
        child_env = dict(env)
        if name in ("droid_sim-mn-native", "robodojo-mn-native"):
            child_env["OMNI_KIT_ACCEPT_EULA"] = "YES"  # Existing project-only user acceptance.
        command = [str(ENVS / name / "bin/python"), __file__, "--child", name]
        if args.cuda:
            command.append("--cuda")
        try:
            result = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=args.import_timeout_s)
            lines = [line for line in result.stdout.splitlines() if line.startswith("READINESS_JSON=")]
            if len(lines) != 1:
                from probe_mn_components import categories
                return {"environment": name, "passed": False, "exit_code": result.returncode,
                        "public_failure_categories": categories(result.stdout + result.stderr)}
            row = json.loads(lines[0].split("=", 1)[1])
            row["exit_code"] = result.returncode
            return row
        except subprocess.TimeoutExpired:
            return {"environment": name, "passed": False, "diagnostic_import_timeout": True}
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(probe, name) for name in args.environments]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (args.output / (row["environment"] + ".json")).write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps({"environment": row["environment"], "passed": row["passed"]}), flush=True)
    receipt = {"kind": "offline_dependency_import_and_small_CUDA_not_inference_or_task_acceptance",
               "at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(r["passed"] for r in rows),
               "environments": rows, "raw_logs_saved": False, "DreamZero_deferred": True,
               "pi05_arx_mn_empty_training_scaffold_not_selected_for_inference": True}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": receipt["passed"], "results": [{"environment": r["environment"], "passed": r["passed"]} for r in rows]}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
