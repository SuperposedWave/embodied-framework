"""Read-only native configuration/dependency probe, never loads model weights.

The invoking historical environment is diagnostic ONLY: it is not certified for
the new fork. Capture exception type/missing module, never arbitrary messages,
environment values, remote URLs or credentials.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = WORKSPACE / "upstream/xpolicy-reference-mn/policy/Pi_05/openpi"
    pyproject = source / "pyproject.toml"
    packages = {}
    for name in ("jax", "flax", "orbax-checkpoint", "torch", "torchvision", "torchcodec", "transformers", "lerobot", "pytest"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    receipt = {"kind": "native_config_dependency_probe_not_real_validation", "at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "python": platform.python_version(), "installed_versions": packages,
        "source_pyproject_sha256": hashlib.sha256(pyproject.read_bytes()).hexdigest(),
        "source_requirements": tomllib.loads(pyproject.read_text())["project"]["dependencies"],
        "historical_environment_changed": False, "new_fork_dependency_stack_certified": False,
        "model_loaded": False, "gpu_inference": False, "simulator_started": False, "passed": False,
        "isaaclab_importable": util.find_spec("isaaclab") is not None}
    sys.path[:0] = [str(source / "src"), str(source / "packages/openpi-client/src")]
    try:
        from openpi.training import config
        selected = config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
        receipt.update(passed=True, config_name=selected.name, action_horizon=selected.model.action_horizon,
            pi05=selected.model.pi05)
    except Exception as error:
        receipt.update(error_type=type(error).__name__, missing_module=error.name if isinstance(error, ModuleNotFoundError) else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt.get(key) for key in ("passed", "error_type", "missing_module", "config_name", "new_fork_dependency_stack_certified")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
