"""Constrain only this task's new OpenWAM environment to native-allowed CUDA12.8.

The unconstrained install selected a CUDA13 PyTorch build incompatible with the
current shared driver. No source checkout, driver or historical venv is changed.
Direct route is temporary for this command, then original proxy fallback.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from prepare_mn_native_environment import PROXIES, public_failure, public_details, write

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    if args.timeout_s < 1:
        raise ValueError("positive timeout required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    target = WORKSPACE / "upstream/envs/openwam_arx-mn-native"
    previous = json.loads((ROOT / "outputs/mn-extension-openwam_arx-env-20261007-run1/receipt.json").read_text())
    if not previous["passed"] or Path(previous["environment"]).resolve() != target.resolve():
        raise ValueError("refuse to modify an environment not created by this task")
    sys.path.insert(0, str(ROOT))
    from embodied_eval.native.assets_common import _revision
    repo = WORKSPACE / "upstream/xpolicy-reference-mn"
    _revision(repo, previous["source_revision"])
    requirements = tomllib.loads((repo / "policy/OpenWAM/OpenWAM/pyproject.toml").read_text())["project"]["dependencies"]
    from packaging.requirements import Requirement
    selected = {"torch": "2.10.0", "torchvision": "0.25.0"}
    for name, version in selected.items():
        requirement = next(Requirement(item) for item in requirements if Requirement(item).name == name)
        if version not in requirement.specifier:
            raise ValueError("selected runtime version violates the pinned native source")
    uv = WORKSPACE / "upstream/envs/downloader/bin/uv"
    command = [str(uv), "pip", "install", "--python", str(target / "bin/python"),
               "torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
               "--index-url", "https://download.pytorch.org/whl/cu128",
               "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"]
    record = {"kind": "native_allowed_torch_cuda_compatibility_constraint_not_model_validation",
              "component": "openwam_arx", "environment": str(target),
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "selected_packages": {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"},
              "versions_within_pinned_native_requirements": True,
              "source_revision": previous["source_revision"], "attempts": [],
              "source_modified": False, "historical_environments_modified": False,
              "global_proxy_modified": False, "shared_driver_modified": False,
              "model_loaded": False, "runtime_validated": False, "raw_installer_logs_persisted": False}
    for route in ("temporary_direct", "original_proxy_environment"):
        env = os.environ.copy()
        if route == "temporary_direct":
            for key in PROXIES:
                env.pop(key, None)
        env.update(UV_HTTP_TIMEOUT="120", UV_NO_PROGRESS="1")
        try:
            result = subprocess.run(command, env=env, capture_output=True, timeout=args.timeout_s)
            attempt = {"route": route, "exit_code": result.returncode}
            if result.returncode:
                attempt.update(failure_class=public_failure(result.stderr), public_details=public_details(result.stderr))
            record["attempts"].append(attempt)
            if result.returncode == 0:
                record["passed"] = True
                break
        except subprocess.TimeoutExpired:
            record["attempts"].append({"route": route, "failure_class": "bounded_install_timeout"})
        print(route + ": incomplete", flush=True)
    if record["passed"]:
        check = subprocess.run([str(uv), "pip", "check", "--python", str(target / "bin/python")], capture_output=True, timeout=60)
        record["dependency_check_exit_code"] = check.returncode
        inventory = subprocess.run([str(target / "bin/python"), "-c",
            "import importlib.metadata as m,json;print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"],
            capture_output=True, timeout=60)
        if inventory.returncode == 0:
            write(args.output / "installed-package-versions.json", json.loads(inventory.stdout))
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _revision(repo, previous["source_revision"])
    write(args.output / "receipt.json", record)
    print(json.dumps(record), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
