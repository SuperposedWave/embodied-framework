"""Add the source-declared HDF5 import dependency to our own Dojo env only.

No native installer, historical env, system package, proxy or driver changes.
The fixed fork imports h5py in core but declares it in isaaclab_rl metadata.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from download_stage4_assets import PROXY_KEYS
from prepare_robodojo_environment import ROOT, WORKSPACE, TARGET, bounded, isaac_requirements
from prepare_mn_native_environment import public_failure, public_details
from probe_mn_components import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    previous = args.environment_from.resolve()
    previous.relative_to(ROOT / "outputs")
    receipt = json.loads((previous / "receipt.json").read_text())
    if not receipt["passed"] or receipt["environment"] != str(TARGET) or receipt["phase"] != "native":
        raise ValueError("helper-owned installed native environment required")
    source = previous / "native-install-source/IsaacLab/source/isaaclab_rl/setup.py"
    if "h5py" not in isaac_requirements(source):
        raise ValueError("native HDF5 declaration changed")
    args.output.mkdir(parents=True, exist_ok=False)
    command = [str(WORKSPACE / "upstream/envs/downloader/bin/uv"), "pip", "install", "--python", str(TARGET / "bin/python"),
        "--constraint", str(previous / "constraints.txt"), "h5py", "--index-url", "https://pypi.org/simple"]
    record = {"kind": "source_declared_native_runtime_import_completion_not_scene_validation", "passed": False,
        "environment": str(TARGET), "declared_requirement": "h5py", "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "attempts": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "global_proxy_modified": False,
        "shared_driver_modified": False, "historical_environments_modified": False, "raw_logs_saved": False}
    for route in ("temporary_direct", "original_proxy_environment"):
        environment = dict(os.environ, UV_NO_PROGRESS="1")
        if route == "temporary_direct":
            for key in PROXY_KEYS:
                environment.pop(key, None)
        code, _, stderr = bounded(command, environment, 300)
        attempt = {"route": route, "exit_code": code}
        if code != 0:
            attempt.update(failure_class=public_failure(stderr), public_details=public_details(stderr))
        record["attempts"].append(attempt)
        if code == 0:
            record["passed"] = True
            break
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", record)
    print(json.dumps({"passed": record["passed"], "requirement": "h5py"}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
