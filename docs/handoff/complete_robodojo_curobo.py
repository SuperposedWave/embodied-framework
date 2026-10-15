"""Restore native editable CuRobo installation of our immutable-code copy.

The ordinary wheel omitted config YAMLs; native install.sh uses editable mode.
No source edit, native rule change, new version selection or shared-driver edit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from download_stage4_assets import PROXY_KEYS
from prepare_robodojo_environment import ROOT, WORKSPACE, TARGET, bounded
from prepare_mn_native_environment import public_failure, public_details
from probe_mn_components import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = args.environment_from.resolve()
    previous.relative_to(ROOT / "outputs")
    receipt = json.loads((previous / "receipt.json").read_text())
    if not receipt["passed"] or receipt["environment"] != str(TARGET) or receipt["phase"] != "native":
        raise ValueError("helper-owned installed native environment required")
    original = WORKSPACE / "upstream/robodojo-curobo-mn"
    from embodied_eval.mn.assets_common import _revision
    _revision(original, "d17b54ce32cba095c0b000c4c58777075d11de0e")
    source = previous / "native-install-source/curobo"
    files = subprocess.run(["git", "-C", str(original), "ls-files", "-z"], capture_output=True, check=True).stdout.split(b"\0")
    hashes = {}
    for row in files:
        if not row:
            continue
        relative = row.decode()
        if not (original / relative).is_file():
            continue
        digest = hashlib.sha256((original / relative).read_bytes()).hexdigest()
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("native tracked install copy differs from fixed reference")
        hashes[relative] = digest
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    command = [str(WORKSPACE / "upstream/envs/downloader/bin/uv"), "pip", "install", "--python", str(TARGET / "bin/python"),
        "--editable", str(source) + "[cu12]", "--no-deps", "--no-build-isolation"]
    record = {"kind": "native_editable_install_restores_fixed_config_resources_not_scene_validation", "passed": False,
        "native_revision": "d17b54ce32cba095c0b000c4c58777075d11de0e", "tracked_install_copy_sha256": hashes,
        "configuration_rules_modified": False, "attempts": [], "raw_logs_saved": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "global_proxy_modified": False, "shared_driver_modified": False}
    for route in ("temporary_direct", "original_proxy_environment"):
        environment = dict(os.environ, UV_NO_PROGRESS="1", SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="0.0.0.dev0")
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
    print(json.dumps({"passed": record["passed"], "tracked_reference_files_verified": len(hashes)}), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
