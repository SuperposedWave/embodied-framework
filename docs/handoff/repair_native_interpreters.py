"""Repair explicitly listed broken Linux venv interpreters, not dependencies.

Back up original configuration and symlink targets before mutation. Require the
same CPython major/minor and preserve every installed distribution metadata
file. No global Python alias, upstream source, weight, proxy or GPU is changed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENVS = ROOT.parent / "upstream/envs"
TOOLS = ROOT.parent / "upstream/toolchains/python-managed"
BROKEN = ("dreamzero-mn-native", "droid_sim-mn-native", "lingbot-stage4",
          "mn-usd-audit", "openpi-stage4", "openpi-vlabench-stage4",
          "openwam_arx-mn-native", "pi05-arx-mn", "pi05-arx-mn-inference")
TARGETS = {"3.11": TOOLS / "cpython-3.11.17-linux-x86_64-gnu/bin/python3.11",
           "3.12": TOOLS / "cpython-3.12.15-linux-x86_64-gnu/bin/python3.12"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata_files(directory):
    return {str(path.relative_to(directory)): sha(path)
            for path in sorted(directory.glob("lib/python*/site-packages/*.dist-info/METADATA"))}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--environments", nargs="+", choices=BROKEN, default=BROKEN)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "explicit_native_venv_link_repair_not_dependency_or_inference_acceptance",
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "applied": args.apply,
              "passed": False, "environments": [], "DreamZero_inference": False,
              "global_python_proxy_driver_or_sources_modified": False,
              "original_patch_versions_not_recoverable_from_generic_venv_config": True}
    try:
        for name in args.environments:
            directory = ENVS / name
            executable = directory / "bin/python"
            cfg = directory / "pyvenv.cfg"
            if not executable.is_symlink() or executable.exists():
                raise ValueError("only_explicitly_broken_interpreter_symlinks_may_be_repaired")
            old_target = str(executable.readlink())
            if not old_target.startswith("/root/.local/share/uv/python/cpython-"):
                raise ValueError("unexpected_old_interpreter_target")
            original = cfg.read_bytes()
            version = "3.12" if "3.12-" in old_target else "3.11" if "3.11-" in old_target else None
            if version not in TARGETS:
                raise ValueError("unreviewed_python_ABI")
            target = TARGETS[version]
            info = subprocess.run([str(target), "-I", "-c",
                "import sys,json;print(json.dumps({'version':sys.version.split()[0], 'minor':'.'.join(map(str,sys.version_info[:2]))}))"],
                capture_output=True, text=True, check=True, timeout=30)
            identity = json.loads(info.stdout)
            if identity["minor"] != version:
                raise ValueError("same_major_minor_required")
            before = metadata_files(directory)
            backup = args.output / name
            backup.mkdir()
            (backup / "pyvenv.cfg.original").write_bytes(original)
            write(backup / "original-links.json", {str(x.name): str(x.readlink())
                for x in sorted((directory / "bin").glob("python*")) if x.is_symlink()})
            write(backup / "installed-metadata-before.json", before)
            row = {"environment": name, "old_python_target": old_target,
                   "new_python_target": str(target), "python": identity,
                   "python_binary_sha256": sha(target), "applied": False,
                   "installed_distributions": len(before), "configuration_backup": str(backup),
                   "passed": False}
            report["environments"].append(row)
            write(args.output / "receipt.json", report)
            if args.apply:
                text = original.decode()
                old_home = next(line.split("=", 1)[1].strip() for line in text.splitlines() if line.startswith("home ="))
                replacement = text.replace("home = " + old_home, "home = " + str(target.parent), 1)
                if replacement == text:
                    raise ValueError("exact_home_replacement_required")
                # The output contains the recoverable original before either mutation.
                cfg.write_text(replacement)
                temporary = directory / "bin/python.repair-new"
                if temporary.exists() or temporary.is_symlink():
                    raise ValueError("refuse_existing_temporary_link")
                temporary.symlink_to(target)
                temporary.replace(executable)
                row["applied"] = True
                result = subprocess.run([str(executable), "-I", "-c",
                    "import sys,json,ssl,sqlite3; print(json.dumps({'version':sys.version.split()[0], 'prefix':sys.prefix, 'base_prefix':sys.base_prefix}))"],
                    capture_output=True, text=True, timeout=30)
                row["interpreter_exit_code"] = result.returncode
                if result.returncode == 0:
                    row["runtime_identity"] = json.loads(result.stdout)
                    row["prefix_is_original_venv"] = Path(row["runtime_identity"]["prefix"]).resolve() == directory.resolve()
                row["package_metadata_unchanged"] = before == metadata_files(directory)
                row["configuration_sha256"] = sha(cfg)
                row["passed"] = result.returncode == 0 and row["prefix_is_original_venv"] and row["package_metadata_unchanged"]
            else:
                row["passed"] = True
            write(args.output / "receipt.json", report)
            print(json.dumps({k: row[k] for k in ("environment", "applied", "passed", "installed_distributions")}), flush=True)
        report["passed"] = all(row["passed"] for row in report["environments"])
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "receipt.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
