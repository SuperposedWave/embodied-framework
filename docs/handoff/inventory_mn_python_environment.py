"""Read installed distribution names/versions and dependency consistency only.

No imports of native runtime packages, environment-variable dump, direct_url
metadata, private URLs, credentials, GPU operations or environment mutations.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())
    check = subprocess.run([str(args.uv), "pip", "check", "--python", sys.executable], capture_output=True, timeout=60)
    # Dependency facts only, no executable paths, indexes or environment values.
    facts = []
    text = (check.stdout + check.stderr).decode("utf-8", errors="replace")
    for line in text.splitlines():
        clean = line.strip()
        if not any(word in clean.lower() for word in ("http", "proxy", "token", "password", "key=", "://")) and re.match(r"(?:The package|Package |Found \d+ incompatibilit)", clean):
            facts.append(clean)
    receipt = {"kind": "installed_python_distribution_inventory_not_native_validation", "at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "python": platform.python_version(), "installed_packages": packages,
        "installed_distribution_count": len(packages), "dependency_check_exit_code": check.returncode,
        "dependency_consistency_passed": check.returncode == 0, "runtime_packages_imported": False,
        "public_dependency_incompatibility_facts": facts,
        "model_loaded": False, "simulator_started": False, "environment_modified": False,
        "raw_installer_log_persisted": False, "package_direct_url_metadata_persisted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("installed_distribution_count", "dependency_consistency_passed", "interpreter")}))
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
