"""Retry a failed frozen-lock install as DRY RUN; save only public categories."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from prepare_mn_native_environment import PROXIES, public_failure, public_details

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existing = json.loads((args.preparation / "receipt.json").read_text())
    if existing["component"] != "droid_sim" or existing["passed"]:
        raise ValueError("require the known failed DROID installation")
    env = os.environ.copy()
    for key in PROXIES:
        env.pop(key, None)
    env["UV_PROJECT_ENVIRONMENT"] = existing["environment"]
    env["UV_HTTP_TIMEOUT"] = "60"
    command = [str(ROOT.parent / "upstream/envs/downloader/bin/uv"), "sync", "--project",
        str((args.preparation / "runtime-project").resolve()), "--frozen", "--no-install-project", "--no-dev", "--dry-run", "--python", "3.11"]
    record = {"kind": "native_lock_dry_run_public_failure_diagnostic", "at_utc": datetime.now(timezone.utc).isoformat(),
        "component": "droid_sim", "preparation": str(args.preparation), "source_revision": existing["source_revision"],
        "route": "temporary_direct", "environment_modified": False, "raw_log_persisted": False,
        "global_proxy_modified": False, "model_loaded": False, "simulator_started": False}
    try:
        result = subprocess.run(command, env=env, capture_output=True, timeout=150)
        record.update(exit_code=result.returncode, failure_class=public_failure(result.stderr) if result.returncode else None,
            public_details=public_details(result.stderr), dry_run_passed=result.returncode == 0)
    except subprocess.TimeoutExpired:
        record.update(failure_class="bounded_dry_run_timeout", dry_run_passed=False)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
