"""Fetch the two exact RoboDojo gitlinks into NEW read-only reference trees.

Never runs upstream installers or updates their submodules to a moving branch.
Git/network logs remain private in memory, and proxies are changed per command.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from run_cosmos_droid_bounded import stop_owned

from download_stage4_assets import PROXY_KEYS
from prepare_mn_native_environment import write

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
REFERENCES = {
    "IsaacLab": ("afca7b09d60d8beb9c1cb28b43066499940b969b", "robodojo-isaaclab-mn"),
    "curobo": ("d17b54ce32cba095c0b000c4c58777075d11de0e", "robodojo-curobo-mn"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument("--route-timeout", type=int, default=180)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    if args.route_timeout < 1:
        raise ValueError("positive route timeout required")
    previous = None
    if args.retry_from:
        preparation = args.retry_from.resolve()
        preparation.relative_to(ROOT / "outputs")
        previous = json.loads((preparation / "receipt.json").read_text())
        if previous["kind"] != "fixed_native_gitlink_preparation_not_dependency_or_runtime_validation" or previous["passed"]:
            raise ValueError("require this helper's failed source preparation")
    from embodied_eval.mn.assets_common import _revision
    parent = WORKSPACE / "upstream/RoboDojo"
    _revision(parent, "266130a3ec41ba9b20b0e5648d2f3542f219c06c")
    records = []
    for name, (revision, directory) in REFERENCES.items():
        link = subprocess.run(["git", "ls-tree", "HEAD", "third_party/" + name],
            cwd=parent, capture_output=True, text=True, check=True).stdout.split()
        if link[:3] != ["160000", "commit", revision]:
            raise ValueError("fixed native gitlink mismatch")
        target = WORKSPACE / "upstream" / directory
        row = {"name": name, "revision": revision, "path": str(target), "attempts": [], "passed": False}
        records.append(row)
        retry = previous and any(not r["passed"] and r["path"] == str(target) and r["revision"] == revision for r in previous["records"])
        if target.exists() and not retry:
            _revision(target, revision)
            row.update(passed=True, existing_fixed_clean_reference=True)
            continue
        if not retry:
            target.mkdir()
            subprocess.run(["git", "init", "--quiet"], cwd=target, capture_output=True, check=True)
        else:
            status = subprocess.run(["git", "status", "--porcelain"], cwd=target, capture_output=True, check=True)
            if status.stdout:
                raise ValueError("incomplete own source tree contains unexpected files")
        for route in ("temporary_direct", "original_proxy_environment"):
            env = os.environ.copy()
            if route == "temporary_direct":
                for key in PROXY_KEYS:
                    env.pop(key, None)
            env["GIT_TERMINAL_PROMPT"] = "0"
            command = ["git"]
            if route == "temporary_direct":
                command += ["-c", "http.proxy=", "-c", "https.proxy="]
            command += ["fetch", "--filter=blob:none", "--depth=1", "--no-tags", "https://github.com/yuechen0614/" + name + ".git", revision]
            process = subprocess.Popen(command, cwd=target, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                _, stderr = process.communicate(timeout=args.route_timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                stop_owned(process)
                _, stderr = process.communicate()
                code = "timeout"
            row["attempts"].append({"route": route, "exit_code": code,
                "proxy_transport_failed": b"proxy" in stderr.lower(), "tls_failed": b"certificate" in stderr.lower()})
            if code == 0:
                subprocess.run(["git", "checkout", "--quiet", "--detach", revision], cwd=target,
                    capture_output=True, check=True)
                _revision(target, revision)
                row["passed"] = True
                break
    receipt = {"kind": "fixed_native_gitlink_preparation_not_dependency_or_runtime_validation",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(r["passed"] for r in records),
        "records": records, "parent_source_modified": False, "global_proxy_modified": False,
        "shared_driver_modified": False, "upstream_installer_executed": False}
    write(output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "references": len(records)}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
