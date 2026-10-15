"""Fixed-commit GitHub archive fallback, explicitly NOT a git checkout.

Preserves failed git evidence. TLS-served source bytes are locally SHA-sealed;
that local digest is not claimed as an independently published checksum.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile

from download_stage4_assets import PROXY_KEYS
from prepare_mn_native_environment import write
from prepare_robodojo_environment import bounded

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
REVISION = "afca7b09d60d8beb9c1cb28b43066499940b969b"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive_tree(record):
    root = Path(record["path"]).resolve()
    inventory = Path(record["file_inventory"])
    if sha(inventory) != record["file_inventory_sha256"]:
        raise ValueError("fixed archive inventory changed")
    rows = json.loads(inventory.read_text())
    expected = {r["file"] for r in rows}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if expected != actual:
        raise ValueError("fixed archive file inventory changed")
    for row in rows:
        path = root / row["file"]
        path.resolve().relative_to(root)
        if sha(path) != row["sha256"]:
            raise ValueError("fixed archive source bytes changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-source-receipt", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((args.git_source_receipt / "receipt.json").read_text())
    curobo = next(r for r in previous["records"] if r["name"] == "curobo")
    if not curobo["passed"]:
        raise ValueError("fixed CuRobo gitlink required")
    from embodied_eval.mn.assets_common import _revision
    _revision(curobo["path"], curobo["revision"])
    parent = WORKSPACE / "upstream/RoboDojo"
    _revision(parent, "266130a3ec41ba9b20b0e5648d2f3542f219c06c")
    destination = WORKSPACE / "upstream/robodojo-isaaclab-archive-mn"
    if destination.exists():
        raise ValueError("new archive reference directory required")
    archive = output / "isaaclab-fixed-commit.tar.gz"
    receipt = {"kind": "fixed_native_gitlink_preparation_not_dependency_or_runtime_validation",
        "passed": False, "records": [curobo], "archive_attempts": [], "global_proxy_modified": False,
        "parent_source_modified": False, "shared_driver_modified": False, "upstream_installer_executed": False}
    for route in ("temporary_direct", "original_proxy_environment"):
        env = os.environ.copy()
        if route == "temporary_direct":
            for key in PROXY_KEYS:
                env.pop(key, None)
        code, _, _ = bounded(["curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout", "15",
            "--max-time", "600", "--continue-at", "-", "--output", str(archive),
            "https://codeload.github.com/yuechen0614/IsaacLab/tar.gz/" + REVISION], env, 620)
        receipt["archive_attempts"].append({"route": route, "exit_code": code})
        if code == 0:
            with tarfile.open(archive) as package:
                top = "IsaacLab-" + REVISION
                if any(not (m.name == top or m.name.startswith(top + "/")) for m in package.getmembers()):
                    raise ValueError("fixed archive root mismatch")
                unpacked = output / "unpacked"
                package.extractall(unpacked, filter="data")
                (unpacked / top).rename(destination)
            rows = [{"file": str(p.relative_to(destination)), "sha256": sha(p)}
                for p in sorted(destination.rglob("*")) if p.is_file()]
            inventory = output / "source-file-sha256.json"
            write(inventory, rows)
            receipt["records"].append({"name": "IsaacLab", "revision": REVISION, "path": str(destination),
                "passed": True, "source_kind": "tls_fixed_commit_archive_locally_sha256_sealed_not_git_checkout",
                "archive_sha256": sha(archive), "publisher_archive_hash_independently_available": False,
                "archive_bytes": archive.stat().st_size, "file_inventory": str(inventory),
                "file_inventory_sha256": sha(inventory), "source_file_count": len(rows)})
            receipt["passed"] = True
            break
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "git_checkout": False}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
