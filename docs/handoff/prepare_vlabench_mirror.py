"""Bounded anonymous fixed-revision asset mirror download, never run a benchmark.

Only child processes bypass proxies. A mirror checksum is NOT proof of identity
with the unavailable original Google Drive ZIPs. All files are verified against
the pinned HF git/LFS metadata before publishing into the native assets tree.
Existing files are compared, never overwritten; downloads remain resumable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

REPO = "lerobot/vlabench-assets"
REVISION = "08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4"
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("obj", "scenes"):
        raise ValueError("unsafe_mirror_path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("unsafe_mirror_symlink")
    return path


def worker(args):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).dataset_info(REPO, revision=REVISION, files_metadata=True, timeout=60, token=False)
    if info.sha != REVISION:
        raise ValueError("mirror_revision_mismatch")
    items = []
    for item in info.siblings:
        if not item.rfilename.startswith(("obj/", "scenes/")):
            continue
        checked_path(args.cache, item.rfilename)
        if item.lfs is not None:
            items.append({"relative": item.rfilename, "size": item.size, "sha256": item.lfs.sha256})
        else:
            items.append({"relative": item.rfilename, "size": item.size, "git_blob": item.blob_id})
    if not items or sum(item["size"] for item in items) > 40 * 1024 ** 3:
        raise ValueError("unexpected_mirror_manifest")
    manifest = {"repo_id": REPO, "revision": REVISION, "files": items}
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / "mirror-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Ordinary git blobs can be obtained in one compressed pack rather than
    # 32,104 separate HTTP downloads. Git LFS pointers are NOT actual assets.
    git_source = args.cache.with_name(args.cache.name + "-git")
    if (git_source / ".git").is_dir():
        head = subprocess.run(["git", "-C", str(git_source), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip() == REVISION:
            for item in items:
                if "git_blob" not in item:
                    continue
                source = checked_path(git_source, item["relative"])
                blob = hashlib.sha1(f"blob {item['size']}\0".encode())
                if not source.is_file() or source.stat().st_size != item["size"]:
                    raise ValueError("mirror_git_blob_mismatch")
                with source.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        blob.update(block)
                if blob.hexdigest() != item["git_blob"]:
                    raise ValueError("mirror_git_blob_mismatch")
                target = checked_path(args.cache, item["relative"])
                if target.exists():
                    if sha(target) != sha(source):
                        raise ValueError("existing_native_asset_conflict_no_overwrite")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, target)
    pending = [item["relative"] for item in items
               if not (args.cache / item["relative"]).is_file()
               or (args.cache / item["relative"]).stat().st_size != item["size"]]
    if pending:
        snapshot_download(REPO, repo_type="dataset", revision=REVISION, token=False,
            allow_patterns=pending, local_dir=args.cache, max_workers=args.workers, etag_timeout=20)
    records = []
    for item in items:
        source = checked_path(args.cache, item["relative"])
        if not source.is_file() or source.stat().st_size != item["size"]:
            raise ValueError("mirror_asset_missing_or_size_mismatch")
        digest = sha(source)
        if item.get("sha256"):
            if digest != item["sha256"]:
                raise ValueError("mirror_lfs_checksum_mismatch")
        else:
            blob = hashlib.sha1(f"blob {item['size']}\0".encode())
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    blob.update(block)
            if blob.hexdigest() != item["git_blob"]:
                raise ValueError("mirror_git_blob_mismatch")
        target = checked_path(args.assets, item["relative"])
        if target.exists() and (not target.is_file() or sha(target) != digest):
            raise ValueError("existing_native_asset_conflict_no_overwrite")
        records.append({"relative": item["relative"], "size": item["size"], "sha256": digest})
    # Compare every existing target before the first publication.
    for item in records:
        target = checked_path(args.assets, item["relative"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source = checked_path(args.cache, item["relative"])
            try:
                os.link(source, target)
            except FileExistsError:
                if sha(target) != item["sha256"]:
                    raise ValueError("concurrent_native_asset_conflict")
    for subtree in ("obj/meshes/tablewares/plates", "obj/meshes/containers/basket",
                    "obj/meshes/fruit", "obj/meshes/containers/tray"):
        if not any((args.assets / subtree).rglob("*.xml")):
            raise ValueError("critical_mirror_asset_subtree_empty")
    report = {"passed": True, "repo_id": REPO, "revision": REVISION,
        "file_count": len(records), "total_bytes": sum(item["size"] for item in records),
        "native_assets_directory": str(args.assets), "official_drive_origin_hash_verified": False,
        "native_source_code_modified": False, "records": records}
    (args.cache / "mirror-verified-assets.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seconds-per-route", type=int, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.cache, args.assets = args.cache.resolve(), args.assets.resolve()
    if (args.cache.name != "lerobot-vlabench-assets-08d7a44" or args.assets.name != "assets"
            or args.assets.parent.name != "VLABench" or args.workers < 1 or args.seconds_per_route < 1):
        raise SystemExit("use dedicated cache and VLABench/assets target with positive limits")
    if args.worker:
        try:
            worker(args)
            return 0
        except Exception as error:
            # Persist only a type/status/known local validation code, never raw
            # HF responses, authorization headers or signed transfer URLs.
            known = {"unsafe_mirror_path", "unsafe_mirror_symlink", "mirror_revision_mismatch",
                "unexpected_mirror_manifest", "mirror_asset_missing_or_size_mismatch",
                "mirror_lfs_checksum_mismatch", "mirror_git_blob_mismatch",
                "existing_native_asset_conflict_no_overwrite", "concurrent_native_asset_conflict",
                "critical_mirror_asset_subtree_empty"}
            code = str(error) if isinstance(error, ValueError) and str(error) in known else "mirror_worker_failed"
            response = getattr(error, "response", None)
            args.cache.mkdir(parents=True, exist_ok=True)
            (args.cache / "mirror-worker-error.json").write_text(json.dumps({
                "error_type": type(error).__name__, "error_code": code,
                "http_status": getattr(response, "status_code", None),
                "os_errno": getattr(error, "errno", None),
                "at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            return 1
    if args.receipt.exists():
        raise SystemExit("refusing to overwrite evidence")
    report = {"kind": "asset_mirror_preparation_not_evaluation", "repo_id": REPO,
        "revision": REVISION, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False, "global_proxy_modified": False, "model_loaded": False,
        "official_drive_origin_hash_verified": False, "attempts": []}
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--cache", str(args.cache),
        "--assets", str(args.assets), "--receipt", str(args.receipt), "--workers", str(args.workers)]
    for route in ("temporary_direct", "original_proxy_environment"):
        environment = os.environ.copy()
        environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        if route == "temporary_direct":
            for key in PROXY_KEYS:
                environment.pop(key, None)
        try:
            result = subprocess.run(command, env=environment, capture_output=True, timeout=args.seconds_per_route)
            attempt = {"route": route, "exit_code": result.returncode,
                       "error_code": None if result.returncode == 0 else "mirror_worker_failed_cache_preserved"}
        except subprocess.TimeoutExpired:
            attempt = {"route": route, "exit_code": None, "error_code": "bounded_timeout_cache_preserved"}
        report["attempts"].append(attempt)
        if attempt["exit_code"] == 0:
            verified = args.cache / "mirror-verified-assets.json"
            value = json.loads(verified.read_text())
            report.update(passed=True, file_count=value["file_count"], total_bytes=value["total_bytes"],
                          verified_manifest=str(verified), verified_manifest_sha256=sha(verified))
            break
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.receipt}: passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
