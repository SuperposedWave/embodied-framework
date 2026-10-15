"""Stage the two official public archives without overwriting native source files.

Drive IDs come from cf588fe scripts/download_assets.py. They aren't immutable
revisions: receipts retain local SHA-256 and never claim official origin hashes.
No model/simulator is constructed. All proxy changes affect one subprocess only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import zipfile

FILES = {"obj": "1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-", "scene": "1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw"}
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(kind, cache, target, interpreter, timeout):
    archive = cache / f"{kind}.zip"
    record = {"archive": str(archive), "official_drive_file_id": FILES[kind], "attempts": [],
              "official_archive_digest_available": False}
    try:
        if not archive.exists():
            partial = cache / f"{kind}.zip.stage4.partial"
            if not partial.resolve().is_relative_to(cache):
                raise ValueError("unsafe_partial_path")
            for route in ("temporary_direct", "original_proxy_environment"):
                environment = os.environ.copy()
                if route == "temporary_direct":
                    for key in PROXY_KEYS:
                        environment.pop(key, None)
                try:
                    result = subprocess.run([str(interpreter), "-m", "gdown", "--no-cookies", "--continue",
                        "--output", str(partial), f"https://drive.google.com/uc?id={FILES[kind]}"],
                        env=environment, capture_output=True, timeout=timeout)
                    code = result.returncode
                    failure = next((label for text, label in (
                        ("No module named gdown", "gdown_unavailable"),
                        ("SSLError", "tls_error"), ("ProxyError", "proxy_error"),
                        ("Too many users", "drive_download_quota"),
                        ("Cannot retrieve the public link", "drive_public_link_unavailable"),
                        ("Permission denied", "filesystem_permission_error"))
                        if text in result.stderr.decode(errors="replace")), None)
                except subprocess.TimeoutExpired:
                    code = 124
                    failure = "download_timeout"
                record["attempts"].append({"route": route, "exit_code": code,
                    "failure_class": failure if code else None})
                if code == 0:
                    break
            else:
                raise RuntimeError("bounded_download_failed_partial_preserved")
            if not zipfile.is_zipfile(partial):
                raise ValueError("download_is_not_a_zip")
            os.link(partial, archive)
            partial.unlink()
        record.update(archive_bytes=archive.stat().st_size, archive_sha256=digest(archive))
        staging = Path(tempfile.mkdtemp(prefix=f"stage4-vlabench-{kind}-", dir=target.parent))
        record["staging_directory_retained"] = str(staging)
        entries = []
        with zipfile.ZipFile(archive) as stream:
            members = stream.infolist()
            names = [info.filename for info in members]
            if len(names) != len(set(names)) or sum(info.file_size for info in members) > 100 * 1024**3:
                raise ValueError("duplicate_names_or_oversized_archive")
            for info in members:
                name = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if (name.is_absolute() or ".." in name.parts or ".git" in name.parts
                        or "\\" in info.filename or stat.S_ISLNK(mode) or info.flag_bits & 1):
                    raise ValueError("unsafe_archive_member")
                if info.is_dir():
                    continue
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(info) as incoming, path.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                entries.append({"file": str(name), "size": info.file_size, "sha256": digest(path)})
        # Validate every existing destination BEFORE publishing any new asset.
        for entry in entries:
            destination = target / entry["file"]
            if not destination.resolve().is_relative_to(target):
                raise ValueError("asset_symlink_escapes_target")
            if destination.exists() and digest(destination) != entry["sha256"]:
                raise ValueError("archive_would_modify_existing_native_asset")
        for entry in entries:
            destination = target / entry["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(staging / entry["file"], destination)
            except FileExistsError:
                if digest(destination) != entry["sha256"]:
                    raise ValueError("existing_asset_changed_during_publish")
        record.update(passed=True, entries=entries, entry_count=len(entries))
    except Exception as error:
        record.update(passed=False, error_type=type(error).__name__,
                      error_code=str(error) if isinstance(error, (ValueError, RuntimeError)) else "asset_staging_failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(FILES), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--gdown-python", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-per-route", type=int, default=900)
    args = parser.parse_args()
    cache, target = args.cache.resolve(), args.target.resolve()
    if args.receipt.exists() or cache.name != "VLABench-assets" or target.parts[-2:] != ("VLABench", "assets"):
        raise SystemExit("require dedicated VLABench cache/target and a new receipt")
    if args.timeout_per_route < 1 or not args.gdown_python.is_file():
        raise SystemExit("bounded timeout and an explicit gdown interpreter required")
    revision = subprocess.run(["git", "-C", str(target.parent.parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    if revision.stdout.strip() != "cf588fe60c0c7282174fe979f5913170cfe69017":
        raise SystemExit("VLABench source revision mismatch")
    cache.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    # Dereferencing a venv's Python symlink would bypass its installed packages.
    record = prepare(args.kind, cache, target, args.gdown_python.absolute(), args.timeout_per_route)
    report = {"kind": "native_asset_preparation_not_benchmark_run", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "global_proxy_modified": False,
        "model_loaded": False, "simulator_constructed": False, "record": record}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.kind}: passed={record['passed']} receipt={args.receipt}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
