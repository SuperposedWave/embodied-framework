"""Persistent bounded HTTP ranges; only a whole verified HF object is published.

Response headers, signed redirects, tokens and proxy values never enter evidence.
Small complete chunks survive retries; an incomplete chunk is never trusted.
This is an asset preparation helper, not Runner recovery or an evaluation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from download_stage4_assets import PROXY_KEYS, manifest, verify


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_response(headers, start, end, total):
    """Reject ignored ranges, truncated responses and another object's length."""
    statuses = re.findall(r"^HTTP/\S+\s+(\d+)", headers, re.M)
    ranges = re.findall(r"^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$",
                        headers, re.M | re.I)
    if not statuses or statuses[-1] != "206" or not ranges:
        raise ValueError("range_response_required")
    if tuple(map(int, ranges[-1])) != (start, end, total):
        raise ValueError("content_range_mismatch")


def chunk(directory, item, index, chunk_bytes, seconds):
    start = index * chunk_bytes
    end = min(start + chunk_bytes, item["size"]) - 1
    size = end - start + 1
    path = directory / f"{index:06d}.chunk"
    metadata = path.with_suffix(".json")
    record = {"index": index, "start": start, "end": end, "attempts": []}
    if path.exists() and metadata.exists():
        stored = json.loads(metadata.read_text())
        digest = sha256(path)
        if path.stat().st_size != size or stored != {"size": size, "sha256": digest}:
            raise ValueError("persistent_chunk_integrity_mismatch")
        return {**record, "status": "verified_existing_chunk", "sha256": digest}
    if path.exists() or metadata.exists():
        raise ValueError("incomplete_chunk_publication_preserved")
    for route in ("temporary_direct", "original_proxy_environment"):
        environment = os.environ.copy()
        if route == "temporary_direct":
            for key in PROXY_KEYS:
                environment.pop(key, None)
        attempt = directory / f"{index:06d}.{uuid.uuid4().hex}.attempt"
        # Cache-busting changes transport only; immutable revision and final hash stay fixed.
        url = item["url"] + f"?download=true&range_id={start}-{end}-{time.time_ns()}"
        result = subprocess.run(["curl", "-sS", "-fL", "--http1.1", "--connect-timeout", "15",
            "--max-time", str(seconds), "--max-filesize", str(size), "--range", f"{start}-{end}",
            "--dump-header", "-", "--output", str(attempt), url], env=environment,
            capture_output=True, timeout=seconds + 20)
        record["attempts"].append({"route": route, "exit_code": result.returncode,
                                   "received_bytes": attempt.stat().st_size if attempt.exists() else 0})
        if result.returncode != 0:
            continue  # Preserve this bounded incomplete attempt, never use it as a chunk.
        try:
            validate_response(result.stdout.decode("latin1"), start, end, item["size"])
            if attempt.stat().st_size != size:
                raise ValueError("range_body_size_mismatch")
        except ValueError as error:
            record["attempts"][-1]["response_error"] = str(error)
            continue
        digest = sha256(attempt)
        os.link(attempt, path)  # Exclusive; no overwrite of another writer's bytes.
        with metadata.open("x") as stream:
            json.dump({"size": size, "sha256": digest}, stream)
        attempt.unlink()  # Published complete bytes remain at path.
        return {**record, "status": "complete_chunk", "sha256": digest}
    return {**record, "status": "failed", "error_code": "bounded_ranges_failed_attempts_preserved"}


def transfer(directory, item, workers, chunk_bytes, seconds):
    target = directory / item["relative"]
    if not target.resolve().is_relative_to(directory) or ".." in Path(item["relative"]).parts:
        raise ValueError("unsafe_target_path")
    record = {"file": item["relative"], "expected_bytes": item["size"]}
    if target.exists():
        return {**record, "status": "verified_existing", "sha256": verify(target, item)}
    if "sha256" not in item:
        raise ValueError("range_transport_requires_fixed_sha256")
    cache = directory / ".persistent-http-ranges" / (item["sha256"] + f"-{chunk_bytes}")
    if not cache.resolve().is_relative_to(directory):
        raise ValueError("unsafe_range_cache_path")
    cache.mkdir(parents=True, exist_ok=True)
    count = (item["size"] + chunk_bytes - 1) // chunk_bytes
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(chunk, cache, item, index, chunk_bytes, seconds): index for index in range(count)}
        for future in as_completed(futures):
            try:
                value = future.result()
            except Exception as error:
                value = {"index": futures[future], "status": "failed", "error_type": type(error).__name__}
            records.append(value)
            if len(records) % 20 == 0 or len(records) == count:
                complete = sum(value["status"] != "failed" for value in records)
                print(f"{item['relative']}: {complete}/{count} complete ranges", flush=True)
    record["ranges"] = sorted(records, key=lambda value: value["index"])
    if any(value["status"] == "failed" for value in records):
        return {**record, "status": "failed", "error_code": "incomplete_ranges_preserved"}
    assembled = cache / f"assembled-{uuid.uuid4().hex}"
    with assembled.open("xb") as output:
        for index in range(count):
            with (cache / f"{index:06d}.chunk").open("rb") as source:
                for block in iter(lambda: source.read(8 << 20), b""):
                    output.write(block)
    digest = verify(assembled, item)  # Full publisher size + SHA256, not just chunk hashes.
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(assembled, target)
        status = "downloaded_verified"
    except FileExistsError:
        verify(target, item)  # Concurrent bounded HF transport must have published identical bytes.
        status = "verified_concurrent_publication"
    return {**record, "status": status, "sha256": digest, "persistent_complete_ranges": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=("openwam_robotwin", "lingbot_robotwin", "robotwin_assets"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--max-seconds-per-route", type=int, default=120)
    args = parser.parse_args()
    directory = args.directory.resolve()
    names = {"openwam_robotwin": "OpenWAM-Alpha-Sim-RoboTwin-Full",
             "lingbot_robotwin": "lingbot-vla-v2-6b-robotwin", "robotwin_assets": "RoboTwin2.0-assets"}
    if args.receipt.exists() or directory.name != names[args.pair] or min(args.workers, args.chunk_mib, args.max_seconds_per_route) < 1:
        raise SystemExit("require dedicated directory, new receipt, positive bounds")
    started = datetime.now(timezone.utc).isoformat()
    items = manifest(args.pair)
    records = []
    for item in items:
        try:
            value = transfer(directory, item, args.workers, args.chunk_mib << 20, args.max_seconds_per_route)
        except Exception as error:
            value = {"file": item["relative"], "status": "failed", "error_type": type(error).__name__}
        records.append(value)
    report = {"kind": "verified_persistent_ranges_not_evaluation", "pair": args.pair,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(value["status"] != "failed" for value in records), "records": records,
        "global_proxy_modified": False, "model_loaded": False, "workers": args.workers,
        "chunk_bytes": args.chunk_mib << 20, "max_seconds_per_route": args.max_seconds_per_route}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"receipt={args.receipt} passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
