"""Download ONLY source-defined dedicated checkpoints, never run a policy.

Uses fixed HF revision / GCS object generations and validates public checksums.
Proxy bypass is scoped to each curl process; failures fall back to its original
environment. No tokens, proxy values, or raw curl error output enter receipts.
Existing final files are verified, not overwritten. Partial transfers can resume.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
REVISION = "480d3bc77b298434e69400a884a416d7196e677c"
VLA_FILES = [
    ("checkpoint_step_6000.safetensors", 24813767464, "sha256", "6352f34db37f0a44ff8262f4d401d85d2a7b036799d00c6402ff81363684eb5e"),
    ("config.yaml", 3268, "sha256", "6a70a66d7e6c0c360910ba60f47eae858cb25b6bb32d0c196c3e1fae8244ebc8"),
    ("normalization_stats.npy", 779, "sha256", "6434b2f23e8406d224424b57cca5e13527777dc260de446ec2bdf4876c29af8f"),
    ("tokenizer/google/umt5-xxl/spiece.model", 4548313, "sha256", "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    ("tokenizer/google/umt5-xxl/tokenizer.json", 16837417, "sha256", "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b"),
    ("tokenizer/google/umt5-xxl/special_tokens_map.json", 6623, "git_blob", "14855e7052ffbb595057dfd791d293c1c940db2c"),
    ("tokenizer/google/umt5-xxl/tokenizer_config.json", 61728, "git_blob", "4e1cc1cd85599ce0b47fd0a746af188fe4043ff2"),
]


def manifest(pair):
    if pair in ("openwam_robotwin", "lingbot_robotwin", "robotwin_assets", "qwen3_vl_processor"):
        metadata = json.loads((ROOT / f"embodied_eval/adapters/{pair}_objects.json").read_text())
        prefix = "datasets/" if metadata.get("repo_type") == "dataset" else ""
        return [{**item, "url": f"https://huggingface.co/{prefix}{metadata['repo_id']}/resolve/"
                 f"{metadata['revision']}/{item['relative']}"} for item in metadata["objects"]]
    if pair == "pi05_vlabench":
        metadata = json.loads((ROOT / "embodied_eval/adapters/pi05_vlabench_objects.json").read_text())
        return [{**item, "url": f"https://huggingface.co/{metadata['repo_id']}/resolve/"
                 f"{metadata['revision']}/{item['relative']}"} for item in metadata["objects"]]
    if pair == "pi05_libero":
        metadata = json.loads((ROOT / "embodied_eval/adapters/pi05_libero_objects.json").read_text())
        return [{"relative": item["name"].removeprefix("checkpoints/pi05_libero/"),
            "url": "https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/"
                   + quote(item["name"], safe="") + "?alt=media&generation=" + item["generation"],
            "size": int(item["size"]), "crc32c": item["crc32c"], "md5": item.get("md5Hash"),
            "sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
                      if item["name"].endswith("norm_stats.json") else None}
                for item in metadata["objects"]]
    return [{"relative": name, "size": size, kind: digest,
        "url": f"https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-VLABench/resolve/{REVISION}/{name}"}
            for name, size, kind, digest in VLA_FILES]


def verify(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError("asset_size_mismatch")
    sha = hashlib.sha256()
    md5 = hashlib.md5() if item.get("md5") else None
    git_blob = hashlib.sha1(f"blob {item['size']}\0".encode()) if "git_blob" in item else None
    crc = None
    if item.get("crc32c"):
        import google_crc32c
        crc = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
            for digest in (md5, git_blob, crc):
                if digest is not None:
                    digest.update(block)
    if ((item.get("sha256") and sha.hexdigest() != item["sha256"])
            or (md5 and base64.b64encode(md5.digest()).decode() != item["md5"])
            or (git_blob and git_blob.hexdigest() != item["git_blob"])
            or (crc and base64.b64encode(crc.digest()).decode() != item["crc32c"])):
        raise ValueError("asset_checksum_mismatch")
    return sha.hexdigest()


def transfer(directory, item, max_seconds, transport="curl"):
    relative = PurePosixPath(item["relative"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe_asset_path")
    target = directory / relative
    if not target.resolve().is_relative_to(directory):
        raise ValueError("asset_symlink_escapes_checkpoint")
    record = {"file": str(relative), "expected_bytes": item["size"], "attempts": []}
    try:
        if target.exists():
            record.update(status="verified_existing", sha256=verify(target, item))
            return record
        target.parent.mkdir(parents=True, exist_ok=True)
        if transport == "hf_xet":
            if not item["url"].startswith("https://huggingface.co/"):
                raise ValueError("hf_transport_requires_pinned_hf_object")
            pieces = item["url"].removeprefix("https://huggingface.co/").split("/")
            dataset = pieces[0] == "datasets"
            if dataset:
                pieces.pop(0)
            repo, revision = "/".join(pieces[:2]), pieces[3]
            code = ("import sys; from huggingface_hub import hf_hub_download; "
                    "p=hf_hub_download(repo_id=sys.argv[1],revision=sys.argv[2],filename=sys.argv[3],"
                    "repo_type=sys.argv[4],cache_dir=sys.argv[5],token=False); print('HF_VERIFIED_CANDIDATE '+p)")
            for route in ("temporary_direct", "original_proxy_environment"):
                environment = os.environ.copy()
                environment.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_XET_HIGH_PERFORMANCE="1")
                if route == "temporary_direct":
                    for key in PROXY_KEYS:
                        environment.pop(key, None)
                try:
                    result = subprocess.run([sys.executable, "-c", code, repo, revision, item["relative"],
                        "dataset" if dataset else "model", str(directory / ".verified-download-cache")],
                        env=environment, capture_output=True, text=True, timeout=max_seconds)
                    record["attempts"].append({"route": route, "transport": "hf_xet", "exit_code": result.returncode})
                    lines = [line for line in result.stdout.splitlines() if line.startswith("HF_VERIFIED_CANDIDATE ")]
                    if result.returncode == 0 and lines:
                        candidate = Path(lines[-1].removeprefix("HF_VERIFIED_CANDIDATE "))
                        if not candidate.resolve().is_relative_to(directory / ".verified-download-cache"):
                            raise ValueError("hf_cache_path_escapes_checkpoint")
                        digest = verify(candidate, item)
                        os.link(candidate.resolve(), target)  # Link verified bytes, never an HF snapshot symlink.
                        record.update(status="downloaded_verified", sha256=digest)
                        print(f"verified {relative} ({item['size']} bytes)", flush=True)
                        return record
                except subprocess.TimeoutExpired:
                    record["attempts"].append({"route": route, "transport": "hf_xet", "exit_code": "bounded_timeout"})
            raise RuntimeError("bounded_hf_download_failed_cache_preserved")
        partial = target.with_name(target.name + ".stage4.partial")
        if not partial.resolve().is_relative_to(directory):
            raise ValueError("partial_symlink_escapes_checkpoint")
        for route in ("temporary_direct", "original_proxy_environment"):
            environment = os.environ.copy()
            if route == "temporary_direct":
                for key in PROXY_KEYS:
                    environment.pop(key, None)
            result = subprocess.run(["curl", "-sS", "-fL", "--http1.1", "--connect-timeout", "20",
                "--max-time", str(max_seconds), "--continue-at", "-", "--output", str(partial), item["url"]],
                env=environment, capture_output=True)
            record["attempts"].append({"route": route, "exit_code": result.returncode})
            if result.returncode == 0:
                break
        else:
            raise RuntimeError("bounded_download_failed_partial_preserved")
        digest = verify(partial, item)
        if target.exists():
            raise RuntimeError("final_asset_appeared_during_download_partial_preserved")
        # Link is exclusive (cannot overwrite another writer); unlink only our partial.
        os.link(partial, target)
        partial.unlink()
        record.update(status="downloaded_verified", sha256=digest)
        print(f"verified {relative} ({item['size']} bytes)", flush=True)
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error_code=str(error)
                      if isinstance(error, (ValueError, RuntimeError)) else "asset_operation_failed")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=("pi05_libero", "openwam_vlabench", "pi05_vlabench", "openwam_robotwin", "lingbot_robotwin", "robotwin_assets", "qwen3_vl_processor"), required=True)
    parser.add_argument("--directory", type=Path, required=True, help="Dedicated checkpoint directory")
    parser.add_argument("--receipt", type=Path, required=True, help="New evidence path")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-seconds-per-route", type=int, default=600)
    parser.add_argument("--transport", choices=("curl", "hf_xet"), default="curl")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.receipt.exists() or args.workers < 1 or args.max_seconds_per_route < 1:
        raise SystemExit("receipt must be new and worker/timeout limits positive")
    if directory == Path("/") or directory.parent == directory or directory.name not in (
            "pi05_libero", "OpenWAM-Alpha-Sim-VLABench", "pi05_vlabench_primitive10",
            "OpenWAM-Alpha-Sim-RoboTwin-Full", "lingbot-vla-v2-6b-robotwin", "RoboTwin2.0-assets",
            "Qwen3-VL-4B-Instruct-processor"):
        raise SystemExit("use an explicitly named, dedicated checkpoint directory")
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    items = manifest(args.pair)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda item: transfer(directory, item, args.max_seconds_per_route, args.transport), items))
    report = {"kind": "checkpoint_download_and_integrity_only_not_evaluation", "pair": args.pair,
        "directory": str(directory), "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(record["status"] != "failed" for record in records),
        "expected_total_bytes": sum(item["size"] for item in items),
        "global_proxy_modified": False, "model_loaded": False, "records": records}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"receipt={args.receipt} passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
