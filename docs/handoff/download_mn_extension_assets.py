"""Pinned component assets only: no model inference or benchmark evaluation.

Reuse bounded direct-first/proxy-fallback transfers. Final files are published
only after complete publisher checksum validation; existing files never replaced.
Receipts contain no tokens, proxy values, response headers or signed redirects.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import quote
from download_stage4_assets import transfer

ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = {"cosmos3": "Cosmos3-Nano-Policy-DROID", "dreamzero": "DreamZero-DROID", "openwam_arx": "OpenWAM-Alpha-Sim-RoboDojo",
    "pi05_arx": "pi05_arx_x5_seed0_step59999", "droid_sim": "DROID-sim-environments", "robodojo": "RoboDojo"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=tuple(DIRECTORIES), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds-per-route", type=int, default=180)
    parser.add_argument("--max-file-bytes", type=int, default=None, help="Explicit metadata-only preparation limit")
    parser.add_argument("--transport", choices=("curl", "ranges"), default="curl")
    parser.add_argument("--range-workers", type=int, default=8)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if (directory.name != DIRECTORIES[args.component] or args.receipt.exists()
            or not 1 <= args.workers <= 12 or not 1 <= args.range_workers <= 12 or args.max_seconds_per_route < 1
            or args.max_file_bytes is not None and args.max_file_bytes < 1):
        raise SystemExit("require named component directory, new receipt and positive bounded limits")
    metadata = json.loads((ROOT / "embodied_eval/native/asset_manifests" / (args.component + ".json")).read_text())
    prefix = "datasets/" if metadata["kind"] == "datasets" else ""
    objects = metadata["objects"]
    items = [{**item, "url": "https://huggingface.co/" + prefix + metadata["repository"] + "/resolve/"
        + metadata["revision"] + "/" + quote(item.get("remote", item["relative"]), safe="/")}
        for item in objects if args.max_file_bytes is None or item["size"] <= args.max_file_bytes]
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    def download(item):
        if args.transport == "ranges" and item["size"] > 32 << 20 and "sha256" in item:
            from download_hf_ranges import transfer as ranges
            try:
                return ranges(directory, item, args.range_workers, 8 << 20, args.max_seconds_per_route)
            except Exception as error:
                return {"file": item["relative"], "status": "failed", "error_type": type(error).__name__}
        return transfer(directory, item, args.max_seconds_per_route)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(download, items))
    passed = bool(items) and all(record["status"] != "failed" for record in records)
    report = {"kind": "component_asset_preparation_not_evaluation", "component": args.component,
        "directory": str(directory), "revision": metadata["revision"], "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "complete_component": passed and len(items) == len(objects), "selected_files": len(items),
        "omitted_files": len(objects) - len(items), "selected_bytes": sum(item["size"] for item in items),
        "max_file_bytes": args.max_file_bytes, "model_loaded": False, "real_evaluation": False,
        "transport": args.transport, "workers": args.workers, "range_workers": args.range_workers,
        "global_proxy_modified": False, "records": records}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"receipt={args.receipt} passed={passed} complete_component={report['complete_component']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
