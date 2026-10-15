"""Repair completed failed asset objects; never restart inference or an episode.

Only objects with an already-written failed result may be selected, so the
active preparation no longer owns their range cache. Fixed publisher size/hash
and strict HTTP range checks stay unchanged. Existing failed evidence remains.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from download_hf_ranges import transfer
from prepare_dreamzero_backbone import ROOT, REPO, MODEL_ROOT
from prepare_mn_native_environment import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--file", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--range-workers", type=int, default=8)
    args = parser.parse_args()
    prep = args.preparation.resolve()
    prep.relative_to(ROOT / "outputs")
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if not 1 <= args.workers <= 2 or not 1 <= args.range_workers <= 8 or len(set(args.file)) != len(args.file):
        raise ValueError("bounded independent asset repair required")
    manifest_path = prep / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["repository"] != REPO or manifest["revision"] != "6b73f84e66371cdfe870c72acd6826e1d61cf279":
        raise ValueError("fixed native backbone manifest required")
    items = {item["relative"]: item for item in manifest["objects"]}
    selected = []
    for name in args.file:
        item = items[name]
        evidence = prep / "object-results" / (name.replace("/", "__") + ".json")
        previous = json.loads(evidence.read_text())
        if previous["status"] != "failed" or previous["file"] != name or previous["expected_bytes"] != item["size"]:
            raise ValueError("object must have finished and published its failed bounded result")
        selected.append(dict(item, url=f"https://huggingface.co/{REPO}/resolve/{manifest['revision']}/{quote(name, safe='/')}"))
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    def repair(item):
        try:
            result = transfer(MODEL_ROOT, item, args.range_workers, 8 << 20, 300)
        except Exception as error:
            result = {"file": item["relative"], "status": "failed", "error_type": type(error).__name__}
        write(args.output / (item["relative"].replace("/", "__") + ".json"), result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(repair, selected))
    receipt = {"kind": "selected_finished_failed_asset_range_repair_not_full_backbone_readiness",
        "passed": all(row["status"] != "failed" for row in rows), "records": rows,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "selected_files": args.file,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "direct_first_then_original_proxy_fallback": True, "global_proxy_modified": False,
        "model_loaded": False, "full_native_backbone_readiness_verified": False, "old_evidence_modified": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "selected_files", "full_native_backbone_readiness_verified")}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
