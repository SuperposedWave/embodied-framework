"""Reconcile published complete files, without downloading or starting a model.

Failed download/repair receipts are immutable. Only whole-file verification of
all fixed manifest objects permits a new private native offline cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from prepare_dreamzero_backbone import ROOT, REPO, MODEL_ROOT
from download_stage4_assets import verify
from prepare_mn_native_environment import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prep, args.output = args.manifest_from.resolve(), args.output.resolve()
    prep.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    manifest_bytes = (prep / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    revision = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
    if manifest["repository"] != REPO or manifest["revision"] != revision or len(manifest["objects"]) != 15 or sum(row["size"] for row in manifest["objects"]) != 82243108389:
        raise ValueError("source-fixed_complete_native_backbone_manifest_required")
    if hashlib.sha256(manifest_bytes).hexdigest() != "5dc952f607365dc638b329d4f4c10a072c83571c1fb17a93a0e5e211bdec8cd6":
        raise ValueError("previously_recorded_native_manifest_bytes_changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", manifest)
    started = datetime.now(timezone.utc).isoformat()
    def check(row):
        path = MODEL_ROOT / row["relative"]
        result = {"file": row["relative"], "expected_bytes": row["size"], "status": "failed"}
        try:
            verify(path, row)
            result.update(status="verified_published_complete_object", expected_sha256=row.get("sha256"),
                expected_git_blob=row.get("git_blob"), actual_bytes=path.stat().st_size)
        except Exception as error:
            # Never include an arbitrary upstream error string or URL.
            result["error_type"] = type(error).__name__
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(check, manifest["objects"]))
    passed = all(row["status"] != "failed" for row in records)
    cache = args.output / "offline-hf-cache"
    if passed:
        repo_cache = cache / "models--Wan-AI--Wan2.1-I2V-14B-480P"
        for row in manifest["objects"]:
            target = repo_cache / "snapshots" / revision / row["relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(MODEL_ROOT / row["relative"])
        ref = repo_cache / "refs/main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        with ref.open("x") as stream:
            stream.write(revision)
    receipt = {"kind": "fixed_native_runtime_whole_asset_reconciliation_not_model_validation", "passed": passed,
        "repository": REPO, "revision": revision, "directory": str(MODEL_ROOT),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "selected_files": len(records),
        "selected_bytes": sum(row["size"] for row in manifest["objects"]), "records": records,
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "private_offline_cache": str(cache) if passed else None,
        "offline_main_alias_to_fixed_revision": revision if passed else None,
        "model_loaded": False, "download_or_proxy_changes": False, "old_evidence_modified": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "selected_files", "selected_bytes", "private_offline_cache")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
