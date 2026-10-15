"""Reuse existing local bytes ONLY when the new publisher's hash is identical.

Same filename/size is not sufficient. No source asset or existing target is
overwritten. This does not select a different checkpoint or substitute a model.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from download_stage4_assets import verify
from download_mn_extension_assets import ROOT, DIRECTORIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=tuple(DIRECTORIES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if not source.is_dir() or source == target or target.name != DIRECTORIES[args.component] or args.receipt.exists():
        raise ValueError("explicit source, dedicated target and new receipt required")
    manifest = json.loads((ROOT / "embodied_eval/mn/asset_manifests" / (args.component + ".json")).read_text())
    records = []
    for item in manifest["objects"]:
        src, dst = source / item["relative"], target / item["relative"]
        if not src.resolve().is_relative_to(source) or not dst.resolve().is_relative_to(target):
            raise ValueError("unsafe_asset_path")
        if dst.exists() or not src.is_file() or src.stat().st_size != item["size"]:
            continue
        try:
            digest = verify(src, item)
        except ValueError:
            records.append({"file": item["relative"], "status": "different_bytes_not_reused"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
            status = "publisher_verified_local_bytes_linked"
        except FileExistsError:
            verify(dst, item)
            status = "publisher_verified_concurrent_publication"
        records.append({"file": item["relative"], "status": status, "sha256": digest, "bytes": item["size"]})
        print(f"publisher-verified reuse: {item['relative']}", flush=True)
    receipt = {"kind": "local_publisher_checksum_reuse_not_model_substitution", "at_utc": datetime.now(timezone.utc).isoformat(),
        "component": args.component, "source": str(source), "target": str(target), "revision": manifest["revision"], "records": records,
        "new_checkpoint_identity_preserved": True, "source_assets_modified": False, "existing_targets_overwritten": False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"receipt={args.receipt}")


if __name__ == "__main__":
    main()
