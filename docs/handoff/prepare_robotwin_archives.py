"""Extract only checksum-verified native archives and publish exclusive links.

Rejects traversal/symlinks/duplicate members, checks CRC while reading, records
file SHA-256. No overwrite, no ZIP without its reviewed public LFS checksum.
Can prepare one archive while other downloads remain incomplete.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from download_stage4_assets import verify, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--archive", choices=("embodiments", "objects", "background_texture"), required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory.name != "RoboTwin2.0-assets" or args.receipt.exists():
        raise SystemExit("dedicated assets directory and new receipt required")
    metadata = json.loads((ROOT / "embodied_eval/adapters/robotwin_assets_objects.json").read_text())
    item = next(v for v in metadata["objects"] if v["relative"] == args.archive + ".zip")
    checksum = verify(directory / item["relative"], item)
    stage = directory / (args.archive + "-verified-extracted")
    stage.mkdir(exist_ok=False)
    records = []
    with zipfile.ZipFile(directory / item["relative"]) as archive:
        names = set()
        members = []
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16) or path in names):
                raise ValueError("unsafe_or_duplicate_archive_member")
            names.add(path)
            if path.parts[0] == "__MACOSX":
                continue
            if path.parts[0] != args.archive or not (stage / path).resolve().is_relative_to(stage):
                raise ValueError("unexpected_archive_root")
            members.append((entry, path))
        for entry, path in members:
            target = stage / path
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(entry) as source, target.open("xb") as sink:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    sink.write(block)
                    digest.update(block)
            records.append({"relative": str(path), "size": entry.file_size, "sha256": digest.hexdigest()})
    published = []
    for name in ("RoboTwin-openwam", "RoboTwin-lingbot"):
        target = ROOT.parent / "upstream" / name / "assets" / args.archive
        if os.path.lexists(target):
            raise ValueError("native_asset_target_already_exists_no_overwrite")
        target.symlink_to(stage / args.archive, target_is_directory=True)
        published.append(str(target))
    receipt = {"kind": "one_verified_native_archive_preparation_not_evaluation",
               "checked_at_utc": datetime.now(timezone.utc).isoformat(), "archive": item["relative"],
               "archive_sha256": checksum, "dataset_revision": metadata["revision"],
               "passed": True, "all_robotwin_assets_ready": False, "file_count": len(records),
               "total_bytes": sum(v["size"] for v in records), "published_links": published, "files": records}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(f"verified/extracted {args.archive}: {len(records)} files; not full RoboTwin readiness")


if __name__ == "__main__":
    main()
