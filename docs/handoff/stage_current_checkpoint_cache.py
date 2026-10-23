"""Byte-identical local checkpoint copies for mmap/I/O diagnostics only.

Never copy download caches, partial files, credentials, source trees or whole
user homes. Original assets are immutable; each visible checkpoint file is
stream-copied and SHA256 compared. No model settings or tensor values change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    cache = Path(tempfile.mkdtemp(prefix="worldeval-native-assets-20261023-", dir="/tmp"))
    report = {"kind": "local_byte_identical_checkpoint_cache_not_new_weights_or_inference",
              "passed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "cache_root": str(cache), "checkpoints": {}, "files": [], "network_downloaded": False,
              "shared_assets_modified": False, "weights_normalization_or_config_values_changed": False}
    def save():
        (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for cfgpath in args.configs:
            cfg = json.loads(cfgpath.read_text())
            if cfg["contract_version"] != "0.3":
                raise ValueError("only_current_configurations")
            source = Path(cfg["model"]["options"]["checkpoint_dir"]).resolve()
            source.relative_to(ROOT.parent.parent.parent / "Model")
            if str(source) in report["checkpoints"]:
                continue
            destination = cache / source.name
            destination.mkdir()
            report["checkpoints"][str(source)] = str(destination)
            for directory, subdirs, names in os.walk(source):
                subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
                for name in sorted(names):
                    if name.startswith(".") or name.endswith((".partial", ".incomplete", ".lock")):
                        continue
                    original = Path(directory) / name
                    if not original.is_file():
                        raise ValueError("checkpoint_file_unavailable")
                    relative = original.relative_to(source)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copied = hashlib.sha256()
                    with original.open("rb") as reader, target.open("xb") as writer:
                        for block in iter(lambda: reader.read(8 << 20), b""):
                            copied.update(block)
                            writer.write(block)
                    actual = hashlib.sha256()
                    with target.open("rb") as reader:
                        for block in iter(lambda: reader.read(8 << 20), b""):
                            actual.update(block)
                    if copied.digest() != actual.digest() or target.stat().st_size != original.stat().st_size:
                        raise ValueError("byte_identical_local_copy_required")
                    report["files"].append({"source": str(original), "local": str(target),
                        "bytes": target.stat().st_size, "sha256": actual.hexdigest(), "source_and_local_SHA256_equal": True})
                    save()
                    print(json.dumps({"checkpoint": source.name, "file": str(relative), "bytes": target.stat().st_size, "equal": True}), flush=True)
        report["passed"] = True
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
