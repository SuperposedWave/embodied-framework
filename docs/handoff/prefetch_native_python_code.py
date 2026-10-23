"""Parallel read-only warmup of known pure-Python dependency code.

No imports, cache relocation, bytecode writes, global settings or process
mutation. This is an I/O diagnostic, not an inference or acceptance result.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--environments", nargs="+", choices=("robotwin-stage4", "openwam-py310", "lingbot-stage4"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for name in args.environments:
        directory = ROOT.parent / "upstream/envs" / name
        for site in directory.glob("lib/python*/site-packages"):
            for package in ("sympy", "mpmath", "networkx"):
                files.extend(path for path in (site / package).rglob("*") if path.is_file() and path.suffix in (".py", ".pyc"))
    start = time.monotonic()
    def touch(path):
        value = path.read_bytes()
        return {"file": str(path), "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    with ThreadPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(touch, sorted(set(files))))
    report = {"kind": "readonly_dependency_code_prefetch_not_runtime_change_or_inference", "passed": True,
              "at_utc": datetime.now(timezone.utc).isoformat(), "duration_s": time.monotonic() - start,
              "files": rows, "total_bytes": sum(row["bytes"] for row in rows),
              "imports_or_rng_or_package_or_process_changes": False}
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"files": len(rows), "bytes": report["total_bytes"], "duration_s": report["duration_s"]}))


if __name__ == "__main__":
    main()
