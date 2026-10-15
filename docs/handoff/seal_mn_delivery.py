"""Hash the delivered local source/evidence AFTER documentation is updated.

This is an integrity receipt, not a Git commit, protocol freeze or stage pass.
Old SHA256SUMS and original evidence are never written.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verification = json.loads((args.delivery / "verification.json").read_text())
    if not verification["delivery_integrity_passed"]:
        raise ValueError("repair delivery integrity before sealing")
    files = {ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "pyproject.toml", ROOT / "uv.lock"}
    for folder in ("embodied_eval", "tests", "examples", "docs"):
        files.update(path for path in (ROOT / folder).rglob("*")
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and
            path.suffix in (".py", ".md", ".json", ".txt", ".toml"))
    files.update(path for path in args.delivery.resolve().iterdir() if path.is_file())
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sorted(files)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "SHA256SUMS").write_text(''.join(f"{value}  {path}\n" for path, value in hashes.items()))
    receipt = {"kind": "mn_source_and_delivery_sha256_not_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
        "files": len(hashes), "source_hashes": hashes, "delivery_integrity_passed": True,
        "stage4_formally_accepted": False, "contract_frozen": False,
        "framework_git_commit_created": False, "old_sha256sums_rewritten": False,
        "historical_stage2_unchanged_files": len(verification["historical_stage2_sha256_files"])}
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_hashes"}))


if __name__ == "__main__":
    main()
