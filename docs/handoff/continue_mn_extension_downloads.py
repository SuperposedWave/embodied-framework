"""Finite, direct-first asset continuation after an existing download receipt.

Runs only the pinned component downloader, never model/simulator execution.
This is an asset-preparation process, not Runner recovery or stage approval.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from download_mn_extension_assets import DIRECTORIES

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("cosmos3", "dreamzero", "openwam_arx"), required=True)
    parser.add_argument("--first-receipt", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-resume-rounds", type=int, default=2)
    args = parser.parse_args()
    if args.directory.name != DIRECTORIES[args.component] or not 1 <= args.max_resume_rounds <= 3:
        raise ValueError("require exact component directory and bounded continuation rounds")
    if not args.first_receipt.resolve().is_relative_to(ROOT / "outputs"):
        raise ValueError("require a framework-local first evidence path")
    if args.output.exists():
        raise ValueError("new final continuation receipt required")
    manifest = json.loads((ROOT / "embodied_eval/mn/asset_manifests" / (args.component + ".json")).read_text())
    record = {"kind": "finite_pinned_asset_continuation_not_model_validation", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "component": args.component, "first_receipt": str(args.first_receipt), "revision": manifest["revision"],
        "max_resume_rounds": args.max_resume_rounds, "resumes": [], "model_loaded": False,
        "simulator_started": False, "global_proxy_modified": False, "passed": False}
    deadline = time.monotonic() + 10800
    while not args.first_receipt.is_file() and time.monotonic() < deadline:
        time.sleep(5)
    if args.first_receipt.is_file():
        first = json.loads(args.first_receipt.read_text())
        if first["component"] != args.component or first["revision"] != manifest["revision"] or Path(first["directory"]).resolve() != args.directory.resolve():
            raise ValueError("initial download identity mismatch")
        record["passed"] = first["complete_component"]
        for index in range(1, args.max_resume_rounds + 1):
            if record["passed"]:
                break
            resume_path = args.output.with_name(args.output.stem + f"-resume{index}.json")
            if resume_path.exists():
                raise ValueError("refuse to overwrite a prior resume receipt")
            result = subprocess.run([sys.executable, str(ROOT / "docs/handoff/download_mn_extension_assets.py"),
                "--component", args.component, "--directory", str(args.directory), "--receipt", str(resume_path),
                "--transport", "ranges", "--workers", "4", "--range-workers", "8" if args.component == "openwam_arx" else "4",
                "--max-seconds-per-route", "120"], cwd=ROOT)
            resumed = json.loads(resume_path.read_text()) if resume_path.is_file() else {}
            record["resumes"].append({"round": index, "receipt": str(resume_path), "exit_code": result.returncode,
                "complete_component": resumed.get("complete_component", False)})
            record["passed"] = resumed.get("complete_component", False)
    else:
        record["failure_class"] = "first_download_receipt_wait_deadline_exceeded"
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: record[key] for key in ("component", "passed", "resumes")}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
