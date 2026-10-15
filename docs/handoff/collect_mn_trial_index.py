"""A new evidence index of isolated trials; never copies/rewrites their traces."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="CASE=trial directory with framework/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved = []
    for declaration in args.case:
        name, folder = declaration.split("=", 1)
        if not name.replace("_", "").isalnum():
            raise ValueError("safe evidence label required")
        source = Path(folder).resolve()
        manifest = json.loads((source / "framework/manifest.json").read_text())
        summary = json.loads((source / "framework/summary.json").read_text())
        if manifest["synthetic"] or manifest["config"]["contract_version"] != "0.2":
            raise ValueError("real 0.2 trial required")
        resolved.append((name, source, summary))
    if len(set(name for name, _, _ in resolved)) != len(resolved):
        raise ValueError("duplicate trial label")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, source, _ in resolved:
        (args.output / name).symlink_to(source, target_is_directory=True)
    receipt = {"kind": "isolated_trial_evidence_index_not_runner_concurrency_or_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_traces_rewritten": False, "formal_acceptance": False,
        "results": [{"case": name, "source_directory": str(source), "completed": summary["completed"], "errors": summary["errors"]}
                    for name, source, summary in resolved]}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
