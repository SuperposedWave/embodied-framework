"""Derive a research-only 55-cell ledger without editing the stage-0 snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/handoff/source-materials/compatibility-draft.json"

# Only discoveries with concrete matching assets or executed evidence are promoted.
UPDATES = {
    "AD18": ("accepted_single_trial", "Stage 3 real OpenWAM-LIBERO trial; not an official 50-trial aggregate"),
    "AD14": ("matching_asset_identified_not_run", "Official gs://openpi-assets/checkpoints/pi05_libero/ with norm_stats.json"),
    "AB18": ("matching_asset_identified_not_run", "OpenWAM-Alpha-Sim-VLABench fixed HF revision 480d3bc77b298434e69400a884a416d7196e677c"),
    "AE4": ("matching_asset_identified_not_run", "LingBot-VLA 2.0 RoboTwin checkpoint fixed HF revision 0451855729ec904f970600e0aec8b84661423afe"),
    "AE18": ("matching_asset_identified_not_run", "OpenWAM-Alpha-Sim-RoboTwin-Full fixed HF revision 04b96af53eeeb111c64631f822efcbb82f4b186e"),
    "AF18": ("matching_asset_identified_not_run", "OpenWAM-Alpha-Sim-RoboDojo fixed HF revision 2c1302294e3ba8319bbdb2c803b7a27de9292d03"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    combinations = source["combinations"]
    cells = [item["source_cell"] for item in combinations]
    if len(cells) != 55 or len(set(cells)) != 55 or not set(UPDATES) <= set(cells):
        raise SystemExit("source matrix does not match the reviewed 55 unique cells")
    if args.output.exists():
        raise SystemExit("output already exists")

    rows = []
    for item in combinations:
        cell = item["source_cell"]
        status, reason = UPDATES.get(cell, ("unresolved", item["required_adaptation_or_blocker"]))
        rows.append({"source_cell": cell, "model": item["model_name"],
                     "benchmark": item["benchmark_id"], "stage0_status": item["status"],
                     "stage4_research_status": status, "reason": reason,
                     "real_framework_run": cell == "AD18"})
    counts = {name: sum(row["stage4_research_status"] == name for row in rows)
              for name in ("accepted_single_trial", "matching_asset_identified_not_run", "unresolved")}
    if counts != {"accepted_single_trial": 1, "matching_asset_identified_not_run": 5, "unresolved": 49}:
        raise SystemExit(f"unexpected ledger counts: {counts}")
    result = {"document_kind": "research_coverage_ledger_not_runtime_configuration",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_snapshot": str(SOURCE.relative_to(ROOT)),
              "definitions": {
                  "accepted_single_trial": "Real framework/native comparison passed for one fixed trial; not a full benchmark score",
                  "matching_asset_identified_not_run": "A dedicated published asset was identified but not loaded or evaluated here",
                  "unresolved": "Model identity, protocol, role, matching asset, endpoint or native comparison remains incomplete"},
              "counts": counts, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output}: {counts}")


if __name__ == "__main__":
    main()
