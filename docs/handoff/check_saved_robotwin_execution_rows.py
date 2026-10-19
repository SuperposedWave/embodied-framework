"""Pure production action conversion on saved four-action RoboTwin runs.

No model/environment instance, asset loading, GPU operation or controller call.
This computes expected native rows, not a live capture at the controller.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.execution_audit.resolve().relative_to(ROOT / "outputs")
    args.output.resolve().relative_to(ROOT / "outputs")
    if args.output.exists():
        raise ValueError("new_native_row_conversion_evidence_required")
    import numpy as np
    from embodied_eval.codec import decode
    from embodied_eval.v03 import contracts as c
    from embodied_eval.v03.transforms import action_for_environment
    from embodied_eval.native.native_benchmarks import NativeSimulator
    from docs.handoff.execution_audit import digest_file
    sources = ("embodied_eval/v03/transforms.py", "embodied_eval/native/native_benchmarks.py")
    hashes = {name: digest_file(ROOT / name) for name in sources}
    cases = []
    for route in ("openwam_robotwin", "openwam_dual_physical", "lingbot_robotwin"):
        pair = [json.loads((args.execution_audit / f"recent-{route}-{mode}.json").read_text()) for mode in ("direct", "bridge")]
        converted = []
        for run in pair:
            manifest_path = Path(run["directory"]) / "manifest.json"
            if digest_file(manifest_path) != run["protected_file_sha256"][str(manifest_path)]:
                raise ValueError("saved_manifest_changed")
            manifest = json.loads(manifest_path.read_text())
            space = decode(c.SpaceSpec, manifest["space"])
            rows = []
            for raw in run["actions"]:
                action = decode(c.Action, raw)
                # No physical frame conversion is allowed to need an unrecorded
                # snapshot here. These reviewed actions already target world.
                native_action = action_for_environment(action, space, None)
                row, kind = NativeSimulator.robotwin_native_action(native_action)
                rows.append({"mode": kind, "dtype": str(row.dtype), "values": row.tolist()})
            converted.append(rows)
        if any(len(rows) != 4 for rows in converted):
            raise ValueError("exactly_four_saved_actions_required")
        checks = []
        for i, (a, b) in enumerate(zip(*converted)):
            eq = a["mode"] == b["mode"] and a["dtype"] == b["dtype"] and np.array_equal(a["values"], b["values"])
            public_equal = pair[0]["actions"][i] == pair[1]["actions"][i]
            if public_equal and not eq:
                raise ValueError("equal_public_actions_mapped_to_different_native_rows")
            checks.append({"action_index": i + 1, "public_action_exact": public_equal,
                "computed_native_row_dtype_and_mode_exact": eq,
                "direct": a, "bridge": b,
                "direct_physics_steps": pair[0]["transitions"][i]["physics_steps"],
                "bridge_physics_steps": pair[1]["transitions"][i]["physics_steps"]})
        cases.append({"route": route, "steps": checks,
            "all_four_computed_native_rows_exact": all(x["computed_native_row_dtype_and_mode_exact"] for x in checks)})
    if any(digest_file(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError("production_conversion_source_changed")
    result = {"kind": "pure_saved_public_to_native_action_conversion_not_controller_wire_capture",
        "passed": True, "at_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "production_conversion_source_sha256": hashes,
        "new_inference": False, "new_simulation": False, "controller_called": False,
        "actual_controller_wire_captured": False, "strict_reproduction_gaps_closed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"passed": True, "routes": len(cases), "conversion_sides": 2,
                      "saved_actions_per_side": 4, "new_inference": False}))


if __name__ == "__main__":
    main()
