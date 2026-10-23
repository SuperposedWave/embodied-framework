"""Check public actions against actual native calls, plus the direct reference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from compare_pi05_robotwin_closedloop import recorded_joint_row
    from compare_native_robotwin_closedloop import recorded_dual_pose_row
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    recorded = json.loads((args.case / "controller-exchanges/receipt.json").read_text())
    manifest = json.loads((args.case / "framework/manifest.json").read_text())
    episode = manifest["config"]["episodes"][0]["episode_id"]
    actions = []
    with (args.case / "framework/episodes" / episode / "trace.internal.jsonl").open() as stream:
        for line in stream:
            if line.startswith('{"event":"prediction",'):
                actions.append(json.loads(line)["data"]["action"])
    native = json.loads((args.case / "native/native-result.json").read_text())
    native_rows = [row for row in native["rows"] if row["env_step"] > 0]
    rows = []
    if len(actions) != len(recorded["records"]) or len(actions) != len(native_rows):
        raise ValueError("complete_actual_public_and_reference_inventory_required")
    for action, call, reference in zip(actions, recorded["records"], native_rows):
        if call["action_type"] == "qpos":
            expected = recorded_joint_row(action)
        elif len(action["commands"]) == 1:
            expected = np.asarray(action["commands"][0]["parameters"]["native_row"], dtype=action["native_dtype"])
        else:
            expected = recorded_dual_pose_row(action)
        actual = np.asarray(call["values"], dtype=call["dtype"])
        oracle = np.asarray(reference["native_action"], dtype=reference["native_dtype"])
        rows.append({"action": call["index"], "actual_call_dtype": call["dtype"],
            "public_action_to_actual_call_exact": expected.dtype == actual.dtype and np.array_equal(expected, actual),
            "actual_call_to_fresh_native_reference_exact": actual.dtype == oracle.dtype and np.array_equal(actual, oracle),
            "completed": call["completed"], "one_native_action_counted": call["counter_after"] - call["counter_before"] == 1})
    report = {"kind": "actual_controller_call_values_not_just_computed_conversion", "at_utc": datetime.now(timezone.utc).isoformat(),
              "case": str(args.case), "rows": rows,
              "all_public_to_actual_calls_exact": all(row["public_action_to_actual_call_exact"] for row in rows),
              "all_actual_calls_to_independent_reference_exact": all(row["actual_call_to_fresh_native_reference_exact"] for row in rows),
              "framework_physics_steps": read_physics(args.case / 'framework', episode),
              "native_physics_steps": native.get("native_physics_steps"),
              "extra_model_camera_planning_or_physics_calls": 0}
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    return 0 if report["all_public_to_actual_calls_exact"] else 1


def read_physics(framework, episode):
    return json.loads((framework / "episodes" / episode / "result.json").read_text())["execution_statistics"]["physics_steps"]


if __name__ == "__main__":
    raise SystemExit(main())
