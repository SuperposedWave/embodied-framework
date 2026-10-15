"""Compare passive execution receipts without relaxing strict equality gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_mn_components import write


def numeric(a, b):
    if a is None or b is None:
        return {"exact": a == b, "available_both": False}
    same_shape = a["shape"] == b["shape"]
    values_a, values_b = np.asarray(a["values"], np.float64), np.asarray(b["values"], np.float64)
    return {"exact": a == b, "shape_equal": same_shape, "dtype_equal": a["dtype"] == b["dtype"],
        "maximum_absolute_difference": float(np.max(np.abs(values_a-values_b))) if same_shape and values_a.size else None}


def first_difference(a, b):
    return next((index for index, (x, y) in enumerate(zip(a, b), 1) if x != y),
                min(len(a), len(b)) + 1 if len(a) != len(b) else None)


def compare(a, b):
    if not a["completed"] or not b["completed"] or not a["observer_enabled"] or not b["observer_enabled"]:
        raise ValueError("two_completed_observer_enabled_receipts_required")
    if (a["manifest_sha256"] != b["manifest_sha256"] or a["source_assets_verified"] != b["source_assets_verified"]
            or a["package_versions"] != b["package_versions"] or len(a["actions"]) != len(b["actions"])):
        raise ValueError("same_recorded_trial_assets_packages_and_action_count_required")
    saved_plan_mode = a.get("planning_mode", "unaltered_native_planner") == "saved_native_plan_controlled_diagnostic_not_evaluation"
    if a.get("planning_mode") != b.get("planning_mode") or a.get("plan_source_sha256") != b.get("plan_source_sha256"):
        raise ValueError("same_planning_or_controlled_replay_mode_required")
    rows = []
    for x, y in zip(a["actions"], b["actions"]):
        plans, motion = [], []
        expected_motion = 0 if saved_plan_mode else 2
        if len(x["plans"]) != 2 or len(y["plans"]) != 2 or len(x["motion_gen"]) != expected_motion or len(y["motion_gen"]) != expected_motion:
            raise ValueError("two_successfully_observed_arm_plans_required")
        for p, q in zip(x["plans"], y["plans"]):
            plans.append({"side": p["side"], "same_side": p["side"] == q["side"],
                "native_target_exact": p["target_pose"] == q["target_pose"],
                "initial_full_articulation_qpos_qvel_exact": p["before"] == q["before"],
                "python_numpy_torch_global_rng_exact": p["rng"] == q["rng"],
                "native_result_exact": p["result"] == q["result"],
                "position": numeric(p["result"].get("position"), q["result"].get("position")),
                "velocity": numeric(p["result"].get("velocity"), q["result"].get("velocity"))})
        for p, m, n in zip(x["plans"], x["motion_gen"], y["motion_gen"]):
            inputs = ("start_position", "goal_position", "goal_quaternion_wxyz")
            motion.append({"side": p["side"], "all_recorded_MotionGen_inputs_exact": all(m[key] == n[key] for key in inputs),
                "input_fields": {key: numeric(m[key], n[key]) for key in inputs},
                "python_numpy_torch_global_rng_exact": m["rng"] == n["rng"],
                "output_position": numeric(m.get("position"), n.get("position")),
                "output_velocity": numeric(m.get("velocity"), n.get("velocity")),
                "success_exact": m["success"] == n["success"]})
        controls_first = first_difference(x["controls"], y["controls"])
        physics_first = first_difference(x["physics"], y["physics"])
        first_physics = None
        if physics_first is not None and physics_first <= min(len(x["physics"]), len(y["physics"])):
            i = physics_first - 1
            ca = [entry for entry in x["controls"] if entry["physics_step_before"] <= i]
            cb = [entry for entry in y["controls"] if entry["physics_step_before"] <= i]
            first_physics = {"step_one_based_within_action": physics_first,
                "articulation_before_exact": x["physics"][i]["before"] == y["physics"][i]["before"],
                "controls_through_this_step_exact": ca == cb,
                "before_after": {when: {side: {key: numeric(x["physics"][i][when][side][key], y["physics"][i][when][side][key])
                    for key in ("qpos", "qvel")} for side in ("left", "right")} for when in ("before", "after")}}
        rows.append({"action_index": x["action_index"], "plans": plans, "motion_gen": motion,
            "control_operations": [len(x["controls"]), len(y["controls"])],
            "first_different_control_operation_one_based": controls_first,
            "physics_steps": [len(x["physics"]), len(y["physics"])],
            "first_different_physics_step_one_based": physics_first, "first_physics_difference": first_physics,
            "entire_recorded_action_exact": x == y})
    witnesses = [{"action_index": row["action_index"], "side": m["side"],
                  "global_rng_equal": m["python_numpy_torch_global_rng_exact"]}
        for row in rows for m in row["motion_gen"] if m["all_recorded_MotionGen_inputs_exact"]
        and (not m["output_position"]["exact"] or not m["output_velocity"]["exact"])]
    return {"modes": [a["mode"], b["mode"]], "source_manifest_assets_and_packages_exact": True,
        "planning_mode": a.get("planning_mode", "unaltered_native_planner"),
        "controlled_saved_plan_replay_not_fresh_planning": saved_plan_mode,
        "all_returned_plan_arrays_exact": all(p["native_result_exact"] for row in rows for p in row["plans"]),
        "all_control_operations_exact": all(row["first_different_control_operation_one_based"] is None for row in rows),
        "all_recorded_articulation_physics_states_exact": all(row["first_different_physics_step_one_based"] is None for row in rows),
        "rows": rows, "same_recorded_MotionGen_input_different_output_witnesses": witnesses,
        "native_planner_output_variation_observed_without_Adapter": bool(witnesses) and a["mode"] == b["mode"] == "native",
        "internal_planner_rng_all_state_or_kernel_root_uniquely_identified": False,
        "all_physics_internal_state_captured": False,
        "strict_entire_observation_trajectory_or_historical_failure_reclassified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path / "component-result.json" for path in (args.first, args.second)]
    report = {"kind": "passive_robotwin_execution_boundary_comparison", "synthetic": False,
        "fresh_model_inference": False, "formal_acceptance": False, "at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": compare(*[json.loads(path.read_text()) for path in paths]),
        "sources": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
    write(args.output, report)
    print(json.dumps({"output": str(args.output), "planner_variation_witnesses":
        report["comparison"]["same_recorded_MotionGen_input_different_output_witnesses"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
