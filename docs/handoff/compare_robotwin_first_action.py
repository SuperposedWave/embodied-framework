"""Read saved real first-action observations; never rerun or relax acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    record = json.loads((path / "component-result.json").read_text())
    receipt = json.loads((path / "receipt.json").read_text())
    if not record["completed"] or not receipt["completed"]:
        raise ValueError("completed native operation observation required")
    return record


def compare(a, b):
    plans = []
    pa = [row for row in a["captured_operations"] if row["operation"].endswith("plan_path")]
    pb = [row for row in b["captured_operations"] if row["operation"].endswith("plan_path")]
    if [row["operation"] for row in pa] != [row["operation"] for row in pb]:
        raise ValueError("native arm call sequence changed")
    for left, right in zip(pa, pb):
        fields = {}
        for name in ("position", "velocity"):
            x, y = left["result"].get(name), right["result"].get(name)
            if x is None or y is None:
                fields[name] = {"exact": x == y, "available": False}
                continue
            u, v = np.asarray(x["values"], np.float64), np.asarray(y["values"], np.float64)
            fields[name] = {"exact": x == y, "shape_equal": x["shape"] == y["shape"], "dtype_equal": x["dtype"] == y["dtype"],
                "maximum_absolute_difference": float(np.max(np.abs(u-v))) if u.shape == v.shape and u.size else None}
        plans.append({"arm_call": left["operation"], "target_values_and_dtype_exact": left["target_pose"] == right["target_pose"],
            "initial_full_articulation_qpos_qvel_exact": left["initial_articulation"] == right["initial_articulation"],
            "native_planning_output_entirely_exact": left["result"] == right["result"], "output_fields": fields})
    controls_a = [row for row in a["captured_operations"] if not row["operation"].endswith("plan_path")]
    controls_b = [row for row in b["captured_operations"] if not row["operation"].endswith("plan_path")]
    first = next((index for index, (x, y) in enumerate(zip(controls_a, controls_b), 1) if x != y), None)
    return {"source_manifest_exact": a["manifest_sha256"] == b["manifest_sha256"],
        "recorded_action_exact": a["recorded_action"] == b["recorded_action"], "package_versions_exact": a["package_versions"] == b["package_versions"],
        "native_robot_source_exact": a["native_source_sha256"] == b["native_source_sha256"],
        "native_plans": plans, "control_operations": [len(controls_a), len(controls_b)],
        "native_control_sequence_entirely_exact": controls_a == controls_b, "first_different_control_operation_one_based": first,
        "first_completed_physics_steps": [a["prefix_comparison"]["native_physx_steps"], b["prefix_comparison"]["native_physx_steps"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = read(args.first), read(args.repeat)
    if a["mode"] != "native" or b["mode"] != "native":
        raise ValueError("two independent native-only observations required")
    report = {"kind": "read_only_native_robotwin_planning_difference_investigation", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": False, "fresh_model_inference": False, "formal_acceptance": False, "native_repeat": compare(a, b),
        "historical_full_trajectory_strict_failure_retained": True, "kernel_or_planner_internal_random_state_root_identified": False,
        "entire_historical_difference_uniquely_attributed": False, "acceptance_tolerance_changed": False,
        "meaning": "相同已保存动作、相同起步关节位置和速度，两次不经过Adapter的原生执行也出现规划输出差异；说明差异可在原生规划侧发生，不证明历史全部差异都由同一内部原因造成。"}
    sources = [args.first, args.repeat]
    if args.adapter:
        adapter = read(args.adapter)
        if adapter["mode"] != "adapter":
            raise ValueError("Benchmark component observation required")
        report["adapter_vs_first_native"] = compare(adapter, a)
        report["adapter_vs_repeat_native"] = compare(adapter, b)
        sources.append(args.adapter)
    report["evidence"] = [{"directory": str(path.resolve()), "component_result_sha256":
        hashlib.sha256((path / "component-result.json").read_bytes()).hexdigest()} for path in sources]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report["native_repeat"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
