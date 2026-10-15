"""Read-only exact comparison of two native-only recorded-action executions.

Never claims all historical differences explained, modifies renderer settings,
relaxes pixel tolerance, performs inference or converts a failed replay to pass.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_mn_components import ROOT, write


def load(directory):
    directory = directory.resolve()
    directory.relative_to(ROOT / "outputs")
    result = json.loads((directory / "native-result.json").read_text())
    receipt = json.loads((directory / "receipt.json").read_text())
    if (not result["native_factory_created"] or not result["close_completed"]
            or result["model_or_benchmark_adapter_runtime_used"] or result["fresh_policy_inference"]
            or result.get("exception_chain") or receipt["child_exit_code"] not in (0, 1)):
        raise ValueError("completed_native_only_replay_with_checked_exit_required")
    captures = result["native_image_captures"]
    if len(captures) != result["replayed_actions"] + 1 or [item["env_step"] for item in captures] != list(range(len(captures))):
        raise ValueError("complete_consecutive_native_image_captures_required")
    for item in captures:
        path = directory / item["file"]
        path.resolve().relative_to(directory)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("native_image_capture_checksum_mismatch")
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("robodojo", "droid"), default="robodojo")
    args = parser.parse_args()
    a, ar = load(args.first)
    b, br = load(args.repeat)
    camera_names = ({"cam_head", "cam_left_wrist", "cam_right_wrist"} if args.benchmark == "robodojo"
        else {"external_cam", "external_cam_2", "wrist_cam"})
    if ar["trial"] != br["trial"] or len(a["rows"]) != len(b["rows"]):
        raise ValueError("same_recorded_action_trial_and_complete_rows_required")
    metadata = {"same_current_physical_gpu_and_driver": ar["render_gpu_identity"] == br["render_gpu_identity"],
        "resolved_simulation_config_exact": a["resolved_native_simulation_config"] == b["resolved_native_simulation_config"],
        "reset_camera_extrinsics_exact": a["native_camera_extrinsics"] == b["native_camera_extrinsics"],
        "allowlisted_render_settings_exact": a["allowlisted_render_settings"] == b["allowlisted_render_settings"],
        "native_source_factory_exact": a["native_source_factory_sha256"] == b["native_source_factory_sha256"],
        "native_action_and_physics_counts_exact": a["replayed_actions"] == b["replayed_actions"] and a["independent_observed_physics_steps"] == b["independent_observed_physics_steps"],
        "native_metrics_exact": a["native_metrics"] == b["native_metrics"],
        "nonimage_states_and_feedback_both_exact_to_same_recorded_reference": all(value["exact_equal"]
            for record in (a, b) for row in record["rows"] for name, value in row["fields"].items()
            if name not in camera_names)}
    rows = []
    for left, right in zip(a["native_image_captures"], b["native_image_captures"]):
        with np.load(args.first / left["file"], allow_pickle=False) as x, np.load(args.repeat / right["file"], allow_pickle=False) as y:
            if set(x.files) != camera_names or set(x.files) != set(y.files):
                raise ValueError("exact_native_camera_inventory_required")
            cameras = {}
            for name in sorted(x.files):
                u, v = x[name], y[name]
                if u.dtype != np.uint8 or v.dtype != np.uint8 or u.shape != v.shape:
                    raise ValueError("same_native_RGB_shape_and_dtype_required")
                delta = np.abs(u.astype(np.int16) - v.astype(np.int16))
                cameras[name] = {"shape": list(u.shape), "dtype": str(u.dtype), "exact_equal": bool(np.array_equal(u, v)),
                    "different_scalar_values": int(np.count_nonzero(delta)), "total_scalar_values": int(u.size),
                    "maximum_absolute_pixel_difference": int(delta.max()), "mean_absolute_pixel_difference": float(delta.mean()),
                    "first_native_sha256": hashlib.sha256(u.tobytes()).hexdigest(), "repeat_native_sha256": hashlib.sha256(v.tobytes()).hexdigest()}
            rows.append({"env_step": left["env_step"], "cameras": cameras})
    proof = {"kind": "native_only_exact_" + args.benchmark + "_render_repeat_investigation_not_replay_pass", "at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": False, "fresh_policy_inference": False, "formal_acceptance": False, "controlled_boundary_checks": metadata,
        "native_images_entirely_exact": all(item["exact_equal"] for row in rows for item in row["cameras"].values()),
        "rows": rows, "historical_strict_failure_preserved": True, "renderer_setting_or_pixel_tolerance_changed": False,
        "root_rendering_kernel_or_temporal_state_identified": False, "all_historical_differences_uniquely_attributed": False,
        "meaning": "同一当前GPU、同一保存动作、相同起步相机参数和仿真配置，两次不经过Adapter的原生重放画面也须逐像素比较；差异不能自动归咎于Adapter，也不因此撤销历史严格失败。",
        "sources": [{"directory": str(path.resolve().relative_to(ROOT)),
            "native_result_sha256": hashlib.sha256((path / "native-result.json").read_bytes()).hexdigest(),
            "receipt_sha256": hashlib.sha256((path / "receipt.json").read_bytes()).hexdigest()} for path in (args.first, args.repeat)]}
    args.output.resolve().relative_to(ROOT / "outputs")
    write(args.output, proof)
    print(json.dumps({"controlled_boundary_checks": metadata, "native_images_entirely_exact": proof["native_images_entirely_exact"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
