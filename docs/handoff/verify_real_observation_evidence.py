"""Read-only verification of the saved real raw/public observation arrays.

No model, simulator, sampling, repairs, tolerances or task-success inference.
The raw arrays and public arrays come from the SAME consumed observation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_mn_components import ROOT, write


def verify(directory):
    receipt_path = directory / "receipt.json"
    component_path = directory / "component-result.json"
    receipt = json.loads(receipt_path.read_text())
    component = json.loads(component_path.read_text())
    if (not receipt["passed"] or receipt["exit_code"] != 0 or receipt["supervisor_timed_out"]
            or not component["passed"] or component["synthetic"] or not component["native_close_returned"]
            or component["actual_saved_actions"] != 4 or len(component["snapshots"]) != 5):
        raise ValueError("completed_real_four_action_five_snapshot_projection_required")
    source = ROOT / "embodied_eval/native" / ("droid_simulator.py" if receipt["benchmark"] == "droid_sim" else "robodojo_simulator.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != component["adapter_source_sha256"]:
        raise ValueError("projection_source_changed_since_observation")
    expected_images = {"external_cam", "external_cam_2", "wrist_cam"} if receipt["benchmark"] == "droid_sim" else {
        "cam_head", "cam_left_wrist", "cam_right_wrist"}
    expected_states = {"arm_joint_positions", "gripper_open_fraction"} if receipt["benchmark"] == "droid_sim" else {
        side + suffix for side in ("left", "right") for suffix in ("_joint_positions", "_eef_position", "_eef_quaternion", "_gripper_target")}
    files = {}
    for index, snapshot in enumerate(component["snapshots"]):
        if (not snapshot["passed"] or snapshot["env_step"] != index or snapshot["raw_observations_consumed"] != 1
                or set(snapshot["images"]) != expected_images or set(snapshot["states"]) != expected_states
                or not snapshot["instruction_exact"] or not snapshot["all_native_public_keys_accounted_for"]):
            raise ValueError("incomplete_exact_same_execution_snapshot")
        name = snapshot["file"]
        if name != f"snapshot-{index:04d}.npz":
            raise ValueError("explicit_snapshot_filename_required")
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != snapshot["sha256"]:
            raise ValueError("saved_capture_bytes_changed")
        with np.load(path, allow_pickle=False) as capture:
            if set(capture.files) != {prefix + key for prefix in ("raw_", "public_") for key in expected_images | expected_states}:
                raise ValueError("saved_complete_native_and_public_array_sets_required")
            for key in expected_images | expected_states:
                a, b = capture["raw_" + key], capture["public_" + key]
                dtype = np.dtype("uint8" if key in expected_images else "float64")
                if a.dtype != dtype or b.dtype != dtype or not np.array_equal(a, b):
                    raise ValueError("real_native_public_array_values_or_types_differ")
                if key in expected_images:
                    checks = snapshot["images"][key]
                    if (hashlib.sha256(a.tobytes()).hexdigest() != checks["native_pixel_bytes_sha256"]
                            or hashlib.sha256(b.tobytes()).hexdigest() != checks["public_pixel_bytes_sha256"]
                            or not checks["frame_timestamp_and_sensor_provenance_exact"]):
                        raise ValueError("pixel_hash_or_sensor_metadata_differ")
                elif not snapshot["states"][key]["provenance_and_timestamp_exact"]:
                    raise ValueError("measured_state_and_control_target_provenance_differ")
        files[name] = digest
    return {"kind": "independently_read_saved_same_execution_arrays_not_historical_native_replay",
        "at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "benchmark": receipt["benchmark"],
        "trial": str(directory.relative_to(ROOT)), "synthetic": False, "snapshots": 5,
        "complete_three_cameras_each_snapshot_exact": True, "all_states_each_snapshot_exact": True,
        "required_camera_arm_state_frame_timestamp_and_provenance_checks_passed": True,
        "saved_capture_sha256": files, "adapter_source_sha256": component["adapter_source_sha256"],
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "component_result_sha256": hashlib.sha256(component_path.read_bytes()).hexdigest(),
        "new_model_inference_or_simulation": False, "historical_replay_failure_reclassified_as_pass": False,
        "formal_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.trial, args.output = args.trial.resolve(), args.output.resolve()
    args.trial.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    result = verify(args.trial)
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ("passed", "benchmark", "snapshots")}))


if __name__ == "__main__":
    main()
