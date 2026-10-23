"""Current 0.3 native-reference entry: reuse native loops, never old runners.

The reference loops call fixed-source clients and simulators directly. Only the
comparison-side naming of raw snapshots is projected below, independently of
production Adapter code. Neither historical evidence nor execution is rewritten.
"""
import argparse
import json
from pathlib import Path
import resource


def current_snapshot(original, observation, values, images, acquired_at=None, state_acquired_at=None):
    if observation.get("contract_version") != "0.3":
        raise ValueError("current_reference_requires_current_observation")
    cameras = {
        "agentview_image": "head", "robot0_eye_in_hand_image": "wrist",
        "forward_rgb": "head", "franka_wrist_rgb": "wrist", "right_rgb": "external_right",
        "head_camera": "head", "left_camera": "left_wrist", "right_camera": "right_wrist",
    }
    projected_images = {cameras[name]: value for name, value in images.items()}
    if len(projected_images) != len(images):
        raise ValueError("native_camera_identity_collision")
    projected_values = {}
    for name, value in values.items():
        if name == "joint_drive_targets":
            if len(value) != 14:
                raise ValueError("complete_native_dual_drive_vector_required")
            for side, offset in (("left", 0), ("right", 7)):
                projected_values[side + "_joint_drive_targets"] = value[offset:offset + 6]
                grip = values[side + "_gripper_open_fraction"]
                if float(grip[0]) != float(value[offset + 6]):
                    raise ValueError("native_mixed_and_separate_gripper_targets_disagree")
        else:
            key = {"gripper_positions": "gripper_joint_positions",
                   "left_gripper_open_fraction": "left_gripper_open_fraction_target",
                   "right_gripper_open_fraction": "right_gripper_open_fraction_target"}.get(name, name)
            projected_values[key] = value
    if state_acquired_at:
        raise ValueError("explicit_current_state_timestamp_projection_required")
    return original(observation, projected_values, projected_images, acquired_at, None)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--framework", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--setup-seed", type=int, default=0)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["contract_version"] != "0.3":
        raise ValueError("only_current_configuration_allowed")
    args.output.mkdir(parents=True, exist_ok=False)
    import replay_mn_native_actions as comparison
    original = comparison.snapshot
    comparison.snapshot = lambda *a, **k: current_snapshot(original, *a, **k)
    family = cfg["benchmark"]["adapter"]
    if family in ("libero", "vlabench"):
        from compare_native_singlearm_closedloop import run
    elif family == "robotwin":
        from compare_native_robotwin_closedloop import run
    else:
        raise ValueError("no_reviewed_direct_reference_for_this_environment_in_this_tool")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
