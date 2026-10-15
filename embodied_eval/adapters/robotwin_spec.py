"""Reviewed dual-arm native profiles; raw targets are not measured joint states."""
from .. import contracts as c
from .robotwin_identity import (OPENWAM_REVISION, OPENWAM_ROBOTWIN_REVISION,
                                    LINGBOT_REVISION, LINGBOT_ROBOTWIN_REVISION)

EXTENSION = "robotwin.native_proprio.v1"
SENSORS = ("head_camera", "left_camera", "right_camera")
VARIANTS = {
    "openwam_robotwin": (OPENWAM_REVISION, OPENWAM_ROBOTWIN_REVISION,
        "04b96af53eeeb111c64631f822efcbb82f4b186e", "ee", 16,
        "3eb52f759ebd004feda2772925d587054f9c2f05e1b979c3ff8c94fdd40e5db5"),
    "lingbot_robotwin": (LINGBOT_REVISION, LINGBOT_ROBOTWIN_REVISION,
        "0451855729ec904f970600e0aec8b84661423afe", "qpos", 14,
        "488ec6569c30dd69419358241efdb2e6c3488117768e7028a1e891be83e10db8"),
}


def refs(variant):
    action_type = VARIANTS[variant][3]
    return f"robotwin-native-{action_type}@1", f"robotwin-{action_type}-controller@1"


def space(variant):
    _, revision, _, action_type, width, _ = VARIANTS[variant]
    action, controller = refs(variant)
    properties = {
        "native_row": {"type": "array", "items": {"type": "number"}, "minItems": width, "maxItems": width},
        "native_dtype": {"type": "string", "enum": ["float32", "float64", "int32", "int64"]},
        "acquire_observation_after": {"type": "boolean"},
    }
    return c.SpaceSpec(spec_id=f"robotwin-{action_type}@{revision[:7]}", revision=revision,
        protocol_id="robotwin-demo-clean-adjust-bottle",
        protocol_version="openwam-sealed-native-1" if action_type == "ee" else "lingbot-seeded-instruction-port-1",
        control_period_s=None, timing_mode="measured_native", physics_period_s=.004,
        frames={key: c.FrameSpec(parent_id=None) for key in ("world", *SENSORS)},
        robots={"robot0": c.RobotSpec(embodiment_id="aloha-agilex-dual", base_frame_id="world")},
        sensors={key: c.SensorSpec(modality="rgb", role=role, mounted_on=mount, frame_id=key,
            tensor=c.TensorSpec(dtype="uint8", shape=(240, 320, 3), layout="HWC"), color_order="RGB")
            for key, role, mount in zip(SENSORS, ("head_camera", "left_wrist_camera", "right_wrist_camera"),
                                       ("scene", "robot0", "robot0"))},
        controllers={controller: c.ControllerSpec(frame_id="world",
            parameter_schema={"type": "object", "properties": properties,
                              "required": list(properties), "additionalProperties": False},
            parameter_units={"native_row": "controller_native", "native_dtype": "metadata",
                             "acquire_observation_after": "metadata"})},
        action_specs={action: c.ActionSpec(command_kinds=("controller_command",),
            target_ids=("robot0/base",), controller_refs=(controller,))},
        required_extensions=(EXTENSION,), extensions={EXTENSION: {
            "joint_drive_targets_and_grippers": "14: left joint drive targets rad[6], open fraction[1], right equivalent; NOT measured joint positions",
            "left_endpose_world_wxyz": "7: native world position metres[3], quaternion scalar-first wxyz[4]",
            "right_endpose_world_wxyz": "7: native world position metres[3], quaternion scalar-first wxyz[4]",
            "left_gripper_open_fraction": "native fraction; 0 closed, 1 open",
            "right_gripper_open_fraction": "native fraction; 0 closed, 1 open",
            "captured_at_sim_time_s": "actual acquisition time of this raw state and camera bundle; may be older than current control clock"}})


def capability(config):
    variant = config.adapter
    model_rev, _, hf, _, _, norm = VARIANTS[variant]
    action, _ = refs(variant)
    return c.CapabilitySpec(model_id=f"{variant}@{hf[:7]}", source_revision=model_rev,
        checkpoint_ref="hf-revision:" + hf, normalization_ref="sha256:" + norm,
        roles=("action_policy",), supported_embodiments=("aloha-agilex-dual",),
        required_inputs=tuple(c.InputRequirement(key=key, kind="sensor", modality="rgb", dtype="uint8",
            shape=(240, 320, 3), layout="HWC") for key in SENSORS)
            + (c.InputRequirement(key="instruction", kind="task"),),
        observation_update="decision_boundary", chunk_mode="adapter_owned",
        produced_action_specs=(action,), supported_extensions=(EXTENSION,),
        supported_timing_modes=("measured_native",))


def profile(variant, spec, cap):
    action, _ = refs(variant)
    return c.AdaptationProfile(profile_id=f"{variant}-demo-clean-adjust-bottle@1", revision="native-1",
        benchmark_spec_ref=spec.spec_id, model_id=cap.model_id, model_source_revision=cap.source_revision,
        checkpoint_ref=cap.checkpoint_ref, normalization_ref=cap.normalization_ref,
        role="action_policy", robot_ids=("robot0",),
        input_bindings={**{key: c.InputBinding(kind="sensor", source_id=key) for key in SENSORS},
                        "instruction": c.InputBinding(kind="task", source_id="instruction")},
        action_spec_refs=(action,), discrete_spec_refs=(), tool_refs=(), chunk_mode="adapter_owned",
        execution_horizon_steps=1, timing_mode="measured_native", required_extensions=(EXTENSION,),
        view_policy=c.ModelViewPolicy(sensor_ids=SENSORS, task_fields=("instruction",),
            extension_names=(EXTENSION,), frame_ids=tuple(spec.frames)))
