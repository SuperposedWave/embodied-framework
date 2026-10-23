"""Environment-only pinned protocol metadata; no selected model or weights."""
from .. import environment_records as c

EXTENSION = "robotwin.native_proprio.v1"
SENSORS = ("head_camera", "left_camera", "right_camera")
PROTOCOLS = {'sealed-instruction': ('0aeea2d669c0f8516f4d5785f0aa33ba812c14b4', 'ee', 16), 'seeded-instruction': ('13c3c47ff4312dd62484bcd51be034af55c062d1', 'qpos', 14)}


def refs(variant):
    action_type = PROTOCOLS[variant][1]
    return f"robotwin-native-{action_type}@1", f"robotwin-{action_type}-controller@1"


def space(variant):
    revision, action_type, width = PROTOCOLS[variant]
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
