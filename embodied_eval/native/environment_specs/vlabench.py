"""Environment-only pinned protocol metadata; no selected model or weights."""
from .. import environment_records as c

BENCHMARK_REVISION = "cf588fe60c0c7282174fe979f5913170cfe69017"
TRACK_SHA256 = "61b07440c6f4272705bea0c8d721bc196e744232f48408075613cc9b6f7c8c04"
EXTENSION = "vlabench.native_proprio.v1"
ACTION = "vlabench-native-ee-target@1"
CONTROLLER = "vlabench-franka-native-ik@1"
SENSORS = ("forward_rgb", "franka_wrist_rgb", "right_rgb")
CAMERA_INDICES = (2, 3, 0)


def space(period):
    camera = c.TensorSpec(dtype="uint8", shape=(480, 480, 3), layout="HWC")
    properties = {name: {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}
                  for name, size in (("position_world_m", 3), ("euler_xyz_rad", 3), ("finger_targets_m", 2))}
    frames = {"world": c.FrameSpec(parent_id=None)}
    frames.update({key: c.FrameSpec(parent_id="world", dynamic=True)
                   for key in ("franka_eef", *SENSORS)})
    return c.SpaceSpec(spec_id="vlabench-franka@cf588fe", revision=BENCHMARK_REVISION,
        protocol_id="vlabench-track1-select-fruit", protocol_version="openwam-native-1",
        control_period_s=period, frames=frames,
        robots={"robot0": c.RobotSpec(embodiment_id="vlabench-franka-native-ee",
            base_frame_id="world", end_effectors={"eef": c.EndEffectorSpec(frame_id="franka_eef",
                                                                          tcp_id="franka/end_effector")})},
        sensors={key: c.SensorSpec(modality="rgb", role=role, mounted_on=mount, frame_id=key,
                    tensor=camera, color_order="RGB", available_rate_hz=1 / period)
                 for key, role, mount in zip(SENSORS, ("head_camera", "left_wrist_camera", "right_camera"),
                                             ("scene", "robot0", "scene"))},
        controllers={CONTROLLER: c.ControllerSpec(frame_id="world",
            parameter_schema={"type": "object", "properties": properties,
                              "required": list(properties), "additionalProperties": False},
            parameter_units={"position_world_m": "m", "euler_xyz_rad": "rad",
                             "finger_targets_m": "m"})},
        action_specs={ACTION: c.ActionSpec(command_kinds=("controller_command",),
            target_ids=("robot0/eef",), controller_refs=(CONTROLLER,))},
        required_extensions=(EXTENSION,), extensions={EXTENSION: {
            "ee_state_world_wxyz": "8 values: world position m[3], quaternion wxyz[4], native closed-flag[1]",
            "robot_base_world_m": "3 world-position values in metres; exact native converter offset",
            "instruction": "public native task instruction sampled for this policy decision"}})
