"""VLABench Franka native EE evaluation; control period is read from the environment."""
from .. import contracts as c

MODEL_REVISION = "48bd67b89d489b14d03b8d92bc66e65d306df32e"
BENCHMARK_REVISION = "cf588fe60c0c7282174fe979f5913170cfe69017"
HF_REVISION = "480d3bc77b298434e69400a884a416d7196e677c"
CHECKPOINT_SHA256 = "6352f34db37f0a44ff8262f4d401d85d2a7b036799d00c6402ff81363684eb5e"
NORMALIZATION_SHA256 = "6434b2f23e8406d224424b57cca5e13527777dc260de446ec2bdf4876c29af8f"
CONFIG_SHA256 = "6a70a66d7e6c0c360910ba60f47eae858cb25b6bb32d0c196c3e1fae8244ebc8"
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


def capability(config):
    return c.CapabilitySpec(model_id="openwam-alpha-sim-vlabench@480d3bc",
        source_revision=MODEL_REVISION, checkpoint_ref="sha256:" + CHECKPOINT_SHA256,
        normalization_ref="sha256:" + NORMALIZATION_SHA256, roles=("action_policy",),
        supported_embodiments=("vlabench-franka-native-ee",),
        required_inputs=tuple(c.InputRequirement(key=key, kind="sensor", modality="rgb", dtype="uint8",
            shape=(480, 480, 3), layout="HWC") for key in SENSORS),
        observation_update="every_env_step", chunk_mode="adapter_owned", produced_action_specs=(ACTION,),
        supported_extensions=(EXTENSION,))


def profile(spec, cap):
    return c.AdaptationProfile(profile_id="openwam-vlabench-track1-select-fruit@1", revision="native-1",
        benchmark_spec_ref=spec.spec_id, model_id=cap.model_id, model_source_revision=cap.source_revision,
        checkpoint_ref=cap.checkpoint_ref, normalization_ref=cap.normalization_ref, role="action_policy",
        robot_ids=("robot0",), input_bindings={key: c.InputBinding(kind="sensor", source_id=key) for key in SENSORS},
        action_spec_refs=(ACTION,), discrete_spec_refs=(), tool_refs=(), chunk_mode="adapter_owned",
        execution_horizon_steps=1, required_extensions=(EXTENSION,),
        view_policy=c.ModelViewPolicy(task_fields=("instruction",), sensor_ids=SENSORS,
            frame_ids=tuple(spec.frames), extension_names=(EXTENSION,)))
