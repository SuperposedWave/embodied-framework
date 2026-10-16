"""Versioned metadata for the narrow OpenWAM-alpha / LIBERO spatial pilot."""
from .. import _legacy_contracts as c

EXTENSION = "libero.native_proprio.v1"
ACTION = "libero-osc7@1"
CONTROLLER = "libero-panda-osc-pose@1"
MODEL_REVISION = "48bd67b89d489b14d03b8d92bc66e65d306df32e"
BENCHMARK_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_SHA256 = "34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a"
NORMALIZATION_SHA256 = "f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e"
MODEL_ID = "openwam-alpha-sim-libero@110b07e4d558"
SPACE_ID = "libero-spatial-panda@8f1084e"
HF_REVISION = "110b07e4d5581c81520a77419cbb0e9371df231b"
TASK_STEM = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
BDDL_SHA256 = "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b"
INIT_SHA256 = "cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a"


def space():
    camera = c.TensorSpec(dtype="uint8", shape=(256, 256, 3), layout="HWC")
    return c.SpaceSpec(
        spec_id=SPACE_ID, revision=BENCHMARK_REVISION,
        protocol_id="libero_spatial", protocol_version="1", control_period_s=.05,
        frames={"world": c.FrameSpec(parent_id=None),
                "agentview_camera": c.FrameSpec(parent_id="world", dynamic=True),
                "eye_in_hand_camera": c.FrameSpec(parent_id="world", dynamic=True),
                "panda_right_hand": c.FrameSpec(parent_id="world", dynamic=True)},
        robots={"robot0": c.RobotSpec(embodiment_id="libero-panda-osc-pose", base_frame_id="world",
            end_effectors={"eef": c.EndEffectorSpec(frame_id="panda_right_hand", tcp_id="grip_site")})},
        sensors={
            "agentview_image": c.SensorSpec(modality="rgb", role="head_camera", mounted_on="scene",
                frame_id="agentview_camera", tensor=camera, color_order="RGB", available_rate_hz=20),
            "robot0_eye_in_hand_image": c.SensorSpec(modality="rgb", role="left_wrist_camera", mounted_on="robot0",
                frame_id="eye_in_hand_camera", tensor=camera, color_order="RGB", available_rate_hz=20)},
        controllers={CONTROLLER: c.ControllerSpec(frame_id="world",
            parameter_schema={"type": "object", "properties": {"osc_action7": {
                "type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1},
                "minItems": 7, "maxItems": 7}}, "required": ["osc_action7"], "additionalProperties": False},
            parameter_units={"osc_action7": "LIBERO controller-native normalized 7D (xyz, rotation, gripper)"})},
        action_specs={ACTION: c.ActionSpec(command_kinds=("controller_command",), target_ids=("robot0/eef",),
                                          controller_refs=(CONTROLLER,))},
        required_extensions=(EXTENSION,),
        extensions={EXTENSION: {
            "eef_position_m": "3D grip_site position in world metres",
            "eef_quaternion_xyzw": "4D right_hand body quaternion, not grip_site orientation",
            "gripper_qpos_m": "2 gripper joint positions in metres"}},
    )


def capability(config):
    return c.CapabilitySpec(
        model_id=MODEL_ID, source_revision=MODEL_REVISION,
        checkpoint_ref=f"sha256:{CHECKPOINT_SHA256}", normalization_ref=f"sha256:{NORMALIZATION_SHA256}",
        roles=("action_policy",), supported_embodiments=("libero-panda-osc-pose",),
        required_inputs=(
            c.InputRequirement(key="agentview_image", kind="sensor", modality="rgb", dtype="uint8",
                               shape=(256, 256, 3), layout="HWC"),
            c.InputRequirement(key="robot0_eye_in_hand_image", kind="sensor", modality="rgb", dtype="uint8",
                               shape=(256, 256, 3), layout="HWC"),
            c.InputRequirement(key="instruction", kind="task")),
        observation_update="every_env_step", chunk_mode="adapter_owned", produced_action_specs=(ACTION,),
        supported_extensions=(EXTENSION,))


def profile(space_spec, model_cap):
    return c.AdaptationProfile(
        profile_id="openwam-alpha-libero-spatial@1", revision="pilot-1",
        benchmark_spec_ref=space_spec.spec_id, model_id=model_cap.model_id,
        model_source_revision=model_cap.source_revision, checkpoint_ref=model_cap.checkpoint_ref,
        normalization_ref=model_cap.normalization_ref, role="action_policy", robot_ids=("robot0",),
        input_bindings={key: c.InputBinding(kind="sensor", source_id=key)
                        for key in ("agentview_image", "robot0_eye_in_hand_image")}
                       | {"instruction": c.InputBinding(kind="task", source_id="instruction")},
        action_spec_refs=(ACTION,), discrete_spec_refs=(), tool_refs=(), chunk_mode="adapter_owned",
        execution_horizon_steps=1,
        view_policy=c.ModelViewPolicy(task_fields=("instruction",),
            sensor_ids=("agentview_image", "robot0_eye_in_hand_image"), robot_fields=(),
            frame_ids=tuple(space_spec.frames), extension_names=(EXTENSION,)),
        required_extensions=(EXTENSION,))
