"""Environment-only pinned protocol metadata; no selected model or weights."""
from .. import environment_records as c

EXTENSION = "libero.native_proprio.v1"
ACTION = "libero-osc7@1"
CONTROLLER = "libero-panda-osc-pose@1"
BENCHMARK_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SPACE_ID = "libero-spatial-panda@8f1084e"
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
