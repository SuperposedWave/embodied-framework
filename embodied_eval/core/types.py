"""Version-independent physical records; metres, radians, seconds and xyzw.

These records do not select a public protocol. Versioned message envelopes live
in v03; frozen private native envelopes live in native.contracts.
"""
from dataclasses import dataclass, field
from typing import Literal

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Validity = Literal["valid", "invalid"]
DType = Literal["uint8", "uint16", "int32", "float32", "float64", "bool"]
TimingMode = Literal["fixed_period", "measured_native"]
CommandKind = Literal[
    "joint_target", "pose_target", "pose_delta", "twist_command",
    "gripper_command", "controller_command",
]


@dataclass(frozen=True, kw_only=True)
class Transform:
    """T_parent_child: transform child coordinates into parent coordinates."""
    translation_m: Vec3
    quaternion_xyzw: Quat


@dataclass(frozen=True, kw_only=True)
class Pose:
    position_m: Vec3
    quaternion_xyzw: Quat
    frame_id: str


@dataclass(frozen=True, kw_only=True)
class Twist:
    linear_m_s: Vec3
    angular_rad_s: Vec3
    frame_id: str


@dataclass(frozen=True, kw_only=True)
class FrameSpec:
    parent_id: str | None
    static_transform: Transform | None = None
    dynamic: bool = False


@dataclass(frozen=True, kw_only=True)
class TimedTransform:
    transform: Transform
    captured_at_s: float


@dataclass(frozen=True, kw_only=True)
class TensorSpec:
    dtype: DType
    shape: tuple[int, ...]
    layout: str


@dataclass(frozen=True, kw_only=True)
class DataRef:
    storage: Literal["file", "shared_memory", "remote_blob"]
    key: str
    byte_length: int
    sha256: str


@dataclass(frozen=True, kw_only=True)
class Tensor:
    dtype: DType
    shape: tuple[int, ...]
    layout: str
    data: object | None = None
    data_ref: DataRef | None = None


@dataclass(frozen=True, kw_only=True)
class SensorSpec:
    modality: Literal["rgb", "depth", "point_cloud", "audio", "tactile"]
    role: str
    mounted_on: str
    frame_id: str
    tensor: TensorSpec
    unit: str | None = None
    color_order: Literal["RGB"] | None = None
    available_rate_hz: float | None = None


@dataclass(frozen=True, kw_only=True)
class JointSpec:
    joint_type: Literal["revolute", "prismatic"]
    position_unit: Literal["rad", "m"]
    velocity_unit: Literal["rad/s", "m/s"]
    effort_unit: Literal["Nm", "N"]
    position_limits: tuple[float, float] | None = None


@dataclass(frozen=True, kw_only=True)
class EndEffectorSpec:
    frame_id: str
    tcp_id: str
    gripper_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class GripperSpec:
    modes: tuple[Literal["open_fraction", "width_m", "joint_targets", "force_n"], ...]
    max_width_m: float | None = None
    joint_ids: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class RobotSpec:
    embodiment_id: str
    base_frame_id: str
    joints: dict[str, JointSpec] = field(default_factory=dict)
    end_effectors: dict[str, EndEffectorSpec] = field(default_factory=dict)
    grippers: dict[str, GripperSpec] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ControllerSpec:
    frame_id: str
    parameter_schema: dict[str, object]
    parameter_units: dict[str, str]
    parameter_domain: Literal["controller_native"] = "controller_native"


@dataclass(frozen=True, kw_only=True)
class ActionSpec:
    command_kinds: tuple[CommandKind, ...]
    target_ids: tuple[str, ...]
    controller_refs: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Budgets:
    env_steps: int | None = None
    sim_time_s: float | None = None
    model_calls: int | None = None
    tool_calls: int | None = None
    wall_time_s: float | None = None


@dataclass(frozen=True, kw_only=True)
class PublicTask:
    instruction: str
    goal_image: Tensor | None = None
    goal_location: Pose | None = None


@dataclass(frozen=True, kw_only=True)
class SensorSample:
    captured_at_s: float
    frame_id: str
    validity: Validity
    provenance: Literal["sensor"]
    payload: Tensor | None = None
    invalid_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class JointState:
    position: float | None = None
    velocity: float | None = None
    effort: float | None = None
    validity: Validity = "valid"
    invalid_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class GripperState:
    mode: Literal["open_fraction", "width_m", "joint_targets", "force_n"]
    value: float | dict[str, float]


@dataclass(frozen=True, kw_only=True)
class EndEffectorState:
    pose: Pose | None = None
    gripper_state: GripperState | None = None


@dataclass(frozen=True, kw_only=True)
class RobotState:
    joints: dict[str, JointState] = field(default_factory=dict)
    end_effectors: dict[str, EndEffectorState] = field(default_factory=dict)
    base_pose: Pose | None = None
    base_twist: Twist | None = None


@dataclass(frozen=True, kw_only=True)
class JointTarget:
    command_id: str
    target_id: str
    mode: Literal["position", "velocity", "effort"]
    value: float
    unit: str
    kind: Literal["joint_target"] = "joint_target"


@dataclass(frozen=True, kw_only=True)
class PoseTarget:
    command_id: str
    target_id: str
    pose: Pose
    kind: Literal["pose_target"] = "pose_target"


@dataclass(frozen=True, kw_only=True)
class PoseDelta:
    command_id: str
    target_id: str
    delta: Transform
    frame_id: str
    reference_policy: Literal["chunk_start", "step_observed", "previous_command"]
    composition: Literal["pre_multiply", "post_multiply"]
    reference_observation_id: str | None = None
    kind: Literal["pose_delta"] = "pose_delta"


@dataclass(frozen=True, kw_only=True)
class TwistCommand:
    command_id: str
    target_id: str
    twist: Twist
    kind: Literal["twist_command"] = "twist_command"


@dataclass(frozen=True, kw_only=True)
class GripperCommand:
    command_id: str
    target_id: str
    mode: Literal["open_fraction", "width_m", "joint_targets", "force_n"]
    value: float | dict[str, float]
    kind: Literal["gripper_command"] = "gripper_command"


@dataclass(frozen=True, kw_only=True)
class ControllerCommand:
    command_id: str
    target_id: str
    controller_profile_ref: str
    parameters: dict[str, object]
    parameter_domain: Literal["controller_native"] = "controller_native"
    kind: Literal["controller_command"] = "controller_command"


@dataclass(frozen=True, kw_only=True)
class ExecutionFeedback:
    request_id: str
    status: Literal["executed", "partially_executed", "rejected", "planning_failed", "execution_failed"]
    consumed_env_steps: int
    elapsed_sim_time_s: float
    call_id: str | None = None
    command_ids: tuple[str, ...] = ()
    code: Literal["ok", "invalid_argument", "planning_failed", "execution_failed", "budget_exhausted"] = "ok"
    consumed_physics_steps: int | None = None


@dataclass(frozen=True, kw_only=True)
class Metric:
    value: float | bool | None
    unit: str
    source: str
    valid: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class ExecutionStatistics:
    env_steps: int
    sim_time_s: float
    model_calls: int
    tool_calls: int
    model_forward_count: int | None = None
    physics_steps: int | None = None


Command = JointTarget | PoseTarget | PoseDelta | TwistCommand | GripperCommand | ControllerCommand


@dataclass(frozen=True, kw_only=True)
class Action:
    commands: tuple[Command, ...]
    acquire_observation_after: bool = True
    native_protocol: str | None = None
    native_dtype: Literal["float32", "float64", "int32", "int64"] | None = None
    source_euler_xyz_rad: dict[str, tuple[float, float, float]] = field(default_factory=dict)
