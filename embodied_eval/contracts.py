"""Contract v0.1. Units and controller-native commands remain distinct."""
from dataclasses import dataclass, field
from typing import Literal

CONTRACT_VERSION = "0.1"
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Validity = Literal["valid", "invalid"]
DType = Literal["uint8", "uint16", "int32", "float32", "float64", "bool"]
TimingMode = Literal["fixed_period", "measured_native"]
CommandKind = Literal[
    "joint_target", "pose_target", "pose_delta", "twist_command",
    "gripper_command", "controller_command",
]
PublicFeedbackField = Literal[
    "request_id", "call_id", "command_ids", "status", "consumed_env_steps",
    "elapsed_sim_time_s", "code",
]


@dataclass(frozen=True, kw_only=True)
class Context:
    run_id: str
    env_id: str
    episode_id: str
    message_id: str
    contract_version: Literal["0.1"] = CONTRACT_VERSION


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
class DiscreteSpec:
    vocabulary: tuple[str, ...]
    target_ids: tuple[str, ...]
    duration_s: float
    intermediate_observations_supported: bool = False


@dataclass(frozen=True, kw_only=True)
class ToolSpec:
    argument_schema: dict[str, object]
    target_ids: tuple[str, ...]
    budget_behavior: Literal["per_control_step"] = "per_control_step"
    intermediate_observations_supported: bool = False


@dataclass(frozen=True, kw_only=True)
class SpaceSpec:
    spec_id: str
    revision: str
    protocol_id: str
    protocol_version: str
    control_period_s: float | None
    frames: dict[str, FrameSpec]
    robots: dict[str, RobotSpec]
    sensors: dict[str, SensorSpec]
    action_specs: dict[str, ActionSpec] = field(default_factory=dict)
    controllers: dict[str, ControllerSpec] = field(default_factory=dict)
    discrete_specs: dict[str, DiscreteSpec] = field(default_factory=dict)
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    timing_mode: TimingMode = "fixed_period"
    physics_period_s: float | None = None
    required_extensions: tuple[str, ...] = ()
    extensions: dict[str, object] = field(default_factory=dict)
    contract_version: Literal["0.1"] = CONTRACT_VERSION


@dataclass(frozen=True, kw_only=True)
class InputRequirement:
    key: str
    kind: Literal["sensor", "joint", "end_effector", "base", "task"]
    modality: str | None = None
    dtype: DType | None = None
    shape: tuple[int, ...] | None = None
    layout: str | None = None
    unit: str | None = None
    frame_id: str | None = None
    max_age_s: float | None = None


@dataclass(frozen=True, kw_only=True)
class InputBinding:
    kind: Literal["sensor", "joint", "end_effector", "base", "task"]
    source_id: str
    field_name: str | None = None


@dataclass(frozen=True, kw_only=True)
class CapabilitySpec:
    model_id: str
    source_revision: str
    checkpoint_ref: str
    normalization_ref: str
    roles: tuple[Literal["action_policy", "agent", "world_model"], ...]
    required_inputs: tuple[InputRequirement, ...]
    supported_embodiments: tuple[str, ...]
    observation_update: Literal["every_env_step", "decision_boundary"]
    chunk_mode: Literal["runner_owned", "adapter_owned"]
    produced_action_specs: tuple[str, ...] = ()
    produced_discrete_specs: tuple[str, ...] = ()
    produced_tools: tuple[str, ...] = ()
    can_stop: bool = False
    produced_predictions: tuple[Literal["future_image", "future_video", "future_state"], ...] = ()
    supported_extensions: tuple[str, ...] = ()
    state_scope: Literal["episode"] = "episode"
    batch_supported: bool = False
    supported_timing_modes: tuple[TimingMode, ...] = ("fixed_period",)
    contract_version: Literal["0.1"] = CONTRACT_VERSION


@dataclass(frozen=True, kw_only=True)
class ModelViewPolicy:
    task_fields: tuple[Literal["instruction", "goal_image", "goal_location"], ...] = ("instruction",)
    sensor_ids: tuple[str, ...] = ()
    robot_fields: tuple[str, ...] = ()
    frame_ids: tuple[str, ...] = ()
    feedback_fields: tuple[PublicFeedbackField, ...] = (
        "request_id", "call_id", "command_ids", "status",
        "consumed_env_steps", "elapsed_sim_time_s", "code",
    )
    extension_names: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class AdaptationProfile:
    profile_id: str
    revision: str
    benchmark_spec_ref: str
    model_id: str
    model_source_revision: str
    checkpoint_ref: str
    normalization_ref: str
    role: Literal["action_policy", "agent", "world_model"]
    robot_ids: tuple[str, ...]
    input_bindings: dict[str, InputBinding]
    action_spec_refs: tuple[str, ...]
    discrete_spec_refs: tuple[str, ...]
    tool_refs: tuple[str, ...]
    chunk_mode: Literal["runner_owned", "adapter_owned"]
    execution_horizon_steps: int
    view_policy: ModelViewPolicy
    frame_conversions: tuple[tuple[str, str], ...] = ()
    required_extensions: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    timing_mode: TimingMode = "fixed_period"
    contract_version: Literal["0.1"] = CONTRACT_VERSION


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
class TaskSpec:
    task_id: str
    protocol_id: str
    protocol_version: str
    episode_id: str
    public: PublicTask
    initialization_ref: str
    observation_policy_ref: str
    budgets: Budgets
    evaluation_config_ref: str
    evaluation_private: dict[str, object] = field(default_factory=dict)
    contract_version: Literal["0.1"] = CONTRACT_VERSION


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
class Observation:
    context: Context
    observation_id: str
    spec_ref: str
    env_step: int
    sim_time_s: float
    sensors: dict[str, SensorSample]
    robots: dict[str, RobotState]
    frame_transforms: dict[str, TimedTransform] = field(default_factory=dict)
    extensions: dict[str, object] = field(default_factory=dict)


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


Command = JointTarget | PoseTarget | PoseDelta | TwistCommand | GripperCommand | ControllerCommand


@dataclass(frozen=True, kw_only=True)
class ActionStep:
    index: int
    duration_s: float
    commands: tuple[Command, ...]


@dataclass(frozen=True, kw_only=True)
class ActionSequence:
    action_spec_ref: str
    execution_horizon_steps: int
    steps: tuple[ActionStep, ...]
    type: Literal["action_sequence"] = "action_sequence"


@dataclass(frozen=True, kw_only=True)
class NativeAction:
    """One opaque native controller call; duration is measured by the benchmark."""
    action_spec_ref: str
    commands: tuple[ControllerCommand, ...]
    kind: Literal["native_action"] = "native_action"


@dataclass(frozen=True, kw_only=True)
class DiscreteAction:
    discrete_spec_ref: str
    target_id: str
    action: str
    duration_s: float
    type: Literal["discrete_action"] = "discrete_action"


@dataclass(frozen=True, kw_only=True)
class ToolCall:
    call_id: str
    tool_ref: str
    target_id: str
    arguments: dict[str, object]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, kw_only=True)
class StopRequest:
    reason: str
    type: Literal["stop_request"] = "stop_request"


Decision = ActionSequence | NativeAction | DiscreteAction | ToolCall | StopRequest


@dataclass(frozen=True, kw_only=True)
class FuturePrediction:
    modality: Literal["future_image", "future_video", "future_state"]
    relative_time_s: float
    payload: Tensor
    provenance: Literal["model_prediction"] = "model_prediction"


@dataclass(frozen=True, kw_only=True)
class PredictionDiagnostics:
    raw_output_ref: str | None = None
    transformations: tuple[str, ...] = ()
    model_forward_count: int | None = None


@dataclass(frozen=True, kw_only=True)
class Prediction:
    context: Context
    request_id: str
    based_on_observation_id: str
    profile_ref: str
    decision: Decision | None = None
    predictions: tuple[FuturePrediction, ...] = ()
    diagnostics: PredictionDiagnostics = field(default_factory=PredictionDiagnostics)


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
class StepResult:
    context: Context
    observation: Observation
    execution_feedback: ExecutionFeedback
    terminated: bool
    truncated: bool
    intermediate_observations: tuple[Observation, ...] = ()
    end_reason: str | None = None
    evaluation_private: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ModelInput:
    context: Context
    task_public: PublicTask
    observation_view: Observation
    execution_feedback_view: dict[str, object] | None = None
    candidate_actions: ActionSequence | None = None


@dataclass(frozen=True, kw_only=True)
class ModelEndContext:
    context: Context
    reason: Literal["environment_terminated", "budget_exhausted", "model_stop", "run_error", "aborted"]


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


@dataclass(frozen=True, kw_only=True)
class EpisodeResult:
    episode_id: str
    run_manifest_ref: str
    run_status: Literal["completed", "error", "aborted"]
    termination_reason: str
    metrics: dict[str, Metric]
    execution_statistics: ExecutionStatistics
    artifact_refs: tuple[str, ...] = ()
    contract_version: Literal["0.1"] = CONTRACT_VERSION
