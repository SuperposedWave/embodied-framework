"""Environment-local loader records. Not another public protocol/version."""
from dataclasses import dataclass, field
from typing import Literal
from ..v03.contracts import Context

from ..core.types import (
    Vec3,
    Quat,
    Validity,
    DType,
    TimingMode,
    CommandKind,
    Command,
    Transform,
    Pose,
    Twist,
    FrameSpec,
    TimedTransform,
    TensorSpec,
    DataRef,
    Tensor,
    SensorSpec,
    JointSpec,
    EndEffectorSpec,
    GripperSpec,
    RobotSpec,
    ControllerSpec,
    ActionSpec,
    Budgets,
    PublicTask,
    SensorSample,
    JointState,
    GripperState,
    EndEffectorState,
    RobotState,
    JointTarget,
    PoseTarget,
    PoseDelta,
    TwistCommand,
    GripperCommand,
    ControllerCommand,
    ExecutionFeedback,
    Metric,
    ExecutionStatistics
)


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
    timing_mode: TimingMode = "fixed_period"
    physics_period_s: float | None = None
    required_extensions: tuple[str, ...] = ()
    extensions: dict[str, object] = field(default_factory=dict)


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
class ActionStep:
    index: int
    duration_s: float
    commands: tuple[Command, ...]


@dataclass(frozen=True, kw_only=True)
class NativeAction:
    """One opaque native controller call; duration is measured by the benchmark."""
    action_spec_ref: str
    commands: tuple[ControllerCommand, ...]
    kind: Literal["native_action"] = "native_action"


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
