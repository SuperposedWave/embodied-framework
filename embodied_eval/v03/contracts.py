"""Canonical public keys, physical samples and independently declared adapters.

There is deliberately no user-supplied BindingConfig. Geometry/commands retain
the existing SI/xyzw semantics; native vectors never become untyped poses.
"""
from dataclasses import dataclass, field
from typing import Literal
from .. import _legacy_contracts as leaf

VERSION = "0.3"


@dataclass(frozen=True, kw_only=True)
class Context:
    run_id: str
    env_id: str
    episode_id: str
    message_id: str
    contract_version: Literal["0.3"] = VERSION


@dataclass(frozen=True, kw_only=True)
class SensorSpec(leaf.SensorSpec):
    native_source: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ReferenceSite:
    robot_id: str
    kind: Literal["measured_position", "measured_orientation", "control_target", "tcp"]
    origin: str
    native_source: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class EndEffectorSpec(leaf.EndEffectorSpec):
    control_site_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class RobotSpec(leaf.RobotSpec):
    end_effectors: dict[str, EndEffectorSpec] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class StateSpec:
    semantic: str
    unit: str
    frame_id: str | None
    tensor: leaf.TensorSpec
    origin: str
    robot_id: str = "robot0"
    provenance: Literal["measurement", "controller_target"] = "measurement"
    site_id: str | None = None
    joint_ids: tuple[str, ...] = ()
    component_units: tuple[str, ...] = ()
    native_source: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class StateSample:
    captured_at_s: float
    payload: leaf.Tensor | None
    provenance: Literal["measurement", "controller_target"] = "measurement"
    validity: Literal["valid", "invalid"] = "valid"
    invalid_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class SpaceSpec:
    spec_id: str
    revision: str
    protocol_id: str
    protocol_version: str
    timing_mode: Literal["fixed_period", "measured_native"]
    control_period_s: float | None
    frames: dict[str, leaf.FrameSpec]
    robots: dict[str, RobotSpec]
    sensors: dict[str, SensorSpec]
    state_fields: dict[str, StateSpec]
    action_specs: dict[str, leaf.ActionSpec]
    controllers: dict[str, leaf.ControllerSpec] = field(default_factory=dict)
    native_protocols: dict[str, str] = field(default_factory=dict)
    physics_period_s: float | None = None
    required_action_targets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    action_joint_modes: dict[str, tuple[Literal["position", "velocity", "effort"], ...]] = field(default_factory=dict)
    view_aliases: dict[str, str] = field(default_factory=dict)
    reference_sites: dict[str, ReferenceSite] = field(default_factory=dict)
    # Known conventions are descriptions, not guessed transforms/calibration.
    frame_conventions: dict[str, str] = field(default_factory=dict)
    contract_version: Literal["0.3"] = VERSION


@dataclass(frozen=True, kw_only=True)
class InputSpec:
    key: str
    public_path: str
    kind: Literal["sensor", "state", "task"]
    semantic: str | None = None
    unit: str | None = None
    tensor: leaf.TensorSpec | None = None
    max_age_s: float | None = None
    frame_id: str | None = None
    required: bool = True
    provenance: Literal["measurement", "controller_target"] | None = None
    site_id: str | None = None
    joint_ids: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class OutputSpec:
    command_kinds: tuple[str, ...]
    # Keys are PUBLIC target IDs, never model-private slots.
    targets: dict[str, Literal["end_effector", "gripper", "joint", "base"]]
    native_protocol: str | None = None
    native_signature: str | None = None
    joint_modes: tuple[Literal["position", "velocity", "effort"], ...] = ()
    gripper_modes: dict[str, tuple[Literal["open_fraction", "width_m", "joint_targets", "force_n"], ...]] = field(default_factory=dict)
    pose_frame_id: str = "world"
    control_sites: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class CapabilitySpec:
    model_id: str
    source_revision: str
    checkpoint_ref: str
    normalization_ref: str
    io_profile_id: str
    inputs: tuple[InputSpec, ...]
    output: OutputSpec
    supported_timing_modes: tuple[Literal["fixed_period", "measured_native"], ...]
    training_embodiments: tuple[str, ...] = ()
    observation_update: Literal["every_env_step", "decision_boundary"] = "every_env_step"
    contract_version: Literal["0.3"] = VERSION


@dataclass(frozen=True, kw_only=True)
class ResolvedProfile:
    profile_id: str
    model_id: str
    space_ref: str
    action_spec_ref: str
    input_paths: dict[str, str]
    target_ids: tuple[str, ...]
    frame_conversions: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()
    contract_version: Literal["0.3"] = VERSION


@dataclass(frozen=True, kw_only=True)
class Observation:
    context: Context
    observation_id: str
    spec_ref: str
    env_step: int
    sim_time_s: float
    sensors: dict[str, leaf.SensorSample]
    states: dict[str, StateSample]
    public_instruction: str | None = None
    frame_transforms: dict[str, leaf.TimedTransform] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ModelInput:
    context: Context
    observation_id: str
    sim_time_s: float
    task_public: leaf.PublicTask
    sensors: dict[str, leaf.SensorSample]
    states: dict[str, StateSample]
    state_specs: dict[str, StateSpec] = field(default_factory=dict)
    sensor_specs: dict[str, SensorSpec] = field(default_factory=dict)
    view_aliases: dict[str, str] = field(default_factory=dict)
    frame_transforms: dict[str, leaf.TimedTransform] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Action:
    commands: tuple[leaf.Command, ...]
    acquire_observation_after: bool = True
    native_protocol: str | None = None
    native_dtype: Literal["float32", "float64", "int32", "int64"] | None = None
    source_euler_xyz_rad: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Prediction:
    context: Context
    based_on_observation_id: str
    action: Action
    model_forward_count: int | None = None
    transformations: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TaskSpec:
    task_id: str
    episode_id: str
    public: leaf.PublicTask
    budgets: leaf.Budgets
    initialization_ref: str
    evaluation_config_ref: str
    evaluation_private: dict[str, object] = field(default_factory=dict)
    contract_version: Literal["0.3"] = VERSION


@dataclass(frozen=True, kw_only=True)
class StepResult:
    observation: Observation
    execution_feedback: leaf.ExecutionFeedback
    terminated: bool
    truncated: bool
    contract_version: Literal["0.3"] = VERSION
