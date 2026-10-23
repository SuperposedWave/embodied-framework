"""TEST ONLY: frozen 0.2 evidence decoder, excluded from runtime distribution.

Geometry, tensor and commands come from core.types. The retained 0.2 stamps
support private native IPC and historical decoding, NOT a public run/config
fallback. New public observations and messages are exclusively v03 records.
"""
from dataclasses import dataclass, field
from typing import Literal
from embodied_eval.core.types import (Action, ActionSpec, Budgets, Command, ControllerSpec, FrameSpec,
                         Metric, PublicTask, RobotSpec, SensorSample, SensorSpec,
                         Tensor, TensorSpec, ExecutionFeedback, TimedTransform)

VERSION = "0.2"


@dataclass(frozen=True, kw_only=True)
class Context:
    run_id: str
    env_id: str
    episode_id: str
    message_id: str
    contract_version: Literal["0.2"] = VERSION


@dataclass(frozen=True, kw_only=True)
class StateSpec:
    semantic: Literal["eef_position", "eef_orientation", "gripper_position", "gripper_open_fraction",
                      "base_position", "joint_position", "joint_drive_target", "gripper_drive_target"]
    unit: str
    frame_id: str | None
    tensor: TensorSpec
    origin: str
    # An origin identifies the measured site/body or commanded actuator, never
    # an invented common TCP. Controller targets MUST NOT masquerade as qpos.


@dataclass(frozen=True, kw_only=True)
class StateSample:
    captured_at_s: float
    payload: Tensor
    provenance: Literal["measurement", "controller_target"] = "measurement"


@dataclass(frozen=True, kw_only=True)
class SpaceSpec:
    spec_id: str
    revision: str
    protocol_id: str
    protocol_version: str
    timing_mode: Literal["fixed_period", "measured_native"]
    control_period_s: float | None
    frames: dict[str, FrameSpec]
    robots: dict[str, RobotSpec]
    sensors: dict[str, SensorSpec]
    state_fields: dict[str, StateSpec]
    action_specs: dict[str, ActionSpec]
    controllers: dict[str, ControllerSpec] = field(default_factory=dict)
    native_protocols: dict[str, str] = field(default_factory=dict)
    physics_period_s: float | None = None
    required_action_targets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    action_joint_modes: dict[str, tuple[Literal["position", "velocity", "effort"], ...]] = field(default_factory=dict)
    contract_version: Literal["0.2"] = VERSION


@dataclass(frozen=True, kw_only=True)
class InputSpec:
    key: str
    kind: Literal["sensor", "state", "task"]
    semantic: str | None = None
    unit: str | None = None
    tensor: TensorSpec | None = None
    max_age_s: float | None = None
    frame_id: str | None = None
    # Tensor shape describes acceptable PUBLIC samples, not model resize size.


@dataclass(frozen=True, kw_only=True)
class OutputSpec:
    command_kinds: tuple[str, ...]
    target_slots: dict[str, Literal["end_effector", "gripper", "joint", "base"]]
    native_protocol: str | None = None
    native_signature: str | None = None
    joint_modes: tuple[Literal["position", "velocity", "effort"], ...] = ()
    gripper_modes: dict[str, tuple[Literal["open_fraction", "width_m", "joint_targets", "force_n"], ...]] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class CapabilitySpec:
    model_id: str
    source_revision: str
    checkpoint_ref: str
    normalization_ref: str
    inputs: tuple[InputSpec, ...]
    output: OutputSpec
    supported_timing_modes: tuple[Literal["fixed_period", "measured_native"], ...]
    training_embodiments: tuple[str, ...] = ()
    observation_update: Literal["every_env_step", "decision_boundary"] = "every_env_step"
    contract_version: Literal["0.2"] = VERSION


@dataclass(frozen=True, kw_only=True)
class BindingConfig:
    inputs: dict[str, str]
    targets: dict[str, str]
    action_spec_ref: str
    frame_conversions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, kw_only=True)
class BoundProfile:
    profile_id: str
    binding: BindingConfig
    model_id: str
    space_ref: str
    warnings: tuple[str, ...] = ()
    contract_version: Literal["0.2"] = VERSION


@dataclass(frozen=True, kw_only=True)
class Observation:
    context: Context
    observation_id: str
    spec_ref: str
    env_step: int
    sim_time_s: float
    sensors: dict[str, SensorSample]
    states: dict[str, StateSample]
    public_instruction: str | None = None
    frame_transforms: dict[str, TimedTransform] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ModelInput:
    context: Context
    observation_id: str
    sim_time_s: float
    task_public: PublicTask
    # Keys are model logical slots, not simulator names. No private fields.
    sensors: dict[str, SensorSample]
    states: dict[str, StateSample]
    state_specs: dict[str, StateSpec] = field(default_factory=dict)
    frame_transforms: dict[str, TimedTransform] = field(default_factory=dict)




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
    public: PublicTask
    budgets: Budgets
    initialization_ref: str
    evaluation_config_ref: str
    evaluation_private: dict[str, object] = field(default_factory=dict)
    contract_version: Literal["0.2"] = VERSION


@dataclass(frozen=True, kw_only=True)
class StepResult:
    observation: Observation
    execution_feedback: ExecutionFeedback
    terminated: bool
    truncated: bool
    contract_version: Literal["0.2"] = VERSION
