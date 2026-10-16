"""EmbodiedEval: current canonical public protocol 0.3."""
from .v03.contracts import (
    Context, CapabilitySpec, ModelInput, Observation, Prediction, SpaceSpec,
    StepResult, TaskSpec, Action, ResolvedProfile, InputSpec, OutputSpec,
)
from .v03.interfaces import BenchmarkAdapter, ModelAdapter

CONTRACT_VERSION = "0.3"
__all__ = [
    "CONTRACT_VERSION", "Context", "CapabilitySpec", "ModelInput", "Observation",
    "Prediction", "SpaceSpec", "StepResult", "TaskSpec", "Action", "ResolvedProfile",
    "InputSpec", "OutputSpec", "BenchmarkAdapter", "ModelAdapter",
]
