"""Current-protocol execution declarations; no adapter or simulator imports."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class ExecutionPolicy:
    # Unknown official permissions may preserve a reference run, never certify it.
    require_verified: bool = False
    consume_actions: int | None = None


@dataclass(frozen=True, kw_only=True)
class ExecutionConstraint:
    authority: Literal["fixed", "range", "policy_owned", "reference_default", "unknown"] = "unknown"
    value: int | float | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[int | float, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class BenchmarkExecution:
    scope: dict[str, str]
    consume_actions: ExecutionConstraint = field(default_factory=ExecutionConstraint)
    action_period_s: ExecutionConstraint = field(default_factory=ExecutionConstraint)


@dataclass(frozen=True, kw_only=True)
class ModelExecution:
    # Prediction horizon is deliberately separate from the consumed prefix.
    generated_actions: int | None = None
    preferred_consume_actions: int | None = None
    reference_consume_actions: int | None = None
    minimum_consume_actions: int | None = None
    maximum_consume_actions: int | None = None
    adjustable: bool = False
    cache_owner: Literal["adapter", "server", "unknown"] = "unknown"
    action_period_s: float | None = None
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ExecutionPlan:
    consume_actions: int | None
    action_period_s: float | None
    cache_owner: str
    change_requested: bool
    protocol_verified: bool
    scope: dict[str, str]
    decisions: dict[str, str]
    unresolved: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    extension: Literal["benchmark-first-execution-policy@1"] = "benchmark-first-execution-policy@1"
