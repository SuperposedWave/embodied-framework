"""Pure per-parameter resolution, independent of model/benchmark identities."""
import math
from ..codec import decode, to_data
from ..errors import ContractError
from ..core.validation import require
from .execution_types import (ExecutionPolicy, ExecutionConstraint, ModelExecution,
                              BenchmarkExecution, ExecutionPlan)


def validate_policy(policy):
    decode(ExecutionPolicy, policy)
    require(policy.consume_actions is None or policy.consume_actions > 0,
            "invalid_execution_policy", "execution.consume_actions", "positive action count required")


def _constraint(rule, path, integer=False):
    decode(ExecutionConstraint, rule)
    for value in (rule.value, rule.minimum, rule.maximum, *rule.choices):
        require(value is None or (type(value) in (int, float) and math.isfinite(value) and value > 0
                and (not integer or type(value) is int)),
                "invalid_execution_declaration", path, "positive finite values with correct units required")
    require(rule.minimum is None or rule.maximum is None or rule.minimum <= rule.maximum,
            "invalid_execution_declaration", path, "inverted bounds")
    require(rule.authority != "fixed" or rule.value is not None,
            "invalid_execution_declaration", path, "fixed rule needs a value")
    require(rule.authority != "range" or bool(rule.choices) or
            (rule.minimum is not None and rule.maximum is not None),
            "invalid_execution_declaration", path, "range needs choices or both bounds")
    require(rule.authority == "unknown" or bool(rule.provenance),
            "missing_execution_provenance", path, "non-unknown authority needs source evidence")


def resolve_execution(space, capability, policy):
    """Return plan or a precise rejection. Never clamps or invents a period."""
    validate_policy(policy)
    model = capability.execution or ModelExecution()
    bench = space.execution or BenchmarkExecution(scope={"protocol": space.protocol_id,
        "version": space.protocol_version, "revision": space.revision})
    decode(ModelExecution, model)
    decode(BenchmarkExecution, bench)
    if capability.execution is not None and capability.temporal is not None:
        previous = capability.temporal
        require("unknown" in (previous.replanning_owner, model.cache_owner) or
                previous.replanning_owner == model.cache_owner,
                "conflicting_execution_declarations", "model.execution.cache_owner", "two declarations disagree about the sole cache owner")
        if previous.native_action_period_s is not None and model.action_period_s is not None:
            require(math.isclose(previous.native_action_period_s, model.action_period_s, rel_tol=0, abs_tol=1e-12),
                    "conflicting_execution_declarations", "model.execution.action_period_s", "model timing declarations disagree")
    require(bool(bench.scope), "missing_execution_scope", "execution", "named protocol scope required")
    for field in ("generated_actions", "preferred_consume_actions", "reference_consume_actions", "minimum_consume_actions", "maximum_consume_actions"):
        value = getattr(model, field)
        require(value is None or type(value) is int and value > 0,
                "invalid_execution_declaration", "model.execution." + field, "positive action count required")
    low, high, preferred = model.minimum_consume_actions, model.maximum_consume_actions, model.preferred_consume_actions
    require(low is None or high is None or low <= high,
            "invalid_execution_declaration", "model.execution", "inverted model bounds")
    require(preferred is None or (low is None or preferred >= low) and (high is None or preferred <= high),
            "invalid_execution_declaration", "model.execution", "native preference outside declared capability")
    require(model.generated_actions is None or preferred is None or preferred <= model.generated_actions,
            "invalid_execution_declaration", "model.execution", "consumed prefix exceeds declared generation length")
    require(not model.adjustable or (model.cache_owner == "adapter" and low is not None and high is not None),
            "unsupported_cache_control", "model.execution", "adjustable prefix requires one adapter-owned cache and explicit bounds")
    require(model.adjustable or model.reference_consume_actions is None or preferred == model.reference_consume_actions,
            "unsupported_execution_change", "model.execution", "non-adjustable profile cannot change its native reference through model options")
    _constraint(bench.consume_actions, "benchmark.execution.consume_actions", True)
    _constraint(bench.action_period_s, "benchmark.execution.action_period_s")
    unresolved, warnings, decisions = [], [], {}
    rule = bench.consume_actions
    selected = preferred if policy.consume_actions is None else policy.consume_actions
    if rule.authority == "fixed":
        require(policy.consume_actions is None or policy.consume_actions == rule.value,
                "execution_rule_conflict", "execution.consume_actions", "explicit override conflicts with fixed benchmark rule")
        selected = rule.value
    elif rule.authority == "range":
        require(selected is not None and (not rule.choices or selected in rule.choices)
                and (rule.minimum is None or selected >= rule.minimum)
                and (rule.maximum is None or selected <= rule.maximum),
                "execution_rule_conflict", "execution.consume_actions", "native preference/override outside benchmark allowance; no silent clamping")
    elif rule.authority in ("reference_default", "unknown"):
        # Changing this could turn a native reference into an unapproved experiment.
        require(policy.consume_actions is None or policy.consume_actions == preferred,
                "execution_permission_unknown", "execution.consume_actions", "no source-backed permission to change the native execution rule")
        require(model.reference_consume_actions is None or selected == model.reference_consume_actions,
                "execution_permission_unknown", "model.execution", "model options changed the reference prefix; benchmark permission is still unknown")
        unresolved.append("benchmark.consume_actions:" + rule.authority)
    if selected != preferred:
        require(model.adjustable and selected is not None and low <= selected <= high,
                "unsupported_execution_change", "execution.consume_actions", "model cannot apply this prefix without changing native/server semantics")
    decisions["consume_actions"] = rule.authority + (":explicit_override" if policy.consume_actions is not None else ":benchmark_value" if rule.authority == "fixed" else ":model_native")
    if selected is None:
        unresolved.append("model.consume_actions:server/native_reference_unverified")
    period_rule = bench.action_period_s
    period = period_rule.value if period_rule.authority == "fixed" else space.control_period_s
    model_period = model.action_period_s
    require(model_period is None or type(model_period) in (int, float) and math.isfinite(model_period) and model_period > 0,
            "invalid_execution_declaration", "model.execution.action_period_s", "positive known physical period or null required")
    if space.timing_mode == "measured_native":
        require(period_rule.authority != "fixed", "execution_timing_conflict", "execution.action_period_s",
                "variable-duration native transition cannot promise a fixed duration")
        period = None
        decisions["action_period_s"] = "measured_native:record_actual_duration_per_transition"
    else:
        if period_rule.authority == "range":
            require(period is not None and (not period_rule.choices or period in period_rule.choices)
                    and (period_rule.minimum is None or period >= period_rule.minimum)
                    and (period_rule.maximum is None or period <= period_rule.maximum),
                    "execution_timing_conflict", "execution.action_period_s", "actual environment period is outside protocol allowance")
        if period_rule.authority == "fixed":
            require(space.control_period_s is not None and math.isclose(period, space.control_period_s, rel_tol=0, abs_tol=1e-12),
                    "execution_timing_conflict", "execution.action_period_s", "actual environment period differs from declared protocol")
        elif period_rule.authority in ("reference_default", "unknown"):
            unresolved.append("benchmark.action_period_s:" + period_rule.authority)
        if model_period is not None and period is not None:
            require(math.isclose(model_period, period, rel_tol=0, abs_tol=1e-12),
                    "execution_timing_conflict", "execution.action_period_s", "weight action period differs from environment; resampling is not implemented")
        decisions["action_period_s"] = period_rule.authority + ":environment_period_unchanged"
    if model_period is None:
        unresolved.append("model.action_period_s:training_semantics_unverified")
    if not model.provenance:
        unresolved.append("model.execution:missing_source_provenance")
    if model.cache_owner == "unknown":
        unresolved.append("model.cache_owner:unknown")
    if unresolved:
        warnings.append("Native-reference execution only; unresolved permissions/timing are not official protocol compliance.")
    require(not policy.require_verified or not unresolved,
            "unverified_execution_protocol", "execution", "; ".join(unresolved))
    return ExecutionPlan(consume_actions=selected, action_period_s=period,
        cache_owner=model.cache_owner, change_requested=selected != preferred,
        protocol_verified=not unresolved, scope=dict(bench.scope), decisions=decisions,
        unresolved=tuple(unresolved), warnings=tuple(warnings))


def assess(space, capability, policy):
    try:
        return to_data(resolve_execution(space, capability, policy)), []
    except ContractError as error:
        return None, [to_data(error.issue)]


def apply_execution(model, plan):
    """Require an acknowledgement of changes; never pretends to configure a server."""
    method = getattr(model, "configure_execution", None)
    if method is None:
        require(not plan.change_requested, "execution_application_missing", "model", "adapter has no execution application hook")
        return {"status": "native_reference_preserved", "server_verified": False,
                "applied_consume_actions": None, "cache_owner": plan.cache_owner}
    receipt = method(plan)
    require(isinstance(receipt, dict) and receipt.get("cache_owner") == plan.cache_owner,
            "invalid_execution_ack", "model", "cache owner acknowledgement required")
    require(not plan.change_requested or (receipt.get("status") == "applied" and
            type(receipt.get("applied_consume_actions")) is int and receipt["applied_consume_actions"] == plan.consume_actions),
            "invalid_execution_ack", "model", "requested prefix was not applied; do not run with an ignored setting")
    return receipt
