"""Environment-protocol declarations; never receives a model selection."""
from ..v03.execution_types import BenchmarkExecution, ExecutionConstraint


def benchmark_execution(space, *, period_verified=False, options=None):
    scope = {"protocol": space.protocol_id, "version": space.protocol_version,
             "revision": space.revision, "space": space.spec_id}
    scope["robots"] = ",".join(sorted(robot.embodiment_id for robot in space.robots.values()))
    scope["controllers"] = ",".join(sorted(space.controllers))
    for key in ("protocol", "task_name", "task_config", "track", "scene", "layout_index"):
        if options and key in options:
            scope[key] = str(options[key])
    # Existing policy scripts are evidence of reference behaviour, not proof
    # that an entire benchmark requires/allows one particular consumed prefix.
    consume = ExecutionConstraint(authority="unknown", provenance={
        "reason": "No reviewed benchmark-wide consumed-prefix permission for this named protocol.",
        "rule": "Preserve the selected model's source-matched native reference."})
    period = ExecutionConstraint(authority="unknown", provenance={
        "reason": "Variable transition duration" if space.timing_mode == "measured_native"
                  else "Static fallback is not an actual loaded control period."})
    if space.timing_mode == "fixed_period" and period_verified:
        period = ExecutionConstraint(authority="fixed", value=space.control_period_s,
            provenance={"revision": space.revision, "protocol": space.protocol_version,
                "source": "Environment control configuration / loaded control_timestep; not model FPS."})
    return BenchmarkExecution(scope=scope, consume_actions=consume, action_period_s=period)
