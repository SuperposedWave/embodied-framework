"""Structural and semantic checks for contract v0.1."""
import math
import tests.legacy_support.frozen._legacy_contracts as c
from embodied_eval.codec import decode, to_data
from embodied_eval.core.validation import require, same_episode, quaternion, pose, unique, check_schema, schema_value, target_table, validate_tensor, shape_matches, validate_gripper, validate_structure


def validate_space(space: c.SpaceSpec):
    decode(c.SpaceSpec, space)
    validate_structure(space)
    targets = target_table(space)
    for ref, discrete in space.discrete_specs.items():
        unique(discrete.vocabulary, ref)
        require(bool(discrete.vocabulary) and discrete.duration_s > 0, "invalid_discrete_spec", ref, "vocabulary and duration required")
        ticks = discrete.duration_s / space.control_period_s
        require(math.isclose(ticks, round(ticks), abs_tol=1e-9), "timing_mismatch", ref, "discrete duration must cover an integer number of control ticks")
        require(set(discrete.target_ids) <= targets.keys(), "unknown_target", ref, "unknown discrete target")
    for ref, tool in space.tools.items():
        check_schema(tool.argument_schema, ref)
        require(set(tool.target_ids) <= targets.keys(), "unknown_target", ref, "unknown tool target")
    for name in (*space.extensions, *space.required_extensions):
        require("." in name, "invalid_extension", name, "extensions must use a namespace")
    require(set(space.required_extensions) <= space.extensions.keys(),
            "missing_extension", "space.extensions", "required extension payload missing")
    return space


def validate_observation(obs: c.Observation, space: c.SpaceSpec):
    decode(c.Observation, obs)
    require(obs.spec_ref == space.spec_id, "spec_mismatch", "observation.spec_ref", "wrong SpaceSpec")
    require(obs.env_step >= 0 and obs.sim_time_s >= 0, "invalid_time", "observation", "negative step/time")
    for sid, sample in obs.sensors.items():
        require(sid in space.sensors, "unknown_sensor", sid, "not in SpaceSpec")
        spec = space.sensors[sid]
        require(sample.frame_id == spec.frame_id, "frame_mismatch", sid, "sensor frame changed")
        require(0 <= sample.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", sid, "future/negative sample time")
        if sample.validity == "invalid":
            require(sample.payload is None and bool(sample.invalid_reason), "invalid_sample", sid, "invalid sample needs reason and no payload")
        else:
            require(sample.payload is not None and sample.invalid_reason is None, "invalid_sample", sid, "valid sample needs payload")
            t = validate_tensor(sample.payload)
            require(t.dtype == spec.tensor.dtype and t.layout == spec.tensor.layout and shape_matches(t.shape, spec.tensor.shape),
                    "tensor_spec_mismatch", sid, "dtype/layout/shape differs from SpaceSpec")
    for rid, state in obs.robots.items():
        require(rid in space.robots, "unknown_robot", rid, "not in SpaceSpec")
        spec = space.robots[rid]
        for jid, joint in state.joints.items():
            require(jid in spec.joints, "unknown_joint", f"{rid}/{jid}", "not in SpaceSpec")
            values = (joint.position, joint.velocity, joint.effort)
            if joint.validity == "invalid":
                require(all(v is None for v in values) and bool(joint.invalid_reason),
                        "invalid_state", jid, "invalid state needs reason and no values")
            else:
                require(any(v is not None for v in values) and joint.invalid_reason is None,
                        "invalid_state", jid, "valid state needs at least one measurement")
        for eid, eff in state.end_effectors.items():
            require(eid in spec.end_effectors, "unknown_target", eid, "not in SpaceSpec")
            if eff.pose:
                pose(eff.pose, space, f"{rid}/{eid}")
            if eff.gripper_state:
                gid = spec.end_effectors[eid].gripper_id
                require(gid is not None, "missing_gripper", eid, "no declared gripper")
                validate_gripper(eff.gripper_state.mode, eff.gripper_state.value, spec.grippers[gid], eid, spec.joints)
        if state.base_pose:
            pose(state.base_pose, space, rid + ".base_pose")
        if state.base_twist:
            require(state.base_twist.frame_id in space.frames, "unknown_frame", rid, "unknown twist frame")
    for fid, transform in obs.frame_transforms.items():
        require(fid in space.frames and space.frames[fid].dynamic, "unexpected_transform", fid, "not a declared dynamic frame")
        require(0 <= transform.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", fid, "invalid transform time")
        quaternion(transform.transform.quaternion_xyzw, fid)
    for name in obs.extensions:
        require("." in name, "invalid_extension", name, "namespace required")
    return obs


def validate_task(task: c.TaskSpec, space: c.SpaceSpec):
    decode(c.TaskSpec, task)
    require((task.protocol_id, task.protocol_version) == (space.protocol_id, space.protocol_version),
            "protocol_mismatch", "task", "task protocol differs from SpaceSpec")
    for name, value in to_data(task.budgets).items():
        require(value is None or value >= 0, "invalid_budget", name, "negative budget")
    require(space.timing_mode != "measured_native" or task.budgets.sim_time_s is None,
            "unsupported_native_budget", "budgets.sim_time_s", "opaque native actions cannot enforce a simulation-time limit")
    if task.public.goal_image:
        validate_tensor(task.public.goal_image)
    if task.public.goal_location:
        pose(task.public.goal_location, space, "task.public.goal_location")
    return task


def validate_prediction(pred: c.Prediction, obs: c.Observation, space: c.SpaceSpec,
                        cap: c.CapabilitySpec, profile: c.AdaptationProfile, *, request_context: c.Context | None = None):
    decode(c.Prediction, pred)
    require(bool(pred.request_id) and pred.context.message_id == pred.request_id,
            "request_mismatch", "prediction.request_id", "response context message_id must equal request_id")
    same_episode(pred.context, obs.context)
    if request_context is not None:
        same_episode(pred.context, request_context)
        require(pred.request_id == request_context.message_id, "request_mismatch", "prediction.request_id", "not the pending request")
    require(pred.based_on_observation_id == obs.observation_id,
            "stale_prediction", "prediction.based_on_observation_id", "prediction is based on a different observation")
    require(pred.profile_ref == profile.profile_id, "profile_mismatch", "prediction.profile_ref", "wrong profile")
    require(profile.role in cap.roles, "role_mismatch", "profile.role", "unsupported role")
    if profile.role != "world_model":
        require(pred.decision is not None, "missing_decision", "prediction", "policy/agent requires an executable decision")
    else:
        require(pred.decision is None and bool(pred.predictions), "invalid_world_prediction", "prediction",
                "pure world_model produces predictions for a separately configured consumer")
    for future in pred.predictions:
        require(future.modality in cap.produced_predictions and future.relative_time_s >= 0,
                "invalid_world_prediction", "prediction.predictions", "unsupported modality or negative horizon")
        validate_tensor(future.payload)
    if pred.diagnostics.model_forward_count is not None:
        require(pred.diagnostics.model_forward_count >= 0, "invalid_count", "diagnostics", "negative forward count")
    decision = pred.decision
    targets = target_table(space)
    if isinstance(decision, (c.ActionSequence, c.DiscreteAction, c.ToolCall)):
        require(space.timing_mode == profile.timing_mode == "fixed_period",
                "timing_mode_mismatch", "decision", "fixed-period decisions cannot execute in measured-native mode")
    if isinstance(decision, c.NativeAction):
        require(space.timing_mode == profile.timing_mode == "measured_native"
                and "measured_native" in cap.supported_timing_modes,
                "timing_mode_mismatch", "decision", "native actions require negotiated measured-native timing")
        require(profile.chunk_mode == cap.chunk_mode == "adapter_owned" and profile.execution_horizon_steps == 1,
                "double_chunk_owner", "decision", "native action is exactly one native call")
        require(decision.action_spec_ref in profile.action_spec_refs
                and decision.action_spec_ref in cap.produced_action_specs
                and decision.action_spec_ref in space.action_specs,
                "unsupported_action", "decision.action_spec_ref", "not declared by benchmark/model/profile")
        spec = space.action_specs[decision.action_spec_ref]
        require(bool(decision.commands), "empty_step", "decision.commands", "commands required")
        unique([cmd.target_id for cmd in decision.commands], "decision.targets")
        unique([cmd.command_id for cmd in decision.commands], "decision.command_ids")
        for cmd in decision.commands:
            require(cmd.target_id in spec.target_ids and cmd.kind in spec.command_kinds,
                    "unsupported_action", cmd.command_id, "target or command kind not accepted")
            require(cmd.controller_profile_ref in spec.controller_refs,
                    "unknown_controller", cmd.command_id, "controller not accepted by action spec")
            schema_value(space.controllers[cmd.controller_profile_ref].parameter_schema,
                         cmd.parameters, cmd.command_id + ".parameters")
    if isinstance(decision, c.ActionSequence):
        require(decision.action_spec_ref in profile.action_spec_refs and decision.action_spec_ref in cap.produced_action_specs,
                "unsupported_action", "decision.action_spec_ref", "not declared by model/profile")
        require(decision.action_spec_ref in space.action_specs, "unsupported_action", "decision", "not accepted by benchmark")
        spec = space.action_specs[decision.action_spec_ref]
        require(0 < decision.execution_horizon_steps <= len(decision.steps),
                "invalid_horizon", "decision", "horizon exceeds sequence or is zero")
        require(decision.execution_horizon_steps <= profile.execution_horizon_steps,
                "invalid_horizon", "decision", "horizon exceeds agreed profile")
        if profile.chunk_mode == "adapter_owned":
            require(len(decision.steps) == decision.execution_horizon_steps == 1,
                    "double_chunk_owner", "decision", "adapter-owned chunk must expose exactly one step")
        ids = []
        for i, step in enumerate(decision.steps):
            require(step.index == i, "invalid_sequence", "decision.steps", "indices must start at zero and be contiguous")
            # v0.1 executes one logical control tick per action step.
            require(math.isclose(step.duration_s, space.control_period_s, abs_tol=1e-9),
                    "timing_mismatch", f"steps[{i}]", "step duration must match the benchmark control tick")
            require(bool(step.commands), "empty_step", f"steps[{i}]", "commands required")
            unique([cmd.target_id for cmd in step.commands], f"steps[{i}].targets")
            for cmd in step.commands:
                ids.append(cmd.command_id)
                require(cmd.target_id in spec.target_ids and cmd.kind in spec.command_kinds,
                        "unsupported_action", cmd.command_id, "target or command kind is not accepted")
                kind, target = targets[cmd.target_id]
                if isinstance(cmd, c.JointTarget):
                    require(kind == "joint", "target_kind_mismatch", cmd.target_id, "joint command needs a joint")
                    unit = getattr(target, cmd.mode + "_unit")
                    require(cmd.unit == unit, "unit_mismatch", cmd.command_id, f"expected {unit}")
                    if cmd.mode == "position" and target.position_limits:
                        require(target.position_limits[0] <= cmd.value <= target.position_limits[1],
                                "action_out_of_bounds", cmd.command_id, "joint position exceeds limits")
                elif isinstance(cmd, (c.PoseTarget, c.PoseDelta)):
                    require(kind == "end_effector", "target_kind_mismatch", cmd.target_id, "pose command needs an end effector")
                    if isinstance(cmd, c.PoseTarget):
                        pose(cmd.pose, space, cmd.command_id)
                    else:
                        require(cmd.frame_id in space.frames, "unknown_frame", cmd.command_id, cmd.frame_id)
                        quaternion(cmd.delta.quaternion_xyzw, cmd.command_id)
                        if cmd.reference_policy == "chunk_start":
                            require(cmd.reference_observation_id == obs.observation_id,
                                    "invalid_delta_reference", cmd.command_id, "chunk reference must be the source observation")
                        else:
                            require(cmd.reference_observation_id is None, "invalid_delta_reference", cmd.command_id,
                                    "per-step/previous-command reference is resolved by the executor")
                elif isinstance(cmd, c.TwistCommand):
                    require(kind in ("end_effector", "base") and cmd.twist.frame_id in space.frames,
                            "target_kind_mismatch", cmd.command_id, "twist needs base/eef and known frame")
                elif isinstance(cmd, c.GripperCommand):
                    require(kind == "gripper", "target_kind_mismatch", cmd.target_id, "gripper command needs gripper")
                    validate_gripper(cmd.mode, cmd.value, target, cmd.command_id, space.robots[cmd.target_id.split("/")[0]].joints)
                elif isinstance(cmd, c.ControllerCommand):
                    require(cmd.controller_profile_ref in spec.controller_refs,
                            "unknown_controller", cmd.command_id, "controller profile not accepted by action spec")
                    schema_value(space.controllers[cmd.controller_profile_ref].parameter_schema,
                                 cmd.parameters, cmd.command_id + ".parameters")
        unique(ids, "decision.command_ids")
    elif isinstance(decision, c.DiscreteAction):
        require(decision.discrete_spec_ref in profile.discrete_spec_refs and decision.discrete_spec_ref in cap.produced_discrete_specs
                and decision.discrete_spec_ref in space.discrete_specs, "unsupported_discrete", "decision", "undeclared vocabulary")
        spec = space.discrete_specs[decision.discrete_spec_ref]
        require(decision.action in spec.vocabulary and decision.target_id in spec.target_ids,
                "unsupported_discrete", "decision", "unknown token/target")
        require(math.isclose(decision.duration_s, spec.duration_s, abs_tol=1e-9),
                "timing_mismatch", "decision", "duration differs from vocabulary profile")
    elif isinstance(decision, c.ToolCall):
        require(decision.tool_ref in profile.tool_refs and decision.tool_ref in cap.produced_tools
                and decision.tool_ref in space.tools, "unsupported_tool", "decision", "undeclared tool")
        tool = space.tools[decision.tool_ref]
        require(decision.target_id in tool.target_ids, "unknown_target", "decision.target_id", "tool cannot operate this target")
        schema_value(tool.argument_schema, decision.arguments, "decision.arguments")
    elif isinstance(decision, c.StopRequest):
        require(cap.can_stop, "unsupported_stop", "decision", "model did not declare stop support")
    return pred


def validate_step_result(result: c.StepResult, previous: c.Observation, space: c.SpaceSpec,
                         *, expected_prediction: c.Prediction | None = None, capability: c.CapabilitySpec | None = None):
    decode(c.StepResult, result)
    same_episode(result.context, previous.context)
    same_episode(result.context, result.observation.context)
    validate_observation(result.observation, space)
    feedback = result.execution_feedback
    if expected_prediction is not None:
        require(feedback.request_id == expected_prediction.request_id, "request_mismatch", "feedback", "wrong request response")
        decision = expected_prediction.decision
        if isinstance(decision, c.ToolCall):
            require(feedback.call_id == decision.call_id, "call_mismatch", "feedback.call_id", "wrong tool response")
        elif isinstance(decision, c.ActionSequence):
            command_ids = {cmd.command_id for step in decision.steps[:decision.execution_horizon_steps] for cmd in step.commands}
            require(set(feedback.command_ids) <= command_ids, "command_mismatch", "feedback.command_ids", "command not scheduled by this prediction")
        elif isinstance(decision, c.NativeAction):
            require(feedback.call_id is None and set(feedback.command_ids) <= {cmd.command_id for cmd in decision.commands},
                    "command_mismatch", "feedback.command_ids", "native feedback must reference the pending command")
    require(feedback.consumed_env_steps >= 0 and feedback.elapsed_sim_time_s >= 0,
            "invalid_execution_count", "feedback", "negative consumption")
    require(result.observation.env_step - previous.env_step == feedback.consumed_env_steps,
            "execution_count_mismatch", "feedback", "actual environment step delta differs")
    require(math.isclose(result.observation.sim_time_s - previous.sim_time_s, feedback.elapsed_sim_time_s, abs_tol=1e-9),
            "execution_time_mismatch", "feedback", "actual simulation time delta differs")
    if space.timing_mode == "measured_native":
        require(feedback.consumed_physics_steps is not None and feedback.consumed_physics_steps >= 0,
                "invalid_execution_count", "feedback.consumed_physics_steps", "actual physical step count is required")
        require(feedback.consumed_env_steps <= 1, "invalid_execution_count", "feedback", "one native call per decision")
        require(feedback.consumed_env_steps > 0 or feedback.consumed_physics_steps == 0,
                "invalid_execution_count", "feedback", "physical progress requires a consumed native action")
        require(math.isclose(feedback.elapsed_sim_time_s, feedback.consumed_physics_steps * space.physics_period_s, abs_tol=1e-9),
                "execution_time_mismatch", "feedback", "native time must equal measured physical steps times the physical period")
        require(not result.intermediate_observations, "invalid_observation_history", "intermediate_observations",
                "native mode reports only its final decision-boundary observation")
        if feedback.status == "executed":
            require(feedback.consumed_env_steps == 1, "invalid_execution_count", "feedback", "executed native action consumes one call")
        if feedback.status == "rejected":
            require(feedback.consumed_env_steps == feedback.consumed_physics_steps == 0,
                    "invalid_execution_count", "feedback", "rejected native request consumes nothing")
    else:
        require(math.isclose(feedback.elapsed_sim_time_s, feedback.consumed_env_steps * space.control_period_s, abs_tol=1e-9),
                "execution_time_mismatch", "feedback", "simulation time differs from logical control ticks")
        if feedback.consumed_physics_steps is not None:
            require(space.physics_period_s is not None and feedback.consumed_physics_steps >= 0
                    and math.isclose(feedback.elapsed_sim_time_s, feedback.consumed_physics_steps * space.physics_period_s, abs_tol=1e-9),
                    "execution_time_mismatch", "feedback", "optional physical counts must also match simulation time")
    expected_intermediates = max(0, feedback.consumed_env_steps - 1)
    if result.intermediate_observations or (capability is not None and capability.observation_update == "every_env_step"):
        require(len(result.intermediate_observations) == expected_intermediates, "missing_step_observations",
                "intermediate_observations", "one observation for every executed tick before the final observation")
    for offset, observation in enumerate(result.intermediate_observations, start=1):
        same_episode(observation.context, previous.context)
        validate_observation(observation, space)
        require(observation.env_step == previous.env_step + offset
                and math.isclose(observation.sim_time_s, previous.sim_time_s + offset * space.control_period_s, abs_tol=1e-9),
                "invalid_observation_history", "intermediate_observations", "history must be ordered, contiguous and within this execution")
    if feedback.status in ("rejected", "planning_failed") and space.timing_mode == "fixed_period":
        require(feedback.consumed_env_steps == 0, "invalid_execution_count", "feedback", "pre-execution rejection consumes no environment ticks")
    if feedback.status == "executed":
        require(feedback.code == "ok", "feedback_status_mismatch", "feedback", "executed status requires ok code")
    require(bool(result.end_reason) == (result.terminated or result.truncated),
            "missing_end_reason", "result", "end reason required only at termination")
    return result


def validate_episode_result(result: c.EpisodeResult):
    decode(c.EpisodeResult, result)
    for key, metric in result.metrics.items():
        require(metric.valid == (metric.value is not None), "invalid_metric", key, "unavailable metric uses null")
        require((not metric.valid) == bool(metric.unavailable_reason),
                "missing_metric_reason", key, "unavailable metric needs a reason")
        if metric.unit == "boolean" and metric.value is not None:
            require(type(metric.value) is bool, "invalid_metric", key, "boolean metric must be bool")
    stats = result.execution_statistics
    require(all(v is None or v >= 0 for v in to_data(stats).values()),
            "invalid_count", "execution_statistics", "negative count/time")
    return result
