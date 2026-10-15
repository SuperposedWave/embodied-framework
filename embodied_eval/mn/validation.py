"""Shared semantic checks; no model or simulator names belong here."""
from dataclasses import fields
import math
from .. import contracts as leaf
from ..codec import decode
from ..validation import (require, validate_tensor, validate_space as validate_leaf_space,
                          quaternion, schema_value, target_table, validate_gripper, shape_matches)
from . import contracts as c


def tensor_matches(tensor, spec, path):
    validate_tensor(tensor)
    require(tensor.dtype == spec.dtype and tensor.layout == spec.layout and shape_matches(tensor.shape, spec.shape),
            "tensor_spec_mismatch", path, "payload differs from declared public sample")


def validate_space(space):
    decode(c.SpaceSpec, space)
    # Structural geometry/controller checks reuse the original leaf semantics.
    projection = {f.name: getattr(space, f.name) for f in fields(leaf.SpaceSpec)
                  if hasattr(space, f.name) and f.name != "contract_version"}
    validate_leaf_space(leaf.SpaceSpec(**projection))
    for key, spec in space.state_fields.items():
        require(bool(spec.unit) and bool(spec.tensor.shape) and all(n>0 for n in spec.tensor.shape),
                "invalid_state_spec", key, "unit and positive declared dimensions required")
        require(spec.frame_id is None or spec.frame_id in space.frames, "unknown_frame", key, "state frame absent")
        require(bool(spec.origin), "missing_provenance", key, "measurement/command origin required")
        if spec.semantic in ("eef_position", "base_position", "gripper_position"):
            require(spec.unit == "m", "unit_mismatch", key, "canonical positions are metres")
        if spec.semantic == "eef_orientation":
            require(spec.unit == "quaternion_xyzw" and spec.tensor.shape == (4,), "unit_mismatch", key, "canonical quaternion order is xyzw")
    return space


def same_episode(a, b):
    require((a.run_id, a.env_id, a.episode_id) == (b.run_id, b.env_id, b.episode_id),
            "cross_episode", "context", "message belongs to a different episode")


def validate_observation(obs, space):
    decode(c.Observation, obs)
    require(obs.spec_ref == space.spec_id and obs.context.message_id == obs.observation_id,
            "observation_identity_mismatch", "observation", "wrong space or message identity")
    require(obs.env_step >= 0 and obs.sim_time_s >= 0, "invalid_time", "observation", "nonnegative clocks required")
    require(set(obs.sensors) <= set(space.sensors) and set(obs.states) <= set(space.state_fields),
            "unknown_sample", "observation", "sample is not declared by public space")
    for frame,sample in obs.frame_transforms.items():
        require(frame in space.frames and space.frames[frame].dynamic,"unexpected_transform",frame,"only declared dynamic transforms are sampled")
        require(0<=sample.captured_at_s<=obs.sim_time_s+1e-9,"invalid_time",frame,"invalid transform time")
        quaternion(sample.transform.quaternion_xyzw,frame)
    for key, sample in obs.sensors.items():
        require(0 <= sample.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", key, "invalid acquisition time")
        require(sample.frame_id == space.sensors[key].frame_id, "frame_mismatch", key, "wrong sensor frame")
        if sample.validity == "valid":
            require(sample.payload is not None, "missing_payload", key, "valid sensor requires tensor")
            tensor_matches(sample.payload, space.sensors[key].tensor, key)
    for key, sample in obs.states.items():
        spec = space.state_fields[key]
        require(0 <= sample.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", key, "invalid acquisition time")
        require(sample.provenance == ("controller_target" if spec.semantic in ("joint_drive_target","gripper_drive_target") else "measurement"),
                "state_provenance_mismatch", key, "controller targets are not measured qpos")
        tensor_matches(sample.payload, spec.tensor, key)
        if spec.semantic == "eef_orientation" and sample.payload.data is not None:
            quaternion(sample.payload.data, key)
    return obs


def validate_prediction(pred, request, obs, space, cap, profile):
    decode(c.Prediction, pred)
    same_episode(pred.context, request)
    require(pred.context.message_id == request.message_id, "wrong_request", "prediction", "response is not for pending request")
    require(pred.based_on_observation_id == obs.observation_id, "stale_prediction", "prediction", "wrong source observation")
    require(pred.model_forward_count is None or pred.model_forward_count >= 0, "invalid_count", "model_forward_count", "nonnegative forward count")
    action = pred.action
    require(action.native_protocol == cap.output.native_protocol, "native_protocol_mismatch", "action", "model changed output domain")
    spec = space.action_specs[profile.binding.action_spec_ref]
    require(bool(action.commands), "empty_action", "action", "at least one command required")
    targets = target_table(space)
    require(len({cmd.command_id for cmd in action.commands}) == len(action.commands), "duplicate_id", "commands", "unique command IDs required")
    require(len({cmd.target_id for cmd in action.commands}) == len(action.commands), "duplicate_target", "commands", "one command per target")
    require(set(cmd.target_id for cmd in action.commands) == set(profile.binding.targets.values()),
            "target_binding_mismatch", "commands", "do not silently omit an arm, joint or gripper")
    poses={cmd.target_id:cmd for cmd in action.commands if isinstance(cmd,leaf.PoseTarget)}
    require(set(action.source_euler_xyz_rad)<=set(poses),"rotation_hint_mismatch","source_euler_xyz_rad","hint must address a pose command")
    if action.source_euler_xyz_rad:
        from .rotations import euler_quaternion_xyzw
        for target,euler in action.source_euler_xyz_rad.items():
            expected=euler_quaternion_xyzw(euler)
            actual=poses[target].pose.quaternion_xyzw
            require(abs(abs(sum(a*b for a,b in zip(expected,actual)))-1.)<1e-12,
                    "rotation_hint_mismatch",target,"lossless Euler hint must represent the public quaternion")
    for cmd in action.commands:
        slot=next(key for key,target_id in profile.binding.targets.items() if target_id==cmd.target_id)
        require(cmd.kind in cap.output.command_kinds and cmd.kind in spec.command_kinds
                and cmd.target_id in spec.target_ids, "unsupported_action", cmd.command_id, "undeclared kind or target")
        kind, target = targets[cmd.target_id]
        if isinstance(cmd, leaf.JointTarget):
            require(cmd.mode in cap.output.joint_modes,"unsupported_joint_mode",cmd.command_id,"model changed declared joint mode")
            require(kind == "joint" and cmd.unit == getattr(target, cmd.mode + "_unit"), "unit_mismatch", cmd.command_id, "joint unit mismatch")
            if cmd.mode == "position" and target.position_limits:
                require(target.position_limits[0] <= cmd.value <= target.position_limits[1], "action_out_of_bounds", cmd.command_id, "joint target outside limits")
        elif isinstance(cmd, (leaf.PoseTarget, leaf.PoseDelta)):
            require(kind == "end_effector", "target_kind_mismatch", cmd.command_id, "pose needs an end effector")
            pose = cmd.pose if isinstance(cmd, leaf.PoseTarget) else cmd.delta
            quaternion(pose.quaternion_xyzw, cmd.command_id)
            require((cmd.pose.frame_id if isinstance(cmd, leaf.PoseTarget) else cmd.frame_id) in space.frames,
                    "unknown_frame", cmd.command_id, "undeclared pose coordinates")
            if isinstance(cmd, leaf.PoseDelta):
                require(cmd.reference_observation_id == obs.observation_id if cmd.reference_policy == "chunk_start"
                        else cmd.reference_observation_id is None, "invalid_delta_reference", cmd.command_id, "wrong delta reference")
        elif isinstance(cmd, leaf.GripperCommand):
            require(cmd.mode in cap.output.gripper_modes.get(slot,()),"unsupported_gripper_mode",cmd.command_id,"model changed declared gripper mode")
            require(kind == "gripper", "target_kind_mismatch", cmd.command_id, "gripper command needs a gripper")
            validate_gripper(cmd.mode, cmd.value, target, cmd.command_id, space.robots[cmd.target_id.split("/")[0]].joints)
        elif isinstance(cmd, leaf.ControllerCommand):
            require(cmd.controller_profile_ref in spec.controller_refs, "unknown_controller", cmd.command_id, "undeclared controller")
            schema_value(space.controllers[cmd.controller_profile_ref].parameter_schema, cmd.parameters, cmd.command_id)
        elif isinstance(cmd, leaf.TwistCommand):
            require(kind in ("base", "end_effector") and cmd.twist.frame_id in space.frames,
                    "target_kind_mismatch", cmd.command_id, "twist needs base/eef and declared frame")
    return pred


def validate_step(reply, previous, request, action, space):
    decode(c.StepResult, reply)
    obs, feedback = reply.observation, reply.execution_feedback
    validate_observation(obs, space)
    same_episode(obs.context, request)
    require(feedback.request_id == request.message_id, "wrong_request", "feedback", "wrong pending request")
    require(set(feedback.command_ids) <= {x.command_id for x in action.commands}, "command_mismatch", "feedback", "unscheduled command")
    require(feedback.consumed_env_steps in (0, 1) and feedback.elapsed_sim_time_s >= 0, "invalid_execution_count", "feedback", "one call per action")
    if feedback.status == "executed":
        require(feedback.consumed_env_steps == 1 and set(feedback.command_ids) == {x.command_id for x in action.commands},
                "incomplete_execution_feedback", "feedback", "executed means one action and all declared commands, not silently omitted targets")
    require(obs.env_step - previous.env_step == feedback.consumed_env_steps
            and math.isclose(obs.sim_time_s - previous.sim_time_s, feedback.elapsed_sim_time_s, abs_tol=1e-8),
            "execution_count_mismatch", "feedback", "returned clocks differ from actual consumption")
    if space.timing_mode == "measured_native":
        require(feedback.consumed_physics_steps is not None and feedback.consumed_physics_steps >= 0,
                "invalid_execution_count", "physics_steps", "actual physics count required")
        require(math.isclose(feedback.elapsed_sim_time_s, feedback.consumed_physics_steps * space.physics_period_s, abs_tol=1e-8),
                "execution_time_mismatch", "physics_steps", "time must match actual counted physics steps")
    else:
        require(math.isclose(feedback.elapsed_sim_time_s, feedback.consumed_env_steps * space.control_period_s, abs_tol=1e-8),
                "execution_time_mismatch", "feedback", "fixed-period clock mismatch")
    return reply
