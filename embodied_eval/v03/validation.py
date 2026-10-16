"""Strict canonical names, measured provenance and physical reference sites."""
from dataclasses import fields
import math
from .. import _legacy_contracts as leaf
from ..codec import decode
from .._legacy_validation import require, quaternion, validate_tensor, shape_matches, target_table
from .._legacy_validation import validate_space as validate_leaf_space
from . import contracts as c

CAMERAS = {"head", "wrist", "left_wrist", "right_wrist", "external_front", "external_left",
           "external_right", "external_0", "external_1"}
STATES = {
    "joint_positions": ("joint_position", None, "measurement"),
    "joint_drive_targets": ("joint_drive_target", None, "controller_target"),
    "joint_velocities": ("joint_velocity", None, "measurement"),
    "joint_efforts": ("joint_effort", None, "measurement"),
    "eef_position": ("eef_position", "m", "measurement"),
    "eef_quaternion": ("eef_orientation", "quaternion_xyzw", "measurement"),
    "tcp_position": ("tcp_position", "m", "measurement"),
    "tcp_quaternion": ("tcp_orientation", "quaternion_xyzw", "measurement"),
    "gripper_joint_positions": ("gripper_position", None, "measurement"),
    "gripper_width": ("gripper_width", "m", "measurement"),
    "gripper_open_fraction": ("gripper_open_fraction", "open_fraction", "measurement"),
    "gripper_open_fraction_target": ("gripper_drive_target", "open_fraction", "controller_target"),
    "gripper_width_target": ("gripper_width_target", "m", "controller_target"),
    "gripper_joint_drive_targets": ("gripper_joint_drive_target", None, "controller_target"),
    "gripper_force_target": ("gripper_force_target", "N", "controller_target"),
    "robot_base_position": ("base_position", "m", "measurement"),
    "robot_base_quaternion": ("base_orientation", "quaternion_xyzw", "measurement"),
}


def leaf_space(space):
    """Reuse structural geometry checks WITHOUT changing shared historical types."""
    def project(cls, value):
        return cls(**{f.name: getattr(value, f.name) for f in fields(cls)})
    robots = {}
    for rid, robot in space.robots.items():
        robots[rid] = leaf.RobotSpec(embodiment_id=robot.embodiment_id, base_frame_id=robot.base_frame_id,
            joints=robot.joints, grippers=robot.grippers,
            end_effectors={key: project(leaf.EndEffectorSpec, value) for key, value in robot.end_effectors.items()})
    return leaf.SpaceSpec(spec_id=space.spec_id, revision=space.revision,
        protocol_id=space.protocol_id, protocol_version=space.protocol_version,
        control_period_s=space.control_period_s, timing_mode=space.timing_mode,
        physics_period_s=space.physics_period_s, frames=space.frames, robots=robots,
        sensors={key: project(leaf.SensorSpec, value) for key, value in space.sensors.items()},
        controllers=space.controllers, action_specs=space.action_specs)


def tensor_matches(tensor, spec, path):
    validate_tensor(tensor)
    require(tensor.dtype == spec.dtype and tensor.layout == spec.layout and shape_matches(tensor.shape, spec.shape),
            "tensor_spec_mismatch", path, "payload differs from declared physical sample")


def state_base(key):
    # Remove at most ONE side. A compound left_right_* name is not a public
    # left/right field and must not be silently treated as the underlying state.
    return key[5:] if key.startswith("left_") else key[6:] if key.startswith("right_") else key


def validate_space(space):
    decode(c.SpaceSpec, space)
    validate_leaf_space(leaf_space(space))
    require(set(space.robots) == {"robot0"}, "unsupported_robot_scope", "robots", "one explicitly controlled robot0, single or dual arm")
    require("world" in space.frames, "missing_frame", "world", "public world frame required")
    targets = target_table(space)
    for key, spec in space.sensors.items():
        role = key
        for suffix in ("_depth", "_point_cloud", "_audio", "_tactile"):
            role = role.removesuffix(suffix)
        require(role in CAMERAS and spec.role == role, "noncanonical_sensor", key, "use a canonical public camera role")
        require(bool(spec.native_source), "missing_provenance", key, "declared source required")
    require(set(space.view_aliases) <= {"main"}, "unknown_view_alias", "view_aliases", "only the explicit main alias is standardized")
    require(all(key in space.sensors for key in space.view_aliases.values()), "missing_view", "view_aliases", "alias must address an actual public sensor, never another alias")
    for sid, site in space.reference_sites.items():
        require(site.robot_id in space.robots and bool(site.origin), "invalid_reference_site", sid, "robot and actual origin required")
    for rid, robot in space.robots.items():
        for eid, end in robot.end_effectors.items():
            if end.control_site_id is not None:
                require(end.control_site_id in space.reference_sites and
                        space.reference_sites[end.control_site_id].kind == "control_target",
                        "unknown_control_site", rid + "/" + eid, "explicit physical controller reference required")
    for key, spec in space.state_fields.items():
        base = state_base(key)
        require(base in STATES and not (base.startswith("robot_base") and key != base),
                "noncanonical_state", key, "canonical single/left/right state key required")
        semantic, unit, provenance = STATES[base]
        require(spec.semantic == semantic and spec.provenance == provenance,
                "state_provenance_mismatch", key, "measurement and control targets are distinct fields")
        require(spec.robot_id in space.robots and bool(spec.origin) and bool(spec.native_source),
                "missing_provenance", key, "actual robot/origin/source required")
        require(spec.frame_id is None or spec.frame_id in space.frames, "unknown_frame", key, "state frame absent")
        require(bool(spec.tensor.shape) and all(n > 0 for n in spec.tensor.shape), "invalid_state_spec", key, "positive declared dimensions required")
        require(unit is None or spec.unit == unit, "unit_mismatch", key, "canonical units required")
        if semantic.endswith("orientation"):
            require(spec.tensor.shape == (4,) and spec.frame_id is not None, "invalid_state_spec", key, "xyzw has four components in an explicit frame")
        if semantic in ("eef_position", "tcp_position", "base_position"):
            require(spec.tensor.shape == (3,) and spec.frame_id is not None, "invalid_state_spec", key, "position has three metric components in an explicit frame")
        if semantic in ("eef_position", "eef_orientation", "tcp_position", "tcp_orientation"):
            require(spec.site_id in space.reference_sites, "missing_reference_site", key, "measured position/orientation site must be explicit")
            expected_kind = "measured_orientation" if semantic.endswith("orientation") else "measured_position"
            require(space.reference_sites[spec.site_id].kind in (expected_kind, "tcp"),
                    "reference_site_mismatch", key, "measurement cannot be relabelled as a control target")
        if "joint" in semantic or semantic == "gripper_position":
            require(spec.tensor.shape == (len(spec.joint_ids),) and bool(spec.joint_ids)
                    and len(set(spec.joint_ids)) == len(spec.joint_ids), "joint_order_missing", key, "each component needs a unique physical joint ID")
            require(all(j in targets and targets[j][0] == "joint" for j in spec.joint_ids), "unknown_joint", key, "state joints must actually be declared")
            attr = "velocity_unit" if semantic == "joint_velocity" else "effort_unit" if semantic == "joint_effort" else "position_unit"
            units = tuple(getattr(targets[j][1], attr) for j in spec.joint_ids)
            require((spec.component_units == units and spec.unit == "per_component") if len(set(units)) > 1
                    else spec.unit == units[0] and (not spec.component_units or spec.component_units == units),
                    "unit_mismatch", key, "joint component units/order must match physical joint declarations")
    for ref in (*space.required_action_targets, *space.action_joint_modes):
        require(ref in space.action_specs, "unknown_action_spec", ref, "execution constraints must address a declared format")
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
            "unknown_sample", "observation", "native/undeclared sample keys are not public fields")
    for frame, sample in obs.frame_transforms.items():
        require(frame in space.frames and space.frames[frame].dynamic, "unexpected_transform", frame, "only declared dynamic transforms are sampled")
        require(0 <= sample.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", frame, "invalid transform acquisition time")
        quaternion(sample.transform.quaternion_xyzw, frame)
    for key, sample in obs.sensors.items():
        require(0 <= sample.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", key, "invalid acquisition time")
        require(sample.frame_id == space.sensors[key].frame_id, "frame_mismatch", key, "wrong sensor frame")
        if sample.validity == "valid":
            require(sample.payload is not None and sample.invalid_reason is None, "missing_payload", key, "valid sensor needs actual data")
            tensor_matches(sample.payload, space.sensors[key].tensor, key)
        else:
            require(bool(sample.invalid_reason), "missing_invalid_reason", key, "invalid sample needs a reason")
    for key, sample in obs.states.items():
        spec = space.state_fields[key]
        require(0 <= sample.captured_at_s <= obs.sim_time_s + 1e-9, "invalid_time", key, "invalid acquisition time")
        require(sample.provenance == spec.provenance, "state_provenance_mismatch", key, "control targets are not measured qpos")
        if sample.validity == "invalid":
            require(bool(sample.invalid_reason) and sample.payload is None, "invalid_state_sample", key, "invalid state must not carry fabricated values")
            continue
        require(sample.payload is not None and sample.invalid_reason is None, "missing_payload", key, "valid state requires tensor")
        tensor_matches(sample.payload, spec.tensor, key)
        if spec.semantic.endswith("orientation") and sample.payload.data is not None:
            quaternion(sample.payload.data, key)
        if spec.unit == "open_fraction" and sample.payload.data is not None:
            require(all(-1e-8 <= v <= 1 + 1e-8 for v in sample.payload.data), "state_out_of_bounds", key, "measured/clipped state opening is not an unbounded action request")
    return obs


def validate_prediction(pred, request, obs, space, cap, profile):
    from .._legacy_validation import schema_value, validate_gripper
    decode(c.Prediction, pred)
    same_episode(pred.context, request)
    require(pred.context.message_id == request.message_id, "wrong_request", "prediction", "response is not for pending request")
    require(pred.based_on_observation_id == obs.observation_id, "stale_prediction", "prediction", "wrong source observation")
    require(pred.model_forward_count is None or pred.model_forward_count >= 0, "invalid_count", "model_forward_count", "nonnegative forward count")
    action = pred.action
    require(action.native_protocol == cap.output.native_protocol, "native_protocol_mismatch", "action", "model changed output domain")
    spec = space.action_specs[profile.action_spec_ref]
    require(bool(action.commands), "empty_action", "action", "at least one command required")
    targets = target_table(space)
    require(len({cmd.command_id for cmd in action.commands}) == len(action.commands), "duplicate_id", "commands", "unique command IDs required")
    require(len({cmd.target_id for cmd in action.commands}) == len(action.commands), "duplicate_target", "commands", "one command per target")
    require(set(cmd.target_id for cmd in action.commands) == set(profile.target_ids),
            "target_binding_mismatch", "commands", "do not silently omit an arm, joint or gripper")
    poses={cmd.target_id:cmd for cmd in action.commands if isinstance(cmd,leaf.PoseTarget)}
    require(set(action.source_euler_xyz_rad)<=set(poses),"rotation_hint_mismatch","source_euler_xyz_rad","hint must address a pose command")
    if action.source_euler_xyz_rad:
        from ..mn.rotations import euler_quaternion_xyzw
        for target,euler in action.source_euler_xyz_rad.items():
            expected=euler_quaternion_xyzw(euler)
            actual=poses[target].pose.quaternion_xyzw
            require(abs(abs(sum(a*b for a,b in zip(expected,actual)))-1.)<1e-12,
                    "rotation_hint_mismatch",target,"lossless Euler hint must represent the public quaternion")
    for cmd in action.commands:
        slot=cmd.target_id
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
            if isinstance(cmd, leaf.PoseTarget):
                require(cmd.pose.frame_id == cap.output.pose_frame_id, "frame_mismatch", cmd.command_id,
                        "model output must use its declared public pose frame")
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
    require(len(set(feedback.command_ids)) == len(feedback.command_ids), "duplicate_id", "feedback.command_ids", "one execution record per command")
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
