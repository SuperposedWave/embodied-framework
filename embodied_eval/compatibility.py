"""Static compatibility evidence and checks of required dynamic model inputs."""
from dataclasses import dataclass
from . import contracts as c
from .codec import decode
from .errors import ContractError, Issue
from .validation import require, validate_space, validate_observation, shape_matches


@dataclass(frozen=True)
class CompatibilityReport:
    issues: tuple[Issue, ...]

    @property
    def compatible(self):
        return not self.issues

    def require_compatible(self):
        if self.issues:
            item = self.issues[0]
            raise ContractError(item.code, item.path, item.message)


def binding_path(binding):
    if binding.kind == "sensor":
        return binding.source_id
    if binding.kind == "task":
        return binding.source_id
    parts = binding.source_id.split("/")
    require(len(parts) == 2, "invalid_binding", binding.source_id, "use robot/entity")
    rid, entity = parts
    if binding.kind == "joint":
        return f"{rid}/joints/{entity}/{binding.field_name}"
    if binding.kind == "end_effector":
        return f"{rid}/end_effectors/{entity}/{binding.field_name}"
    return f"{rid}/base_{binding.field_name}"


def binding_semantics(binding, space):
    """Return source units/frame/tensor; no implicit conversions."""
    if binding.kind == "sensor":
        require(binding.field_name is None, "invalid_binding", binding.source_id, "sensor binding has no field")
        require(binding.source_id in space.sensors, "missing_sensor", binding.source_id, "sensor not declared")
        s = space.sensors[binding.source_id]
        return s.unit, s.frame_id, s.tensor, s.modality
    if binding.kind == "task":
        require(binding.source_id in ("instruction", "goal_image", "goal_location") and binding.field_name is None,
                "invalid_binding", binding.source_id, "unknown public task field")
        return None, None, None, None
    parts = binding.source_id.split("/")
    require(len(parts) == 2 and parts[0] in space.robots, "invalid_binding", binding.source_id, "use declared robot/entity")
    rid, entity = parts
    r = space.robots[rid]
    if binding.kind == "joint":
        require(entity in r.joints and binding.field_name in ("position", "velocity", "effort"),
                "invalid_binding", binding.source_id, "unknown joint or field")
        return getattr(r.joints[entity], binding.field_name + "_unit"), None, None, None
    if binding.kind == "end_effector":
        require(entity in r.end_effectors and binding.field_name in ("pose", "gripper_state"),
                "invalid_binding", binding.source_id, "unknown end effector or field")
        require(binding.field_name != "gripper_state" or r.end_effectors[entity].gripper_id is not None,
                "invalid_binding", binding.source_id, "no gripper declared")
        # End-effector frame identifies the TCP, not the coordinates of each observed pose.
        return None, None, None, None
    require(entity == "base" and binding.field_name in ("pose", "twist"),
            "invalid_binding", binding.source_id, "base binding uses robot/base with pose or twist")
    return None, None, None, None


def check_compatibility(space: c.SpaceSpec, cap: c.CapabilitySpec, profile: c.AdaptationProfile):
    issues = []
    def check(ok, code, path, message):
        if not ok:
            issues.append(Issue(code, path, message))
    try:
        validate_space(space)
        decode(c.CapabilitySpec, cap)
        decode(c.AdaptationProfile, profile)
    except ContractError as e:
        return CompatibilityReport((e.issue,))
    for field, actual, expected in (
        ("benchmark_spec_ref", profile.benchmark_spec_ref, space.spec_id),
        ("model_id", profile.model_id, cap.model_id),
        ("model_source_revision", profile.model_source_revision, cap.source_revision),
        ("checkpoint_ref", profile.checkpoint_ref, cap.checkpoint_ref),
        ("normalization_ref", profile.normalization_ref, cap.normalization_ref),
        ("chunk_mode", profile.chunk_mode, cap.chunk_mode),
    ):
        check(actual == expected, "profile_mismatch", field, f"expected {expected!r}; got {actual!r}")
    check(bool(profile.profile_id and profile.revision), "missing_revision", "profile", "versioned profile required")
    check(bool(cap.model_id and cap.source_revision and cap.checkpoint_ref and cap.normalization_ref),
          "missing_provenance", "capability", "model revision, checkpoint and normalization identity required")
    check(not profile.unresolved, "unresolved_profile", "profile.unresolved", repr(profile.unresolved))
    check(profile.timing_mode == space.timing_mode and profile.timing_mode in cap.supported_timing_modes,
          "timing_mode_mismatch", "timing_mode", "benchmark, model capability and profile must agree")
    check(bool(cap.supported_timing_modes) and len(set(cap.supported_timing_modes)) == len(cap.supported_timing_modes),
          "invalid_timing", "supported_timing_modes", "nonempty unique timing modes required")
    if space.timing_mode == "measured_native":
        check(profile.chunk_mode == "adapter_owned" and profile.execution_horizon_steps == 1,
              "double_chunk_owner", "profile", "one native call per decision; the model adapter owns caching")
        check(cap.observation_update == "decision_boundary", "observation_history_mismatch", "capability",
              "native actions expose only decision-boundary observations, not every physical step")
    check(profile.role in cap.roles, "unsupported_role", "role", "role not supported by model")
    check(bool(profile.robot_ids), "missing_robot", "robot_ids", "select at least one robot")
    check(len(set(profile.robot_ids)) == len(profile.robot_ids), "duplicate_id", "robot_ids", "duplicate robot")
    for rid in profile.robot_ids:
        check(rid in space.robots, "unknown_robot", rid, "robot not declared")
        if rid in space.robots:
            check(space.robots[rid].embodiment_id in cap.supported_embodiments,
                  "unsupported_embodiment", rid, "checkpoint does not declare this embodiment")
    check(profile.execution_horizon_steps > 0, "invalid_horizon", "execution_horizon_steps", "positive horizon required")
    check(profile.chunk_mode != "adapter_owned" or profile.execution_horizon_steps == 1,
          "double_chunk_owner", "execution_horizon_steps", "adapter-owned cache exposes one step at a time")
    for name, refs, supported, accepted in (
        ("action_spec_refs", profile.action_spec_refs, cap.produced_action_specs, space.action_specs),
        ("discrete_spec_refs", profile.discrete_spec_refs, cap.produced_discrete_specs, space.discrete_specs),
        ("tool_refs", profile.tool_refs, cap.produced_tools, space.tools),
    ):
        check(set(refs) <= set(supported) and set(refs) <= set(accepted), "unsupported_output", name,
              "profile output must be declared by both adapters")
        for ref in refs:
            if ref in accepted:
                check(all(t.split("/")[0] in profile.robot_ids for t in accepted[ref].target_ids),
                      "robot_scope_mismatch", name + "." + ref, "output may address only selected robots")
    if profile.role == "action_policy":
        check(bool(profile.action_spec_refs or profile.discrete_spec_refs), "missing_output", "profile",
              "action policy needs an action or discrete output")
    if profile.role == "agent":
        check(bool(profile.tool_refs or profile.action_spec_refs or profile.discrete_spec_refs or cap.can_stop),
              "missing_output", "profile", "agent has no executable decision")
    if profile.role == "world_model":
        check(bool(cap.produced_predictions), "missing_output", "capability", "world model needs prediction modalities")
        check(not (profile.action_spec_refs or profile.discrete_spec_refs or profile.tool_refs),
              "world_model_role_mismatch", "profile", "pure prediction role has no executable decisions")
    required_ext = set(space.required_extensions) | set(profile.required_extensions)
    check(required_ext <= set(cap.supported_extensions), "unsupported_extension", "required_extensions", "model cannot interpret extension")
    check(required_ext <= set(space.extensions), "missing_extension", "required_extensions", "extension definition absent")
    check(required_ext <= set(profile.view_policy.extension_names), "hidden_required_input", "required_extensions", "required extension is not public")
    check(set(profile.view_policy.extension_names) <= set(space.extensions),
          "unknown_extension", "view_policy", "public extension has no definition")
    for source, target in profile.frame_conversions:
        check(source in space.frames and target in space.frames, "unknown_frame", "frame_conversions", "conversion frames must be declared")
        check(source in profile.view_policy.frame_ids and target in profile.view_policy.frame_ids,
              "hidden_required_input", "frame_conversions", "conversion frames must be public")
        if source in space.frames and target in space.frames:
            def root(frame):
                while space.frames[frame].parent_id is not None:
                    frame = space.frames[frame].parent_id
                return frame
            check(root(source) == root(target), "disconnected_frames", "frame_conversions", "no declared transform path")
    view = profile.view_policy
    for fid in view.frame_ids:
        check(fid in space.frames, "unknown_frame", "view_policy.frame_ids", fid)
        if fid in space.frames:
            parent = space.frames[fid].parent_id
            check(parent is None or parent in view.frame_ids, "hidden_required_input", fid, "public frame graph needs public parents")
    for sid in view.sensor_ids:
        check(sid in space.sensors, "unknown_sensor", "view_policy.sensor_ids", sid)
        if sid in space.sensors:
            check(space.sensors[sid].frame_id in view.frame_ids, "hidden_required_input", sid, "sensor frame must be public")
    for rid in profile.robot_ids:
        if rid in space.robots:
            robot = space.robots[rid]
            check(robot.base_frame_id in view.frame_ids and all(e.frame_id in view.frame_ids for e in robot.end_effectors.values()),
                  "hidden_required_input", rid, "selected robot coordinate metadata needs public frames")
    for ref in profile.action_spec_refs:
        if ref in space.action_specs:
            for controller in space.action_specs[ref].controller_refs:
                check(space.controllers[controller].frame_id in view.frame_ids,
                      "hidden_required_input", controller, "controller frame must be public")
    keys = [req.key for req in cap.required_inputs]
    check(len(keys) == len(set(keys)), "duplicate_id", "required_inputs", "duplicate input key")
    check(set(keys) == set(profile.input_bindings), "input_binding_mismatch", "input_bindings", "bind exactly the required input keys")
    for req in cap.required_inputs:
        if req.key not in profile.input_bindings:
            continue
        binding = profile.input_bindings[req.key]
        try:
            check(binding.kind == req.kind, "input_kind_mismatch", req.key, "required kind differs")
            unit, frame, tensor, modality = binding_semantics(binding, space)
            if binding.kind == "sensor":
                check(binding.source_id in view.sensor_ids, "hidden_required_input", req.key, "sensor not model-visible")
            elif binding.kind == "task":
                check(binding.source_id in view.task_fields, "hidden_required_input", req.key, "task field not model-visible")
            else:
                check(binding.source_id.split("/")[0] in profile.robot_ids, "robot_scope_mismatch", req.key, "input robot not selected")
                check(binding_path(binding) in view.robot_fields, "hidden_required_input", req.key, "robot field not model-visible")
            if req.unit is not None:
                check(req.unit == unit, "unit_mismatch", req.key, "model requirements describe canonical input, not native preprocessing")
            if req.frame_id is not None:
                check(req.frame_id in view.frame_ids, "hidden_required_input", req.key, "required frame is not public")
                if frame is not None:
                    check(req.frame_id == frame or (frame, req.frame_id) in profile.frame_conversions,
                          "frame_mismatch", req.key, "declare explicit frame conversion")
            if tensor:
                for name in ("dtype", "layout"):
                    expected = getattr(req, name)
                    check(expected is None or expected == getattr(tensor, name), "tensor_spec_mismatch", req.key, f"wrong {name}")
                check(req.shape is None or tensor.shape == req.shape or shape_matches(tensor.shape, req.shape),
                      "tensor_spec_mismatch", req.key, "shape mismatch requires an adapter-native preprocessing declaration, not an implicit resize")
            else:
                check(req.dtype is None and req.shape is None and req.layout is None,
                      "input_semantics_mismatch", req.key, "tensor constraints currently apply to sensors only")
            check(req.modality is None or req.modality == modality, "input_semantics_mismatch", req.key, "wrong modality")
            check(req.max_age_s is None or req.max_age_s >= 0, "invalid_timing", req.key, "negative maximum input age")
            check(req.max_age_s is None or req.kind == "sensor", "input_semantics_mismatch", req.key,
                  "v0.1 robot state is sampled at observation time; age constraint applies to sensors")
        except ContractError as e:
            issues.append(e.issue)
    # Check even optional visible robot paths, preventing misspelled allowlists from silently hiding fields.
    for path in view.robot_fields:
        try:
            parts = path.split("/")
            if len(parts) == 4 and parts[1] in ("joints", "end_effectors"):
                kind = "joint" if parts[1] == "joints" else "end_effector"
                binding_semantics(c.InputBinding(kind=kind, source_id=f"{parts[0]}/{parts[2]}", field_name=parts[3]), space)
            elif len(parts) == 2 and parts[1] in ("base_pose", "base_twist"):
                binding_semantics(c.InputBinding(kind="base", source_id=f"{parts[0]}/base", field_name=parts[1][5:]), space)
            else:
                raise ContractError("invalid_binding", path, "unknown robot visibility path")
            check(parts[0] in profile.robot_ids, "robot_scope_mismatch", path, "robot not selected")
        except ContractError as e:
            issues.append(e.issue)
    for ref in profile.tool_refs:
        if ref in space.tools and cap.observation_update == "every_env_step":
            check(space.tools[ref].intermediate_observations_supported, "observation_history_mismatch", ref,
                  "long tool needs intermediate observations for this model")
    for ref in profile.discrete_spec_refs:
        if ref in space.discrete_specs and cap.observation_update == "every_env_step":
            discrete = space.discrete_specs[ref]
            check(discrete.duration_s <= space.control_period_s or discrete.intermediate_observations_supported,
                  "observation_history_mismatch", ref, "multi-tick discrete action needs intermediate observations")
    return CompatibilityReport(tuple(issues))


def validate_model_input(model_input: c.ModelInput, space: c.SpaceSpec, cap: c.CapabilitySpec, profile: c.AdaptationProfile):
    decode(c.ModelInput, model_input)
    check_compatibility(space, cap, profile).require_compatible()
    from .validation import same_episode
    same_episode(model_input.context, model_input.observation_view.context)
    obs = validate_observation(model_input.observation_view, space)
    view = profile.view_policy
    require(set(obs.sensors) <= set(view.sensor_ids) and set(obs.extensions) <= set(view.extension_names)
            and set(obs.frame_transforms) <= set(view.frame_ids) and set(obs.robots) <= set(profile.robot_ids),
            "visibility_violation", "model_input", "undeclared visible observation data")
    for field in ("instruction", "goal_image", "goal_location"):
        value = getattr(model_input.task_public, field)
        require(field in view.task_fields or value is None or (field == "instruction" and value == ""),
                "visibility_violation", "task_public." + field, "task field is private")
    require(model_input.execution_feedback_view is None
            or set(model_input.execution_feedback_view) <= set(view.feedback_fields),
            "visibility_violation", "execution_feedback_view", "unexpected feedback field")
    if model_input.execution_feedback_view is not None:
        decode(c.ExecutionFeedback, model_input.execution_feedback_view | {
            key: value for key, value in dict(request_id="omitted", status="executed", consumed_env_steps=0,
                                            elapsed_sim_time_s=0.0).items()
            if key not in model_input.execution_feedback_view})
    require(model_input.candidate_actions is None, "reserved_input", "candidate_actions",
            "planner candidate-action input needs a phase 5C protocol; v0.1 leaves it reserved")
    for rid, state in obs.robots.items():
        for jid, joint in state.joints.items():
            for field in ("position", "velocity", "effort"):
                require(getattr(joint, field) is None or f"{rid}/joints/{jid}/{field}" in view.robot_fields,
                        "visibility_violation", rid + "/" + jid + "/" + field, "robot field is private")
        for eid, eff in state.end_effectors.items():
            for field in ("pose", "gripper_state"):
                require(getattr(eff, field) is None or f"{rid}/end_effectors/{eid}/{field}" in view.robot_fields,
                        "visibility_violation", rid + "/" + eid + "/" + field, "robot field is private")
        for field in ("base_pose", "base_twist"):
            require(getattr(state, field) is None or f"{rid}/{field}" in view.robot_fields,
                    "visibility_violation", rid + "/" + field, "robot field is private")
        measured = [e.pose for e in state.end_effectors.values()] + [state.base_pose, state.base_twist]
        for value in measured:
            require(value is None or value.frame_id in view.frame_ids,
                    "visibility_violation", rid, "pose/twist coordinates must be public")
    for key in profile.required_extensions:
        require(key in obs.extensions, "missing_required_input", key, "required dynamic extension absent")
    for req in cap.required_inputs:
        b = profile.input_bindings[req.key]
        if b.kind == "sensor":
            sample = obs.sensors.get(b.source_id)
            require(sample is not None and sample.validity == "valid", "missing_required_input", req.key, "required sensor unavailable")
            require(req.max_age_s is None or obs.sim_time_s - sample.captured_at_s <= req.max_age_s + 1e-9,
                    "stale_input", req.key, "sensor exceeds maximum age")
        elif b.kind == "task":
            value = getattr(model_input.task_public, b.source_id)
            require(value is not None and (b.source_id != "instruction" or bool(value)),
                    "missing_required_input", req.key, "required task field unavailable")
        else:
            rid, entity = b.source_id.split("/")
            state = obs.robots.get(rid)
            require(state is not None, "missing_required_input", req.key, "robot state absent")
            if b.kind == "joint":
                joint = state.joints.get(entity)
                require(joint is not None and joint.validity == "valid", "missing_required_input", req.key, "joint state unavailable")
                value = getattr(joint, b.field_name)
            elif b.kind == "end_effector":
                eff = state.end_effectors.get(entity)
                value = getattr(eff, b.field_name) if eff is not None else None
            else:
                value = getattr(state, "base_" + b.field_name)
            require(value is not None, "missing_required_input", req.key, "required state field absent")
            if req.frame_id is not None and isinstance(value, (c.Pose, c.Twist)):
                require(value.frame_id == req.frame_id or (value.frame_id, req.frame_id) in profile.frame_conversions,
                        "frame_mismatch", req.key, "required pose/twist frame conversion undeclared")
    for source, target in profile.frame_conversions:
        for frame in (source, target):
            cursor = frame
            while cursor is not None:
                spec = space.frames[cursor]
                require(not spec.dynamic or cursor in obs.frame_transforms, "missing_transform", cursor, "dynamic transform absent")
                cursor = spec.parent_id
    return model_input
