"""Deterministic public-path matching, never per-combination wiring."""
from dataclasses import replace
import copy
from ..codec import decode
from ..errors import Issue
from ..core.validation import require, shape_matches, target_table
from ..core.compatibility import CompatibilityReport
from . import contracts as c


def input_key(space, req):
    prefix, dot, key = req.public_path.partition(".")
    require(bool(dot and key) and "." not in key, "invalid_public_path", req.key, "use one explicit public sensor/state/task path")
    if prefix == "views" and req.kind == "sensor":
        require(key in space.view_aliases, "missing_view", req.key, "no explicit public default view")
        return space.view_aliases[key]
    require(prefix == {"sensor": "sensors", "state": "states", "task": "task_public"}[req.kind],
            "invalid_public_path", req.key, "path domain differs from input kind")
    require(req.kind != "task" or key in ("instruction", "goal_image", "goal_location"),
            "private_input_forbidden", req.key, "only declared public task fields may reach a model")
    return key


def connected(space, source, target):
    def root(frame):
        seen = set()
        while frame in space.frames and space.frames[frame].parent_id is not None:
            if frame in seen:
                return None
            seen.add(frame)
            frame = space.frames[frame].parent_id
        return frame if frame in space.frames else None
    return root(source) is not None and root(source) == root(target)


def compatibility(space, cap, control_mode=None, temporal=None):
    decode(c.SpaceSpec, space)
    decode(c.CapabilitySpec, cap)
    issues, warnings, paths, conversions = [], [], {}, set()
    def check(ok, code, path, message):
        if not ok:
            issues.append(Issue(code, path, message))
    check(all((cap.model_id, cap.io_profile_id, cap.source_revision, cap.checkpoint_ref, cap.normalization_ref)),
          "missing_provenance", "capability", "named source/weight/normalization and I/O identity required")
    check(space.timing_mode in cap.supported_timing_modes, "timing_mode_mismatch", "timing_mode", "model and simulator execution timing differ")
    check(len({r.key for r in cap.inputs}) == len(cap.inputs), "duplicate_id", "inputs", "unique requirement IDs required")
    for req in cap.inputs:
        try:
            key = input_key(space, req)
        except ValueError as error:
            if req.required or error.issue.code != "missing_view":
                issues.append(error.issue)
            continue
        paths[req.key] = ("task_public." if req.kind == "task" else "sensors." if req.kind == "sensor" else "states.") + key
        if req.kind == "task":
            continue
        table = space.sensors if req.kind == "sensor" else space.state_fields
        if key not in table and not req.required:
            continue
        check(key in table, "missing_required_input", req.public_path, "required canonical field is absent")
        if key not in table:
            continue
        spec = table[key]
        actual = spec.modality if req.kind == "sensor" else spec.semantic
        check(req.semantic is None or req.semantic == actual, "input_semantics_mismatch", req.public_path, "public physical meaning differs")
        check(req.unit is None or req.unit == spec.unit, "unit_mismatch", req.public_path, "native preprocessing may not invent public units")
        check(req.max_age_s is None or req.max_age_s >= 0, "invalid_timing", req.public_path, "nonnegative input age required")
        if req.tensor:
            check(req.tensor.dtype == spec.tensor.dtype and req.tensor.layout == spec.tensor.layout
                  and shape_matches(spec.tensor.shape, req.tensor.shape), "tensor_spec_mismatch", req.public_path, "shape/dtype/layout differ")
        if req.kind == "state":
            check(req.provenance is None or req.provenance == spec.provenance, "state_provenance_mismatch", req.public_path, "control targets do not satisfy measured inputs")
            check(req.site_id is None or req.site_id == spec.site_id, "reference_site_mismatch", req.public_path, "coordinate changes cannot guess a different tool reference point")
            check(not req.joint_ids or req.joint_ids == spec.joint_ids, "joint_order_mismatch", req.public_path, "physical joint IDs/order differ")
        if req.frame_id is not None and req.frame_id != spec.frame_id:
            check(connected(space, spec.frame_id, req.frame_id), "disconnected_frames", req.public_path, "no declared coordinate transform path")
            if connected(space, spec.frame_id, req.frame_id):
                conversions.add((spec.frame_id, req.frame_id))
    targets = target_table(space)
    check(bool(cap.output.targets and cap.output.command_kinds), "missing_output", "output", "nonempty typed commands and public targets required")
    for tid, kind in cap.output.targets.items():
        check(tid in targets, "unknown_target", tid, "required public action target absent")
        if tid not in targets:
            continue
        check(targets[tid][0] == kind, "target_kind_mismatch", tid, "joint/eef/gripper/base differ")
        if kind == "gripper" and "gripper_command" in cap.output.command_kinds:
            modes = cap.output.gripper_modes.get(tid, ())
            check(bool(modes) and set(modes) <= set(targets[tid][1].modes), "unsupported_gripper_mode", tid, "no calibrated conversion between gripper modes")
        if kind == "end_effector" and "pose_target" in cap.output.command_kinds:
            eff = targets[tid][1]
            check(bool(eff.control_site_id), "missing_control_site", tid, "physical action reference must be known")
            check(tid not in cap.output.control_sites or cap.output.control_sites[tid] == eff.control_site_id,
                  "reference_site_mismatch", tid, "model output and controller reference sites differ")
            if eff.frame_id != cap.output.pose_frame_id:
                check(connected(space, cap.output.pose_frame_id, eff.frame_id), "disconnected_frames", tid, "no output coordinate conversion path")
                if connected(space, cap.output.pose_frame_id, eff.frame_id):
                    conversions.add((cap.output.pose_frame_id, eff.frame_id))
    if cap.output.native_protocol:
        check(bool(cap.output.native_signature) and space.native_protocols.get(cap.output.native_protocol) == cap.output.native_signature,
              "native_protocol_mismatch", "output", "named native extensions require identical meaning, not only equal vector dimensions")
    else:
        check(cap.output.native_signature is None, "invalid_native_protocol", "output", "signature needs protocol")
    candidates = []
    for ref, spec in space.action_specs.items():
        if not set(cap.output.command_kinds) <= set(spec.command_kinds):
            continue
        if not set(cap.output.targets) <= set(spec.target_ids):
            continue
        if not set(space.required_action_targets.get(ref, ())) <= set(cap.output.targets):
            continue
        if "joint_target" in cap.output.command_kinds and not (cap.output.joint_modes and
                set(cap.output.joint_modes) <= set(space.action_joint_modes.get(ref, ()))):
            continue
        candidates.append(ref)
    if control_mode is not None:
        check(control_mode in candidates, "unsupported_action", "control_mode", "explicit environment mode does not match declared outputs")
        selected = control_mode if control_mode in candidates else None
    else:
        check(bool(candidates), "unsupported_action", "output", "no complete executable action format")
        check(len(candidates) <= 1, "ambiguous_control_mode", "control_mode", "multiple execution formats; select a named environment mode explicitly")
        selected = candidates[0] if len(candidates) == 1 else None
    for rid in {t.split("/")[0] for t in cap.output.targets}:
        if rid in space.robots and space.robots[rid].embodiment_id not in cap.training_embodiments:
            warnings.append(f"training_coverage_unverified:{rid}:{space.robots[rid].embodiment_id}")
    from .temporal import assess
    timing_issues, timing_warnings, _ = assess(space, cap, paths, temporal)
    issues.extend(timing_issues)
    warnings.extend(timing_warnings)
    return CompatibilityReport(tuple(issues), tuple(warnings)), paths, tuple(sorted(conversions)), selected


def resolve(space, cap, control_mode=None, temporal=None):
    report, paths, conversions, selected = compatibility(space, cap, control_mode, temporal)
    report.require_compatible()
    return c.ResolvedProfile(profile_id=cap.model_id + "::" + space.spec_id, model_id=cap.model_id,
        space_ref=space.spec_id, action_spec_ref=selected, input_paths=paths,
        target_ids=tuple(cap.output.targets), frame_conversions=conversions, warnings=report.warnings)


def model_input(obs, task, cap, profile, space, history=None):
    sensors, states = {}, {}
    for req in cap.inputs:
        if req.kind == "task":
            continue
        path = profile.input_paths.get(req.key)
        key = path.split(".", 1)[1] if path else None
        table = obs.sensors if req.kind == "sensor" else obs.states
        if key not in table and not req.required:
            continue
        require(key in table, "missing_required_input", req.public_path, "required dynamic sample absent")
        sample = table[key]
        if not req.required and (sample.validity != "valid" or sample.payload is None):
            continue
        require(sample.validity == "valid" and sample.payload is not None, "missing_required_input", req.public_path, "required sample invalid")
        require(0 <= obs.sim_time_s - sample.captured_at_s + 1e-9 and
                (req.max_age_s is None or obs.sim_time_s - sample.captured_at_s <= req.max_age_s + 1e-9),
                "stale_input", req.public_path, "invalid/stale acquisition time")
        (sensors if req.kind == "sensor" else states)[key] = sample
    instruction = obs.public_instruction if obs.public_instruction is not None else task.public.instruction
    public = replace(task.public, instruction=instruction)
    for req in cap.inputs:
        if req.kind == "task" and req.required:
            require(bool(getattr(public, input_key(space, req))), "missing_required_input", req.public_path, "public task field unavailable")
    view = copy.deepcopy(c.ModelInput(context=obs.context, observation_id=obs.observation_id,
        sim_time_s=obs.sim_time_s, task_public=public, sensors=sensors, states=states,
        state_specs={key: space.state_fields[key] for key in states},
        sensor_specs={key: space.sensors[key] for key in sensors},
        view_aliases={alias: key for alias, key in space.view_aliases.items() if key in sensors},
        frame_transforms=obs.frame_transforms))
    needs_history = any(r.history is not None and r.key in profile.input_paths
                        and profile.input_paths[r.key].split(".", 1)[1] in
                        (sensors if r.kind == "sensor" else states) for r in cap.inputs)
    require(not needs_history or history is not None, "missing_history_buffer", "model_input",
            "declared history cannot be silently replaced by the latest snapshot")
    return history.assemble(view) if history is not None else view
