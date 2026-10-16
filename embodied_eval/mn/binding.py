"""Semantic compatibility and public projection, independent of training labels."""
from dataclasses import dataclass, replace
import copy
from ..codec import decode
from ..errors import Issue
from .._legacy_validation import require, shape_matches, target_table
from . import contracts as c


@dataclass(frozen=True)
class CompatibilityReport:
    issues: tuple[Issue, ...]
    warnings: tuple[str, ...]

    @property
    def compatible(self):
        return not self.issues

    def require_compatible(self):
        if self.issues:
            from ..errors import ContractError
            i = self.issues[0]
            raise ContractError(i.code, i.path, i.message)


def check_compatibility(space, cap, binding):
    decode(c.SpaceSpec, space)
    decode(c.CapabilitySpec, cap)
    decode(c.BindingConfig, binding)
    issues, warnings = [], []

    def check(ok, code, path, message):
        if not ok:
            issues.append(Issue(code, path, message))

    check(all((cap.model_id,cap.source_revision,cap.checkpoint_ref,cap.normalization_ref)),
          "missing_provenance","capability","model/source/checkpoint/normalization identities required")
    check(bool(cap.output.target_slots and cap.output.command_kinds),"missing_output","capability","executable action targets and types required")
    check(not cap.output.native_protocol or bool(cap.output.native_signature),"invalid_native_protocol","output","native protocol requires a nonempty signature")

    check(space.timing_mode in cap.supported_timing_modes, "timing_mode_mismatch", "timing_mode", "model and simulator timing differ")
    slots = [r.key for r in cap.inputs]
    check(len(set(slots)) == len(slots), "duplicate_id", "inputs", "duplicate logical input")
    check(set(binding.inputs) == set(slots), "input_binding_mismatch", "binding.inputs", "bind all and only required model inputs")
    for req in cap.inputs:
        source = binding.inputs.get(req.key)
        if source is None:
            continue
        table = space.sensors if req.kind == "sensor" else space.state_fields
        if req.kind == "task":
            check(source == "instruction", "invalid_binding", req.key, "only public instruction is supported")
            continue
        check(source in table, "missing_required_input", req.key, f"no public {req.kind} named {source}")
        if source not in table:
            continue
        spec = table[source]
        actual_semantic = spec.modality if req.kind == "sensor" else spec.semantic
        check(req.semantic is None or req.semantic == actual_semantic, "input_semantics_mismatch", req.key, f"expected {req.semantic}, got {actual_semantic}")
        check(req.unit is None or req.unit == spec.unit, "unit_mismatch", req.key, "public units differ; native preprocessing is model-owned")
        if req.frame_id is not None:
            check(req.frame_id == spec.frame_id or (spec.frame_id, req.frame_id) in binding.frame_conversions,
                  "frame_mismatch", req.key, "explicit input frame conversion required")
        check(req.max_age_s is None or req.max_age_s >= 0, "invalid_timing", req.key, "nonnegative age required")
        if req.tensor:
            check(req.tensor.dtype == spec.tensor.dtype and req.tensor.layout == spec.tensor.layout
                  and shape_matches(spec.tensor.shape, req.tensor.shape), "tensor_spec_mismatch", req.key, "public tensor shape/dtype/layout differ")
    check(set(binding.targets) == set(cap.output.target_slots), "target_binding_mismatch", "binding.targets", "no missing or silently dropped action targets")
    targets = target_table(space)
    check(len(set(binding.targets.values())) == len(binding.targets), "duplicate_target", "binding.targets", "two logical outputs cannot overwrite the same target")
    spec = space.action_specs.get(binding.action_spec_ref)
    check(spec is not None, "unsupported_action", "action_spec_ref", "simulator does not declare this action format")
    if spec:
        check(set(cap.output.command_kinds) <= set(spec.command_kinds), "unsupported_action", "command_kinds", "simulator cannot execute all model command types")
        check(set(space.required_action_targets.get(binding.action_spec_ref,()))<=set(binding.targets.values()),
              "missing_execution_target","binding.targets","simulator's atomic action requires all listed targets")
        if "joint_target" in cap.output.command_kinds:
            check(bool(cap.output.joint_modes) and set(cap.output.joint_modes)<=set(space.action_joint_modes.get(binding.action_spec_ref,())),
                  "unsupported_joint_mode","joint_modes","position/velocity/effort execution modes must be declared by both sides")
    for slot, target in binding.targets.items():
        check(target in targets, "unknown_target", slot, target)
        if target in targets and slot in cap.output.target_slots:
            check(targets[target][0] == cap.output.target_slots[slot], "target_kind_mismatch", slot, "joint/eef/gripper/base target differs")
            if targets[target][0]=="gripper" and "gripper_command" in cap.output.command_kinds:
                modes=cap.output.gripper_modes.get(slot,())
                check(bool(modes) and set(modes)<=set(targets[target][1].modes),"unsupported_gripper_mode",slot,"gripper width/fraction/force modes differ")
        if spec:
            check(target in spec.target_ids, "unsupported_action", slot, "target is outside the action format")
    if cap.output.native_protocol:
        check(space.native_protocols.get(cap.output.native_protocol) == cap.output.native_signature,
              "native_protocol_mismatch", "native_protocol", "opaque native arrays require an identical declared signature")
    else:
        check(cap.output.native_signature is None, "invalid_native_protocol", "output", "signature requires protocol")
    for source, target in binding.frame_conversions:
        check(source in space.frames and target in space.frames, "unknown_frame", "frame_conversions", "declare public conversion frames")
        if source in space.frames and target in space.frames:
            def root(f):
                seen = set()
                while space.frames[f].parent_id is not None:
                    if f in seen:
                        return None
                    seen.add(f)
                    f = space.frames[f].parent_id
                    if f not in space.frames:
                        return None
                return f
            check(root(source) is not None and root(source) == root(target), "disconnected_frames", "frame_conversions", "no public transform path")
    for rid in {t.split("/")[0] for t in binding.targets.values()}:
        if rid in space.robots and space.robots[rid].embodiment_id not in cap.training_embodiments:
            warnings.append(f"training_coverage_unverified:{rid}:{space.robots[rid].embodiment_id}")
    return CompatibilityReport(tuple(issues), tuple(warnings))


def resolve(space, cap, binding):
    report = check_compatibility(space, cap, binding)
    report.require_compatible()
    return c.BoundProfile(profile_id=f"{cap.model_id}::{space.spec_id}", binding=binding,
                         model_id=cap.model_id, space_ref=space.spec_id, warnings=report.warnings)


def model_input(observation, task, cap, profile):
    # Project into logical slots, deep-copy once, and never pass TaskSpec/private.
    sensors, states = {}, {}
    for req in cap.inputs:
        key = profile.binding.inputs[req.key]
        if req.kind == "task":
            continue
        table = observation.sensors if req.kind == "sensor" else observation.states
        require(key in table, "missing_required_input", req.key, "required dynamic sample absent")
        sample = table[key]
        if req.kind == "sensor":
            require(sample.validity == "valid" and sample.payload is not None, "missing_required_input", req.key, "invalid sensor")
        require(req.max_age_s is None or observation.sim_time_s - sample.captured_at_s <= req.max_age_s + 1e-9,
                "stale_input", req.key, "sample exceeds declared age")
        (sensors if req.kind == "sensor" else states)[req.key] = sample
    instruction = observation.public_instruction if observation.public_instruction is not None else task.public.instruction
    require(not any(r.kind == "task" for r in cap.inputs) or bool(instruction), "missing_required_input", "instruction", "public task text unavailable")
    return copy.deepcopy(c.ModelInput(context=observation.context, observation_id=observation.observation_id,
        sim_time_s=observation.sim_time_s, task_public=replace(task.public, instruction=instruction), sensors=sensors, states=states,
        frame_transforms=observation.frame_transforms))
