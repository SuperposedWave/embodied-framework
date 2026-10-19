"""Weight-profile-owned public paths and private native encoding/caches.

No benchmark ID, object or options are accepted. Reusing the existing native
codecs is intentional: it preserves released preprocessing, precision and RNG.
The old logical slot names exist only in this model-private encoding boundary.
"""
from dataclasses import fields, replace
from ..core import types as leaf
from ..native import contracts as native
from ..core.validation import require
from . import contracts as c
from .context import native_context

STATE_PATHS = {"position": "eef_position", "quaternion": "eef_quaternion",
    "fingers": "gripper_joint_positions", "grip": "gripper_open_fraction",
    "base": "robot_base_position", "joints": "joint_positions"}
for _side in ("left", "right"):
    STATE_PATHS.update({_side + "_position": _side + "_eef_position",
        _side + "_quaternion": _side + "_eef_quaternion",
        _side + "_grip": _side + "_gripper_open_fraction_target",
        _side + "_joints": _side + "_joint_positions"})


def profile_paths(profile, original):
    paths = {}
    for req in original.inputs:
        if req.kind == "task":
            paths[req.key] = "task_public.instruction"
        elif req.kind == "sensor":
            name = req.key
            if profile in ("eef10_osc", "state8_osc", "eef10_absolute", "state7_absolute"):
                name = {"left_wrist": "wrist", "right_wrist": "external_right"}.get(name, name)
            if profile == "nano-policy-droid-joint8":
                name = {"exterior_right": "external_1", "exterior_left": "external_0"}.get(name, name)
            if profile == "droid-joint8-causal":
                # Native right_image historically names source camera 0.
                name = {"exterior_right": "external_0", "exterior_left": "external_1"}.get(name, name)
            paths[req.key] = "sensors." + name
        elif req.key != "drive_targets":
            paths[req.key] = "states." + STATE_PATHS[req.key]
    return paths


def public_targets(original):
    return {slot: "robot0/" + ("base" if slot == "dual" else slot) for slot in original.output.target_slots}


def capability(family, options, original):
    profile = options["io_profile"]
    paths, targets, inputs = profile_paths(profile, original), public_targets(original), []
    for req in original.inputs:
        if req.key == "drive_targets":
            for side in ("left", "right"):
                inputs.extend((c.InputSpec(key=side + "_drive", public_path="states." + side + "_joint_drive_targets",
                    kind="state", semantic="joint_drive_target", unit="rad", provenance="controller_target",
                    tensor=leaf.TensorSpec(dtype="float64", shape=(6,), layout="vector"),
                    joint_ids=tuple("robot0/" + side + "_j" + str(i) for i in range(6))),
                    c.InputSpec(key=side + "_grip", public_path="states." + side + "_gripper_open_fraction_target",
                    kind="state", semantic="gripper_drive_target", unit="open_fraction", provenance="controller_target",
                    tensor=leaf.TensorSpec(dtype="float64", shape=(1,), layout="vector"))))
            continue
        values = {f.name: getattr(req, f.name) for f in fields(native.InputSpec)}
        values["public_path"] = paths[req.key]
        if req.kind == "state":
            key = paths[req.key].split(".")[1]
            values["provenance"] = "controller_target" if req.semantic == "gripper_drive_target" else "measurement"
            if req.semantic == "joint_position":
                prefix = "left_" if key.startswith("left_") else "right_" if key.startswith("right_") else ""
                values["joint_ids"] = tuple("robot0/" + prefix + "j" + str(i) for i in range(req.tensor.shape[0]))
            if req.semantic in ("eef_position", "eef_orientation"):
                values["site_id"] = "robot0/" + key.replace("_quaternion", "_orientation") + "_site"
        inputs.append(c.InputSpec(**values))
    out = original.output
    return c.CapabilitySpec(model_id=original.model_id, source_revision=original.source_revision,
        checkpoint_ref=original.checkpoint_ref, normalization_ref=original.normalization_ref,
        io_profile_id=profile, inputs=tuple(inputs),
        output=c.OutputSpec(command_kinds=out.command_kinds, targets={targets[k]: v for k, v in out.target_slots.items()},
            native_protocol=out.native_protocol, native_signature=out.native_signature, joint_modes=out.joint_modes,
            gripper_modes={targets[k]: v for k, v in out.gripper_modes.items()},
            control_sites={targets[k]: targets[k] + "_control_site" for k, kind in out.target_slots.items()
                           if kind == "end_effector" and "pose_target" in out.command_kinds}),
        supported_timing_modes=original.supported_timing_modes, training_embodiments=original.training_embodiments,
        observation_update=original.observation_update,
        temporal=c.ModelTiming(native_action_period_s=None,
            replanning_owner="server" if family == "openwam" else "adapter",
            provenance={"revision": original.source_revision, "checkpoint": original.checkpoint_ref,
                "io_profile": profile,
                "rule": "preserve native next-action/chunk cache; wire observations are current snapshots, not assembled histories",
                "action_period_status": "unknown: training FPS/stride has not been proven to define this weight's action duration"}))


def native_state_spec(spec):
    return native.StateSpec(**{f.name: getattr(spec, f.name) for f in fields(native.StateSpec)})


def private_space(space, original, paths):
    def project(cls, value):
        return cls(**{f.name: getattr(value, f.name) for f in fields(cls)})
    robots = {rid: leaf.RobotSpec(embodiment_id=r.embodiment_id, base_frame_id=r.base_frame_id,
        joints=r.joints, grippers=r.grippers,
        end_effectors={key: project(leaf.EndEffectorSpec, end) for key, end in r.end_effectors.items()})
        for rid, r in space.robots.items()}
    sensors = {key: project(leaf.SensorSpec, sensor) for key, sensor in space.sensors.items()}
    states = {key: native_state_spec(value) for key, value in space.state_fields.items()}
    if any(req.key == "drive_targets" for req in original.inputs):
        states["model/drive_targets"] = native.StateSpec(semantic="joint_drive_target", unit="rad_and_open_fraction", frame_id=None,
            tensor=leaf.TensorSpec(dtype="float64", shape=(14,), layout="vector"),
            origin="model-private reconstruction of released drive-target encoding, not a public mixed state")
    return native.SpaceSpec(spec_id=space.spec_id, revision=space.revision, protocol_id=space.protocol_id,
        protocol_version=space.protocol_version, timing_mode=space.timing_mode,
        control_period_s=space.control_period_s, physics_period_s=space.physics_period_s,
        frames=space.frames, robots=robots, sensors=sensors, state_fields=states,
        action_specs=space.action_specs, controllers=space.controllers, native_protocols=space.native_protocols,
        required_action_targets=space.required_action_targets, action_joint_modes=space.action_joint_modes)


def private_input(view, original, paths):
    sensors, states = {}, {}
    for req in original.inputs:
        if req.kind == "task":
            continue
        if req.key == "drive_targets":
            samples = [view.states[side + suffix] for side in ("left", "right")
                       for suffix in ("_joint_drive_targets", "_gripper_open_fraction_target")]
            require(all(s.validity == "valid" and s.payload is not None and s.payload.data is not None
                        and s.provenance == "controller_target" for s in samples),
                    "missing_required_input", "drive_targets", "released encoding needs actual controller targets")
            require(len({s.captured_at_s for s in samples}) == 1 and len({s.payload.dtype for s in samples}) == 1,
                    "state_acquisition_mismatch", "drive_targets", "do not combine different acquisition times or precision")
            values = [value for sample in samples for value in sample.payload.data]
            states[req.key] = native.StateSample(captured_at_s=samples[0].captured_at_s, provenance="controller_target",
                payload=replace(samples[0].payload, shape=(14,), data=values))
            continue
        key = paths[req.key].split(".", 1)[1]
        if req.kind == "sensor":
            sensors[req.key] = view.sensors[key]
        else:
            sample = view.states[key]
            require(sample.validity == "valid" and sample.payload is not None, "missing_required_input", key, "native encoding cannot use invalid states")
            states[req.key] = native.StateSample(captured_at_s=sample.captured_at_s, payload=sample.payload, provenance=sample.provenance)
    return native.ModelInput(context=native_context(view.context), observation_id=view.observation_id,
        sim_time_s=view.sim_time_s, task_public=view.task_public, sensors=sensors, states=states,
        frame_transforms=view.frame_transforms)


class CanonicalModel:
    def __init__(self, family, options, output, entry):
        self.original = entry.describe(options)
        self.cap = capability(family, options, self.original)
        self.paths = profile_paths(options["io_profile"], self.original)
        self.engine = entry.factory(options, output)

    def begin_episode(self, context, task, profile, space):
        self.profile = profile
        inputs = {req.key: "model/drive_targets" if req.key == "drive_targets" else self.paths[req.key].split(".", 1)[1]
                  for req in self.original.inputs}
        bound = native.BoundProfile(profile_id=profile.profile_id, model_id=profile.model_id, space_ref=profile.space_ref,
            warnings=profile.warnings, binding=native.BindingConfig(inputs=inputs, targets=public_targets(self.original),
                action_spec_ref=profile.action_spec_ref, frame_conversions=profile.frame_conversions))
        self.engine.begin_episode(native_context(context), task, bound, private_space(space, self.original, self.paths))

    def observe(self, model_input):
        require(not model_input.history, "unsupported_native_history", "model_input.history",
                "this released current-snapshot encoder must not silently discard a new history requirement")
        self.engine.observe(private_input(model_input, self.original, self.paths))

    def predict(self, context):
        result = self.engine.predict(native_context(context))
        return c.Prediction(context=context, based_on_observation_id=result.based_on_observation_id,
            action=result.action,
            model_forward_count=result.model_forward_count,
            transformations=result.transformations + ("canonical-v03-to-weight-private-encoding",))

    def end_episode(self, context):
        self.engine.end_episode(native_context(context))

    def diagnostics(self):
        return dict(self.engine.diagnostics(), public_contract_version="0.3", model_private_codec_only=True)

    def close(self):
        self.engine.close()
