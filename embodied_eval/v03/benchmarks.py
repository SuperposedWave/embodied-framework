"""Environment-owned canonicalization; never receives model IDs or options.

Existing environment-only loaders/executors retain pinned controller/scoring
behavior. Native field names appear only here and in source metadata.
"""
from dataclasses import fields, replace
from ..core import types as leaf
from ..native import contracts as native
from ..core.validation import require
from . import contracts as c
from .context import native_context, public_context

CAMERA_MAPS = {
    "libero": {"agentview_image": "head", "robot0_eye_in_hand_image": "wrist"},
    "vlabench": {"forward_rgb": "head", "franka_wrist_rgb": "wrist", "right_rgb": "external_right"},
    "robotwin": {"head_camera": "head", "left_camera": "left_wrist", "right_camera": "right_wrist"},
    "robodojo": {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"},
    # Stable source identities, not guessed physical sides. Model profiles state
    # their pinned ordering explicitly; a full base/optical calibration is NOT
    # fabricated from the source variable names.
    "droid_sim": {"external_cam": "external_0", "external_cam_2": "external_1", "wrist_cam": "wrist"},
}


def frame_map(family, space):
    cameras = CAMERA_MAPS[family]
    mapping = {"world": "world"}
    for source, key in cameras.items():
        fid = space.sensors[source].frame_id
        if fid != "world":
            mapping[fid] = "camera/" + key
    for fid in space.frames:
        if fid not in mapping:
            mapping[fid] = "robot0/reference_" + str(len(mapping))
    require(len(set(mapping.values())) == len(mapping), "frame_identity_collision", "frames", "two native frames cannot silently merge")
    return mapping


def state_names(family, old):
    mapping = {key: key for key in old.state_fields if key != "joint_drive_targets"}
    if family == "libero":
        mapping["gripper_positions"] = "gripper_joint_positions"
    if family == "droid_sim":
        mapping["arm_joint_positions"] = "joint_positions"
    for side in ("left", "right"):
        if family == "robotwin":
            mapping[side + "_gripper_open_fraction"] = side + "_gripper_open_fraction_target"
        if family == "robodojo":
            mapping[side + "_gripper_target"] = side + "_gripper_open_fraction_target"
    return mapping


def canonical_space(family, old):
    fmap, cmap = frame_map(family, old), CAMERA_MAPS[family]
    frames = {fmap[key]: replace(spec, parent_id=fmap.get(spec.parent_id)) for key, spec in old.frames.items()}
    # An unknown base orientation is not an identity transform to world.
    frames["robot0/base"] = leaf.FrameSpec(parent_id=None)
    sensors = {}
    for source, key in cmap.items():
        spec = old.sensors[source]
        values = {f.name: getattr(spec, f.name) for f in fields(leaf.SensorSpec)}
        values.update(role=key, frame_id=fmap[spec.frame_id], native_source={
            "field": source, "revision": old.revision, "mount": spec.mounted_on})
        if key in ("wrist", "left_wrist", "right_wrist"):
            values["mounted_on"] = "robot0/" + key
        sensors[key] = c.SensorSpec(**values)
    robots, sites = {}, {}
    for rid, spec in old.robots.items():
        joints = dict(spec.joints)
        if family == "libero":
            joints.update({"gripper_j" + str(i): leaf.JointSpec(joint_type="prismatic", position_unit="m",
                velocity_unit="m/s", effort_unit="N") for i in range(2)})
        ends = {}
        for key, end in spec.end_effectors.items():
            sid = rid + "/" + key + "_control_site"
            # OSC remains a controller-native extension, not a verified physical
            # pose executor. No guessed LIBERO TCP conversion is advertised.
            control = sid if family != "libero" else None
            if control:
                sites[sid] = c.ReferenceSite(robot_id=rid, kind="control_target",
                    origin="pinned native executor target: " + end.tcp_id,
                    native_source={"revision": old.revision, "target": rid + "/" + key})
            ends[key] = c.EndEffectorSpec(frame_id=fmap[end.frame_id], tcp_id=end.tcp_id,
                gripper_id=end.gripper_id, control_site_id=control)
        robots[rid] = c.RobotSpec(embodiment_id=spec.embodiment_id, base_frame_id=rid + "/base",
            joints=joints, end_effectors=ends, grippers=spec.grippers)
    states = {}
    for source, key in state_names(family, old).items():
        spec = old.state_fields[source]
        prefix = "left_" if key.startswith("left_") else "right_" if key.startswith("right_") else ""
        joint_ids, site = (), None
        if spec.semantic in ("joint_position", "joint_drive_target"):
            joint_ids = tuple("robot0/" + prefix + "j" + str(i) for i in range(spec.tensor.shape[0]))
        elif spec.semantic == "gripper_position":
            joint_ids = tuple("robot0/gripper_j" + str(i) for i in range(spec.tensor.shape[0]))
        if spec.semantic in ("eef_position", "eef_orientation"):
            site = "robot0/" + prefix + ("eef_orientation_site" if spec.semantic.endswith("orientation") else "eef_position_site")
            sites[site] = c.ReferenceSite(robot_id="robot0",
                kind="measured_orientation" if spec.semantic.endswith("orientation") else "measured_position",
                origin=spec.origin, native_source={"field": source, "revision": old.revision})
        states[key] = c.StateSpec(semantic=spec.semantic, unit=spec.unit, frame_id=fmap.get(spec.frame_id),
            tensor=spec.tensor, origin=spec.origin, site_id=site, joint_ids=joint_ids,
            provenance="controller_target" if spec.semantic in ("joint_drive_target", "gripper_drive_target") else "measurement",
            native_source={"field": source, "revision": old.revision})
    if "joint_drive_targets" in old.state_fields:
        original = old.state_fields["joint_drive_targets"]
        for side, offset in (("left", 0), ("right", 7)):
            states[side + "_joint_drive_targets"] = c.StateSpec(semantic="joint_drive_target", unit="rad", frame_id=None,
                tensor=replace(original.tensor, shape=(6,)), origin=side + " arm native drive targets, not measured qpos",
                provenance="controller_target", joint_ids=tuple("robot0/" + side + "_j" + str(i) for i in range(6)),
                native_source={"field": "joint_drive_targets", "slice": [offset, offset + 6], "revision": old.revision})
    return c.SpaceSpec(spec_id=old.spec_id + ":canonical-v03", revision=old.revision,
        protocol_id=old.protocol_id, protocol_version=old.protocol_version,
        timing_mode=old.timing_mode, control_period_s=old.control_period_s, physics_period_s=old.physics_period_s,
        frames=frames, robots=robots, sensors=sensors, state_fields=states, controllers=old.controllers,
        action_specs=old.action_specs, native_protocols=old.native_protocols,
        required_action_targets=old.required_action_targets, action_joint_modes=old.action_joint_modes,
        view_aliases={"main": "head"} if "head" in sensors else {}, reference_sites=sites,
        frame_conventions={"world": "right-handed native protocol world; metres/radians/seconds; no cross-benchmark origin identity",
            "robot0/base": "unconnected until a verified full base transform is available"})


def canonical_observation(family, old_space, obs, space=None):
    space = space or canonical_space(family, old_space)
    fmap, names = frame_map(family, old_space), state_names(family, old_space)
    sensors = {CAMERA_MAPS[family][key]: replace(sample, frame_id=fmap[sample.frame_id]) for key, sample in obs.sensors.items()}
    states = {names[key]: c.StateSample(captured_at_s=sample.captured_at_s, payload=sample.payload,
        provenance=sample.provenance) for key, sample in obs.states.items() if key != "joint_drive_targets"}
    if "joint_drive_targets" in obs.states:
        raw = obs.states["joint_drive_targets"]
        require(raw.payload.data is not None and raw.payload.shape == (14,), "invalid_native_state", "drive_targets", "inline pinned mixed vector required")
        for side, offset in (("left", 0), ("right", 7)):
            states[side + "_joint_drive_targets"] = c.StateSample(captured_at_s=raw.captured_at_s,
                payload=replace(raw.payload, shape=(6,), data=raw.payload.data[offset:offset + 6]), provenance="controller_target")
            gripper = states[side + "_gripper_open_fraction_target"]
            require(gripper.payload is not None and gripper.payload.data == [raw.payload.data[offset + 6]]
                    and gripper.captured_at_s == raw.captured_at_s,
                    "inconsistent_native_target", side, "separate and mixed-vector gripper targets differ; do not guess which is true")
    return c.Observation(context=public_context(obs.context), observation_id=obs.observation_id, spec_ref=space.spec_id,
        env_step=obs.env_step, sim_time_s=obs.sim_time_s, sensors=sensors, states=states,
        public_instruction=obs.public_instruction,
        frame_transforms={fmap[key]: value for key, value in obs.frame_transforms.items()})


class CanonicalBenchmark:
    def __init__(self, family, options, output, entry):
        self.family = family
        self.engine = entry.factory(options, output)
        self.latest = None
        try:
            self.describe()
        except Exception:
            self.engine.close()
            raise

    def describe(self):
        self.native_space = self.engine.describe()
        self.space = canonical_space(self.family, self.native_space)
        return self.space

    def enumerate_tasks(self, episodes):
        return [c.TaskSpec(**{f.name: getattr(task, f.name) for f in fields(c.TaskSpec) if f.name != "contract_version"})
                for task in self.engine.enumerate_tasks(episodes)]

    def reset(self, task, context):
        old = native.TaskSpec(**{f.name: getattr(task, f.name) for f in fields(native.TaskSpec) if f.name != "contract_version"})
        result = self.engine.reset(old, native_context(context))
        self.latest = canonical_observation(self.family, self.native_space, result, self.space)
        return self.latest

    def advance(self, action, context, remaining):
        from .transforms import action_for_environment
        converted = action_for_environment(action, self.space, self.latest)
        result = self.engine.advance(converted, native_context(context), remaining)
        self.latest = canonical_observation(self.family, self.native_space, result.observation, self.space)
        return c.StepResult(observation=self.latest, execution_feedback=result.execution_feedback,
                           terminated=result.terminated, truncated=result.truncated)

    def score(self, context):
        return self.engine.score(native_context(context))

    def diagnostics(self):
        return dict(self.engine.diagnostics(), public_contract_version="0.3", canonical_environment_boundary=True)

    def close(self):
        self.engine.close()
