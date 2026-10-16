"""Shared structural/physical checks, independent of message version."""
import math
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable
from . import types as c
from ..codec import decode, json_value
from ..errors import ContractError


def require(condition, code, path, message):
    if not condition:
        raise ContractError(code, path, message)


def quaternion(q, path):
    require(math.isclose(sum(v * v for v in q), 1.0, abs_tol=1e-6),
            "invalid_rotation", path, "expected a unit quaternion_xyzw; normalize explicitly in the adapter")


def pose(p, space, path):
    quaternion(p.quaternion_xyzw, path + ".quaternion_xyzw")
    require(p.frame_id in space.frames, "unknown_frame", path + ".frame_id", p.frame_id)


def unique(values, path):
    require(len(values) == len(set(values)), "duplicate_id", path, "IDs must be unique")


def check_schema(schema, path):
    json_value(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ContractError("invalid_schema", path, str(error)) from error
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef"):
                    require(isinstance(item, str) and item.startswith("#"),
                            "external_schema_ref", path, "v0.1 schemas must be self-contained")
                if key == "format":
                    require(item in FormatChecker.checkers, "unsupported_format", path, str(item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(schema)


def schema_value(schema, value, path):
    check_schema(schema, path + ".schema")
    value = json_value(value, path)
    try:
        # An empty registry prevents retrieval even if nested schema IDs rebase a fragment.
        error = next(Draft202012Validator(schema, format_checker=FormatChecker(), registry=Registry()).iter_errors(value), None)
    except Unresolvable as error:
        raise ContractError("invalid_schema_ref", path, "schema reference cannot be resolved locally") from error
    if error is not None:
        suffix = ".".join(str(v) for v in error.absolute_path)
        raise ContractError("schema_mismatch", path + ("." + suffix if suffix else ""), error.message)


def target_table(space):
    targets = {}
    for rid, robot in space.robots.items():
        targets[f"{rid}/base"] = ("base", robot)
        for collection, kind in ((robot.joints, "joint"), (robot.end_effectors, "end_effector"),
                                 (robot.grippers, "gripper")):
            for key, entry in collection.items():
                target = f"{rid}/{key}"
                require(target not in targets, "duplicate_target", target, "robot entity IDs collide")
                targets[target] = (kind, entry)
    return targets


def validate_tensor(t: c.Tensor):
    decode(c.Tensor, t)
    require(bool(t.shape) and all(n > 0 for n in t.shape), "invalid_shape", "tensor.shape", "positive shape required")
    require((t.data is not None) != (t.data_ref is not None), "ambiguous_payload", "tensor",
            "provide exactly one of data or data_ref")
    if t.data_ref is not None:
        sizes = {"uint8": 1, "uint16": 2, "int32": 4, "float32": 4, "float64": 8, "bool": 1}
        require(t.data_ref.byte_length == math.prod(t.shape) * sizes[t.dtype],
                "invalid_buffer_size", "tensor.data_ref", "ref describes an uncompressed tensor buffer")
        require(len(t.data_ref.sha256) == 64 and all(x in "0123456789abcdef" for x in t.data_ref.sha256),
                "missing_checksum", "tensor.data_ref", "lowercase SHA-256 required")
        return t
    def walk(value, dims, path):
        if dims:
            require(isinstance(value, (list, tuple)) and len(value) == dims[0],
                    "tensor_shape_mismatch", path, "nested data does not match shape")
            if len(dims) == 1 and t.dtype in ("uint8", "uint16", "int32", "bool"):
                # Same per-value checks, with paths built only on failure.
                # Do not cache a result: nested lists remain mutable.
                if t.dtype == "bool":
                    for i, v in enumerate(value):
                        if type(v) is not bool:
                            walk(v, (), f"{path}[{i}]")
                else:
                    lo, hi = {"uint8": (0, 255), "uint16": (0, 65535),
                              "int32": (-2147483648, 2147483647)}[t.dtype]
                    for i, v in enumerate(value):
                        if type(v) is not int or not lo <= v <= hi:
                            walk(v, (), f"{path}[{i}]")
                return
            for i, v in enumerate(value):
                walk(v, dims[1:], f"{path}[{i}]")
            return
        if t.dtype == "bool":
            require(type(value) is bool, "tensor_dtype_mismatch", path, "expected bool")
        elif t.dtype in ("uint8", "uint16", "int32"):
            bounds = {"uint8": (0, 255), "uint16": (0, 65535), "int32": (-2147483648, 2147483647)}
            lo, hi = bounds[t.dtype]
            require(type(value) is int and lo <= value <= hi, "tensor_dtype_mismatch", path, "integer outside dtype range")
        else:
            require(type(value) in (float, int) and math.isfinite(value),
                    "tensor_dtype_mismatch", path, "expected finite number")
            if t.dtype == "float32":
                require(abs(value) <= 3.4028234663852886e38,
                        "tensor_dtype_mismatch", path, "value exceeds float32 range")
    walk(t.data, t.shape, "tensor.data")
    return t


def shape_matches(actual, expected):
    return len(actual) == len(expected) and all(b == -1 or a == b for a, b in zip(actual, expected))


def validate_gripper(mode, value, spec, path, joints=None):
    require(mode in spec.modes, "unsupported_gripper_mode", path, mode)
    if mode == "joint_targets":
        require(isinstance(value, dict) and bool(value), "invalid_gripper", path, "named joint values required")
        require(set(value) == set(spec.joint_ids), "invalid_gripper", path, "provide exactly the declared gripper joints")
        for jid, position in value.items():
            require(type(position) in (int, float) and math.isfinite(position), "invalid_gripper", path, "finite joint position required")
            require(joints is not None and jid in joints, "missing_semantics", path, "gripper joint specs required")
            limits = joints[jid].position_limits
            require(limits is None or limits[0] <= position <= limits[1],
                    "action_out_of_bounds", path, "gripper joint position exceeds limits")
    else:
        require(type(value) in (int, float) and math.isfinite(value), "invalid_gripper", path, "finite scalar required")
        if mode == "open_fraction":
            require(0 <= value <= 1, "invalid_gripper", path, "open_fraction is 0 closed / 1 open")
        else:
            require(value >= 0, "invalid_gripper", path, "negative width/force")
        if mode == "width_m" and spec.max_width_m is not None:
            require(value <= spec.max_width_m, "invalid_gripper", path, "width exceeds declared maximum")


def validate_structure(space):
    require(bool(space.spec_id and space.revision), "missing_revision", "space", "versioned spec identity required")
    if space.timing_mode == "fixed_period":
        require(space.control_period_s is not None and space.control_period_s > 0,
                "invalid_timing", "control_period_s", "fixed control period must be positive")
    else:
        require(space.control_period_s is None, "invalid_timing", "control_period_s",
                "native duration is measured, not a constant control period")
        require(space.physics_period_s is not None, "invalid_timing", "physics_period_s",
                "native execution needs a declared physical step period")
        require(not (getattr(space, "discrete_specs", {}) or getattr(space, "tools", {})), "unsupported_native_output", "space",
                "initial native timing supports only NativeAction, not discrete actions or tools")
    require(space.physics_period_s is None or space.physics_period_s > 0,
            "invalid_timing", "physics_period_s", "physical step period must be positive")
    require(bool(space.frames), "missing_frame", "frames", "at least one root frame required")
    for fid, entry in space.frames.items():
        require(bool(fid), "invalid_id", "frames", "empty ID")
        require(entry.parent_id is None or entry.parent_id in space.frames, "unknown_frame", fid, "unknown parent")
        seen, cursor = set(), fid
        while cursor is not None:
            require(cursor not in seen, "frame_cycle", fid, "frame graph contains a cycle")
            require(cursor in space.frames, "unknown_frame", fid, "unknown ancestor")
            seen.add(cursor)
            cursor = space.frames[cursor].parent_id
        if entry.parent_id is None:
            require(not entry.dynamic and entry.static_transform is None,
                    "invalid_root", fid, "root has no transform to a parent")
        else:
            require(entry.dynamic != (entry.static_transform is not None),
                    "ambiguous_transform", fid, "declare exactly one of static or dynamic transform")
        if entry.static_transform:
            quaternion(entry.static_transform.quaternion_xyzw, fid)
    targets = target_table(space)
    for rid, robot in space.robots.items():
        require(bool(rid) and "/" not in rid, "invalid_id", rid, "robot ID must be nonempty and unqualified")
        for entity in (*robot.joints, *robot.end_effectors, *robot.grippers):
            require(bool(entity) and "/" not in entity and entity != "base",
                    "invalid_id", rid, "entity IDs must be nonempty, unqualified and not reserved base")
        require(robot.base_frame_id in space.frames, "unknown_frame", rid, "unknown base frame")
        for jid, joint in robot.joints.items():
            expected = ("rad", "rad/s", "Nm") if joint.joint_type == "revolute" else ("m", "m/s", "N")
            require((joint.position_unit, joint.velocity_unit, joint.effort_unit) == expected,
                    "unit_mismatch", f"{rid}/{jid}", f"expected {expected}")
            if joint.position_limits:
                require(joint.position_limits[0] <= joint.position_limits[1],
                        "invalid_bounds", f"{rid}/{jid}", "inverted bounds")
        for eid, eff in robot.end_effectors.items():
            require(eff.frame_id in space.frames, "unknown_frame", f"{rid}/{eid}", eff.frame_id)
            require(eff.gripper_id is None or eff.gripper_id in robot.grippers,
                    "unknown_target", f"{rid}/{eid}", "gripper_id is local to the robot")
        for gid, gripper in robot.grippers.items():
            require(bool(gripper.modes), "missing_mode", f"{rid}/{gid}", "gripper modes required")
            unique(gripper.joint_ids, f"{rid}/{gid}.joint_ids")
            require(set(gripper.joint_ids) <= robot.joints.keys(), "unknown_joint", f"{rid}/{gid}", "gripper joints must be declared robot joints")
            require("joint_targets" not in gripper.modes or bool(gripper.joint_ids),
                    "missing_semantics", f"{rid}/{gid}", "joint_targets uses declared joint position units and limits")
            if gripper.max_width_m is not None:
                require(gripper.max_width_m > 0, "invalid_bounds", f"{rid}/{gid}", "width must be positive")
    for sid, sensor in space.sensors.items():
        require(sensor.frame_id in space.frames, "unknown_frame", sid, sensor.frame_id)
        require(bool(sensor.tensor.shape) and all(n > 0 or n == -1 for n in sensor.tensor.shape),
                "invalid_shape", sid, "spec shape uses positive dimensions or -1")
        if sensor.available_rate_hz is not None:
            require(sensor.available_rate_hz > 0, "invalid_timing", sid, "sample rate must be positive")
        if sensor.modality == "rgb":
            require(sensor.color_order == "RGB" and sensor.tensor.layout in ("HWC", "CHW"),
                    "invalid_rgb", sid, "RGB with HWC or CHW is required")
            channel = 2 if sensor.tensor.layout == "HWC" else 0
            require(len(sensor.tensor.shape) == 3 and sensor.tensor.shape[channel] == 3,
                    "invalid_rgb", sid, "RGB requires three channels")
        if sensor.modality == "depth":
            require(sensor.unit == "m", "unit_mismatch", sid, "canonical metric depth uses metres")
    for ref, controller in space.controllers.items():
        require(controller.frame_id in space.frames, "unknown_frame", ref, controller.frame_id)
        check_schema(controller.parameter_schema, ref)
        keys = set(controller.parameter_schema.get("properties", {}))
        require(keys == set(controller.parameter_units), "missing_semantics", ref,
                "declare the unit/domain of every controller parameter")
        require(bool(keys), "missing_semantics", ref, "controller parameters need named properties")
        require(all(bool(unit) for unit in controller.parameter_units.values()),
                "missing_semantics", ref, "parameter units/domain labels cannot be empty")
    for ref, action in space.action_specs.items():
        unique(action.target_ids, ref)
        unique(action.command_kinds, ref + ".command_kinds")
        unique(action.controller_refs, ref + ".controller_refs")
        require(bool(action.command_kinds and action.target_ids), "empty_action_spec", ref, "kinds and targets required")
        require(set(action.target_ids) <= targets.keys(), "unknown_target", ref, "unknown action target")
        require(set(action.controller_refs) <= space.controllers.keys(), "unknown_controller", ref, "unknown controller")
    return space


def same_episode(a, b):
    require((a.run_id, a.env_id, a.episode_id) == (b.run_id, b.env_id, b.episode_id),
            "episode_mismatch", "context", "run/environment/episode identity differs")
