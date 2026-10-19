"""Adapter-owned coordinate math; never a Runner responsibility."""
from dataclasses import replace
from ..core import types as leaf
from ..geometry import compose, multiply_quaternions, rotate_vector
from ..core.transforms import transform_between
from ..core.validation import require, target_table
from .matching import input_key


def inputs_for_model(view, capability, space):
    states, specs = dict(view.states), dict(view.state_specs)
    history = dict(view.history)
    def convert(sample, spec, destination, at):
        transform = transform_between(space, at, spec.frame_id, destination)
        require(sample.payload is not None and sample.payload.data is not None, "payload_unavailable", spec.semantic,
                "inline data required; DataRef reading is not implemented")
        data = sample.payload.data
        if spec.semantic in ("eef_position", "tcp_position", "base_position"):
            data = [a + b for a, b in zip(rotate_vector(transform.quaternion_xyzw, data), transform.translation_m)]
        elif spec.semantic.endswith("orientation"):
            data = list(multiply_quaternions(transform.quaternion_xyzw, tuple(data)))
        else:
            require(False, "unsupported_frame_conversion", spec.semantic, "only explicitly physical position/orientation fields can be transformed")
        return replace(sample, payload=replace(sample.payload, data=data))
    for req in capability.inputs:
        if req.kind != "state" or req.frame_id is None:
            continue
        key = input_key(space, req)
        if key not in states and not req.required:
            continue
        spec = specs[key]
        if spec.frame_id == req.frame_id:
            continue
        states[key] = convert(states[key], spec, req.frame_id, view)
        path = "states." + key
        if path in history:
            history[path] = tuple(replace(entry, sample=convert(entry.sample, spec, req.frame_id,
                replace(view, sim_time_s=entry.sample.captured_at_s, frame_transforms=entry.frame_transforms)))
                for entry in history[path])
        specs[key] = replace(spec, frame_id=req.frame_id)
    return replace(view, states=states, state_specs=specs, history=history)


def action_for_environment(action, space, observation):
    targets = target_table(space)
    commands, hints = [], dict(action.source_euler_xyz_rad)
    for command in action.commands:
        if isinstance(command, leaf.PoseTarget):
            destination = targets[command.target_id][1].frame_id
            if command.pose.frame_id != destination:
                transform = transform_between(space, observation, command.pose.frame_id, destination)
                result = compose(transform, leaf.Transform(translation_m=command.pose.position_m,
                                                          quaternion_xyzw=command.pose.quaternion_xyzw))
                command = replace(command, pose=leaf.Pose(position_m=result.translation_m,
                    quaternion_xyzw=result.quaternion_xyzw, frame_id=destination))
                hints.pop(command.target_id, None)
        commands.append(command)
    return replace(action, commands=tuple(commands), source_euler_xyz_rad=hints)
