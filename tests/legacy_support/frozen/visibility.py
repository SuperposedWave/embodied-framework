"""Allowlisted public snapshots. Never pass TaskSpec or StepResult to a model."""
from copy import deepcopy
from dataclasses import replace
import tests.legacy_support.frozen._legacy_contracts as c
from embodied_eval.codec import to_data
from tests.legacy_support.frozen._legacy_validation import require, same_episode


def public_task(task, policy):
    return c.PublicTask(
        instruction=task.public.instruction if "instruction" in policy.task_fields else "",
        goal_image=deepcopy(task.public.goal_image) if "goal_image" in policy.task_fields else None,
        goal_location=deepcopy(task.public.goal_location) if "goal_location" in policy.task_fields else None,
    )


def public_space(space, profile):
    """Static robot metadata is public; scene frames and extensions are allowlisted."""
    v = profile.view_policy
    actions = {key: deepcopy(space.action_specs[key]) for key in profile.action_spec_refs}
    controller_ids = {key for a in actions.values() for key in a.controller_refs}
    return replace(
        space,
        frames={key: deepcopy(space.frames[key]) for key in v.frame_ids},
        robots={key: deepcopy(space.robots[key]) for key in profile.robot_ids},
        sensors={key: deepcopy(space.sensors[key]) for key in v.sensor_ids},
        action_specs=actions,
        controllers={key: deepcopy(space.controllers[key]) for key in controller_ids},
        discrete_specs={key: deepcopy(space.discrete_specs[key]) for key in profile.discrete_spec_refs},
        tools={key: deepcopy(space.tools[key]) for key in profile.tool_refs},
        extensions={key: deepcopy(space.extensions[key]) for key in v.extension_names},
    )


def build_model_input(task: c.TaskSpec, observation: c.Observation, profile: c.AdaptationProfile,
                      feedback: c.ExecutionFeedback | None = None, *, context: c.Context | None = None):
    require(task.episode_id == observation.context.episode_id, "episode_mismatch", "task", "wrong episode")
    context = context or observation.context
    same_episode(context, observation.context)
    v = profile.view_policy
    sensors = {}
    for key in v.sensor_ids:
        if key in observation.sensors:
            sample = deepcopy(observation.sensors[key])
            sensors[key] = replace(sample, invalid_reason="unavailable" if sample.validity == "invalid" else None)
    robots = {}
    for rid in profile.robot_ids:
        source = observation.robots.get(rid)
        if source is None:
            continue
        joints, effectors = {}, {}
        for jid, joint in source.joints.items():
            selected = {field: deepcopy(getattr(joint, field)) for field in ("position", "velocity", "effort")
                        if f"{rid}/joints/{jid}/{field}" in v.robot_fields}
            if selected and (joint.validity == "invalid" or any(value is not None for value in selected.values())):
                joints[jid] = c.JointState(**selected, validity=joint.validity,
                    invalid_reason="unavailable" if joint.validity == "invalid" else None)
        for eid, eff in source.end_effectors.items():
            selected = {field: deepcopy(getattr(eff, field)) for field in ("pose", "gripper_state")
                        if f"{rid}/end_effectors/{eid}/{field}" in v.robot_fields}
            if selected:
                effectors[eid] = c.EndEffectorState(**selected)
        robots[rid] = c.RobotState(
            joints=joints, end_effectors=effectors,
            base_pose=deepcopy(source.base_pose) if f"{rid}/base_pose" in v.robot_fields else None,
            base_twist=deepcopy(source.base_twist) if f"{rid}/base_twist" in v.robot_fields else None,
        )
    view = replace(
        observation, sensors=sensors, robots=robots,
        frame_transforms={key: deepcopy(value) for key, value in observation.frame_transforms.items() if key in v.frame_ids},
        extensions={key: deepcopy(value) for key, value in observation.extensions.items() if key in v.extension_names},
    )
    raw_feedback = to_data(feedback) if feedback is not None else None
    feedback_view = ({key: raw_feedback[key] for key in v.feedback_fields} if raw_feedback is not None else None)
    return c.ModelInput(context=context, task_public=public_task(task, v),
                        observation_view=view, execution_feedback_view=feedback_view)
