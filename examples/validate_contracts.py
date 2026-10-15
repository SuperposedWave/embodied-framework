"""Generate and check actual wire examples. This is a contract check, not a runner."""
from dataclasses import replace
import json
from pathlib import Path
from embodied_eval import contracts as c
from embodied_eval.codec import dumps, loads, to_data
from embodied_eval.compatibility import check_compatibility, validate_model_input
from embodied_eval.validation import (
    validate_space, validate_task, validate_observation, validate_prediction,
    validate_step_result, validate_episode_result,
)
from embodied_eval.visibility import public_space, build_model_input
from .contract_sample import make_case, prediction, sequence


def build_examples():
    space, cap, profile, task, obs = make_case()
    check_compatibility(space, cap, profile).require_compatible()
    validate_space(space)
    validate_task(task, space)
    validate_observation(obs, space)
    view = build_model_input(task, obs, profile)
    validate_model_input(view, space, cap, profile)
    model_space = public_space(space, profile)
    validate_space(model_space)
    agent = replace(profile, profile_id="synthetic-agent@1", role="agent", tool_refs=("move_eef@1",))
    world = replace(profile, profile_id="synthetic-world@1", role="world_model",
                    action_spec_refs=(), discrete_spec_refs=(), tool_refs=())
    pose = lambda x: c.Pose(position_m=(x, 0., .3), quaternion_xyzw=(0., 0., 0., 1.), frame_id="base")
    cases = [
        ("single_step", profile, prediction(obs, profile, sequence())),
        ("chunk_8_execute_2", profile, prediction(obs, profile, sequence(8, 2))),
        ("dual_arm", profile, prediction(obs, profile, sequence(commands=lambda i: (
            c.PoseTarget(command_id="left-pose", target_id="robot0/left_eef", pose=pose(-.2)),
            c.PoseTarget(command_id="right-pose", target_id="robot0/right_eef", pose=pose(.2)))))),
        ("controller_native", profile, prediction(obs, profile, sequence(commands=lambda i: (
            c.ControllerCommand(command_id="controller-0", target_id="robot0/left_eef",
                controller_profile_ref="normalized-controller@1", parameters={"delta": [0., .1, 0.]}),)))),
        ("discrete", profile, prediction(obs, profile, c.DiscreteAction(discrete_spec_ref="navigation@1",
            target_id="robot0/base", action="forward", duration_s=.05))),
        ("tool", agent, prediction(obs, agent, c.ToolCall(call_id="call-0", tool_ref="move_eef@1",
            target_id="robot0/left_eef", arguments={"position_m": [.1, 0., .3], "frame_id": "base", "max_ticks": 3}))),
        ("model_stop", profile, prediction(obs, profile, c.StopRequest(reason="No further action."))),
        ("world_prediction", world, prediction(obs, world, None, predictions=(
            c.FuturePrediction(modality="future_state", relative_time_s=.1,
                payload=c.Tensor(dtype="float32", shape=(2,), layout="F", data=[.1, .2])),))),
    ]
    for _, prof, pred in cases:
        check_compatibility(space, cap, prof).require_compatible()
        validate_prediction(loads(c.Prediction, dumps(pred)), obs, space, cap, prof)
    next_obs = replace(obs, context=replace(obs.context, message_id="obs-3"), observation_id="obs-3", env_step=3, sim_time_s=.15,
        sensors={key: replace(sample, captured_at_s=.15) for key, sample in obs.sensors.items()},
        frame_transforms={key: replace(transform, captured_at_s=.15) for key, transform in obs.frame_transforms.items()})
    feedback = c.ExecutionFeedback(request_id="request-0", call_id="call-0", status="executed",
        consumed_env_steps=3, elapsed_sim_time_s=.15)
    step = c.StepResult(context=next_obs.context, observation=next_obs, execution_feedback=feedback,
        terminated=False, truncated=False, evaluation_private={"success": False},
        intermediate_observations=tuple(replace(next_obs, context=replace(obs.context, message_id=f"obs-{tick}"),
            observation_id=f"obs-{tick}", env_step=tick, sim_time_s=tick * .05,
            sensors={key: replace(sample, captured_at_s=tick * .05) for key, sample in obs.sensors.items()},
            frame_transforms={key: replace(transform, captured_at_s=tick * .05) for key, transform in obs.frame_transforms.items()})
            for tick in (1, 2)))
    validate_step_result(step, obs, space)
    result = c.EpisodeResult(episode_id=task.episode_id, run_manifest_ref="synthetic-manifest@1",
        run_status="completed", termination_reason="model_stop",
        metrics={"success": c.Metric(value=False, unit="boolean", source="synthetic-scorer@1", valid=True)},
        execution_statistics=c.ExecutionStatistics(env_steps=3, sim_time_s=.15, model_calls=2, tool_calls=1))
    validate_episode_result(result)
    return {
        "fixture_only": True, "space": to_data(space), "capability": to_data(cap),
        "profile": to_data(profile), "task": to_data(task), "observation": to_data(obs),
        "model_space": to_data(model_space), "model_input": to_data(view),
        "prediction_cases": [{"name": name, "profile": to_data(prof), "prediction": to_data(pred)} for name, prof, pred in cases],
        "tool_step_result": to_data(step), "stopped_episode_result": to_data(result),
    }


if __name__ == "__main__":
    examples = build_examples()
    target = Path(__file__).with_name("contract_examples.json")
    target.write_text(json.dumps(examples, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(f"Validated {len(examples['prediction_cases'])} prediction cases, public view and result contracts.")
    print(target)
