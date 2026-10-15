"""Shared synthetic fixtures and contract example helpers."""
from dataclasses import replace
from embodied_eval import contracts as c
from embodied_eval.adapters.mock_spec import make_case, IDENTITY


def prediction(obs, profile, decision, *, predictions=(), request_id="request-0"):
    return c.Prediction(context=replace(obs.context, message_id=request_id), request_id=request_id,
        based_on_observation_id=obs.observation_id, profile_ref=profile.profile_id,
        decision=decision, predictions=predictions)


def sequence(length=1, horizon=1, *, commands=None):
    return c.ActionSequence(action_spec_ref="robot-control@1", execution_horizon_steps=horizon,
        steps=tuple(c.ActionStep(index=i, duration_s=.05, commands=commands(i) if commands else (
            c.JointTarget(command_id=f"left-{i}", target_id="robot0/left_j0", mode="position", value=.1*i, unit="rad"),))
        for i in range(length)))
