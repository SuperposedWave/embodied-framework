"""Deterministic policy/agent exercising the public model interface only."""
from collections import deque
from copy import deepcopy
from dataclasses import replace

import tests.legacy_support.frozen._legacy_contracts as c
from embodied_eval.errors import AdapterError
from tests.legacy_support.frozen.adapters.mock_spec import make_case, IDENTITY


class MockModel:
    def __init__(self, config, tool_ticks=3):
        self.config, self.tool_ticks = config, tool_ticks
        _, cap, _, _, _ = make_case()
        required = cap.required_inputs
        if config.scenario == "dual_arm":
            required += tuple(c.InputRequirement(key=sid, kind="sensor", modality="rgb")
                              for sid in ("left_wrist_rgb", "right_wrist_rgb"))
        self.cap = replace(cap, model_id="mock-model", source_revision="mock-model-1",
            roles=("agent",) if config.scenario == "tool" else ("action_policy",), produced_predictions=(),
            checkpoint_ref="builtin-mock@1", normalization_ref="identity@1", required_inputs=required,
            observation_update=config.observation_update,
            chunk_mode="adapter_owned" if config.scenario == "adapter_cache" else "runner_owned")
        self.events, self.closed = [], False
        self.context = None
        self._cache, self._history = deque(), []
        self.observe_count = self.predict_count = self.forward_count = 0

    def describe(self):
        return deepcopy(self.cap)

    def _fault(self, name):
        return self.config.fault == name and (self.config.fault_episode_id is None
            or self.context.episode_id == self.config.fault_episode_id)

    def begin_episode(self, context, task, space, profile):
        self.context, self.task, self.space, self.profile = context, deepcopy(task), deepcopy(space), deepcopy(profile)
        self._cache.clear()
        self._history.clear()
        self.latest = None
        self.observe_count = self.predict_count = self.forward_count = 0
        self.events = [{"event": "begin", "episode_id": context.episode_id, "cache_at_begin": 0, "history_at_begin": 0}]
        if self._fault("begin_error"):
            raise AdapterError("model_service", "begin_failed", "synthetic initialization error")

    def observe(self, model_input):
        self.latest = deepcopy(model_input)
        self._history.append(model_input.observation_view.observation_id)
        self.observe_count += 1
        self.events.append({"event": "observe", "observation_id": model_input.observation_view.observation_id,
                            "env_step": model_input.observation_view.env_step})

    def _commands(self, index, request_id):
        obs = self.latest.observation_view
        scenario = self.config.scenario
        command_id = f"{request_id}:command:{index}"
        if scenario == "dual_arm":
            return tuple(c.PoseTarget(command_id=command_id + ":" + side, target_id=f"robot0/{side}_eef",
                pose=c.Pose(position_m=(x, .1, .3), quaternion_xyzw=(0., 0., 0., 1.), frame_id="base"))
                for side, x in (("left", -.3), ("right", .3)))
        if scenario == "pose_delta":
            return (c.PoseDelta(command_id=command_id, target_id="robot0/left_eef",
                delta=replace(IDENTITY, translation_m=(.1, 0., 0.)), frame_id="base",
                reference_policy=self.config.delta_reference, composition=self.config.delta_composition,
                reference_observation_id=obs.observation_id if self.config.delta_reference == "chunk_start" else None),)
        # Tail is deliberately distinct so tests can detect reuse of discarded predictions.
        value = obs.robots["robot0"].joints["left_j0"].position + .01*(index+1)
        return (c.JointTarget(command_id=command_id, target_id="robot0/left_j0", mode="position", value=value,
                              unit="deg" if self._fault("invalid_unit") else "rad"),)

    def predict(self, request_context):
        self.predict_count += 1
        self.events.append({"event": "predict", "request_id": request_context.message_id})
        if self._fault("service_error"):
            raise AdapterError("model_service", "service_unavailable", "synthetic service failure")
        scenario, obs = self.config.scenario, self.latest.observation_view
        forwards = 1
        if scenario == "stop":
            decision = c.StopRequest(reason="Synthetic stop.")
        elif scenario == "tool":
            feedback = self.latest.execution_feedback_view or {}
            if feedback.get("status") == "planning_failed":
                decision = c.StopRequest(reason="Synthetic planner rejected the request.")
            else:
                decision = c.ToolCall(call_id=request_context.message_id + ":tool", tool_ref="move_eef@1",
                    target_id="robot0/left_eef",
                    arguments={"position_m": [.1, .1, .3], "frame_id": "base", "max_ticks": self.tool_ticks})
        elif scenario == "discrete":
            decision = c.DiscreteAction(discrete_spec_ref="navigation@1", target_id="robot0/base",
                                        action="forward", duration_s=self.space.control_period_s)
        else:
            if scenario == "adapter_cache":
                forwards = 0
                if not self._cache:
                    self._cache.extend(self._commands(i, request_context.message_id) for i in range(self.config.chunk_length))
                    forwards = 1
                commands = self._cache.popleft()
                # Each exported command has the current request identity.
                commands = tuple(replace(cmd, command_id=request_context.message_id + ":" + str(i)) for i, cmd in enumerate(commands))
                steps = (c.ActionStep(index=0, duration_s=self.space.control_period_s, commands=commands),)
            else:
                length = self.config.chunk_length if scenario in ("chunk", "pose_delta") else 1
                steps = tuple(c.ActionStep(index=i, duration_s=self.space.control_period_s,
                    commands=self._commands(i, request_context.message_id)) for i in range(length))
            decision = c.ActionSequence(action_spec_ref="robot-control@1",
                execution_horizon_steps=min(len(steps), self.profile.execution_horizon_steps), steps=steps)
        self.forward_count += forwards
        pred = c.Prediction(context=request_context, request_id=request_context.message_id,
            based_on_observation_id=obs.observation_id, profile_ref=self.profile.profile_id, decision=decision,
            diagnostics=c.PredictionDiagnostics(model_forward_count=forwards if self.config.report_forward_count else None))
        if self._fault("stale_observation"):
            pred = replace(pred, based_on_observation_id="previous-observation")
        if self._fault("cross_episode"):
            pred = replace(pred, context=replace(pred.context, episode_id="previous-episode"))
        if self._fault("wrong_request"):
            pred = replace(pred, request_id="previous-request", context=replace(pred.context, message_id="previous-request"))
        return pred

    def end_episode(self, context):
        self.events.append({"event": "end", "context": {"episode_id": context.context.episode_id, "reason": context.reason}})
        self._history.clear()
        self._cache.clear()
        self.latest = None
        if self._fault("end_error"):
            raise AdapterError("model_service", "end_failed", "synthetic end error")

    def close(self):
        self.closed = True
        self.events.append({"event": "close"})
        if self.context is not None and self._fault("close_error"):
            raise AdapterError("infrastructure", "model_close_failed", "synthetic close error")

    def diagnostics(self):
        return {"observe_count": self.observe_count, "predict_count": self.predict_count,
                "forward_count": self.forward_count, "cache_after_end": len(self._cache),
                "history_after_end": list(self._history), "closed": self.closed, "events": deepcopy(self.events)}
