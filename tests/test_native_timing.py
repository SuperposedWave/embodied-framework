"""Synthetic tests of approved measured-native timing; no real simulator/model."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from embodied_eval import contracts as c
from embodied_eval.artifacts import ArtifactWriter
from embodied_eval.codec import decode, dumps, loads, to_data
from embodied_eval.compatibility import check_compatibility
from embodied_eval.errors import AdapterError, ContractError
from embodied_eval.runner import run_episode
from embodied_eval.validation import validate_space, validate_task, validate_prediction, validate_step_result
from embodied_eval.visibility import build_model_input
from examples.contract_sample import make_case, prediction, sequence


def native_case():
    space, cap, profile, task, obs = make_case()
    space = replace(space, control_period_s=None, physics_period_s=.004,
                    timing_mode="measured_native", discrete_specs={}, tools={})
    cap = replace(cap, chunk_mode="adapter_owned", supported_timing_modes=("measured_native",),
                  produced_discrete_specs=(), produced_tools=())
    profile = replace(profile, chunk_mode="adapter_owned", timing_mode="measured_native",
                      execution_horizon_steps=1, discrete_spec_refs=(), tool_refs=())
    task = replace(task, budgets=c.Budgets(env_steps=3, model_calls=10))
    return space, cap, profile, task, obs


def native_action(request_id="req"):
    return c.NativeAction(action_spec_ref="robot-control@1", commands=(c.ControllerCommand(
        command_id=request_id + ":cmd", target_id="robot0/base",
        controller_profile_ref="normalized-controller@1", parameters={"delta": [0., .1, 0.]}),))


def reply_for(previous, request, physical_steps=25, *, status="executed", actions=1, terminated=False):
    seconds = physical_steps * .004
    obs = replace(previous, context=request, observation_id=request.message_id + ":obs",
                  env_step=previous.env_step + actions, sim_time_s=previous.sim_time_s + seconds,
                  sensors={key: replace(value, captured_at_s=previous.sim_time_s + seconds)
                           for key, value in previous.sensors.items()},
                  frame_transforms={key: replace(value, captured_at_s=previous.sim_time_s + seconds)
                                    for key, value in previous.frame_transforms.items()})
    feedback = c.ExecutionFeedback(request_id=request.message_id, status=status, consumed_env_steps=actions,
        elapsed_sim_time_s=seconds, consumed_physics_steps=physical_steps, command_ids=(request.message_id + ":cmd",),
        code={"executed": "ok", "planning_failed": "planning_failed", "execution_failed": "execution_failed",
              "rejected": "invalid_argument"}.get(status, "ok"))
    return c.StepResult(context=request, observation=obs, execution_feedback=feedback, terminated=terminated,
                        truncated=False, end_reason="native_success" if terminated else None)


class NativeTimingTests(unittest.TestCase):
    def setUp(self):
        self.space, self.cap, self.profile, self.task, self.obs = native_case()
        self.request = replace(self.obs.context, message_id="req")

    def fails(self, code, function, *args, **kwargs):
        with self.assertRaises(ContractError) as error:
            function(*args, **kwargs)
        self.assertEqual(error.exception.issue.code, code)

    def test_legacy_wire_defaults_remain_fixed_and_physics_unknown(self):
        space, cap, profile, _, _ = make_case()
        for cls, value, removed in ((c.SpaceSpec, space, ("timing_mode", "physics_period_s")),
                                   (c.CapabilitySpec, cap, ("supported_timing_modes",)),
                                   (c.AdaptationProfile, profile, ("timing_mode",))):
            raw = to_data(value)
            for key in removed:
                raw.pop(key)
            decoded = decode(cls, raw)
            self.assertEqual(decoded, value)
        feedback = decode(c.ExecutionFeedback, {"request_id": "req", "status": "executed",
                                               "consumed_env_steps": 1, "elapsed_sim_time_s": .05})
        self.assertIsNone(feedback.consumed_physics_steps)
        stats = decode(c.ExecutionStatistics, {"env_steps": 1, "sim_time_s": .05, "model_calls": 1, "tool_calls": 0})
        self.assertIsNone(stats.physics_steps)

    def test_native_decision_round_trip_and_no_duration_field(self):
        pred = prediction(self.obs, self.profile, native_action(), request_id="req")
        self.assertEqual(loads(c.Prediction, dumps(pred)), pred)
        validate_prediction(pred, self.obs, self.space, self.cap, self.profile)
        raw = to_data(native_action()) | {"duration_s": .05}
        self.fails("unknown_field", decode, c.NativeAction, raw)

    def test_native_command_is_controller_only(self):
        raw = to_data(native_action())
        raw["commands"] = to_data(sequence().steps[0].commands)
        self.fails("unknown_field", decode, c.NativeAction, raw)

    def test_valid_native_compatibility(self):
        validate_space(self.space)
        self.assertTrue(check_compatibility(self.space, self.cap, self.profile).compatible)

    def test_native_space_rejects_constant_control_period_or_missing_physics(self):
        for changes in ({"control_period_s": .05}, {"physics_period_s": None}, {"physics_period_s": 0.}):
            with self.subTest(changes=changes):
                self.fails("invalid_timing", validate_space, replace(self.space, **changes))
        self.fails("invalid_timing", validate_space, replace(make_case()[0], control_period_s=None))

    def test_native_mode_does_not_extend_discrete_or_tool_semantics(self):
        legacy = make_case()[0]
        for changes in ({"discrete_specs": legacy.discrete_specs}, {"tools": legacy.tools}):
            with self.subTest(changes=changes):
                self.fails("unsupported_native_output", validate_space, replace(self.space, **changes))

    def test_three_way_timing_negotiation_and_single_chunk_owner(self):
        for cap, profile, code in (
            (replace(self.cap, supported_timing_modes=("fixed_period",)), self.profile, "timing_mode_mismatch"),
            (self.cap, replace(self.profile, timing_mode="fixed_period"), "timing_mode_mismatch"),
            (self.cap, replace(self.profile, execution_horizon_steps=2), "double_chunk_owner"),
            (replace(self.cap, observation_update="every_env_step"), self.profile, "observation_history_mismatch"),
        ):
            with self.subTest(code=code):
                self.assertIn(code, {issue.code for issue in check_compatibility(self.space, cap, profile).issues})

    def test_native_mode_rejects_any_simulation_time_limit(self):
        for limit in (0., 1., 60.):
            self.fails("unsupported_native_budget", validate_task,
                       replace(self.task, budgets=c.Budgets(sim_time_s=limit)), self.space)

    def test_fixed_decision_and_native_decision_cannot_cross_modes(self):
        self.fails("timing_mode_mismatch", validate_prediction,
            prediction(self.obs, self.profile, sequence()), self.obs, self.space, self.cap, self.profile)
        space, cap, profile, _, obs = make_case()
        self.fails("timing_mode_mismatch", validate_prediction,
            prediction(obs, profile, native_action()), obs, space, cap, profile)

    def test_native_controller_schema_target_and_unique_ids_checked(self):
        action = native_action()
        command = action.commands[0]
        for changed, code in ((replace(command, target_id="other/base"), "unsupported_action"),
                              (replace(command, controller_profile_ref="missing@1"), "unknown_controller"),
                              (replace(command, parameters={"delta": [2., 0., 0.]}), "schema_mismatch")):
            self.fails(code, validate_prediction, prediction(self.obs, self.profile,
                replace(action, commands=(changed,))), self.obs, self.space, self.cap, self.profile)
        self.fails("duplicate_id", validate_prediction, prediction(self.obs, self.profile,
            replace(action, commands=(command, command))), self.obs, self.space, self.cap, self.profile)

    def test_variable_durations_use_measured_physical_steps(self):
        for count in (1, 25, 173):
            validate_step_result(reply_for(self.obs, self.request, count), self.obs, self.space, capability=self.cap)

    def test_missing_negative_or_fabricated_physical_count_is_rejected(self):
        reply = reply_for(self.obs, self.request)
        for count, code in ((None, "invalid_execution_count"), (-1, "invalid_execution_count"),
                            (1, "execution_time_mismatch")):
            self.fails(code, validate_step_result, replace(reply,
                execution_feedback=replace(reply.execution_feedback, consumed_physics_steps=count)), self.obs, self.space)

    def test_planning_failure_may_consume_one_action_and_zero_physics(self):
        validate_step_result(reply_for(self.obs, self.request, 0, status="planning_failed"), self.obs, self.space)

    def test_rejected_action_cannot_hide_consumption_or_multiple_calls(self):
        self.fails("invalid_execution_count", validate_step_result,
                   reply_for(self.obs, self.request, 0, status="rejected"), self.obs, self.space)
        self.fails("invalid_execution_count", validate_step_result,
                   reply_for(self.obs, self.request, 25, actions=2), self.obs, self.space)

    def test_no_fabricated_intermediate_observations(self):
        reply = reply_for(self.obs, self.request)
        self.fails("invalid_observation_history", validate_step_result,
                   replace(reply, intermediate_observations=(self.obs,)), self.obs, self.space)

    def test_feedback_matching_and_private_count_not_implicitly_public(self):
        pred = prediction(self.obs, self.profile, native_action(), request_id="req")
        reply = reply_for(self.obs, self.request)
        validate_step_result(reply, self.obs, self.space, expected_prediction=pred)
        self.fails("command_mismatch", validate_step_result, replace(reply,
            execution_feedback=replace(reply.execution_feedback, command_ids=("other",))),
            self.obs, self.space, expected_prediction=pred)
        view = build_model_input(self.task, reply.observation, self.profile, reply.execution_feedback)
        self.assertNotIn("consumed_physics_steps", view.execution_feedback_view)


class FakeNativeModel:
    def __init__(self, profile):
        self.profile = profile
        self.history = []
        self.calls = 0
        self.ends = []

    def begin_episode(self, context, task, space, profile):
        self.history.clear()
        self.calls = 0

    def observe(self, model_input):
        self.obs = model_input.observation_view
        self.history.append(self.obs.env_step)

    def predict(self, request):
        self.calls += 1
        return replace(prediction(self.obs, self.profile, native_action(request.message_id), request_id=request.message_id),
                       diagnostics=c.PredictionDiagnostics(model_forward_count=1))

    def end_episode(self, context):
        self.ends.append(context.reason)


class FakeNativeBenchmark:
    def __init__(self, obs, durations=(25, 100, 7), *, success_at=None, fail_at=None, lose_at=None):
        self.initial = obs
        self.durations = durations
        self.success_at, self.fail_at, self.lose_at = success_at, fail_at, lose_at
        self.calls = 0
        self.scored = 0

    def reset(self, task, context):
        self.calls = 0
        self.obs = replace(self.initial, context=context)
        return self.obs

    def advance(self, action, request, remaining):
        assert isinstance(action, c.NativeAction)
        assert remaining.sim_time_s is None
        count = self.durations[self.calls]
        self.calls += 1
        if self.calls == self.lose_at:
            raise AdapterError("execution", "lost_native_reply", "synthetic unknown count",
                               execution_started=True, state_known=False)
        reply = reply_for(self.obs, request, count,
                          status="execution_failed" if self.calls == self.fail_at else "executed",
                          terminated=self.calls == self.success_at)
        self.obs = reply.observation
        return reply

    def score(self, context):
        self.scored += 1
        return {"success": c.Metric(value=self.calls == self.success_at, unit="boolean", source="synthetic", valid=True)}


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.space, self.cap, self.profile, self.task, self.obs = native_case()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = 0

    def execute(self, benchmark, task=None, model=None):
        self.index += 1
        model = model or FakeNativeModel(self.profile)
        writer = ArtifactWriter(Path(self.temp.name)/f"run-{self.index}")
        outcome = run_episode(model, benchmark, self.profile, task or self.task, self.space, self.cap,
            writer, "synthetic-run", {"success": c.Metric(value=False, unit="boolean", source="synthetic", valid=True)})
        return outcome, model

    def test_three_actions_different_lengths_count_real_simulation_time(self):
        outcome, model = self.execute(FakeNativeBenchmark(self.obs))
        stats = outcome.result.execution_statistics
        self.assertEqual((stats.env_steps, stats.physics_steps, stats.model_calls), (3, 132, 3))
        self.assertAlmostEqual(stats.sim_time_s, .528)
        self.assertTrue(outcome.statistics_complete)
        self.assertEqual(model.history, [0, 1, 2])

    def test_early_native_success_stops_before_next_cached_row(self):
        benchmark = FakeNativeBenchmark(self.obs, success_at=1)
        outcome, model = self.execute(benchmark)
        self.assertEqual(outcome.result.termination_reason, "environment_terminated")
        self.assertEqual((benchmark.calls, model.calls, outcome.result.execution_statistics.physics_steps), (1, 1, 25))

    def test_known_partial_failure_keeps_actual_progress_and_score(self):
        benchmark = FakeNativeBenchmark(self.obs, fail_at=2)
        outcome, _ = self.execute(benchmark)
        self.assertEqual(outcome.result.run_status, "error")
        self.assertTrue(outcome.statistics_complete)
        self.assertEqual((outcome.result.execution_statistics.env_steps,
                          outcome.result.execution_statistics.physics_steps), (2, 125))
        self.assertEqual(benchmark.scored, 1)

    def test_unknown_partial_failure_preserves_only_last_confirmed_count(self):
        benchmark = FakeNativeBenchmark(self.obs, lose_at=2)
        outcome, _ = self.execute(benchmark)
        self.assertFalse(outcome.statistics_complete)
        self.assertEqual(outcome.result.termination_reason, "execution_state_unknown")
        self.assertEqual((outcome.result.execution_statistics.env_steps,
                          outcome.result.execution_statistics.physics_steps), (1, 25))
        self.assertIsNone(outcome.result.metrics["success"].value)
        self.assertEqual(benchmark.scored, 0)

    def test_model_call_budget_and_episode_isolation(self):
        benchmark = FakeNativeBenchmark(self.obs)
        task = replace(self.task, budgets=c.Budgets(env_steps=3, model_calls=1))
        outcome, model = self.execute(benchmark, task)
        self.assertEqual(outcome.result.execution_statistics.physics_steps, 25)
        second, _ = self.execute(benchmark, task, model)
        self.assertEqual(second.result.execution_statistics.physics_steps, 25)
        self.assertEqual(model.history, [0])

    def test_native_budget_rejection_happens_before_reset(self):
        benchmark = FakeNativeBenchmark(self.obs)
        outcome, model = self.execute(benchmark, replace(self.task, budgets=c.Budgets(sim_time_s=1)))
        self.assertEqual(outcome.errors[0]["code"], "unsupported_native_budget")
        self.assertEqual((model.calls, benchmark.calls, benchmark.scored), (0, 0, 0))
