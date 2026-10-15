from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from embodied_eval import contracts as c
from embodied_eval.codec import decode, to_data
from embodied_eval.config import RunConfig, ModelConfig, BenchmarkConfig, ProfileConfig, EpisodeConfig
from embodied_eval.errors import ContractError, AdapterError
from embodied_eval.runner import run


class RunnerTests(unittest.TestCase):
    def test_late_prediction_is_not_executed_after_cooperative_wall_budget(self):
        from embodied_eval.adapters.mock_model import MockModel
        clock = [0.]
        original = MockModel.predict
        def late_predict(model, context):
            result = original(model, context)
            clock[0] = .02
            return result
        config = replace(self.config, model=ModelConfig(scenario="stop"),
            episodes=(replace(self.config.episodes[0], budgets=c.Budgets(wall_time_s=.01)),))
        with patch("embodied_eval.runner.time.monotonic", side_effect=lambda: clock[0]), patch.object(MockModel, "predict", late_predict):
            outcome = self.execute(config)
        self.assertEqual(outcome.results[0].termination_reason, "budget_exhausted")
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 0)
        self.assertEqual(outcome.results[0].execution_statistics.model_calls, 1)

    def test_unknown_execution_state_is_explicit_and_not_scored_as_failure(self):
        from embodied_eval.adapters.mock_benchmark import MockBenchmark
        def lost_reply(benchmark, call, request, remaining):
            benchmark._execute(1, request, remaining, tool=call)
            raise AdapterError("infrastructure", "connection_lost", "synthetic lost response",
                               execution_started=True, state_known=False)
        with patch.object(MockBenchmark, "execute_tool", lost_reply):
            outcome = self.execute(self.tool_config())
        row = outcome.summary["episodes"][0]
        self.assertFalse(row["execution_statistics_complete"])
        self.assertEqual(row["execution_statistics_scope"], "last_confirmed")
        self.assertEqual(outcome.results[0].termination_reason, "execution_state_unknown")
        self.assertIsNone(outcome.results[0].metrics["success"].value)
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 0)
        self.assertEqual(self.diagnostic(outcome)["benchmark"]["env_steps"], 1)

    def test_reused_observation_identity_is_rejected_before_history_update(self):
        self.check_reused_observation_identity()

    def test_cleanup_error_does_not_hide_unknown_execution_state(self):
        from embodied_eval.adapters.mock_benchmark import MockBenchmark
        def lost_reply(benchmark, call, request, remaining):
            benchmark._execute(1, request, remaining, tool=call)
            raise AdapterError("infrastructure", "connection_lost", "synthetic lost response",
                               execution_started=True, state_known=False)
        config = replace(self.tool_config(), model=ModelConfig(scenario="tool", fault="end_error"))
        with patch.object(MockBenchmark, "execute_tool", lost_reply):
            outcome = self.execute(config)
        self.assertEqual(outcome.results[0].termination_reason, "execution_state_unknown")
        self.assertFalse(outcome.summary["episodes"][0]["execution_statistics_complete"])
        self.assertEqual([error["code"] for error in outcome.summary["episodes"][0]["errors"]], ["connection_lost", "end_failed"])

    def check_reused_observation_identity(self):
        from embodied_eval.adapters.mock_benchmark import MockBenchmark
        original = MockBenchmark._snapshot
        def repeated_id(benchmark):
            observation = original(benchmark)
            if benchmark.steps > 0:
                observation = replace(observation, observation_id="ep-0:obs:1")
            return observation
        with patch.object(MockBenchmark, "_snapshot", repeated_id):
            outcome = self.execute()
        self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["code"], "observation_id_reused")
        self.assertEqual(self.diagnostic(outcome)["model"]["observe_count"], 1)
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 1)

    def test_adapter_budget_overrun_retains_known_actual_ticks(self):
        from embodied_eval.adapters.mock_benchmark import MockBenchmark
        original = MockBenchmark.execute_tool
        def ignore_budget(benchmark, call, request, remaining):
            return original(benchmark, call, request, replace(remaining, env_steps=99, sim_time_s=99.))
        with patch.object(MockBenchmark, "execute_tool", ignore_budget):
            outcome = self.execute(self.tool_config())
        self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["code"], "budget_overrun")
        self.assertTrue(outcome.summary["episodes"][0]["execution_statistics_complete"])
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 6)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.counter = 0
        self.config = RunConfig(run_id="unit-run",
            model=ModelConfig(scenario="single"),
            episodes=(EpisodeConfig(episode_id="ep-0", budgets=c.Budgets(env_steps=3, model_calls=10, tool_calls=0)),))

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, config=None):
        self.counter += 1
        return run(config or self.config, self.root/f"run-{self.counter}")

    def read(self, outcome, name, episode="ep-0"):
        return json.loads((outcome.output/"episodes"/episode/name).read_text())

    def trace(self, outcome, episode="ep-0"):
        return [json.loads(line) for line in (outcome.output/"episodes"/episode/"trace.internal.jsonl").read_text().splitlines()]

    def diagnostic(self, outcome, episode="ep-0"):
        return self.read(outcome, "diagnostics.internal.json", episode)

    def test_single_step_lifecycle_and_three_control_ticks(self):
        outcome = self.execute()
        result = outcome.results[0]
        stats = result.execution_statistics
        self.assertEqual((stats.env_steps, stats.model_calls, stats.model_forward_count), (3, 3, 3))
        self.assertAlmostEqual(stats.sim_time_s, .15)
        diag = self.diagnostic(outcome)
        self.assertEqual(diag["model"]["observe_count"], 4)
        self.assertTrue(diag["model"]["closed"])
        self.assertTrue(diag["benchmark"]["closed"])
        self.assertEqual(diag["model"]["events"][0]["event"], "begin")
        self.assertEqual([event["event"] for event in diag["model"]["events"]][-2:], ["end", "close"])
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertIs(result.metrics["success"].value, False)
        self.assertEqual(outcome.exit_code, 0)

    def test_runner_chunk_discards_tail_and_repredicts(self):
        config = replace(self.config, model=ModelConfig(scenario="chunk", chunk_length=8),
            episodes=(EpisodeConfig(episode_id="ep-0", budgets=c.Budgets(env_steps=6, sim_time_s=.3, model_calls=6, tool_calls=0)),))
        outcome = self.execute(config)
        stats = outcome.results[0].execution_statistics
        self.assertEqual((stats.env_steps, stats.model_calls, stats.model_forward_count), (6, 3, 3))
        diag = self.diagnostic(outcome)
        self.assertEqual(diag["model"]["observe_count"], 7)
        self.assertAlmostEqual(diag["benchmark"]["robot"]["joints"]["left_j0"]["position"], .06)
        predictions = [row["data"] for row in self.trace(outcome) if row["event"] == "prediction"]
        self.assertEqual([len(pred["decision"]["steps"]) for pred in predictions], [8, 8, 8])
        self.assertEqual([row["env_step"] for row in diag["model"]["events"] if row["event"] == "observe"], list(range(7)))

    def test_internal_cache_and_histories_reset_between_episodes(self):
        config = replace(self.config, model=ModelConfig(scenario="adapter_cache"),
            profile=ProfileConfig(chunk_mode="adapter_owned", execution_horizon_steps=1),
            episodes=self.config.episodes + (replace(self.config.episodes[0], episode_id="ep-1"),))
        outcome = self.execute(config)
        for index, result in enumerate(outcome.results):
            self.assertEqual((result.execution_statistics.model_calls, result.execution_statistics.model_forward_count), (3, 1))
            diag = self.diagnostic(outcome, f"ep-{index}")
            self.assertEqual(diag["model"]["cache_after_end"], 0)
            self.assertEqual(diag["model"]["history_after_end"], [])
            self.assertEqual(diag["model"]["events"][0]["cache_at_begin"], 0)
            self.assertEqual(diag["benchmark"]["reset_count"], index+1)  # same pair reused on clean episodes
            self.assertAlmostEqual(diag["benchmark"]["robot"]["joints"]["left_j0"]["position"], .03)

    def test_decision_boundary_observes_only_before_predict(self):
        config = replace(self.config, model=ModelConfig(scenario="chunk", observation_update="decision_boundary"),
            episodes=(replace(self.config.episodes[0], budgets=c.Budgets(env_steps=6, model_calls=6)),))
        outcome = self.execute(config)
        diag = self.diagnostic(outcome)
        self.assertEqual(diag["model"]["observe_count"], 3)
        self.assertEqual([e["env_step"] for e in diag["model"]["events"] if e["event"] == "observe"], [0, 2, 4])

    def test_model_call_budget_does_not_cancel_the_already_requested_chunk(self):
        config = replace(self.config, model=ModelConfig(scenario="chunk"),
            episodes=(replace(self.config.episodes[0], budgets=c.Budgets(env_steps=6, model_calls=1)),))
        outcome = self.execute(config)
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 2)
        self.assertEqual(outcome.results[0].execution_statistics.model_calls, 1)

    def test_zero_and_subtick_simulation_budgets_do_not_predict(self):
        for budgets in (c.Budgets(env_steps=0), c.Budgets(model_calls=0), c.Budgets(sim_time_s=.049), c.Budgets(wall_time_s=0.)):
            with self.subTest(budgets=budgets):
                outcome = self.execute(replace(self.config, episodes=(replace(self.config.episodes[0], budgets=budgets),)))
                self.assertEqual(outcome.results[0].execution_statistics.model_calls, 0)
                self.assertEqual(outcome.results[0].execution_statistics.env_steps, 0)
                self.assertEqual(self.diagnostic(outcome)["model"]["observe_count"], 1)

    def test_dual_arm_commands_and_three_cameras_are_separate(self):
        config = replace(self.config, model=ModelConfig(scenario="dual_arm"),
                         episodes=(replace(self.config.episodes[0], budgets=c.Budgets(env_steps=1)),))
        outcome = self.execute(config)
        diag = self.diagnostic(outcome)
        effectors = diag["benchmark"]["robot"]["end_effectors"]
        self.assertEqual(effectors["left_eef"]["pose"]["position_m"], [-.3, .1, .3])
        self.assertEqual(effectors["right_eef"]["pose"]["position_m"], [.3, .1, .3])
        self.assertEqual(len(diag["benchmark"]["controls"][0]["command_ids"]), 2)
        public = json.loads((outcome.output/"episodes/ep-0/model_input.public.jsonl").read_text().splitlines()[0])
        self.assertEqual(set(public["observation_view"]["sensors"]), {"head_rgb", "left_wrist_rgb", "right_wrist_rgb"})
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 1)

    def test_pose_delta_reference_and_nonzero_rotation(self):
        for reference in ("chunk_start", "step_observed", "previous_command"):
            for composition in ("pre_multiply", "post_multiply"):
                with self.subTest(reference=reference, composition=composition):
                    config = replace(self.config, model=ModelConfig(scenario="pose_delta", chunk_length=2,
                        delta_reference=reference, delta_composition=composition),
                        episodes=(replace(self.config.episodes[0], budgets=c.Budgets(env_steps=2)),))
                    outcome = self.execute(config)
                    pose = self.diagnostic(outcome)["benchmark"]["robot"]["end_effectors"]["left_eef"]["pose"]
                    displacement = .1 if reference == "chunk_start" else .2
                    expected = (-.2 + displacement, 0., .3) if composition == "pre_multiply" else (-.2, displacement, .3)
                    for actual, desired in zip(pose["position_m"], expected):
                        self.assertAlmostEqual(actual, desired)

    def test_discrete_navigation_uses_base_coordinates(self):
        config = replace(self.config, model=ModelConfig(scenario="discrete"))
        outcome = self.execute(config)
        self.assertAlmostEqual(self.diagnostic(outcome)["benchmark"]["robot"]["base_pose"]["position_m"][0], .3)
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 3)

    def tool_config(self, **benchmark_options):
        return replace(self.config, model=ModelConfig(scenario="tool"),
            benchmark=BenchmarkConfig(**benchmark_options),
            episodes=(replace(self.config.episodes[0], budgets=c.Budgets(env_steps=5, sim_time_s=.25, model_calls=5, tool_calls=3)),))

    def test_long_tool_consumes_three_then_two_ticks(self):
        outcome = self.execute(self.tool_config())
        self.assertEqual(outcome.exit_code, 0)
        result = outcome.results[0]
        stats = result.execution_statistics
        self.assertEqual((stats.env_steps, stats.model_calls, stats.tool_calls), (5, 2, 2))
        self.assertAlmostEqual(stats.sim_time_s, .25)
        steps = [row["data"] for row in self.trace(outcome) if row["event"] == "step_result"]
        self.assertEqual([row["execution_feedback"]["consumed_env_steps"] for row in steps], [3, 2])
        self.assertEqual(steps[-1]["execution_feedback"]["status"], "partially_executed")
        self.assertEqual(steps[-1]["execution_feedback"]["code"], "budget_exhausted")
        self.assertEqual(self.diagnostic(outcome)["model"]["observe_count"], 6)
        self.assertEqual([len(step["intermediate_observations"]) for step in steps], [2, 1])

    def test_tool_call_budget_and_missing_history(self):
        config = self.tool_config()
        config = replace(config, episodes=(replace(config.episodes[0], budgets=c.Budgets(env_steps=5, tool_calls=1)),))
        outcome = self.execute(config)
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 3)
        self.assertEqual(outcome.results[0].execution_statistics.model_calls, 1)
        outcome = self.execute(self.tool_config(fault="missing_history"))
        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 3)
        self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["code"], "missing_step_observations")
        self.assertEqual(self.diagnostic(outcome)["model"]["predict_count"], 1)

    def test_tool_without_required_history_rejected_before_reset(self):
        config = self.tool_config(intermediate_observations_supported=False)
        with patch("embodied_eval.adapters.mock_benchmark.MockBenchmark.reset") as reset:
            with self.assertRaises(ContractError) as error:
                self.execute(config)
            reset.assert_not_called()
        self.assertEqual(error.exception.issue.code, "observation_history_mismatch")

    def test_planning_rejection_returns_feedback_and_no_environment_ticks(self):
        outcome = self.execute(self.tool_config(fault="planning_failed"))
        self.assertEqual(outcome.exit_code, 0)
        stats = outcome.results[0].execution_statistics
        self.assertEqual((stats.env_steps, stats.tool_calls, stats.model_calls), (0, 1, 2))
        self.assertEqual(outcome.results[0].termination_reason, "model_stop")
        self.assertIs(outcome.results[0].metrics["success"].value, False)

    def test_model_stop_is_completed_without_implying_success(self):
        outcome = self.execute(replace(self.config, model=ModelConfig(scenario="stop")))
        self.assertEqual(outcome.results[0].termination_reason, "model_stop")
        self.assertEqual(outcome.results[0].run_status, "completed")
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 0)
        self.assertIs(outcome.results[0].metrics["success"].value, False)

    def test_native_mock_success_cancels_chunk_tail(self):
        config = replace(self.config, model=ModelConfig(scenario="chunk"), benchmark=BenchmarkConfig(success_at_step=1))
        outcome = self.execute(config)
        self.assertEqual(outcome.results[0].termination_reason, "environment_terminated")
        self.assertEqual(outcome.results[0].execution_statistics.env_steps, 1)
        self.assertIs(outcome.results[0].metrics["success"].value, True)

    def test_private_information_never_reaches_model_hooks_or_inputs(self):
        outcome = self.execute()
        public = (outcome.output/"episodes/ep-0/model_input.public.jsonl").read_text()
        hook_data = [row["data"] for row in self.trace(outcome) if row["event"] in ("begin_public", "end_public")]
        for secret in ("secret_goal", "987654321", "private-init", "private-evaluation", "bench.private", "private_goal"):
            self.assertNotIn(secret, public)
            self.assertNotIn(secret, json.dumps(hook_data))
        self.assertIn("987654321", (outcome.output/"episodes/ep-0/trace.internal.jsonl").read_text())
        end = next(row["data"] for row in self.trace(outcome) if row["event"] == "end_public")
        self.assertEqual(set(end), {"context", "reason"})

    def test_invalid_model_responses_execute_no_commands(self):
        for fault, code in (("stale_observation", "stale_prediction"), ("cross_episode", "episode_mismatch"),
                            ("wrong_request", "request_mismatch"), ("invalid_unit", "unit_mismatch")):
            with self.subTest(fault=fault):
                outcome = self.execute(replace(self.config, model=ModelConfig(scenario="single", fault=fault)))
                self.assertEqual(outcome.exit_code, 1)
                self.assertEqual(outcome.results[0].execution_statistics.env_steps, 0)
                self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["category"], "model_output")
                self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["code"], code)
                self.assertTrue(self.diagnostic(outcome)["benchmark"]["closed"])

    def test_partial_execution_error_preserves_consumption(self):
        config = self.tool_config(fault="partial_execution_error", fault_after_step=1)
        outcome = self.execute(config)
        stats = outcome.results[0].execution_statistics
        self.assertEqual((stats.env_steps, stats.model_calls, stats.tool_calls), (1, 1, 1))
        self.assertAlmostEqual(stats.sim_time_s, .05)
        self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["category"], "execution")
        self.assertTrue(outcome.summary["episodes"][0]["execution_statistics_complete"])
        self.assertEqual(outcome.exit_code, 1)

    def test_service_error_has_unknown_forward_count(self):
        outcome = self.execute(replace(self.config, model=ModelConfig(scenario="single", fault="service_error")))
        self.assertEqual(outcome.results[0].execution_statistics.model_calls, 1)
        self.assertIsNone(outcome.results[0].execution_statistics.model_forward_count)
        self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["category"], "model_service")

    def test_forward_count_unknown_is_not_guessed_from_predict_count(self):
        outcome = self.execute(replace(self.config, model=ModelConfig(scenario="single", report_forward_count=False)))
        self.assertEqual(outcome.results[0].execution_statistics.model_calls, 3)
        self.assertIsNone(outcome.results[0].execution_statistics.model_forward_count)

    def test_scoring_failure_uses_null_not_task_failure(self):
        outcome = self.execute(replace(self.config, benchmark=BenchmarkConfig(fault="score_error")))
        metric = outcome.results[0].metrics["success"]
        self.assertIsNone(metric.value)
        self.assertFalse(metric.valid)
        self.assertEqual(metric.unavailable_reason, "scorer_failed")
        self.assertEqual(outcome.results[0].run_status, "error")
        self.assertTrue(self.diagnostic(outcome)["model"]["closed"])

    def test_begin_end_reset_and_close_failures_still_release_both_adapters(self):
        cases = [
            replace(self.config, model=ModelConfig(scenario="single", fault="begin_error")),
            replace(self.config, model=ModelConfig(scenario="single", fault="end_error")),
            replace(self.config, model=ModelConfig(scenario="single", fault="close_error")),
            replace(self.config, benchmark=BenchmarkConfig(fault="reset_error")),
            replace(self.config, benchmark=BenchmarkConfig(fault="close_error")),
        ]
        for config in cases:
            with self.subTest(model=config.model.fault, benchmark=config.benchmark.fault):
                outcome = self.execute(config)
                self.assertEqual(outcome.exit_code, 1)
                diag = self.diagnostic(outcome)
                self.assertTrue(diag["model"]["closed"])
                self.assertTrue(diag["benchmark"]["closed"])

    def test_error_rebuilds_pair_before_next_episode(self):
        config = replace(self.config, model=ModelConfig(scenario="single", fault="service_error", fault_episode_id="ep-0"),
            episodes=self.config.episodes + (replace(self.config.episodes[0], episode_id="ep-1"),))
        outcome = self.execute(config)
        self.assertEqual([result.run_status for result in outcome.results], ["error", "completed"])
        self.assertTrue(self.diagnostic(outcome, "ep-0")["model"]["closed"])
        self.assertEqual(self.diagnostic(outcome, "ep-1")["benchmark"]["reset_count"], 1)
        self.assertEqual(outcome.results[1].execution_statistics.env_steps, 3)
        self.assertAlmostEqual(self.diagnostic(outcome, "ep-1")["benchmark"]["robot"]["joints"]["left_j0"]["position"], .03)

    def test_output_directory_is_never_overwritten(self):
        output = self.root/"existing"
        output.mkdir()
        marker = output/"marker.txt"
        marker.write_text("preserve")
        with self.assertRaises(ContractError) as error:
            run(self.config, output)
        self.assertEqual(error.exception.issue.code, "output_exists")
        self.assertEqual(marker.read_text(), "preserve")

    def test_manifest_records_synthetic_sources_dependencies_and_configuration(self):
        outcome = self.execute()
        manifest = json.loads((outcome.output/"manifest.json").read_text())
        self.assertTrue(manifest["synthetic"])
        self.assertIn("runner.py", manifest["source_sha256"])
        self.assertEqual(manifest["config"]["episodes"][0]["seed"], 0)
        self.assertEqual(manifest["execution"]["workers"], 1)
        self.assertEqual(manifest["profile"]["chunk_mode"], "runner_owned")
        for result in outcome.results:
            for reference in result.artifact_refs:
                self.assertTrue(Path(reference).exists(), reference)

    def test_config_rejects_unknown_fields_unsafe_ids_and_unbounded_runs(self):
        for raw in (to_data(self.config) | {"extra": 1}, to_data(self.config) | {"run_id": "../outside"},
                    to_data(self.config) | {"episodes": [{"episode_id": "ep-0", "budgets": {}}]}):
            with self.subTest(raw=raw):
                with self.assertRaises(ContractError):
                    config = decode(RunConfig, raw)
                    self.execute(config)


class CliTests(unittest.TestCase):
    def test_cli_exit_codes_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = RunConfig(run_id="cli-test", model=ModelConfig(scenario="stop"),
                             episodes=(EpisodeConfig(episode_id="ep-0"),))
            config_path = root/"config.json"
            def invoke(config, target):
                config_path.write_text(json.dumps(to_data(config)))
                return subprocess.run([sys.executable, "-m", "embodied_eval", "run", "--config", str(config_path),
                    "--output", str(target)], capture_output=True, text=True)
            ok = invoke(base, root/"success")
            self.assertEqual(ok.returncode, 0, ok.stderr)
            existing = invoke(base, root/"success")
            self.assertEqual(existing.returncode, 2)
            error = invoke(replace(base, benchmark=BenchmarkConfig(fault="score_error")), root/"error")
            self.assertEqual(error.returncode, 1, error.stderr)
            rejected = invoke(replace(base, profile=ProfileConfig(chunk_mode="adapter_owned")), root/"rejected")
            self.assertEqual(rejected.returncode, 2)
            self.assertFalse((root/"rejected").exists())


if __name__ == "__main__":
    unittest.main()
