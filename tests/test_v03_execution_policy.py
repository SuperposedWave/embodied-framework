"""Synthetic execution-policy gates; no GPU, inference or real benchmark claims."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from types import SimpleNamespace

from embodied_eval.codec import decode, to_data
from embodied_eval.errors import ContractError
from embodied_eval.core.registry import AdapterRegistry
from embodied_eval.v03 import contracts as c
from embodied_eval.v03.config import RunConfig, AdapterConfig, EpisodeConfig
from embodied_eval.v03.execution import resolve_execution, apply_execution
from embodied_eval.v03.preflight import check
from embodied_eval.v03.runner import run
from embodied_eval.v03.synthetic import register, capability, space, SyntheticModel, SyntheticBenchmark

SOURCE = {"source": "SYNTHETIC declaration, not official benchmark evidence"}


def model(**changes):
    declaration = c.ModelExecution(generated_actions=8, preferred_consume_actions=4,
        minimum_consume_actions=1, maximum_consume_actions=8, adjustable=True,
        cache_owner="adapter", action_period_s=.05, provenance=SOURCE)
    cap = capability({"cache_rows": 4})
    declaration = replace(declaration, **changes)
    return replace(cap, execution=declaration, temporal=replace(cap.temporal, replanning_owner=declaration.cache_owner))


def environment(authority="policy_owned", **rule):
    return replace(space({}), execution=c.BenchmarkExecution(scope={"protocol": "synthetic-test@1"},
        consume_actions=c.ExecutionConstraint(authority=authority, provenance=SOURCE, **rule),
        action_period_s=c.ExecutionConstraint(authority="fixed", value=.05, provenance=SOURCE)))


class ExecutionResolution(unittest.TestCase):
    def resolve(self, env=None, cap=None, **policy):
        return resolve_execution(env or environment(), cap or model(), c.ExecutionPolicy(**policy))

    def rejects(self, code, env=None, cap=None, **policy):
        with self.assertRaises(ContractError) as error:
            self.resolve(env, cap, **policy)
        self.assertEqual(error.exception.issue.code, code)

    def test_fixed_benchmark_selects_supported_prefix(self):
        plan = self.resolve(environment("fixed", value=2), require_verified=True)
        self.assertEqual(plan.consume_actions, 2)
        self.assertTrue(plan.change_requested and plan.protocol_verified)

    def test_fixed_rule_rejects_conflicting_user_override(self):
        self.rejects("execution_rule_conflict", environment("fixed", value=2), consume_actions=3)

    def test_fixed_rule_cannot_take_over_server_cache(self):
        self.rejects("unsupported_execution_change", environment("fixed", value=2),
                     model(adjustable=False, cache_owner="server"))

    def test_range_prefers_native_prefix(self):
        self.assertEqual(self.resolve(environment("range", minimum=2, maximum=6)).consume_actions, 4)

    def test_range_never_silently_clamps_native_prefix(self):
        self.rejects("execution_rule_conflict", environment("range", minimum=1, maximum=2))

    def test_range_allows_supported_explicit_experiment(self):
        self.assertEqual(self.resolve(environment("range", choices=(2, 4)), consume_actions=2).consume_actions, 2)

    def test_range_rejects_unlisted_choice(self):
        self.rejects("execution_rule_conflict", environment("range", choices=(2, 4)), consume_actions=3)

    def test_policy_owned_uses_model_preference(self):
        self.assertEqual(self.resolve(require_verified=True).consume_actions, 4)

    def test_model_capability_rejects_unsupported_prefix(self):
        self.rejects("unsupported_execution_change", consume_actions=9)

    def test_reference_default_is_not_a_hard_benchmark_rule(self):
        plan = self.resolve(environment("reference_default", value=7))
        self.assertEqual(plan.consume_actions, 4)
        self.assertFalse(plan.protocol_verified)

    def test_unknown_keeps_native_and_explicitly_records_gap(self):
        plan = self.resolve(environment("unknown"))
        self.assertEqual(plan.consume_actions, 4)
        self.assertIn("benchmark.consume_actions:unknown", plan.unresolved)

    def test_unknown_does_not_authorize_an_experiment(self):
        self.rejects("execution_permission_unknown", environment("unknown"), consume_actions=2)

    def test_model_options_cannot_bypass_unknown_benchmark_permission(self):
        self.rejects("execution_permission_unknown", environment("unknown"),
                     model(reference_consume_actions=8))

    def test_known_numbers_without_model_provenance_cannot_pass_strict_gate(self):
        self.rejects("unverified_execution_protocol", cap=model(provenance={}), require_verified=True)

    def test_strict_gate_rejects_unknown_authority(self):
        self.rejects("unverified_execution_protocol", environment("unknown"), require_verified=True)

    def test_unknown_weight_action_period_cannot_pass_strict_gate(self):
        self.rejects("unverified_execution_protocol", cap=model(action_period_s=None), require_verified=True)

    def test_mismatched_period_rejected_without_resampling(self):
        self.rejects("execution_timing_conflict", cap=model(action_period_s=.1))

    def test_period_rule_and_loaded_environment_must_agree(self):
        self.rejects("execution_timing_conflict", replace(environment(), control_period_s=.1))

    def test_per_parameter_authority_not_whole_benchmark_flag(self):
        env = environment("policy_owned")
        env = replace(env, execution=replace(env.execution, action_period_s=c.ExecutionConstraint()))
        plan = self.resolve(env)
        self.assertEqual(plan.consume_actions, 4)
        self.assertIn("benchmark.action_period_s:unknown", plan.unresolved)

    def test_conflicting_cache_and_action_time_declarations_rejected(self):
        for temporal in (replace(model().temporal, replanning_owner="server"),
                         replace(model().temporal, native_action_period_s=.1)):
            self.rejects("conflicting_execution_declarations", cap=replace(model(), temporal=temporal))

    def test_period_range_is_actually_checked(self):
        env = environment()
        env = replace(env, execution=replace(env.execution, action_period_s=c.ExecutionConstraint(
            authority="range", minimum=.1, maximum=.2, provenance=SOURCE)))
        self.rejects("execution_timing_conflict", env)

    def test_variable_transitions_do_not_invent_seconds_per_chunk(self):
        env = replace(environment(), timing_mode="measured_native", control_period_s=None,
            physics_period_s=.004)
        env = replace(env, execution=replace(env.execution, action_period_s=c.ExecutionConstraint()))
        self.assertIsNone(self.resolve(env).action_period_s)

    def test_variable_transition_cannot_advertise_fixed_duration(self):
        self.rejects("execution_timing_conflict", replace(environment(), timing_mode="measured_native", control_period_s=None))

    def test_generated_and_consumed_actions_are_distinct(self):
        self.assertEqual(model().execution.generated_actions, 8)
        self.assertEqual(self.resolve().consume_actions, 4)

    def test_integer_rule_survives_json_roundtrip(self):
        rule = c.ExecutionConstraint(authority="fixed", value=2, provenance=SOURCE)
        self.assertIs(type(decode(c.ExecutionConstraint, to_data(rule)).value), int)

    def test_boolean_and_zero_policy_rejected(self):
        for value in (True, 0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ContractError):
                self.resolve(consume_actions=value)

    def test_no_authority_without_evidence(self):
        env = environment()
        env = replace(env, execution=replace(env.execution, consume_actions=c.ExecutionConstraint(authority="fixed", value=2)))
        self.rejects("missing_execution_provenance", env)

    def test_inverted_capability_and_invalid_generation_rejected(self):
        for changes in ({"maximum_consume_actions": 2}, {"generated_actions": 2},
                        {"cache_owner": "server"}, {"action_period_s": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                self.resolve(cap=model(**changes))

    def test_unacknowledged_change_rejected(self):
        plan = self.resolve(environment("fixed", value=2))
        adapter = Mock()
        adapter.configure_execution.return_value = {"status": "applied", "applied_consume_actions": 4, "cache_owner": "adapter"}
        with self.assertRaises(ContractError):
            apply_execution(adapter, plan)

    def test_missing_application_hook_rejected_for_change(self):
        with self.assertRaises(ContractError):
            apply_execution(object(), self.resolve(environment("fixed", value=2)))

    def test_server_reference_does_not_claim_remote_configuration(self):
        plan = self.resolve(environment("unknown"), model(adjustable=False, cache_owner="server"))
        receipt = apply_execution(object(), plan)
        self.assertFalse(receipt["server_verified"])
        self.assertIsNone(receipt["applied_consume_actions"])


class ExecutionRuntime(unittest.TestCase):
    def setUp(self):
        self.registry = AdapterRegistry()
        register(self.registry)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, m="synthetic_policy_a", b="synthetic_simulator_a", **policy):
        return RunConfig(run_id="execution-test", model=AdapterConfig(adapter=m, options={"cache_rows": 4}),
            benchmark=AdapterConfig(adapter=b, options={"period_s": .05}),
            episodes=(EpisodeConfig(episode_id="ep", budgets=c.leaf.Budgets(env_steps=5, model_calls=5)),),
            execution=c.ExecutionPolicy(**policy))

    def test_four_cross_combinations_and_third_plugin_without_runner_changes(self):
        for i, m in enumerate(self.registry.models):
            for j, b in enumerate(self.registry.benchmarks):
                with self.subTest(model=m, benchmark=b):
                    result = run(self.config(m, b, consume_actions=2), self.root / f"cross-{i}-{j}", self.registry)
                    self.assertEqual(result.exit_code, 0)
                    self.assertEqual(result.results[0]["execution_statistics"]["model_forward_count"], 3)
        self.registry.register_model("third-model", replace(self.registry.model("synthetic_policy_a"),
            describe=lambda o: replace(capability(o), model_id="synthetic-third")))
        self.registry.register_benchmark("third-environment", self.registry.benchmark("synthetic_simulator_a"))
        self.assertEqual(run(self.config("third-model", "third-environment", consume_actions=2),
                             self.root / "third", self.registry).exit_code, 0)

    def test_runtime_saves_plan_application_and_keeps_counters_separate(self):
        result = run(self.config(consume_actions=2), self.root / "receipt", self.registry)
        evidence = json.loads((result.output / "execution-plan.json").read_text())
        self.assertEqual(evidence["runtime"]["consume_actions"], 2)
        self.assertEqual(evidence["application"]["applied_consume_actions"], 2)
        self.assertEqual(result.results[0]["execution_statistics"]["env_steps"], 5)
        self.assertFalse(evidence["application"]["server_verified"])

    def test_preflight_does_not_construct_either_adapter(self):
        for table in (self.registry.models, self.registry.benchmarks):
            for name, entry in list(table.items()):
                table[name] = replace(entry, factory=Mock(side_effect=AssertionError("GPU factory in preflight")))
        self.assertTrue(check(self.config(), self.registry)["structurally_compatible"])

    def test_runtime_rechecks_actual_period_before_model_factory(self):
        class ChangedPeriod(SyntheticBenchmark):
            def describe(self):
                return replace(super().describe(), control_period_s=.1)
        entry = self.registry.benchmark("synthetic_simulator_a")
        self.registry.benchmarks["synthetic_simulator_a"] = replace(entry, factory=ChangedPeriod)
        with self.assertRaises(ContractError):
            run(self.config(), self.root / "period-changed", self.registry)
        self.assertFalse((self.root / "period-changed").exists())

    def test_variable_duration_keeps_actual_physics_counters(self):
        cfg = self.config(consume_actions=2)
        cfg = replace(cfg, benchmark=replace(cfg.benchmark, options={"timing_mode": "measured_native"}))
        result = run(cfg, self.root / "variable", self.registry)
        stats = result.results[0]["execution_statistics"]
        self.assertEqual((stats["env_steps"], stats["physics_steps"]), (5, 25))
        self.assertAlmostEqual(stats["sim_time_s"], .1)

    def test_real_catalogues_remain_separate_and_gpu_free(self):
        root = Path(__file__).resolve().parents[1]
        for module, forbidden in (("model_catalog", "benchmark_catalog"), ("benchmark_catalog", "model_catalog")):
            command = "import sys; import embodied_eval.native." + module + "; assert 'embodied_eval.native." + forbidden + "' not in sys.modules; assert not any(n in sys.modules for n in ('numpy','torch','sapien','mujoco'))"
            response = subprocess.run([sys.executable, "-c", command], cwd=root, capture_output=True, text=True)
            self.assertEqual(response.returncode, 0, response.stderr)

    def test_causal_model_does_not_claim_arbitrary_prefix_adjustment(self):
        from embodied_eval.native.execution_profiles import model_execution
        opts = {"io_profile": "droid-joint8-causal"}
        self.assertFalse(model_execution("dreamzero", opts).adjustable)
        with self.assertRaises(ContractError):
            cap = replace(capability({}), execution=model_execution("dreamzero", {**opts, "open_loop_horizon": 2}))
            resolve_execution(space({}), cap, c.ExecutionPolicy())

    def test_only_current_cli_commands_are_published(self):
        response = subprocess.run([sys.executable, "-m", "embodied_eval", "--help"], capture_output=True, text=True)
        self.assertEqual(response.returncode, 0)
        self.assertIn("{run,check}", response.stdout)
        self.assertNotIn("migrate", response.stdout)

    def test_runtime_uses_one_context_type_and_rejects_historical_context(self):
        from embodied_eval import ExecutionPolicy
        from embodied_eval.contracts import ModelExecution
        self.assertIs(ExecutionPolicy, c.ExecutionPolicy)
        self.assertIs(ModelExecution, c.ModelExecution)
        from embodied_eval.native import contracts as private, environment_records
        from embodied_eval.v03.context import native_context
        from tests.legacy_support.frozen.native_records_v02 import Context as OldContext
        self.assertIs(private.Context, c.Context)
        self.assertIs(environment_records.Context, c.Context)
        self.assertIs(private.Prediction, c.Prediction)
        self.assertIs(private.TaskSpec, c.TaskSpec)
        with self.assertRaises(ContractError):
            native_context(OldContext(run_id="r", env_id="e", episode_id="ep", message_id="m"))

    def test_native_openpi_applies_prefix_once_without_an_outer_cache(self):
        import numpy as np
        from embodied_eval.native.native_models import NativeActionModel
        from embodied_eval.core.types import Action
        adapter = NativeActionModel("openpi", {"io_profile": "state8_osc", "repo": "/synthetic", "checkpoint_dir": "/synthetic"})
        adapter.configure_execution(2)
        adapter.peer = Mock()
        adapter.peer.infer.return_value = {"actions": np.arange(350, dtype=np.float32).reshape(50, 7)}
        adapter.latest = SimpleNamespace(task_public=SimpleNamespace(instruction="synthetic"), observation_id="o")
        adapter.profile = adapter.space = None
        adapter.native_input = Mock(return_value={"synthetic": True})
        adapter.decode_action = Mock(return_value=Action(commands=()))
        context = c.Context(run_id="r", env_id="e", episode_id="ep", message_id="m")
        with patch("embodied_eval.native.transforms.convert_pose_action", side_effect=lambda action, *args: action):
            for _ in range(3):
                adapter.predict(context)
        self.assertEqual(adapter.peer.infer.call_count, 2)
        self.assertEqual(adapter.requests, 2)
        self.assertEqual(adapter.reply_rows[0]["reply_action_rows"], 50)
        self.assertEqual(adapter.reply_rows[0]["adapter_enqueued_rows"], 2)
        self.assertEqual(len(adapter.cache), 1)
        self.assertEqual([call.kwargs["acquire"] for call in adapter.decode_action.call_args_list], [False, True, False])
        with self.assertRaises(ContractError):
            adapter.configure_execution(1)

    def test_environment_loader_has_no_model_repository_or_paired_variant(self):
        from embodied_eval.adapters.robotwin_benchmark import RobotwinBenchmark
        import inspect
        source = inspect.getsource(RobotwinBenchmark)
        self.assertNotIn("model_repo", inspect.signature(RobotwinBenchmark).parameters)
        for value in ("openwam_robotwin", "lingbot_robotwin", "benchmarks.robotwin.eval_manifest"):
            self.assertNotIn(value, source)


if __name__ == "__main__":
    unittest.main()
