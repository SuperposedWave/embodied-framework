"""Source-defined pair wiring with test doubles; NOT real RoboTwin evidence."""
from dataclasses import replace
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from embodied_eval import contracts as c
from embodied_eval.codec import to_data, decode
from embodied_eval.compatibility import check_compatibility
from embodied_eval.config import RunConfig, ModelConfig, BenchmarkConfig, EpisodeConfig, ProfileConfig, validate_config
from embodied_eval.errors import AdapterError, ContractError
from embodied_eval.validation import validate_observation, validate_prediction, validate_step_result, validate_task
from embodied_eval.visibility import build_model_input
from embodied_eval.artifacts import ArtifactWriter
from embodied_eval.runner import run_episode

try:
    import numpy as np
    from embodied_eval.adapters import robotwin_spec as spec
    from embodied_eval.adapters.robotwin_model import RobotwinModel
    from embodied_eval.adapters.robotwin_benchmark import RobotwinBenchmark, _RngDomain
    from embodied_eval.adapters.native_robotwin_executor import LingbotNativeActionCache, MeasuredRobotwinExecutor
    from test_robotwin_executor import FakeRobotwin
except ImportError:
    np = None


def config(variant="lingbot_robotwin"):
    return RunConfig(run_id="robotwin-pair-test", model=ModelConfig(adapter=variant,
        openwam_repo="/native/openwam", lingbot_repo="/native/lingbot", checkpoint_dir="/model/robotwin"),
        benchmark=BenchmarkConfig(adapter="robotwin_lingbot" if variant == "lingbot_robotwin" else "robotwin_openwam",
                                  robotwin_repo="/native/robotwin", task_name="adjust_bottle"),
        profile=ProfileConfig(chunk_mode="adapter_owned", execution_horizon_steps=1),
        episodes=(EpisodeConfig(episode_id="trial0", seed=0, budgets=c.Budgets(env_steps=400, model_calls=400)),))


class PublicFakeEnv(FakeRobotwin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_lim = 400

    def get_obs(self):
        self.observation_calls += 1
        return {"observation": {key: {"rgb": np.full((240, 320, 3), self.observation_calls, np.uint8),
                                      "private_segmentation": "hidden"} for key in spec.SENSORS},
                "joint_action": {"vector": np.arange(14, dtype=np.float64)},
                "endpose": {"left_endpose": [1, 2, 3, 1, 0, 0, 0], "right_endpose": [4, 5, 6, 1, 0, 0, 0],
                            "left_gripper": .4, "right_gripper": .6}, "expert_plan": "hidden"}

    def close_env(self):
        pass


class Bridge:
    def __init__(self, width=14):
        self.width, self.calls, self.resets = width, 0, 0
        self.inputs = []

    def reset(self, *args):
        self.resets += 1

    def predict(self, obs, instruction):
        self.calls += 1
        self.inputs.append(obs)
        row = np.arange(self.width, dtype=np.float32)
        return np.stack([row, row + 1, row + 2]) if self.width == 14 else row

    def close(self):
        pass

    def diagnostics(self):
        return {"server_identity_verified": False}


@unittest.skipIf(np is None, "optional numerical dependencies not installed")
class RobotwinPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = config()
        self.bench = RobotwinBenchmark(self.cfg.benchmark, self.cfg.model.adapter, self.temp.name)
        self.bench.runtime = Path(self.temp.name)
        self.bench.env = PublicFakeEnv()
        self.bench.executor = MeasuredRobotwinExecutor(self.bench.env,
            benchmark_revision=spec.VARIANTS[self.cfg.model.adapter][1], action_type="qpos")
        self.bench.selected_seed = 100000
        self.bench.seed_index = 0
        self.bench.instruction = "Lift the bottle."
        self.bench._task_created = True
        self.context = c.Context(run_id="run", env_id="env0", episode_id="trial0", message_id="begin")
        self.task = c.TaskSpec(task_id="adjust_bottle/episode_000", episode_id="trial0",
            protocol_id=self.bench.space.protocol_id, protocol_version=self.bench.space.protocol_version,
            public=c.PublicTask(instruction=self.bench.instruction), initialization_ref="demo_clean/seed-index-0/expert-valid-100000",
            observation_policy_ref="public@1", budgets=c.Budgets(env_steps=3, model_calls=3), evaluation_config_ref="native@1")
        self.obs = self.bench.reset(self.task, self.context)
        self.model = RobotwinModel(self.cfg.model)
        self.bridge = Bridge()
        self.model.bridge = self.bridge
        self.model.cache = LingbotNativeActionCache(self.bridge)
        self.cap = self.model.describe()
        self.profile = spec.profile(self.cfg.model.adapter, self.bench.space, self.cap)
        self.model.begin_episode(self.context, self.task.public, self.bench.space, self.profile)
        self.model.observe(build_model_input(self.task, self.obs, self.profile))

    def predict(self, index=1):
        return self.model.predict(replace(self.context, message_id=f"req{index}"))

    def test_both_source_defined_configurations_validate(self):
        for variant in spec.VARIANTS:
            validate_config(config(variant))

    def test_describe_negotiates_actual_sapien_period_before_first_execution(self):
        self.bench.env.scene.period = float(np.float32(.004))
        self.bench.executor = MeasuredRobotwinExecutor(self.bench.env,
            benchmark_revision=spec.VARIANTS[self.cfg.model.adapter][1], action_type="qpos")
        actual = self.bench.describe()
        self.assertEqual(actual.physics_period_s, self.bench.env.scene.period)
        self.assertNotEqual(actual.physics_period_s, .004)
        reply = self.bench.advance(self.predict().decision, replace(self.context, message_id="execute"), c.Budgets(env_steps=1))
        self.assertEqual(reply.execution_feedback.elapsed_sim_time_s, 3 * actual.physics_period_s)
        validate_step_result(reply, self.obs, actual)

    def test_describe_rejects_non_native_physics_period(self):
        self.bench.env.scene.period = .005
        with self.assertRaises(AdapterError):
            self.bench.describe()

    def test_native_setup_directory_does_not_precreate_artifact_writer_root(self):
        output = Path(self.temp.name) / "run-output"
        benchmark = RobotwinBenchmark(self.cfg.benchmark, self.cfg.model.adapter, output)
        self.assertFalse(benchmark.runtime.is_relative_to(output))
        benchmark.runtime.mkdir(parents=True, exist_ok=False)
        self.assertFalse(output.exists())
        writer = ArtifactWriter(output)
        self.assertTrue(writer.root.is_dir())

    def test_cross_family_pair_rejected(self):
        with self.assertRaises(ContractError):
            validate_config(replace(self.cfg, benchmark=replace(self.cfg.benchmark, adapter="robotwin_openwam")))

    def test_seed_index_and_native_action_limits_frozen(self):
        for change in (replace(self.cfg.episodes[0], seed=7), replace(self.cfg.episodes[0], budgets=c.Budgets(env_steps=300, model_calls=400))):
            with self.assertRaises(ContractError):
                validate_config(replace(self.cfg, episodes=(change,)))

    def test_no_hard_simulation_time_even_zero(self):
        with self.assertRaises(ContractError):
            validate_task(replace(self.task, budgets=c.Budgets(sim_time_s=0)), self.bench.space)

    def test_randomized_not_silently_clean(self):
        with self.assertRaises(ContractError):
            validate_config(replace(self.cfg, benchmark=replace(self.cfg.benchmark, task_config="demo_randomized")))

    def test_native_space_and_profile_compatibility_both_families(self):
        for variant in spec.VARIANTS:
            cfg = config(variant)
            space, cap = spec.space(variant), spec.capability(cfg.model)
            self.assertTrue(check_compatibility(space, cap, spec.profile(variant, space, cap)).compatible)

    def test_raw_targets_not_mislabeled_measured_joints(self):
        validate_observation(self.obs, self.bench.space)
        self.assertEqual(self.obs.robots["robot0"], c.RobotState())
        self.assertEqual(self.obs.extensions[spec.EXTENSION]["joint_drive_targets_and_grippers"], list(range(14)))

    def test_native_quaternion_layout_preserved(self):
        self.assertEqual(self.obs.extensions[spec.EXTENSION]["left_endpose_world_wxyz"], [1, 2, 3, 1, 0, 0, 0])

    def test_private_native_fields_never_enter_model_input(self):
        native = self.model.native_input(self.model.latest)
        self.assertNotIn("expert_plan", native)
        self.assertEqual(set(native["observation"]["head_camera"]), {"rgb"})

    def test_chunk_rows_dtype_order_and_one_policy_query(self):
        rows = [self.predict(i).decision.commands[0].parameters for i in range(1, 4)]
        self.assertEqual(self.bridge.calls, 1)
        self.assertEqual([r["acquire_observation_after"] for r in rows], [False, False, True])
        self.assertTrue(all(r["native_dtype"] == "float32" for r in rows))
        np.testing.assert_array_equal(rows[2]["native_row"], np.arange(14, dtype=np.float32) + 2)

    def test_prediction_is_valid_measured_native_wire(self):
        prediction = self.predict()
        validate_prediction(prediction, self.obs, self.bench.space, self.cap, self.profile, request_context=prediction.context)
        decoded = decode(c.Prediction, to_data(prediction))
        self.assertIsInstance(decoded.decision, c.NativeAction)

    def test_no_extra_camera_inside_chunk_stale_time_is_honest(self):
        pred = self.predict()
        reply = self.bench.advance(pred.decision, pred.context, c.Budgets(env_steps=3, model_calls=2))
        validate_step_result(reply, self.obs, self.bench.space, expected_prediction=pred, capability=self.cap)
        self.assertEqual(self.bench.env.observation_calls, 1)
        self.assertEqual(reply.observation.sim_time_s, .012)
        self.assertEqual(reply.observation.sensors["head_camera"].captured_at_s, 0)
        self.assertEqual(reply.observation.extensions[spec.EXTENSION]["captured_at_sim_time_s"], 0)

    def test_camera_acquired_after_last_row_only_when_continuing(self):
        for i in range(1, 4):
            pred = self.predict(i)
            reply = self.bench.advance(pred.decision, pred.context, c.Budgets(env_steps=10-i, model_calls=10-i))
            self.model.observe(build_model_input(self.task, reply.observation, self.profile, reply.execution_feedback))
        self.assertEqual(self.bench.env.observation_calls, 2)
        self.assertEqual(reply.observation.extensions[spec.EXTENSION]["captured_at_sim_time_s"], .096)

    def test_final_budget_does_not_add_native_observation(self):
        for i in range(1, 4):
            pred = self.predict(i)
            reply = self.bench.advance(pred.decision, pred.context, c.Budgets(env_steps=4-i, model_calls=3-i))
            self.model.observe(build_model_input(self.task, reply.observation, self.profile))
        self.assertEqual(self.bench.env.observation_calls, 1)

    def test_native_success_no_extra_success_or_camera_call(self):
        self.bench.env.success_at = (1, 2)
        pred = self.predict()
        reply = self.bench.advance(pred.decision, pred.context, c.Budgets(env_steps=3, model_calls=2))
        self.assertTrue(reply.terminated)
        self.assertEqual(self.bench.env.success_checks, 0)
        self.assertEqual(self.bench.env.observation_calls, 1)
        self.assertTrue(self.bench.score(pred.context)["success"].value)

    def test_known_partial_failure_returns_measured_counts(self):
        self.bench.env.error_after = (1, 2)
        pred = self.predict()
        reply = self.bench.advance(pred.decision, pred.context, c.Budgets(env_steps=3, model_calls=2))
        self.assertEqual(reply.execution_feedback.status, "execution_failed")
        self.assertEqual(reply.execution_feedback.consumed_physics_steps, 2)
        self.assertEqual(reply.observation.sim_time_s, .008)

    def test_dtype_loss_rejected_before_physics(self):
        pred = self.predict()
        cmd = pred.decision.commands[0]
        params = {**cmd.parameters, "native_dtype": "int32", "native_row": [.5] * 14}
        with self.assertRaises(AdapterError):
            self.bench.advance(replace(pred.decision, commands=(replace(cmd, parameters=params),)), pred.context, c.Budgets(env_steps=3))
        self.assertEqual(self.bench.env.scene.calls, 0)

    def test_model_cannot_cross_episode(self):
        with self.assertRaises(ContractError):
            self.model.predict(replace(self.context, episode_id="other"))

    def test_end_and_begin_reset_full_chunk_cache(self):
        self.predict()
        self.model.end_episode(c.ModelEndContext(context=self.context, reason="budget_exhausted"))
        self.assertEqual(self.model.cache.rows, [])
        self.model.begin_episode(self.context, self.task.public, self.bench.space, self.profile)
        self.assertEqual(self.bridge.resets, 2)

    def test_second_reset_cannot_reuse_fixed_scene(self):
        with self.assertRaises(AdapterError):
            self.bench.reset(self.task, self.context)

    def test_rng_domain_restores_caller_and_advances_private_cursor(self):
        py, state = random.getstate(), np.random.get_state()
        domain = _RngDomain(42)
        with domain.activate():
            first = (random.random(), np.random.random())
        with domain.activate():
            second = (random.random(), np.random.random())
        self.assertNotEqual(first, second)
        self.assertEqual(random.getstate(), py)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        other = _RngDomain(42)
        with other.activate():
            self.assertEqual(first, (random.random(), np.random.random()))

    def test_openwam_one_row_uses_every_decision_camera_boundary(self):
        cfg = config("openwam_robotwin")
        model = RobotwinModel(cfg.model)
        model.bridge = Bridge(16)
        space, cap = spec.space(cfg.model.adapter), model.describe()
        profile = spec.profile(cfg.model.adapter, space, cap)
        model.begin_episode(self.context, self.task.public, space, profile)
        observation = replace(self.obs, spec_ref=space.spec_id)
        model.observe(build_model_input(self.task, observation, profile))
        pred = model.predict(replace(self.context, message_id="ee-request"))
        validate_prediction(pred, observation, space, cap, profile, request_context=pred.context)
        self.assertTrue(pred.decision.commands[0].parameters["acquire_observation_after"])
        self.assertEqual(len(pred.decision.commands[0].parameters["native_row"]), 16)

    def test_full_serial_runner_native_accounting_with_fake_adapters(self):
        self.bench.context = None  # Start the one prepared test scene through Runner.
        from embodied_eval.robotwin_pairs import fallback_metrics
        writer = ArtifactWriter(Path(self.temp.name) / "fake-run")
        outcome = run_episode(self.model, self.bench, self.profile, self.task, self.bench.space,
                              self.cap, writer, "run", fallback_metrics(self.cfg))
        self.assertEqual(outcome.result.run_status, "completed")
        self.assertEqual(outcome.result.execution_statistics.env_steps, 3)
        self.assertEqual(outcome.result.execution_statistics.physics_steps, 24)
        self.assertEqual(outcome.result.execution_statistics.sim_time_s, .096)
        self.assertEqual(self.bridge.calls, 1)
        self.assertEqual(self.bench.env.observation_calls, 2)  # setUp + actual Runner reset only
        self.assertEqual(self.model.cache.rows, [])

    def test_registry_provenance_not_verified_before_preflight(self):
        from embodied_eval.config import run_provenance
        info = run_provenance(self.cfg)
        self.assertFalse(info["server_identity_verified"])
        self.assertFalse(info["asset_checks_completed"])
        self.assertFalse(info["native_scene_meter_real_verified"])

    def test_invalid_expert_scan_bound_rejected(self):
        for limit in (0, 101):
            with self.assertRaises(ContractError):
                validate_config(replace(self.cfg, benchmark=replace(self.cfg.benchmark, max_seed_candidates=limit)))
