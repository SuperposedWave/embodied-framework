"""Fake-scene checks only; do not prove real SAPIEN integration or native scores."""
from types import SimpleNamespace
import unittest

from embodied_eval import _legacy_contracts as c
from embodied_eval.errors import AdapterError

try:
    import numpy as np
    from embodied_eval.adapters.native_robotwin_executor import MeasuredRobotwinExecutor, LingbotNativeActionCache
    from embodied_eval.adapters.native_robotwin_models import OPENWAM_ROBOTWIN_REVISION, LINGBOT_ROBOTWIN_REVISION
except ImportError:
    np = None


class FakeScene:
    def __init__(self, fail_at=None, change_period_at=None):
        self.calls = 0
        self.period = .004
        self.fail_at, self.change_period_at = fail_at, change_period_at
        self.render_updates = 0

    def get_timestep(self):
        return self.period

    def update_render(self):
        self.render_updates += 1

    def step(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("fake error; native state may already have advanced")
        if self.calls == self.change_period_at:
            self.period = .008


class FakeRobotwin:
    def __init__(self, durations=(3, 17, 4), *, scene=None, success_at=None, error_after=None):
        self.scene = scene or FakeScene()
        self.take_action_cnt = 0
        self.step_lim = 3
        self.eval_success = False
        self.durations = durations
        self.success_at, self.error_after = success_at, error_after
        self.actions = []
        self.observation_calls = self.success_checks = 0

    def get_obs(self):
        self.observation_calls += 1
        return {"should_not": "be acquired by the counter"}

    def check_success(self):
        self.success_checks += 1
        return self.eval_success

    def take_action(self, action, action_type):
        duration = self.durations[self.take_action_cnt]
        self.take_action_cnt += 1
        self.actions.append((action.copy(), action_type))
        self.scene.update_render()
        for index in range(duration):
            self.scene.step()
            if self.success_at == (self.take_action_cnt, index + 1):
                self.eval_success = True
                break
            if self.error_after == (self.take_action_cnt, index + 1):
                raise RuntimeError("known partial native error")


@unittest.skipIf(np is None, "optional real-client numerical dependencies not installed")
class RobotwinExecutorTests(unittest.TestCase):
    def executor(self, env, action_type="qpos", revision=None):
        return MeasuredRobotwinExecutor(env, benchmark_revision=revision or LINGBOT_ROBOTWIN_REVISION,
                                       action_type=action_type)

    def row(self, size=14):
        return np.arange(size, dtype=np.float32) / 10

    def test_different_macro_lengths_count_native_and_physical_steps_separately(self):
        env = FakeRobotwin()
        executor = self.executor(env)
        receipts = [executor.execute(self.row(), c.Budgets(env_steps=3-index)) for index in range(3)]
        self.assertEqual([item.consumed_physics_steps for item in receipts], [3, 17, 4])
        self.assertEqual(receipts[-1].cumulative_env_steps, 3)
        self.assertEqual(receipts[-1].cumulative_physics_steps, 24)
        self.assertAlmostEqual(receipts[-1].cumulative_sim_time_s, .096)
        self.assertTrue(receipts[-1].native_action_limit_reached)

    def test_meter_restores_scene_and_does_not_add_camera_or_success_calls(self):
        env = FakeRobotwin()
        original = env.scene
        self.executor(env).execute(self.row(), c.Budgets(env_steps=1))
        self.assertIs(env.scene, original)
        self.assertEqual((env.observation_calls, env.success_checks, original.render_updates), (0, 0, 1))

    def test_exact_native_array_dtype_and_order_preserved(self):
        env = FakeRobotwin()
        row = self.row()
        self.executor(env).execute(row, c.Budgets(env_steps=1))
        sent, action_type = env.actions[0]
        self.assertEqual(action_type, "qpos")
        self.assertEqual(sent.dtype, np.float32)
        np.testing.assert_array_equal(sent, row)
        self.assertIsNot(sent, row)

    def test_sapien_float32_period_is_retained_not_rounded_to_nominal(self):
        scene = FakeScene()
        scene.period = float(np.float32(.004))
        env = FakeRobotwin(scene=scene)
        executor = self.executor(env)
        receipt = executor.execute(self.row(), c.Budgets(env_steps=1))
        self.assertEqual(executor.physics_period_s, scene.period)
        self.assertEqual(receipt.elapsed_sim_time_s, 3 * scene.period)
        self.assertNotEqual(receipt.elapsed_sim_time_s, 3 * .004)

    def test_bootstrap_rejects_a_genuinely_different_source_period(self):
        scene = FakeScene()
        scene.period = .00400001
        with self.assertRaises(AdapterError):
            self.executor(FakeRobotwin(scene=scene))

    def test_openwam_ee16_is_a_separate_source_and_action_domain(self):
        env = FakeRobotwin()
        executor = self.executor(env, action_type="ee", revision=OPENWAM_ROBOTWIN_REVISION)
        executor.execute(self.row(16), c.Budgets(env_steps=1))
        self.assertEqual(env.actions[0][1], "ee")
        with self.assertRaises(AdapterError):
            executor.execute(self.row(14), c.Budgets(env_steps=1))

    def test_success_inside_macro_retains_only_completed_physical_steps(self):
        env = FakeRobotwin(durations=(100,), success_at=(1, 7))
        executor = self.executor(env)
        receipt = executor.execute(self.row(), c.Budgets(env_steps=3))
        self.assertEqual(receipt.consumed_physics_steps, 7)
        self.assertTrue(receipt.official_success)
        with self.assertRaises(AdapterError) as error:
            executor.execute(self.row(), c.Budgets(env_steps=2))
        self.assertEqual(error.exception.code, "native_episode_finished")
        self.assertEqual(len(env.actions), 1)

    def test_known_exception_returns_actual_partial_receipt(self):
        env = FakeRobotwin(durations=(100,), error_after=(1, 7))
        original = env.scene
        receipt = self.executor(env).execute(self.row(), c.Budgets(env_steps=1))
        self.assertEqual((receipt.consumed_env_steps, receipt.consumed_physics_steps), (1, 7))
        self.assertEqual(receipt.error_type, "RuntimeError")
        self.assertIs(env.scene, original)

    def test_thrown_physics_call_invalidates_counter_no_false_precision_or_resume(self):
        env = FakeRobotwin(scene=FakeScene(fail_at=2))
        original = env.scene
        executor = self.executor(env)
        with self.assertRaises(AdapterError) as error:
            executor.execute(self.row(), c.Budgets(env_steps=1))
        self.assertFalse(error.exception.state_known)
        self.assertEqual(executor.physics_steps, 0)
        self.assertIsNone(executor.last_receipt)
        self.assertIs(env.scene, original)
        with self.assertRaises(AdapterError) as second:
            executor.execute(self.row(), c.Budgets(env_steps=1))
        self.assertEqual(second.exception.code, "native_state_unknown")

    def test_changed_physical_period_fails_closed(self):
        env = FakeRobotwin(scene=FakeScene(change_period_at=1))
        executor = self.executor(env)
        with self.assertRaises(AdapterError) as error:
            executor.execute(self.row(), c.Budgets(env_steps=1))
        self.assertFalse(error.exception.state_known)
        self.assertFalse(executor.valid)

    def test_budget_rejected_before_native_call_including_zero_simulation_limit(self):
        for budgets in (c.Budgets(env_steps=0), c.Budgets(wall_time_s=0), c.Budgets(sim_time_s=0), c.Budgets(sim_time_s=10)):
            env = FakeRobotwin()
            executor = self.executor(env)
            with self.assertRaises(AdapterError):
                executor.execute(self.row(), budgets)
            self.assertEqual((env.take_action_cnt, env.scene.calls), (0, 0))

    def test_external_counter_change_is_not_silently_absorbed(self):
        env = FakeRobotwin()
        executor = self.executor(env)
        env.take_action_cnt = 1
        with self.assertRaises(AdapterError) as error:
            executor.execute(self.row(), c.Budgets(env_steps=1))
        self.assertFalse(error.exception.state_known)
        self.assertEqual(len(env.actions), 0)

    def test_unreviewed_source_or_nonfresh_episode_or_timestep_is_rejected(self):
        with self.assertRaises(ValueError):
            self.executor(FakeRobotwin(), revision="other")
        env = FakeRobotwin()
        env.take_action_cnt = 1
        with self.assertRaises(AdapterError):
            self.executor(env)
        env = FakeRobotwin()
        env.scene.period = .01
        with self.assertRaises(AdapterError):
            self.executor(env)

    def test_diagnostics_never_claim_registration_or_real_meter_verification(self):
        diagnostic = self.executor(FakeRobotwin()).diagnostics()
        self.assertFalse(diagnostic["framework_registered"])
        self.assertFalse(diagnostic["real_scene_meter_verified"])
        self.assertFalse(diagnostic["source_identity_verified"])


@unittest.skipIf(np is None, "optional real-client numerical dependencies not installed")
class RobotwinCacheTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.resets = []
        self.chunk = np.arange(42, dtype=np.float32).reshape(3, 14)
        def infer(observation, instruction):
            self.calls.append((observation, instruction))
            return self.chunk.copy()
        self.bridge = SimpleNamespace(predict=infer, reset=lambda: self.resets.append(True))
        self.cache = LingbotNativeActionCache(self.bridge)

    def test_every_row_used_in_order_no_duplicate_prediction_within_chunk(self):
        self.cache.begin_episode()
        self.assertTrue(self.cache.needs_observation)
        rows = [self.cache.next_action({"fresh": True} if index == 0 else None, "task") for index in range(3)]
        np.testing.assert_array_equal(np.stack(rows), self.chunk)
        self.assertTrue(self.cache.needs_observation)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(rows[0].dtype, np.float32)

    def test_episode_end_discards_tail_and_next_episode_starts_first_row(self):
        self.cache.begin_episode()
        first = self.cache.next_action({}, "task")
        self.cache.end_episode()
        self.assertFalse(self.cache.active)
        self.assertEqual(self.cache.rows, [])
        self.cache.begin_episode()
        second = self.cache.next_action({}, "task")
        np.testing.assert_array_equal(first, second)
        self.assertEqual((len(self.resets), len(self.calls)), (2, 2))

    def test_changed_instruction_cannot_silently_advance_stale_chunk(self):
        self.cache.begin_episode()
        self.cache.next_action({}, "task")
        with self.assertRaises(AdapterError):
            self.cache.next_action(None, "different")
        self.assertEqual(self.cache.rows, [])
        self.assertEqual(len(self.calls), 1)

    def test_bad_chunk_not_cached_and_begin_failure_does_not_reuse_old_rows(self):
        self.cache.begin_episode()
        self.chunk[2, 0] = float("nan")
        with self.assertRaises(AdapterError):
            self.cache.next_action({}, "task")
        self.assertEqual(self.cache.rows, [])
        self.chunk[2, 0] = 28
        self.cache.next_action({}, "task")
        def failed_reset():
            raise RuntimeError("synthetic reset failed")
        self.bridge.reset = failed_reset
        with self.assertRaises(RuntimeError):
            self.cache.begin_episode()
        self.assertFalse(self.cache.active)
        self.assertEqual(self.cache.rows, [])
