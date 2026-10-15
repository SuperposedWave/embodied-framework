"""Synthetic architecture/control probes; never counted as real task scores."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from embodied_eval import contracts as leaf
from embodied_eval.errors import ContractError
from embodied_eval.mn import contracts as c
from embodied_eval.mn.binding import resolve, check_compatibility
from embodied_eval.mn.native_catalog import capability, space as native_space
from embodied_eval.mn.native_models import NativeActionModel
from embodied_eval.mn.native_benchmarks import NativeSimulator
from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator, space, extract_environment_class, METHODS
from embodied_eval.mn.validation import validate_space, validate_prediction
from embodied_eval.mn.joint_native import RAW_GRIPPER_PROTOCOL


class DualExtensionTests(unittest.TestCase):
    def setUp(self):
        self.context = c.Context(run_id="r", env_id="e", episode_id="ep", message_id="request")
        self.model_options = {"repo": "/absent", "checkpoint_dir": "/absent", "io_profile": "arx_joint14"}
        self.bench_options = {"repo": "/absent"}
        self.space = space(self.bench_options)
        self.cap = capability("openpi", self.model_options)

    def binding(self, robotwin=False):
        inputs = {"instruction": "instruction", "head": "head_camera" if robotwin else "cam_head",
            "left_wrist": "left_camera" if robotwin else "cam_left_wrist",
            "right_wrist": "right_camera" if robotwin else "cam_right_wrist"}
        for side in ("left", "right"):
            inputs[side + "_joints"] = side + "_joint_positions"
            inputs[side + "_grip"] = side + ("_gripper_open_fraction" if robotwin else "_gripper_target")
        return c.BindingConfig(inputs=inputs, targets={key: "robot0/" + key for key in self.cap.output.target_slots},
            action_spec_ref="joint-absolute-native-gripper@1" if robotwin else "dual-joint-raw-gripper@1")

    def model(self):
        model = NativeActionModel("openpi", self.model_options)
        model.space, model.profile = self.space, resolve(self.space, self.cap, self.binding())
        return model

    def view(self):
        sensors = {key: leaf.SensorSample(captured_at_s=0., frame_id="world", validity="valid", provenance="sensor",
            payload=leaf.Tensor(dtype="uint8", shape=(2, 3, 3), layout="HWC", data=np.full((2, 3, 3), i, np.uint8).tolist()))
            for i, key in enumerate(("head", "left_wrist", "right_wrist"))}
        states = {}
        for i, side in enumerate(("left", "right")):
            for suffix, values in (("_joints", np.arange(6, dtype=np.float64) + i * 10), ("_grip", np.array([.25 + i * .5]))):
                states[side + suffix] = c.StateSample(captured_at_s=0.,
                    payload=leaf.Tensor(dtype="float64", shape=values.shape, layout="vector", data=values.tolist()))
        return c.ModelInput(context=self.context, observation_id="obs", sim_time_s=0.,
            task_public=leaf.PublicTask(instruction="synthetic dynamic text"), sensors=sensors, states=states)

    def bench(self):
        return RoboDojoSimulator(self.bench_options)

    def test_pi05_dual_can_wire_two_independent_environments(self):
        validate_space(self.space)
        self.assertTrue(check_compatibility(self.space, self.cap, self.binding()).compatible)
        rw = native_space("robotwin", {"repo": "/absent", "protocol": "seeded-instruction", "task_name": "adjust_bottle"})
        report = check_compatibility(rw, self.cap, self.binding(True))
        self.assertTrue(report.compatible)
        self.assertTrue(report.warnings)  # Training identity is NOT a structural gate.

    def test_pi05_input_exact_arm_order_chw_and_float32(self):
        request = self.model().native_input(self.view())
        np.testing.assert_array_equal(request["state"], [0, 1, 2, 3, 4, 5, .25, 10, 11, 12, 13, 14, 15, .75])
        self.assertEqual(request["state"].dtype, np.float32)
        self.assertEqual(request["images"]["cam_high"].shape, (3, 2, 3))
        self.assertEqual(request["images"]["cam_right_wrist"][0, 0, 0], 2)
        self.assertEqual(request["prompt"], "synthetic dynamic text")

    def test_pi05_preserves_unbounded_gripper_to_native_controller(self):
        model, bench = self.model(), self.bench()
        row = np.arange(14, dtype=np.float32) / 16
        row[6], row[13] = 1.2, -.1
        action = model.decode_action(row, self.context, self.view())
        validate_prediction(c.Prediction(context=self.context, based_on_observation_id="obs", action=action),
            self.context, self.view(), self.space, self.cap, model.profile)
        self.assertEqual(action.native_protocol, RAW_GRIPPER_PROTOCOL)
        self.assertTrue(action.acquire_observation_after)
        actual = bench.native_action(action)
        np.testing.assert_array_equal(actual["left_arm_joint_state"], row[:6])
        np.testing.assert_array_equal(actual["right_arm_joint_state"], row[7:13])
        self.assertEqual(actual["left_ee_joint_state"][0], row[6])
        self.assertEqual(actual["right_ee_joint_state"][0], row[13])

    def test_missing_right_hand_is_rejected(self):
        action = self.model().decode_action(np.zeros(14), self.context, self.view())
        with self.assertRaises(ContractError):
            self.bench().native_action(replace(action, commands=action.commands[:7]))

    def test_native_array_dtype_cast_is_lossless(self):
        action = self.model().decode_action(np.zeros(14), self.context, self.view())
        commands = (replace(action.commands[0], value=.1),) + action.commands[1:]
        with self.assertRaises(ContractError):
            self.bench().native_action(replace(action, commands=commands, native_dtype="float32"))

    def test_pi05_full_fifty_row_cache_preserves_order(self):
        model = self.model()
        model.latest = self.view()
        rows = np.arange(700, dtype=np.float32).reshape(50, 14)
        model.peer = SimpleNamespace(infer=lambda data: {"actions": rows})
        for i in range(50):
            prediction = model.predict(replace(self.context, message_id=f"req{i}"))
            self.assertEqual(prediction.action.commands[0].value, rows[i, 0])
            self.assertTrue(prediction.action.acquire_observation_after)
        self.assertEqual(model.requests, 1)
        self.assertFalse(model.cache)

    def test_pi05_short_chunk_is_not_accepted_as_full_checkpoint_profile(self):
        model = self.model()
        model.latest = self.view()
        model.peer = SimpleNamespace(infer=lambda data: {"actions": np.zeros((49, 14), np.float32)})
        with self.assertRaises(ContractError):
            model.predict(self.context)

    @staticmethod
    def extraction_fixture():
        methods = "\n".join(f"    def {name}(self):\n        return 'environment-only'" for name in METHODS if name != "reset")
        return "raise RuntimeError('MODULE MUST NOT EXECUTE')\nclass EvalEnv:\n" + methods + "\n    def reset(self):\n        self.was_reset = True\n        self.model_client.call(func_name='reset')\n"

    def test_native_environment_extraction_does_not_execute_module_or_policy_reset(self):
        cls, hashes = extract_environment_class(self.extraction_fixture(), object, {})
        env = cls()
        env.reset()
        self.assertTrue(env.was_reset)
        self.assertFalse(hasattr(env, "model_client"))
        self.assertEqual(set(hashes), set(METHODS))

    def test_new_policy_dependency_in_source_is_rejected(self):
        source = self.extraction_fixture().replace("return 'environment-only'", "return self.deploy_cfg", 1)
        with self.assertRaises(ContractError):
            extract_environment_class(source, object, {})

    def test_missing_reviewed_environment_method_is_rejected(self):
        with self.assertRaises(ContractError):
            extract_environment_class(self.extraction_fixture().replace("def step(self)", "def changed_step(self)"), object, {})

    def test_control_queue_dependencies_are_extracted_and_called(self):
        source = self.extraction_fixture().replace(
            "    def have_empty(self):\n        return 'environment-only'",
            "    def have_empty(self, env_idx_list=None):\n"
            "        if env_idx_list is None:\n            env_idx_list = list(range(self.num_envs))\n"
            "        return len(self.get_control_empty(env_idx_list=env_idx_list)) != 0",
        ).replace(
            "    def get_control_empty(self):\n        return 'environment-only'",
            "    def get_control_empty(self, env_idx_list=None):\n"
            "        if env_idx_list is None:\n            env_idx_list = list(range(self.num_envs))\n"
            "        return self.robot_manager.control_manager.get_empty(env_idx_list=env_idx_list)",
        )
        cls, hashes = extract_environment_class(source, object, {})
        env = cls()
        env.num_envs = 2
        calls = []
        env.robot_manager = SimpleNamespace(control_manager=SimpleNamespace(
            get_empty=lambda **kwargs: calls.append(kwargs["env_idx_list"]) or [0]))
        self.assertTrue(env.have_empty())
        self.assertTrue(env.have_empty([1]))
        self.assertEqual(calls, [[0, 1], [1]])
        self.assertTrue({"have_empty", "get_control_empty"} <= set(hashes))

    def test_missing_control_queue_dependency_is_rejected_before_native_execution(self):
        for name in ("have_empty", "get_control_empty"):
            with self.subTest(name=name), self.assertRaises(ContractError):
                extract_environment_class(self.extraction_fixture().replace(
                    f"def {name}(self)", f"def missing_{name}(self)"), object, {})

    def test_native_pose_xyz_wxyz_conversion(self):
        commands = tuple(cmd for side in ("left", "right") for cmd in (
            leaf.PoseTarget(command_id=side + "p", target_id="robot0/" + side + "_eef",
                pose=leaf.Pose(position_m=(1., 2., 3.), quaternion_xyzw=(0., 0., 1., 0.), frame_id="world")),
            leaf.GripperCommand(command_id=side + "g", target_id="robot0/" + side + "_grip", mode="open_fraction", value=.5)))
        actual = self.bench().native_action(c.Action(commands=commands))
        self.assertEqual(actual["left_ee_pose"].tolist(), [1., 2., 3., 0., 0., 0., 1.])

    def test_official_score_not_initial_optimistic_flag(self):
        bench = self.bench()
        bench.env = SimpleNamespace(get_score=lambda: None, success=[True], end_flag=[False],
            reward_manager=SimpleNamespace(get_score=lambda: [73.]))
        result = bench.score(self.context)
        self.assertFalse(result["success"].value)
        self.assertEqual(result["score"].value, 73.)
        bench.env.end_flag[0] = True
        self.assertEqual(bench.score(self.context)["score"].value, 100.)

    def test_native_score_collection_does_not_reset_registered_progress(self):
        bench = self.bench()
        def register_again():
            raise AssertionError("score registration must not run after actions")
        bench.env = SimpleNamespace(get_score=register_again, success=[True], end_flag=[False],
            reward_manager=SimpleNamespace(get_score=lambda: [15.]))
        self.assertEqual(bench.score(self.context)["score"].value, 15.)

    def test_reset_registers_native_reward_and_score_after_native_reset(self):
        bench = self.bench()
        events = []
        bench.env = SimpleNamespace(reset=lambda **kwargs: events.append("reset"),
            run_reward=lambda: events.append("reward"), get_score=lambda: events.append("score"),
            episode_nums=1, unstable_envs=set(), reward_manager=SimpleNamespace(
                check_list=[["checks"]], score_list=[["stage0", "stage1"]], score_meta=[{"mode": "transition"}]))
        bench._snapshot = lambda context: events.append("snapshot")
        bench.reset(None, self.context)
        self.assertEqual(events, ["reset", "reward", "score", "snapshot"])

    def test_empty_native_task_checks_rejected_before_first_action(self):
        bench = self.bench()
        bench.env = SimpleNamespace(reset=lambda **kwargs: None, run_reward=lambda: None, get_score=lambda: None,
            episode_nums=1, unstable_envs=set(), reward_manager=SimpleNamespace(
                check_list=[[]], score_list=[[]], score_meta=[{"mode": None}]))
        with self.assertRaises(ContractError) as raised:
            bench.reset(None, self.context)
        self.assertEqual(raised.exception.issue.code, "native_reward_registration_missing")

    def test_measured_native_counts_are_three_separate_values(self):
        bench = self.bench()
        bench.actual_physics_steps = 0
        def execute(row):
            bench.actual_physics_steps += 13
            bench.env.take_action_cnt[0] += 1
        bench.env = SimpleNamespace(take_action=execute, take_action_cnt=[0], end_flag=[False], success=[True])
        bench._snapshot = lambda context: None
        action = self.model().decode_action(np.zeros(14), self.context, self.view())
        result = bench.advance(action, self.context, leaf.Budgets(env_steps=1))
        self.assertEqual(result.execution_feedback.consumed_env_steps, 1)
        self.assertEqual(result.execution_feedback.consumed_physics_steps, 13)
        self.assertAlmostEqual(result.execution_feedback.elapsed_sim_time_s, .052)

    def test_measured_native_hard_time_budget_rejected_before_execution(self):
        with self.assertRaises(ContractError):
            self.bench().advance(None, self.context, leaf.Budgets(sim_time_s=.01))

    def test_native_ik_failure_never_silently_claims_second_arm_executed(self):
        bench = self.bench()
        bench.actual_physics_steps = 0
        def partial(row):
            bench.failed_ik_targets.add("robot0/right_eef")
            bench.actual_physics_steps += 10
            bench.env.take_action_cnt[0] += 1
        bench.env = SimpleNamespace(take_action=partial, take_action_cnt=[0], end_flag=[False], success=[True])
        bench._snapshot = lambda context: None
        commands = tuple(cmd for side in ("left", "right") for cmd in (
            leaf.PoseTarget(command_id=side + "p", target_id="robot0/" + side + "_eef",
                pose=leaf.Pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.), frame_id="world")),
            leaf.GripperCommand(command_id=side + "g", target_id="robot0/" + side + "_grip", mode="open_fraction", value=.5)))
        result = bench.advance(c.Action(commands=commands), self.context, leaf.Budgets(env_steps=1))
        self.assertEqual(result.execution_feedback.status, "execution_failed")
        self.assertEqual(result.execution_feedback.code, "planning_failed")
        from embodied_eval.codec import decode
        self.assertEqual(decode(leaf.ExecutionFeedback, result.execution_feedback), result.execution_feedback)
        self.assertNotIn("rightp", result.execution_feedback.command_ids)
        self.assertEqual(result.execution_feedback.consumed_physics_steps, 10)

    def test_robotwin_measured_joint_projection_never_uses_drive_targets(self):
        simulator = NativeSimulator.__new__(NativeSimulator)
        simulator.family = "robotwin"
        simulator.space = native_space("robotwin", {"repo": "/absent", "protocol": "seeded-instruction", "task_name": "adjust_bottle"})
        simulator.runtime = SimpleNamespace(_scope=nullcontext, env=SimpleNamespace(robot=SimpleNamespace(
            get_left_arm_real_jointState=lambda: [1, 2, 3, 4, 5, 6, .2],
            get_right_arm_real_jointState=lambda: [10, 20, 30, 40, 50, 60, .8])))
        ext = {"captured_at_sim_time_s": 0., "joint_drive_targets_and_grippers": [0.] * 14}
        for side in ("left", "right"):
            ext[side + "_endpose_world_wxyz"] = [0., 0., 0., 1., 0., 0., 0.]
            ext[side + "_gripper_open_fraction"] = .5
        obs = leaf.Observation(context=leaf.Context(run_id="r", env_id="e", episode_id="ep", message_id="o"),
            observation_id="o", spec_ref="old", env_step=1, sim_time_s=.04, sensors={}, robots={},
            extensions={"robotwin.native_proprio.v1": ext})
        result = simulator._project(obs)
        self.assertEqual(result.states["left_joint_positions"].payload.data, [1., 2., 3., 4., 5., 6.])
        self.assertEqual(result.states["right_joint_positions"].provenance, "measurement")
        self.assertEqual(result.states["left_joint_positions"].captured_at_s, .04)
        self.assertEqual(result.states["joint_drive_targets"].provenance, "controller_target")
        self.assertEqual(result.states["joint_drive_targets"].captured_at_s, 0.)


if __name__ == "__main__":
    unittest.main()
