"""Interface/mapping regression only: no weights, remote APIs or GPU simulation.

Passing these tests is not native benchmark or stage-4 acceptance evidence.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tests.legacy_support.frozen import _legacy_contracts as c
from tests.legacy_support.frozen.adapters import openpi_libero_spec as pi, vlabench_spec as vla
from tests.legacy_support.frozen.adapters import openpi_vlabench_spec as pivla
from tests.legacy_support.frozen.compatibility import check_compatibility
from tests.legacy_support.config import (BenchmarkConfig, EpisodeConfig, ModelConfig, ProfileConfig,
                                  RunConfig, load_config, validate_config)
from embodied_eval.errors import AdapterError, ContractError
from tests.legacy_support.frozen.visibility import build_model_input

try:
    import numpy as np
except ImportError:
    np = None


ROOT = Path(__file__).resolve().parents[1]


class Stage4RegistrationTests(unittest.TestCase):
    def test_pi_vlabench_has_separate_protocol_and_checkpoint(self):
        config = load_config(ROOT / "examples/stage4-pi05-vlabench-select-fruit.json")
        spec, cap = pivla.space(.01), pivla.capability(config.model)
        self.assertFalse(check_compatibility(spec, cap, pivla.profile(spec, cap)).issues)
        self.assertEqual(spec.protocol_version, "openpi-native-1")
        self.assertIn("f658a8bec801", cap.checkpoint_ref)
        self.assertNotEqual(cap.normalization_ref, pi.capability(None).normalization_ref)
        with self.assertRaises(ContractError):
            validate_config(replace(config, benchmark=replace(config.benchmark, adapter="vlabench")))

    def test_pi_vlabench_manifest_has_only_inference_objects(self):
        import json
        manifest = json.loads((ROOT / "embodied_eval/adapters/pi05_vlabench_objects.json").read_text())
        self.assertEqual(len(manifest["objects"]), 19)
        self.assertEqual(sum(item["size"] for item in manifest["objects"]), manifest["expected_total_bytes"])
        self.assertTrue(all(item["relative"].startswith(("assets/", "params/")) for item in manifest["objects"]))

    def test_pi_config_and_contract_are_compatible(self):
        config = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
        spec, cap = pi.space(), pi.capability(config.model)
        self.assertFalse(check_compatibility(spec, cap, pi.profile(spec, cap)).issues)
        self.assertEqual(spec.protocol_version, "openpi-native-1")
        self.assertEqual(config.episodes[0].seed, 7)

    def test_vla_config_and_contract_are_compatible(self):
        config = load_config(ROOT / "examples/stage4-openwam-vlabench-select-fruit.json")
        spec, cap = vla.space(.01), vla.capability(config.model)
        self.assertFalse(check_compatibility(spec, cap, vla.profile(spec, cap)).issues)
        self.assertEqual(spec.control_period_s, .01)
        self.assertIn(vla.EXTENSION, spec.required_extensions)

    def test_pairs_cannot_be_crossed(self):
        config = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
        with self.assertRaises(ContractError):
            validate_config(replace(config, benchmark=BenchmarkConfig(adapter="vlabench", vlabench_repo="/native")))

    def test_real_adapters_reject_mock_faults(self):
        config = load_config(ROOT / "examples/stage4-openwam-vlabench-select-fruit.json")
        with self.assertRaises(ContractError):
            validate_config(replace(config, model=replace(config.model, fault="service_error")))

    def test_pi_budget_cannot_inherit_stage3_settings(self):
        config = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
        for ep in (replace(config.episodes[0], seed=42),
                   replace(config.episodes[0], budgets=c.Budgets(env_steps=600, model_calls=600, sim_time_s=30))):
            with self.assertRaises(ContractError):
                validate_config(replace(config, episodes=(ep,)))

    def test_vla_task_and_track_are_fixed(self):
        config = load_config(ROOT / "examples/stage4-openwam-vlabench-select-fruit.json")
        for benchmark in (replace(config.benchmark, task_name="select_toy"),
                          replace(config.benchmark, trial_index=1)):
            with self.assertRaises(ContractError):
                validate_config(replace(config, benchmark=benchmark))

    def test_missing_checkpoint_fails_before_adapter_creation(self):
        from tests.legacy_support.frozen import stage4_pairs
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ContractError):
                stage4_pairs._file(Path(root) / "missing", "0" * 64)

    def test_modified_asset_fails_closed(self):
        from tests.legacy_support.frozen import stage4_pairs
        with self.assertRaises(ContractError):
            stage4_pairs._file(ROOT / "examples/stage4-pi05-libero-spatial.json", "0" * 64)

    def test_vla_public_view_drops_scorer_state(self):
        spec, cap = vla.space(.01), vla.capability(None)
        context = c.Context(run_id="test", env_id="env0", episode_id="ep0", message_id="obs0")
        observation = c.Observation(context=context, observation_id="obs0", spec_ref=spec.spec_id,
            env_step=0, sim_time_s=0, sensors={}, robots={"robot0": c.RobotState()},
            extensions={vla.EXTENSION: {"ee_state_world_wxyz": [0, 0, 1, 1, 0, 0, 0, 1],
                        "robot_base_world_m": [0, -.4, .78], "instruction": "pick fruit"},
                        "private.scorer": {"target_entity": "secret"}})
        task = c.TaskSpec(task_id="fruit", protocol_id=spec.protocol_id, protocol_version=spec.protocol_version,
            episode_id="ep0", public=c.PublicTask(instruction="pick fruit"), initialization_ref="hidden",
            observation_policy_ref="public", budgets=c.Budgets(env_steps=300), evaluation_config_ref="scorer",
            evaluation_private={"secret": "hidden"})
        view = build_model_input(task, observation, vla.profile(spec, cap))
        self.assertEqual(set(view.observation_view.extensions), {vla.EXTENSION})
        self.assertEqual(view.observation_view.extensions[vla.EXTENSION]["instruction"], "pick fruit")


@unittest.skipIf(np is None, "real-client optional dependencies unavailable")
class Stage4MappingTests(unittest.TestCase):
    def test_pi_libero_preserves_trusted_native_numpy_initial_state_loading(self):
        from tests.legacy_support.frozen.adapters.libero_openpi_benchmark import LiberoOpenpiBenchmark
        benchmark = LiberoOpenpiBenchmark.__new__(LiberoOpenpiBenchmark)
        benchmark.config = BenchmarkConfig(libero_repo="/verified-source")
        benchmark.native_task = SimpleNamespace(problem_folder="libero_spatial", init_states_file="task.pruned_init")
        calls = []
        fake_torch = SimpleNamespace(load=lambda path, **kw: calls.append((path, kw)) or ["native-state"])
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(benchmark._load_init_states(), ["native-state"])
        self.assertEqual(calls, [(Path("/verified-source/libero/libero/init_files/libero_spatial/task.pruned_init"),
                                  {"weights_only": False})])

    def context(self):
        return c.Context(run_id="test", env_id="env0", episode_id="ep0", message_id="obs0")

    def pi_vla_model(self, reply=None):
        from tests.legacy_support.frozen.adapters.openpi_vlabench_model import OpenpiVLABenchModel
        model = OpenpiVLABenchModel(ModelConfig(), Path("/unused"), "/unused")
        space = pivla.space(.01)
        model.profile, model.period = pivla.profile(space, pivla.capability(None)), .01
        model.utils = SimpleNamespace(quaternion_to_euler=lambda quat: np.zeros(3))
        sensors = {key: c.SensorSample(captured_at_s=0, frame_id=key, validity="valid", provenance="sensor",
            payload=c.Tensor(dtype="uint8", shape=(1, 1, 3), layout="HWC", data=[[[index] * 3]]))
                   for key, index in zip(vla.SENSORS, vla.CAMERA_INDICES)}
        obs = c.Observation(context=self.context(), observation_id="obs0", spec_ref=space.spec_id,
            env_step=0, sim_time_s=0, sensors=sensors, robots={"robot0": c.RobotState()}, extensions={vla.EXTENSION: {
                "ee_state_world_wxyz": [.1, -.3, .98, 1, 0, 0, 0, 1],
                "robot_base_world_m": [10, 20, 30], "instruction": "current native text"}})
        model.latest = c.ModelInput(context=self.context(), task_public=c.PublicTask(instruction="initial text"),
                                   observation_view=obs)
        class Peer:
            calls = 0
            def infer(self, payload):
                self.calls += 1
                action = np.array([.2, .3, .4, .5, .6, .7, .1], np.float64)
                return {"actions": np.tile(action, (10, 1))} if reply is None else reply
        model.peer = Peer()
        return model

    def test_pi_vlabench_preserves_three_cameras_and_native_fixed_input_offset(self):
        model = self.pi_vla_model()
        payload = model.native_input(model.latest)
        self.assertEqual(set(payload), {"observation/image", "observation/second_image",
            "observation/wrist_image", "observation/state", "prompt"})
        self.assertEqual([payload[key][0, 0, 0] for key in ("observation/image", "observation/second_image",
            "observation/wrist_image")], [2, 0, 3])
        np.testing.assert_allclose(payload["observation/state"], [.1, .1, .2, 0, 0, 0, 1])
        self.assertEqual(payload["prompt"], "current native text")

    def test_pi_vlabench_cache_and_live_world_output_offset(self):
        model = self.pi_vla_model()
        for tick in range(6):
            result = model.predict(replace(self.context(), message_id=f"request{tick}"))
            parameters = result.decision.steps[0].commands[0].parameters
            self.assertEqual(result.decision.steps[0].duration_s, .01)
            np.testing.assert_allclose(parameters["position_world_m"], [10.2, 20.3, 30.4])
            np.testing.assert_allclose(parameters["euler_xyz_rad"], [.5, .6, .7])
            self.assertEqual(parameters["finger_targets_m"], [.04, .04])
            self.assertIsNone(result.diagnostics.model_forward_count)
        self.assertEqual(model.peer.calls, 2)
        model.end_episode(self.context())
        self.assertFalse(model.cache)
        self.assertIsNone(model.latest)

    def test_pi_vlabench_gripper_threshold_is_not_libero_sign_conversion(self):
        actions = np.zeros((5, 7))
        actions[:, -1] = .09999
        model = self.pi_vla_model({"actions": actions})
        result = model.predict(replace(self.context(), message_id="request0"))
        self.assertEqual(result.decision.steps[0].commands[0].parameters["finger_targets_m"], [0., 0.])

    def test_pi_vlabench_world_offset_preserves_native_float32_inplace_rounding(self):
        row = np.array([.12345678, -.3456789, .56789123, .4, .5, .6, .2], dtype=np.float32)
        actions = np.tile(row, (5, 1))
        model = self.pi_vla_model({"actions": actions})
        base = np.asarray(model.latest.observation_view.extensions[vla.EXTENSION]["robot_base_world_m"])
        expected = row[:3].copy()
        expected += base  # Exact fixed examples/vlabench/eval.py Pi0.predict sequence.
        self.assertFalse(np.array_equal(expected.astype(np.float64), row[:3] + base))
        prediction = model.predict(replace(self.context(), message_id="request0"))
        actual = prediction.decision.steps[0].commands[0].parameters["position_world_m"]
        np.testing.assert_array_equal(actual, expected.astype(np.float64))
        np.testing.assert_array_equal(actions[0], row)
        np.testing.assert_array_equal(model.cache[0], row)

    def test_pi_vlabench_world_offset_keeps_native_float64_precision(self):
        row = np.array([.123456789012, -.345678901234, .567891234567, .4, .5, .6, .2], dtype=np.float64)
        model = self.pi_vla_model({"actions": np.tile(row, (5, 1))})
        expected = row[:3].copy()
        expected += np.asarray(model.latest.observation_view.extensions[vla.EXTENSION]["robot_base_world_m"])
        prediction = model.predict(replace(self.context(), message_id="request0"))
        actual = prediction.decision.steps[0].commands[0].parameters["position_world_m"]
        np.testing.assert_array_equal(actual, expected)

    def test_pi_vlabench_rejects_bad_chunks_without_retry(self):
        for actions in (np.zeros((4, 7)), np.zeros((5, 8)), np.full((5, 7), np.nan), np.full((5, 7), 1+1j)):
            model = self.pi_vla_model({"actions": actions})
            with self.assertRaises(AdapterError):
                model.predict(replace(self.context(), message_id="request0"))
            self.assertEqual(model.peer.calls, 1)
            self.assertFalse(model.cache)

    def pi_input(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        sample = c.SensorSample(captured_at_s=0, frame_id="world", validity="valid", provenance="sensor",
            payload=c.Tensor(dtype="uint8", shape=image.shape, layout="HWC", data=image.tolist()))
        obs = c.Observation(context=self.context(), observation_id="obs0", spec_ref=pi.space().spec_id,
            env_step=0, sim_time_s=0, robots={"robot0": c.RobotState()},
            sensors={"agentview_image": sample, "robot0_eye_in_hand_image": sample}, extensions={pi.EXTENSION: {
                "eef_position_m": [0.123456789012, 0, 1], "eef_quaternion_xyzw": [0, 0, 0, 1],
                "gripper_qpos_m": [.02, .02]}})
        return c.ModelInput(context=self.context(), task_public=c.PublicTask(instruction="pick bowl"), observation_view=obs)

    def model(self, reply=None):
        from tests.legacy_support.frozen.adapters.openpi_libero_model import OpenpiLiberoModel
        model = OpenpiLiberoModel(ModelConfig(), Path("/unused"))
        model.profile = pi.profile(pi.space(), pi.capability(None))
        model.period = .05
        model.latest = self.pi_input()
        model.images = SimpleNamespace(resize_with_pad=lambda image, h, w: image,
                                       convert_to_uint8=lambda image: image)
        class Peer:
            calls = 0
            def infer(self, payload):
                self.calls += 1
                return {"actions": np.arange(70, dtype=np.float32).reshape(10, 7)} if reply is None else reply
        model.peer = Peer()
        return model

    def test_pi_native_rotation_state_and_prompt(self):
        model = self.model()
        payload = model.native_input(model.latest)
        np.testing.assert_array_equal(payload["observation/image"], np.arange(12).reshape(2, 2, 3)[::-1, ::-1])
        self.assertEqual(payload["observation/state"].shape, (8,))
        self.assertEqual(payload["observation/state"][0], .123456789012)
        self.assertEqual(payload["prompt"], "pick bowl")

    def test_pi_replans_every_five_actions_and_keeps_unclipped_output(self):
        model = self.model()
        actions = []
        for step in range(6):
            result = model.predict(replace(self.context(), message_id=f"request{step}"))
            actions.append(result.decision.steps[0].commands[0].parameters["osc_action7"])
            self.assertIsNone(result.diagnostics.model_forward_count)
        self.assertEqual(model.peer.calls, 2)
        self.assertEqual(actions[4][-1], 34.)
        self.assertEqual(actions[5], actions[0])

    def test_pi_episode_end_clears_cache_and_observation(self):
        model = self.model()
        model.predict(replace(self.context(), message_id="request0"))
        self.assertEqual(len(model.cache), 4)
        model.end_episode(self.context())
        self.assertEqual(len(model.cache), 0)
        self.assertIsNone(model.latest)

    def test_pi_rejects_bad_chunk_without_caching(self):
        for actions in (np.zeros((4, 7)), np.zeros((5, 8)), np.full((5, 7), np.nan), np.full((5, 7), 1+1j)):
            model = self.model({"actions": actions})
            with self.assertRaises(AdapterError):
                model.predict(replace(self.context(), message_id="request0"))
            self.assertEqual(len(model.cache), 0)

    def test_finite_array_rejects_string_and_complex_payloads(self):
        from tests.legacy_support.frozen.adapters.native_common import finite_array
        for value in (["1", "2"], [1+1j, 2+0j], [float("nan"), 1]):
            with self.assertRaises(AdapterError):
                finite_array(value, (2,), "state", np.float32)

    def test_pi_quaternion_identity_and_half_turn(self):
        from tests.legacy_support.frozen.adapters.openpi_libero_model import quat_to_axisangle
        np.testing.assert_array_equal(quat_to_axisangle([0, 0, 0, 1]), np.zeros(3))
        np.testing.assert_allclose(quat_to_axisangle([1., 0, 0, 0]), [np.pi, 0, 0])

    def test_vla_maps_only_public_native_inputs_and_measured_period(self):
        from tests.legacy_support.frozen.adapters.openwam_vlabench_model import OpenWAMVLABenchModel
        model = OpenWAMVLABenchModel(ModelConfig(), Path("/unused"))
        declared = vla.space(.01)
        model.profile, model.period = vla.profile(declared, vla.capability(None)), .01
        class Policy:
            def predict(self, observation):
                assert set(observation) == {"rgb", "ee_state", "robot_frame", "instruction"}
                assert observation["instruction"] == "current native instruction"
                assert observation["rgb"][:, 0, 0, 0].tolist() == [2, 3, 0]
                return [.1, .2, .3], [.4, .5, .6], [.04, .04]
        model.policy = Policy()
        sensors = {key: c.SensorSample(captured_at_s=0, frame_id=key, validity="valid", provenance="sensor",
            payload=c.Tensor(dtype="uint8", shape=(1, 1, 3), layout="HWC", data=[[[index] * 3]]))
                   for key, index in zip(vla.SENSORS, vla.CAMERA_INDICES)}
        obs = c.Observation(context=self.context(), observation_id="obs0", spec_ref=declared.spec_id,
            env_step=0, sim_time_s=0, sensors=sensors, robots={"robot0": c.RobotState()}, extensions={vla.EXTENSION: {
                "ee_state_world_wxyz": [0, 0, 1, 1, 0, 0, 0, 1], "robot_base_world_m": [0, -.4, .78],
                "instruction": "current native instruction"}, "private": {"score": 1}})
        model.latest = c.ModelInput(context=self.context(), task_public=c.PublicTask(instruction="initial text"), observation_view=obs)
        result = model.predict(replace(self.context(), message_id="request0"))
        self.assertEqual(result.decision.steps[0].duration_s, .01)
        self.assertEqual(result.decision.steps[0].commands[0].parameters["finger_targets_m"], [.04, .04])

    def benchmark(self, *, elapsed=.01, done=True):
        from tests.legacy_support.frozen.adapters.vlabench_benchmark import VLABenchBenchmark
        class Environment:
            physics = SimpleNamespace(clock=0)
            physics.time = lambda: Environment.physics.clock
            robot = SimpleNamespace(get_qpos_from_ee_pos=lambda **kwargs: (False, np.zeros(7)))
            def step(self, action):
                assert action.shape == (9,)
                self.physics.clock += elapsed
                return SimpleNamespace(last=lambda: done)
            def get_observation(self, **kwargs):
                return {"rgb": np.zeros((4, 480, 480, 3), np.uint8), "ee_state": [0, 0, 1, 1, 0, 0, 0, 1]}
            def get_robot_frame_position(self):
                return [0, -.4, .78]
            def get_task_progress(self):
                return .5
            def get_intention_score(self, **kwargs):
                raise KeyError("native missing entity")
        benchmark = VLABenchBenchmark(BenchmarkConfig(vlabench_repo="/unused"))
        benchmark.env, benchmark.context, benchmark.space = Environment(), self.context(), vla.space(.01)
        benchmark.last_instruction = "pick fruit"
        command = c.ControllerCommand(command_id="cmd0", target_id="robot0/eef", controller_profile_ref=vla.CONTROLLER,
            parameters={"position_world_m": [0, 0, 1], "euler_xyz_rad": [0, 0, 0], "finger_targets_m": [.04, .04]})
        return benchmark, c.ActionStep(index=0, duration_s=.01, commands=(command,))

    def test_vla_native_step_and_scoring_keep_unavailable_metric_null(self):
        benchmark, action = self.benchmark()
        utilities = SimpleNamespace(euler_to_quaternion=lambda *values: [1, 0, 0, 0])
        with patch("tests.legacy_support.frozen.adapters.vlabench_benchmark.native_module", return_value=utilities):
            result = benchmark.advance(action, replace(self.context(), message_id="request0"), c.Budgets(env_steps=1))
        self.assertTrue(result.terminated)
        self.assertEqual(result.execution_feedback.consumed_env_steps, 1)
        self.assertEqual(result.execution_feedback.elapsed_sim_time_s, .01)
        score = benchmark.score(self.context())
        self.assertTrue(score["success"].value)
        self.assertEqual(score["progress_score"].value, .5)
        self.assertIsNone(score["intention_score"].value)
        self.assertFalse(score["intention_score"].valid)

    def test_vla_unexpected_native_time_is_not_reported_as_valid_execution(self):
        benchmark, action = self.benchmark(elapsed=.02)
        utilities = SimpleNamespace(euler_to_quaternion=lambda *values: [1, 0, 0, 0])
        with patch("tests.legacy_support.frozen.adapters.vlabench_benchmark.native_module", return_value=utilities):
            with self.assertRaises(AdapterError) as caught:
                benchmark.advance(action, replace(self.context(), message_id="request0"), c.Budgets(env_steps=1))
        self.assertTrue(caught.exception.execution_started)
        self.assertFalse(caught.exception.state_known)

    def test_vla_checks_budget_before_physics(self):
        benchmark, action = self.benchmark()
        with self.assertRaises(AdapterError):
            benchmark.advance(action, replace(self.context(), message_id="request0"), c.Budgets(env_steps=0))
        self.assertEqual(benchmark.env.physics.clock, 0)


@unittest.skipIf(np is None, "real-client optional dependencies unavailable")
class NativeRobotwinBridgeTests(unittest.TestCase):
    def observation(self):
        return {"observation": {key: {"rgb": np.zeros((2, 2, 3), np.uint8)}
                                for key in ("head_camera", "left_camera", "right_camera")},
                "joint_action": {"vector": np.arange(14, dtype=np.float32)},
                "endpose": {"left_endpose": [0, 0, 1, 0, 0, 0, 1],
                            "right_endpose": [0, 0, 1, 0, 0, 0, 1], "left_gripper": 1, "right_gripper": 0},
                "private_score": 1}

    def lingbot(self, action=None):
        from tests.legacy_support.frozen.adapters.native_robotwin_models import LingbotRobotwinNativePolicy, NativeEndpoint
        model = LingbotRobotwinNativePolicy(NativeEndpoint(repo="/unused", port=8923))
        model.peer = SimpleNamespace(infer=lambda payload: {"action": np.zeros((4, 14)) if action is None else action})
        return model

    def test_lingbot_uses_exact_native_keys_without_private_fields(self):
        model = self.lingbot()
        payload = model.native_input(self.observation(), "adjust bottle")
        self.assertEqual(set(payload), {"observation.images.cam_high", "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist", "observation.state", "task"})

    def test_lingbot_preserves_full_chunk_and_is_not_registered(self):
        model = self.lingbot()
        self.assertEqual(model.predict(self.observation(), "adjust bottle").shape, (4, 14))
        self.assertFalse(model.diagnostics()["framework_registered"])

    def test_lingbot_normalizes_one_action_to_a_single_row(self):
        model = self.lingbot(np.zeros(14, dtype=np.float32))
        self.assertEqual(model.predict(self.observation(), "adjust bottle").shape, (1, 14))

    def test_lingbot_rejects_malformed_or_nonfinite_chunk(self):
        for action in (np.zeros((0, 14)), np.zeros((1, 16)), np.full((2, 14), np.nan)):
            with self.assertRaises(AdapterError):
                self.lingbot(action).predict(self.observation(), "adjust bottle")

    def test_lingbot_reset_does_not_select_new_checkpoint(self):
        model = self.lingbot()
        sent = []
        model.peer = SimpleNamespace(infer=lambda payload: sent.append(payload))
        model.reset()
        self.assertEqual(sent, [{"reset": True, "robo_name": "robotwin", "path_to_pi_model": None}])

    def test_openwam_uses_pinned_conversion_and_prompt_without_execution(self):
        from tests.legacy_support.frozen.adapters.native_robotwin_models import OpenWAMRobotwinNativePolicy, NativeEndpoint
        from tests.legacy_support.frozen.adapters.native_common import native_module
        source = ROOT.parent / "upstream/OpenWAM"
        if not source.exists():
            self.skipTest("OpenWAM native reference unavailable")
        model = OpenWAMRobotwinNativePolicy(NativeEndpoint(repo=str(source), port=8924))
        try:
            model.client = native_module(source, "benchmarks.utils.client")
            model.conversion = native_module(source, "benchmarks.utils.action_conversion")
            model.prompt = native_module(source, "benchmarks.robotwin.prompt_template")
        except ImportError:
            self.skipTest("native client optional dependencies unavailable")
        raw = np.asarray([0, 0, 1, 1, 0, 0, 0, 1, 0, 1] * 2, np.float32)
        model.peer = SimpleNamespace(predict_once=lambda payload: {"action": raw})
        model.last_instruction = "adjust bottle"
        payload = model.native_input(self.observation(), "adjust bottle")
        self.assertEqual(len(payload["state"]), 20)
        self.assertEqual(payload["prompt"], model.prompt.format_prompt_for_inference("adjust bottle"))
        np.testing.assert_array_equal(model.predict(self.observation(), "adjust bottle"),
                                      model.conversion.eef20d_to_ee16d(raw))
        self.assertFalse(model.diagnostics()["framework_registered"])


if __name__ == "__main__":
    unittest.main()
