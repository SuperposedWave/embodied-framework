"""Synthetic boundary tests, not real policy inference or simulator scores."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from embodied_eval import _legacy_contracts as leaf
from embodied_eval.errors import ContractError, AdapterError
from embodied_eval.mn import contracts as c
from embodied_eval.mn.additional_catalog import POLICIES, policy_capability, policy_options
from embodied_eval.mn.additional_models import JointPolicyModel
from embodied_eval.mn.binding import resolve, check_compatibility
from embodied_eval.mn.droid_simulator import DroidSimulator, space, metrics
from embodied_eval.mn.validation import validate_space


class AdditionalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.context = c.Context(run_id="r", env_id="e", episode_id="episode", message_id="request")
        self.environment_options = {"repo": "/absent", "assets_dir": "/absent"}
        self.space = space(self.environment_options)

    def opts(self, family):
        return {"repo": "/absent", "codec_repo": "/absent", "checkpoint_dir": "/absent",
                "client_repo": "/absent", "io_profile": POLICIES[family]["profile"]}

    def binding(self):
        return c.BindingConfig(inputs={"exterior_right": "external_cam", "exterior_left": "external_cam_2",
            "wrist": "wrist_cam", "joints": "arm_joint_positions", "grip": "gripper_open_fraction", "instruction": "instruction"},
            targets={**{f"j{i}": f"robot0/j{i}" for i in range(7)}, "grip": "robot0/grip"},
            action_spec_ref="joint7-binary-gripper@1")

    def model(self, family):
        model = JointPolicyModel(family, self.opts(family))
        model.profile = resolve(self.space, model.cap, self.binding())
        model.space = self.space
        model.session_id = "synthetic-session"
        return model

    def view(self):
        images = {name: leaf.SensorSample(frame_id="world", captured_at_s=0., validity="valid", provenance="sensor",
            payload=leaf.Tensor(dtype="uint8", shape=(2, 3, 3), layout="HWC", data=np.full((2, 3, 3), i, np.uint8).tolist()))
            for i, name in enumerate(("exterior_right", "exterior_left", "wrist"))}
        states = {name: c.StateSample(captured_at_s=0., payload=leaf.Tensor(dtype="float64", shape=array.shape,
            layout="vector", data=array.tolist())) for name, array in (("joints", np.arange(7, dtype=np.float64)), ("grip", np.array([.2])))}
        return c.ModelInput(context=self.context, observation_id="obs", sim_time_s=0.,
            task_public=leaf.PublicTask(instruction="synthetic instruction"), sensors=images, states=states)

    def test_two_independent_models_compose_with_named_droid_protocol(self):
        validate_space(self.space)
        for family in POLICIES:
            with self.subTest(family=family):
                self.assertFalse(check_compatibility(self.space, policy_capability(family, self.opts(family)), self.binding()).issues)
        from docs.handoff.prepare_mn_extension import configurations
        examples = configurations()
        # Geometry-named Cosmos views and legacy-named DreamZero slots differ:
        # data wiring follows each reviewed native client, without env branches.
        self.assertEqual(examples["cosmos3_droid_sim"].binding.inputs["exterior_right"], "external_cam_2")
        self.assertEqual(examples["dreamzero_droid_sim"].binding.inputs["exterior_right"], "external_cam")

    def test_missing_required_camera_is_not_filled_black(self):
        binding = self.binding()
        binding = replace(binding, inputs={k: v for k, v in binding.inputs.items() if k != "wrist"})
        self.assertTrue(check_compatibility(self.space, policy_capability("dreamzero", self.opts("dreamzero")), binding).issues)

    def test_missing_joint_not_silently_discarded(self):
        binding = replace(self.binding(), targets={k: v for k, v in self.binding().targets.items() if k != "j6"})
        self.assertTrue(check_compatibility(self.space, policy_capability("cosmos3", self.opts("cosmos3")), binding).issues)

    def test_joint_drive_targets_do_not_satisfy_measured_input(self):
        states = dict(self.space.state_fields)
        states["arm_joint_positions"] = replace(states["arm_joint_positions"], semantic="joint_drive_target")
        report = check_compatibility(replace(self.space, state_fields=states), policy_capability("dreamzero", self.opts("dreamzero")), self.binding())
        self.assertTrue(report.issues)

    def test_options_reject_family_ambiguity(self):
        options = self.opts("cosmos3")
        options["io_profile"] = "nano-reasoner"
        with self.assertRaises(ContractError):
            policy_options("cosmos3", options)

    def test_options_reject_oversized_horizon(self):
        for family in POLICIES:
            with self.assertRaises(ContractError):
                policy_options(family, {**self.opts(family), "open_loop_horizon": POLICIES[family]["horizon"] + 1})

    def test_source_binary_threshold_and_seven_joint_values(self):
        for family in POLICIES:
            model = self.model(family)
            for value, opening in ((-.1, 1.), (.5, 1.), (.50001, 0.), (1.1, 0.)):
                row = np.array([0., .1, .2, .3, .4, .5, .6, value], np.float32)
                action = model.decode_action(row, self.context)
                self.assertEqual([cmd.value for cmd in action.commands[:-1]], row[:-1].tolist())
                self.assertEqual(action.commands[-1].value, opening)
                self.assertEqual(action.native_dtype, "float64" if family == "dreamzero" else "float32")

    def test_nonfinite_action_is_rejected(self):
        row = np.zeros(8, np.float32)
        row[3] = np.nan
        with self.assertRaises(AdapterError):
            self.model("dreamzero").decode_action(row, self.context)

    def test_droid_executor_preserves_native_joint_values_and_binary_sign(self):
        env = DroidSimulator.__new__(DroidSimulator)
        env.space = self.space
        row = np.array([.1, .2, .3, .4, .5, .6, .7, .9], np.float32)
        model = self.model("dreamzero")
        action = model.decode_action(row, self.context)
        actual = env.native_action(action)
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual[:7], row[:7])
        self.assertEqual(actual[-1], 1.)
        action = replace(action, commands=action.commands[:-1] + (replace(action.commands[-1], value=1.),))
        self.assertEqual(env.native_action(action)[-1], 0.)

    def test_executor_rejects_missing_joint(self):
        env = DroidSimulator.__new__(DroidSimulator)
        env.space = self.space
        action = self.model("dreamzero").decode_action(np.zeros(8), self.context)
        with self.assertRaises(ContractError):
            env.native_action(replace(action, commands=action.commands[:-2] + action.commands[-1:]))

    def test_cache_and_short_chunk_rejection(self):
        model = self.model("dreamzero")
        model.horizon = 2
        model.latest = self.view()
        sent = []
        model.native_input = lambda view: {"prompt": view.task_public.instruction}
        model.peer = SimpleNamespace(infer=lambda value: sent.append(value) or {"actions": np.arange(24, dtype=np.float32).reshape(3, 8)})
        model.predict(self.context)
        model.predict(replace(self.context, message_id="request2"))
        self.assertEqual(len(sent), 1)
        self.assertEqual(model.requests, 1)
        self.assertFalse(model.cache)
        model.predict(replace(self.context, message_id="request3"))
        self.assertEqual(len(sent), 2)
        model.end_episode(self.context)
        self.assertFalse(model.cache)
        self.assertIsNone(model.latest)
        model.latest = self.view()
        model.peer.infer = lambda value: {"actions": np.zeros((1, 8), np.float32)}
        with self.assertRaises(ContractError):
            model.predict(self.context)

    def test_fresh_session_per_episode_no_fake_reset_inference(self):
        from unittest.mock import patch
        model = self.model("dreamzero")
        sent = []
        model.peer = SimpleNamespace(infer=lambda data: sent.append(data))
        with patch("embodied_eval.mn.additional_models.native_module", return_value=SimpleNamespace()):
            model.begin_episode(self.context, leaf.PublicTask(instruction="one"), model.profile, model.space)
            first = model.session_id
            model.begin_episode(self.context, leaf.PublicTask(instruction="two"), model.profile, model.space)
        self.assertNotEqual(first, model.session_id)
        self.assertFalse(sent)

    def test_dreamzero_wire_image_order_dtype_and_gripper_sign(self):
        model = self.model("dreamzero")
        calls = []
        model.images = SimpleNamespace(resize_with_pad=lambda data, h, w: calls.append((int(data[0, 0, 0]), h, w)) or data)
        request = model.native_input(self.view())
        self.assertEqual(calls, [(0, 180, 320), (1, 180, 320), (2, 180, 320)])
        self.assertEqual(request["endpoint"], "infer")
        self.assertEqual(request["observation/gripper_position"].dtype, np.float64)
        self.assertAlmostEqual(request["observation/gripper_position"][0], .8)
        np.testing.assert_array_equal(request["observation/cartesian_position"], np.zeros(6))

    def test_no_official_scorer_does_not_become_success_or_failure(self):
        metric = metrics(self.environment_options)["success"]
        self.assertIsNone(metric.value)
        self.assertFalse(metric.valid)
        self.assertEqual(metric.unavailable_reason, "official_task_success_scorer_not_defined")


if __name__ == "__main__":
    unittest.main()
