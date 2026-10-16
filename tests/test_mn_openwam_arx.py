"""Synthetic independent-profile tests; no real weights or robot execution."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from embodied_eval import _legacy_contracts as leaf
from embodied_eval.errors import ContractError
from embodied_eval.mn import contracts as c
from embodied_eval.mn.native_catalog import capability, space as native_space
from embodied_eval.mn.openwam_arx import OpenWAMArxModel
from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator
from embodied_eval.mn.binding import check_compatibility, resolve


class OpenWAMArxTests(unittest.TestCase):
    def setUp(self):
        self.opts = {"repo": "/absent", "checkpoint_dir": "/absent", "io_profile": "eef20_arx_absolute"}
        self.context = c.Context(run_id="r", env_id="e", episode_id="ep", message_id="request")
        self.bench = RoboDojoSimulator({"repo": "/absent"})
        self.model = OpenWAMArxModel(self.opts)
        self.model.space = self.bench.space
        self.model.profile = resolve(self.bench.space, self.model.cap, self.binding())

    def binding(self, robotwin=False):
        inputs = {"instruction": "instruction", "head": "head_camera" if robotwin else "cam_head",
            "left_wrist": "left_camera" if robotwin else "cam_left_wrist", "right_wrist": "right_camera" if robotwin else "cam_right_wrist"}
        for side in ("left", "right"):
            inputs.update({side + "_position": side + "_eef_position", side + "_quaternion": side + "_eef_quaternion",
                side + "_grip": side + ("_gripper_open_fraction" if robotwin else "_gripper_target")})
        return c.BindingConfig(inputs=inputs, targets={key: "robot0/" + key for key in self.model.cap.output.target_slots},
            action_spec_ref="dual-cartesian-absolute@1" if robotwin else "dual-pose-absolute@1")

    def row(self):
        return {side + suffix: np.asarray(values, np.float32) for side in ("left", "right") for suffix, values in (
            ("_ee_pose", [.125, .25, .5, 0., 0., 0., 1.]), ("_ee_joint_state", [.5]))}

    def test_source_defined_absolute_profile_composes_two_dual_benchmarks(self):
        rw = native_space("robotwin", {"repo": "/absent", "protocol": "seeded-instruction", "task_name": "adjust_bottle"})
        report = check_compatibility(rw, self.model.cap, self.binding(True))
        self.assertTrue(report.compatible)
        self.assertTrue(report.warnings)

    def test_new_physical_profile_does_not_relabel_old_ambiguous_native_array(self):
        old = capability("openwam", {**self.opts, "io_profile": "eef20_dual_legacy"})
        self.assertIsNotNone(old.output.native_protocol)
        self.assertIsNone(self.model.cap.output.native_protocol)
        self.assertEqual(self.model.cap.output.command_kinds, ("pose_target", "gripper_command"))

    def test_native_pose_to_public_and_executor_roundtrip_preserves_dtype(self):
        expected = self.row()
        action = self.model.decode_action(expected, self.context)
        self.assertEqual(action.commands[0].pose.quaternion_xyzw, (0., 0., 1., 0.))
        self.assertEqual(action.native_dtype, "float32")
        actual = self.bench.native_action(action)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
            self.assertEqual(actual[key].dtype, expected[key].dtype)

    def test_missing_second_pose_binding_is_rejected(self):
        binding = self.binding()
        binding = replace(binding, targets={key: value for key, value in binding.targets.items() if key != "right_eef"})
        self.assertFalse(check_compatibility(self.bench.space, self.model.cap, binding).compatible)

    def test_episode_boundary_resets_local_cache_not_native_rng(self):
        self.model._connect = lambda: None
        self.model.cache.append(self.row())
        self.model.begin_episode(self.context, leaf.PublicTask(instruction="one"), self.model.profile, self.bench.space)
        self.assertFalse(self.model.cache)
        self.assertIsNone(self.model.latest)

    def fake_engine(self, actions):
        self.model.latest = SimpleNamespace(observation_id="obs")
        self.model.native_input = lambda view: {"state": [0.] * 20}
        self.model.native_rows = lambda chunk: [self.row() for row in chunk]
        self.model.horizon = None
        calls = []
        self.model.server = SimpleNamespace(_obs_preprocessor=SimpleNamespace(preprocess=lambda payload: {
            "image": "synthetic", "prompt": "synthetic", "state": payload["state"]}),
            engine=SimpleNamespace(generate_batch=lambda data: calls.append(data) or {"actions": actions}),
            _policy=SimpleNamespace(_project_binary_dims=lambda data: data))
        return calls

    def test_chunk_cache_is_model_owned_and_forward_count_stays_unknown(self):
        calls = self.fake_engine(np.zeros((1, 3, 20), np.float32))
        for i in range(3):
            prediction = self.model.predict(replace(self.context, message_id=f"r{i}"))
            self.assertIsNone(prediction.model_forward_count)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.model.requests, 1)
        self.assertFalse(self.model.cache)

    def test_wrong_native_output_dimension_is_rejected(self):
        self.fake_engine(np.zeros((1, 3, 80), np.float32))
        with self.assertRaises(ContractError):
            self.model.predict(self.context)

    def test_end_episode_discards_unused_native_actions(self):
        self.model.cache.append(self.row())
        self.model.end_episode(self.context)
        self.assertFalse(self.model.cache)


if __name__ == "__main__":
    unittest.main()
