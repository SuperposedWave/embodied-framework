"""Synthetic checks of the diagnostic; not real simulator evidence."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/handoff"))
from probe_real_observation_copy import compare_snapshot
from embodied_eval.mn import contracts as c
from embodied_eval.mn.droid_simulator import DroidSimulator, space as droid_space
from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator, space as dojo_space


class RealObservationProjectionEvidence(unittest.TestCase):
    def droid(self):
        native = DroidSimulator.__new__(DroidSimulator)
        native.cfg = SimpleNamespace(scene=1)
        native.space = droid_space({"repo": "synthetic-source", "assets_dir": "synthetic-assets"})
        native.sim_time, native.steps = .25, 3
        raw = {"policy": {name: np.full((1, 720, 1280, 3), index + 1, np.uint8)
            for index, name in enumerate(("external_cam", "external_cam_2", "wrist_cam"))}}
        raw["policy"].update(arm_joint_pos=np.arange(7, dtype=np.float32), gripper_pos=np.array([.25], np.float32))
        context = c.Context(run_id="synthetic", env_id="e", episode_id="ep", message_id="o")
        return raw, native._snapshot(raw, context)

    def dojo(self):
        native = RoboDojoSimulator.__new__(RoboDojoSimulator)
        native.space = dojo_space({"repo": "synthetic-source"})
        native.sim_time, native.steps = .16, 4
        raw = {"vision": {name: {"color": np.full((3, 4, 3), index, np.uint8)} for index, name in enumerate(
            ("cam_head", "cam_left_wrist", "cam_right_wrist"))}, "state": {}, "instruction": "synthetic public instruction"}
        for side in ("left", "right"):
            raw["state"].update({side + "_arm_joint_state": np.arange(6, dtype=np.float32),
                side + "_ee_pose": np.array([1., 2., 3., 1., 0., 0., 0.]), side + "_ee_joint_state": np.array([.5])})
        native.env = SimpleNamespace(get_obs=lambda: raw)
        context = c.Context(run_id="synthetic", env_id="e", episode_id="ep", message_id="o")
        return raw, native._snapshot(context)

    def test_droid_three_images_measured_states_inversion_and_timestamps_exact(self):
        raw, public = self.droid()
        _, result = compare_snapshot(raw, public, "droid_sim")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual(public.states["gripper_open_fraction"].payload.data, [.75])

    def test_required_camera_cannot_be_silently_omitted(self):
        raw, public = self.dojo()
        sensors = dict(public.sensors)
        sensors.pop("cam_right_wrist")
        with self.assertRaises(KeyError):
            compare_snapshot(raw, replace(public, sensors=sensors), "robodojo")

    def test_dojo_both_arms_pose_order_and_target_provenance_exact(self):
        raw, public = self.dojo()
        _, result = compare_snapshot(raw, public, "robodojo")
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["states"]), 8)
        states = dict(public.states)
        states["right_gripper_target"] = replace(states["right_gripper_target"], provenance="measurement")
        self.assertFalse(compare_snapshot(raw, replace(public, states=states), "robodojo")[1]["passed"])

    def test_pixel_mutation_or_float_image_dtype_fails_strictly(self):
        raw, public = self.dojo()
        for change_type in (False, True):
            sensors = dict(public.sensors)
            image = sensors["cam_head"]
            data = np.asarray(image.payload.data).copy()
            data[0, 0, 0] += 1
            sensors["cam_head"] = replace(image, payload=replace(image.payload,
                data=data.tolist(), dtype="float32" if change_type else "uint8"))
            self.assertFalse(compare_snapshot(raw, replace(public, sensors=sensors), "robodojo")[1]["passed"])


if __name__ == "__main__":
    unittest.main()
