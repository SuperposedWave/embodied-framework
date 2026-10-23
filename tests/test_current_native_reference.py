"""Synthetic checks of comparison-only naming, never inference evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "docs/handoff/compare_current_native.py"
spec = importlib.util.spec_from_file_location("current_native_reference", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CurrentReferenceTest(unittest.TestCase):
    def project(self, values, images, version="0.3", **kwargs):
        return module.current_snapshot(lambda obs, state, rgb, at, state_at: (state, rgb, at),
            {"contract_version": version}, values, images, **kwargs)

    def test_libero_native_names(self):
        states, images, _ = self.project({"gripper_positions": [0., .02]},
            {"agentview_image": "pixels", "robot0_eye_in_hand_image": "wristpixels"})
        self.assertEqual(states, {"gripper_joint_positions": [0., .02]})
        self.assertEqual(images, {"head": "pixels", "wrist": "wristpixels"})

    def test_vlabench_three_views(self):
        _, images, _ = self.project({}, {"forward_rgb": 1, "franka_wrist_rgb": 2, "right_rgb": 3})
        self.assertEqual(images, {"head": 1, "wrist": 2, "external_right": 3})

    def test_both_arms_and_grippers_preserved_without_mutation(self):
        values = {"joint_drive_targets": list(range(14)), "left_gripper_open_fraction": [6],
                  "right_gripper_open_fraction": [13], "left_joint_positions": [20] * 6,
                  "right_joint_positions": [30] * 6}
        before = copy.deepcopy(values)
        states, _, _ = self.project(values, {"head_camera": 1, "left_camera": 2, "right_camera": 3})
        self.assertEqual(values, before)
        self.assertEqual(states["left_joint_drive_targets"], list(range(6)))
        self.assertEqual(states["right_joint_drive_targets"], list(range(7, 13)))
        self.assertEqual(states["right_gripper_open_fraction_target"], [13])
        self.assertEqual(states["left_joint_positions"], [20] * 6)
        self.assertEqual(states["right_joint_positions"], [30] * 6)

    def test_incomplete_drive_vector_rejected(self):
        with self.assertRaises(ValueError):
            self.project({"joint_drive_targets": [0] * 7}, {})

    def test_mixed_separate_gripper_conflict_rejected(self):
        with self.assertRaises(ValueError):
            self.project({"joint_drive_targets": [0] * 14, "left_gripper_open_fraction": [1]}, {})

    def test_unknown_camera_not_guessed(self):
        with self.assertRaises(KeyError):
            self.project({}, {"ambiguous-camera": 0})

    def test_camera_collision_not_silently_overwritten(self):
        with self.assertRaises(ValueError):
            self.project({}, {"head_camera": 1, "agentview_image": 2})

    def test_current_only_and_no_unprojected_timestamps(self):
        with self.assertRaises(ValueError):
            self.project({}, {}, version="0.2")
        with self.assertRaises(ValueError):
            self.project({}, {}, state_acquired_at={"joint_drive_targets": 0.})


if __name__ == "__main__":
    unittest.main()
