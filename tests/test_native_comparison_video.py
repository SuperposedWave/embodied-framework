"""Synthetic recorder checks, not real model or Benchmark evaluations."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HANDOFF = Path(__file__).resolve().parents[1] / "docs/handoff"
sys.path.insert(0, str(HANDOFF))
from comparison_video import Frames, frame_at
from compare_pi05_robotwin_closedloop import recorded_joint_row


class NativeComparisonVideo(unittest.TestCase):
    def test_frame_selection_uses_simulation_time_not_action_index(self):
        rows = [{"sim_time_s": 0., "env_step": 0}, {"sim_time_s": .12, "env_step": 1}, {"sim_time_s": 1.2, "env_step": 2}]
        self.assertEqual(frame_at(rows, .9), 1)
        self.assertEqual(frame_at(rows, .12), 1)
        self.assertEqual(frame_at(rows, 5.), 2)

    def test_camera_names_cannot_escape_output_or_silently_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            for cameras in (("../private",), ("left", "left"), (), ("left/extra",)):
                target = Path(directory) / "frames"
                with self.assertRaises(ValueError):
                    Frames(target, cameras)
                self.assertFalse(target.exists())

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "image dependencies not installed")
    def test_passive_frames_preserve_pixels_and_timestamps_and_reject_missing_camera(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            frames = Frames(Path(directory) / "frames", ("left", "right"))
            pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            frames.append({"left": pixels, "right": pixels[::-1]}, 0, 0.)
            with Image.open(frames.directory / frames.rows[0]["cameras"]["left"]["file"]) as image:
                self.assertTrue(np.array_equal(np.asarray(image), pixels))
            with self.assertRaises(ValueError):
                frames.append({"left": pixels}, 1, .4)
            frames.append({"left": pixels, "right": pixels}, 1, .4, {"left": 0., "right": .4})
            self.assertEqual(frames.rows[-1]["cameras"]["left"]["captured_at_s"], 0.)
            with self.assertRaises(ValueError):
                frames.append({"left": pixels, "right": pixels}, 2, .3)
            with self.assertRaises(ValueError):
                frames.append({"left": pixels, "right": pixels}, 2, .5, {"left": .6, "right": .5})
            self.assertTrue(frames.close().is_file())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
    def test_native_dual_row_requires_all_commands_and_preserves_native_dtype(self):
        import numpy as np
        commands = []
        for side in ("left", "right"):
            commands.extend({"target_id": f"robot0/{side}_j{i}", "value": float(i)} for i in range(6))
            commands.append({"target_id": f"robot0/{side}_grip", "parameters": {"requested_open_fraction": 1.01}})
        action = {"commands": commands, "native_dtype": "float64"}
        row = recorded_joint_row(action)
        self.assertEqual(row.dtype, np.float64)
        self.assertEqual(row.shape, (14,))
        self.assertEqual(row[6], 1.01)
        with self.assertRaises(ValueError):
            recorded_joint_row(dict(action, commands=commands[:-1]))
        with self.assertRaises(ValueError):
            recorded_joint_row(dict(action, commands=commands[:-1] + [commands[0]]))


if __name__ == "__main__":
    unittest.main()
