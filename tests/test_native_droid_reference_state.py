"""Synthetic native-reference packing checks, not a real policy/score test."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from compare_native_cosmos_droid_closedloop import native_state


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
class NativeDroidReferenceState(unittest.TestCase):
    def test_keeps_all_seven_joints_and_gripper_with_or_without_batch_axis(self):
        import numpy as np
        for width in (7, 1):
            vector = np.linspace(-0.3, 0.4, width, dtype=np.float32)
            for value in (vector, vector[None]):
                actual = native_state(value, width)
                self.assertEqual(actual.shape, (width,))
                self.assertEqual(actual.dtype, vector.dtype)
                self.assertEqual(actual.tobytes(), vector.tobytes())

    def test_rejects_scalar_missing_joint_extra_environment_and_unknown_dtype(self):
        import numpy as np
        for value in (np.float32(0), np.zeros(6, np.float32), np.zeros((2, 7), np.float32),
                np.zeros((1, 1, 7), np.float32), np.zeros(7, np.float64), np.full(7, np.nan, np.float32)):
            with self.assertRaises(ValueError):
                native_state(value, 7)


if __name__ == "__main__":
    unittest.main()
