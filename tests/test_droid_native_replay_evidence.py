"""Synthetic diagnostic conversion tests; no model, simulator or score."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class DroidNativeReplayEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT / "docs/handoff"))
        spec = importlib.util.spec_from_file_location("native_droid_replay_evidence", ROOT / "docs/handoff/replay_droid_native_actions.py")
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def action(self, opening=.5):
        return {"commands": [dict(kind="joint_target", target_id=f"robot0/j{i}", mode="position", unit="rad", value=float(i)) for i in range(7)]
            + [dict(kind="gripper_command", target_id="robot0/grip", mode="open_fraction", value=opening)],
            "native_protocol": None, "native_dtype": "float32"}

    def test_all_joints_half_gripper_and_native_dtype(self):
        action = self.action()
        row = self.helper.native_action(action)
        self.assertEqual(row.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(row, [0, 1, 2, 3, 4, 5, 6, .5])

    def test_missing_joint_and_duplicate_target_rejected(self):
        for changed in (self.action()["commands"][1:], self.action()["commands"] + [self.action()["commands"][0]]):
            with self.assertRaises(ValueError):
                self.helper.native_action(dict(self.action(), commands=changed))

    def test_dtype_loss_and_native_extension_rejected(self):
        action = self.action()
        action["commands"][0]["value"] = .1
        with self.assertRaises(ValueError):
            self.helper.native_action(action)
        with self.assertRaises(ValueError):
            self.helper.native_action(dict(self.action(), native_protocol="unknown-native-array"))


if __name__ == "__main__":
    unittest.main()
