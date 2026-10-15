"""Synthetic standalone diagnostic checks, not an actual simulator replay."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from embodied_eval.codec import to_data
from embodied_eval.mn.native_models import NativeActionModel
from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator
from embodied_eval.mn import contracts as c
from embodied_eval.mn.binding import resolve

directory = Path(__file__).resolve().parents[1] / "docs/handoff"
if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("robodojo_native_replay_evidence", directory / "replay_robodojo_native_actions.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class NativeReplayEvidenceTests(unittest.TestCase):
    def action(self):
        model = NativeActionModel("openpi", {"repo": "/absent", "checkpoint_dir": "/absent", "io_profile": "arx_joint14"})
        executor = RoboDojoSimulator({"repo": "/absent"})
        model.space = executor.space
        model.profile = resolve(executor.space, model.cap, c.BindingConfig(inputs={
            "head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist", "instruction": "instruction",
            **{side + slot: side + source for side in ("left", "right") for slot, source in (("_joints", "_joint_positions"), ("_grip", "_gripper_target"))}},
            targets={key: "robot0/" + key for key in model.cap.output.target_slots}, action_spec_ref="dual-joint-raw-gripper@1"))
        context = c.Context(run_id="synthetic", env_id="test", episode_id="ep", message_id="request")
        raw = np.linspace(-1., 2., 14, dtype=np.float32)
        return to_data(model.decode_action(raw, context, None)), raw

    def test_both_arms_and_unbounded_raw_gripper_preserve_dtype(self):
        action, raw = self.action()
        native = helper.native_action(action)
        self.assertEqual(set(native), {"left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state"})
        restored = np.concatenate([native[key] for key in ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")])
        self.assertEqual(restored.dtype, raw.dtype)
        np.testing.assert_array_equal(restored, raw)
        self.assertGreater(native["right_ee_joint_state"][0], 1.)

    def test_extra_and_duplicate_targets_never_silently_discarded(self):
        action, _ = self.action()
        action["commands"].append(dict(action["commands"][0]))
        with self.assertRaises(ValueError):
            helper.native_action(action)
        action["commands"][-1]["target_id"] = "robot0/extra"
        with self.assertRaises(ValueError):
            helper.native_action(action)

    def test_declared_dtype_loss_rejected(self):
        action, _ = self.action()
        action["commands"][0]["value"] = .1234567890123
        with self.assertRaises(ValueError):
            helper.native_action(action)


if __name__ == "__main__":
    unittest.main()
