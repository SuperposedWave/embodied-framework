"""Synthetic exchange safety checks, no real policy, simulator or task score."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
from embodied_eval.adapters.native_policy_transport import NativePolicyPeer

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / "docs/handoff"
if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("dreamzero_exchange_evidence", directory / "record_dreamzero_policy_exchange.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class DreamZeroExchangeEvidence(unittest.TestCase):
    def request(self):
        return {**{name: np.full((180, 320, 3), index, np.uint8) for index, name in enumerate((
            "observation/exterior_image_0_left", "observation/exterior_image_1_left", "observation/wrist_image_left"))},
            "observation/joint_position": np.arange(7, dtype=np.float64),
            "observation/gripper_position": np.array([.5], np.float64),
            "observation/cartesian_position": np.zeros(6, np.float64),
            "session_id": "synthetic-session", "prompt": "synthetic public instruction", "endpoint": "infer"}

    def test_explicit_public_request_accepted(self):
        helper.check_request(self.request())

    def test_unknown_authentication_key_rejected_before_peer(self):
        value = self.request()
        value["authorization"] = "synthetic-not-a-key"
        with self.assertRaises(ValueError):
            helper.check_request(value)

    def test_wrong_camera_state_dtype_and_compatibility_field_rejected(self):
        for name, replacement in (("observation/joint_position", np.zeros(7, np.float32)),
                ("observation/wrist_image_left", np.zeros((320, 180, 3), np.uint8)),
                ("observation/cartesian_position", np.ones(6, np.float64))):
            value = self.request()
            value[name] = replacement
            with self.assertRaises(ValueError):
                helper.check_request(value)

    def test_original_return_identity_no_retry_and_restoration_after_error(self):
        chunk = np.zeros((8, 8), np.float32)
        peer = SimpleNamespace(codec=SimpleNamespace(packb=lambda _: b"synthetic-codec-bytes"))
        value = self.request()
        calls = []
        def original(actual_peer, actual_request):
            self.assertIs(actual_peer, peer)
            self.assertIs(actual_request, value)
            calls.append(1)
            return chunk
        with TemporaryDirectory() as temporary, patch.object(NativePolicyPeer, "infer", original):
            output = Path(temporary) / "evidence"
            with helper.record_exchanges(output):
                self.assertIs(NativePolicyPeer.infer(peer, value), chunk)
            self.assertIs(NativePolicyPeer.infer, original)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertTrue(receipt["all_observed_exchanges_completed"])
            self.assertFalse(receipt["headers_authentication_metadata_or_raw_errors_saved"])
            self.assertEqual(len(calls), 1)
            failed = Path(temporary) / "failed"
            with self.assertRaises(RuntimeError):
                with helper.record_exchanges(failed):
                    raise RuntimeError("synthetic failure")
            self.assertIs(NativePolicyPeer.infer, original)
            self.assertFalse(json.loads((failed / "receipt.json").read_text())["all_observed_exchanges_completed"])


if __name__ == "__main__":
    unittest.main()
