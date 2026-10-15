"""Synthetic diagnostic recorder checks; no socket, weights or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from embodied_eval.adapters.native_policy_transport import NativePolicyPeer

path = Path(__file__).resolve().parents[1] / "docs/handoff/record_arx_policy_exchange.py"
spec = importlib.util.spec_from_file_location("arx_exchange_evidence", path)
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


class ExchangeEvidenceTests(unittest.TestCase):
    def peer(self):
        peer = NativePolicyPeer.__new__(NativePolicyPeer)
        peer.codec = SimpleNamespace(packb=lambda value: json.dumps(value, default=lambda x: x.tolist()).encode())
        return peer

    def request(self):
        return {"state": np.zeros(14, np.float32), "prompt": "synthetic task", "images": {
            name: np.zeros((3, 2, 3), np.uint8) for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}}

    def test_only_one_inference_and_original_response_identity_preserved(self):
        response = {"actions": np.ones((50, 14), np.float32), "private_extra": "private-test-placeholder"}
        calls = []
        def infer(peer, value):
            calls.append(value)
            return response
        with tempfile.TemporaryDirectory() as temporary, patch.object(NativePolicyPeer, "infer", infer):
            directory = Path(temporary) / "evidence"
            with recorder.record_exchanges(directory):
                self.assertIs(self.peer().infer(self.request()), response)
            self.assertIs(NativePolicyPeer.infer, infer)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("private-test-placeholder", b"".join(p.read_bytes() for p in directory.iterdir()).decode())
            self.assertTrue(json.loads((directory / "receipt.json").read_text())["all_observed_exchanges_completed"])

    def test_auth_field_rejected_before_call_without_persisting_value(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(NativePolicyPeer, "infer") as infer:
            directory = Path(temporary) / "evidence"
            with recorder.record_exchanges(directory), self.assertRaises(ValueError):
                self.peer().infer(dict(self.request(), api_key="private-test-placeholder"))
            infer.assert_not_called()
            self.assertNotIn("private-test-placeholder", (directory / "receipt.json").read_text())
            self.assertFalse(json.loads((directory / "receipt.json").read_text())["all_observed_exchanges_completed"])

    def test_lost_response_not_recorded_as_complete_and_original_method_restored(self):
        def infer(peer, value):
            raise OSError("private-test-placeholder")
        with tempfile.TemporaryDirectory() as temporary, patch.object(NativePolicyPeer, "infer", infer):
            directory = Path(temporary) / "evidence"
            with self.assertRaises(OSError), recorder.record_exchanges(directory):
                self.peer().infer(self.request())
            self.assertIs(NativePolicyPeer.infer, infer)
            receipt = json.loads((directory / "receipt.json").read_text())
            self.assertFalse(receipt["records"][0]["completed"])
            self.assertNotIn("private-test-placeholder", json.dumps(receipt))


if __name__ == "__main__":
    unittest.main()
