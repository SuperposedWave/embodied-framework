"""Synthetic transport compatibility and error-safety checks; no inference."""
import unittest
from unittest.mock import Mock, patch
from embodied_eval.adapters.native_policy_transport import NativePolicyPeer, connection_options


class NativePolicyTransportTests(unittest.TestCase):
    def test_native_websockets12_options_do_not_leak_new_socket_keywords(self):
        def old_connect(uri, *, compression=None, max_size=None, open_timeout=10, close_timeout=10, **kwargs):
            pass
        self.assertEqual(connection_options(old_connect, 30), {
            "compression": None, "max_size": None, "open_timeout": 30, "close_timeout": 5})

    def test_framework_websockets16_disables_proxy_and_keepalive_explicitly(self):
        def new_connect(uri, *, proxy=True, ping_interval=20, **kwargs):
            pass
        options = connection_options(new_connect, 45)
        self.assertIsNone(options["proxy"])
        self.assertIsNone(options["ping_interval"])
        self.assertEqual(options["open_timeout"], 45)

    def test_server_error_text_is_not_propagated_or_retried(self):
        peer = NativePolicyPeer.__new__(NativePolicyPeer)
        peer.socket, peer.codec, peer.timeout_s = Mock(), Mock(), 30
        peer.socket.recv.return_value = "private-test-placeholder"
        with self.assertRaises(RuntimeError) as raised:
            peer.infer({"state": [1]})
        self.assertNotIn("private-test-placeholder", str(raised.exception))
        peer.socket.send.assert_called_once()
        peer.socket.recv.assert_called_once_with(timeout=30)

    def test_binary_reply_uses_native_codec_unchanged(self):
        peer = NativePolicyPeer.__new__(NativePolicyPeer)
        peer.socket, peer.codec, peer.timeout_s = Mock(), Mock(), 30
        peer.socket.recv.return_value = b"native-output"
        peer.codec.unpackb.return_value = {"actions": [1, 2]}
        self.assertEqual(peer.infer({"state": [1]}), {"actions": [1, 2]})
        peer.codec.unpackb.assert_called_once_with(b"native-output")


if __name__ == "__main__":
    unittest.main()
