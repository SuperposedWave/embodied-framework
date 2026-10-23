"""Synthetic private-worker lifecycle tests; no simulator or GPU."""
import unittest
from unittest.mock import Mock
from embodied_eval.errors import AdapterError
from embodied_eval.native.droid_simulator import DroidProcessSimulator


class DroidProcessTests(unittest.TestCase):
    def setUp(self):
        self.bench = DroidProcessSimulator.__new__(DroidProcessSimulator)
        self.bench.closed = False
        self.bench.connection = Mock()
        self.bench.worker = Mock(exitcode=0, pid=123)
        self.bench.worker.is_alive.return_value = True

    def test_successful_reply_has_no_native_object(self):
        self.bench.connection.recv.return_value = {"ok": True, "result": "public-placeholder"}
        self.assertEqual(self.bench._call("reset", "task", "context"), "public-placeholder")
        self.bench.connection.send.assert_called_once_with(("reset", ("task", "context")))

    def test_lost_advance_reply_reports_execution_uncertainty(self):
        self.bench.connection.recv.side_effect = EOFError()
        with self.assertRaises(AdapterError) as raised:
            self.bench._call("advance", "action", "context", "budget")
        self.assertFalse(raised.exception.state_known)
        self.assertTrue(raised.exception.execution_started)
        self.assertEqual(raised.exception.retryability, "never")

    def test_lost_reset_reply_is_not_claimed_as_action_execution(self):
        self.bench.connection.recv.side_effect = EOFError()
        with self.assertRaises(AdapterError) as raised:
            self.bench._call("reset", "task", "context")
        self.assertFalse(raised.exception.execution_started)

    def test_malformed_reply_is_rejected(self):
        self.bench.connection.recv.return_value = {"ok": True}
        with self.assertRaises(AdapterError) as raised:
            self.bench._receive("reset")
        self.assertEqual(raised.exception.code, "droid_invalid_worker_reply")

    def test_close_checks_native_exit_and_is_idempotent(self):
        self.bench.connection.recv.return_value = {"ok": True, "result": None}
        self.bench.close()
        self.bench.close()
        self.bench.connection.send.assert_called_once_with(("close", ()))
        self.assertTrue(self.bench.closed)
        self.bench.connection.close.assert_called_once_with()

    def test_native_close_signal_failure_is_not_success(self):
        self.bench.connection.recv.return_value = {"ok": True, "result": None}
        self.bench.worker.exitcode = -11
        with self.assertRaises(AdapterError) as raised:
            self.bench.close()
        self.assertEqual(raised.exception.code, "droid_native_close_failed")
        self.assertTrue(self.bench.closed)


if __name__ == "__main__":
    unittest.main()
