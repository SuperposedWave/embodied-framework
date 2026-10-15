"""Synthetic teardown/path-isolation checks; no Isaac scene or GPU work."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from embodied_eval.errors import AdapterError, ContractError
from embodied_eval.mn.robodojo_simulator import RoboDojoProcessSimulator, prepare_robot_paths


class RoboDojoProcessTests(unittest.TestCase):
    def proxy(self):
        bench = RoboDojoProcessSimulator.__new__(RoboDojoProcessSimulator)
        bench.closed = bench.failed = False
        bench.connection = Mock()
        bench.worker = Mock(exitcode=0, pid=123)
        bench.worker.is_alive.return_value = True
        return bench

    def test_loss_after_advance_reports_unknown_execution(self):
        bench = self.proxy()
        bench.connection.recv.side_effect = EOFError()
        with self.assertRaises(AdapterError) as raised:
            bench.advance("action", "context", "budget")
        self.assertTrue(raised.exception.execution_started)
        self.assertFalse(raised.exception.state_known)
        self.assertEqual(raised.exception.retryability, "never")

    def test_error_reply_allows_cleanup_only(self):
        bench = self.proxy()
        bench.connection.recv.return_value = {"ok": False, "method": "reset", "error_type": "ValueError", "locations": []}
        with self.assertRaises(AdapterError) as raised:
            bench.reset("task", "context")
        self.assertEqual(raised.exception.native_failure["error_type"], "ValueError")
        with self.assertRaises(ContractError):
            bench.reset("task", "context")
        self.assertEqual(bench.connection.send.call_count, 1)

    def test_wrong_method_reply_rejected(self):
        bench = self.proxy()
        bench.connection.recv.return_value = {"ok": True, "method": "score", "result": None}
        with self.assertRaises(AdapterError):
            bench.reset("task", "context")

    def test_failed_advance_preserves_safe_counts_without_claiming_execution_known(self):
        bench = self.proxy()
        counts = {"action_count": 1, "physics_steps": 10, "end_flag": True, "success_flag": True}
        bench.connection.recv.return_value = {"ok": False, "method": "advance", "error_type": "ContractError",
            "locations": [], "error_code": "execution_count_mismatch", "native_counts": counts}
        with self.assertRaises(AdapterError) as raised:
            bench.advance("action", "context", "budget")
        self.assertEqual(raised.exception.native_failure["native_counts"], counts)
        self.assertEqual(raised.exception.native_failure["error_code"], "execution_count_mismatch")
        self.assertFalse(raised.exception.state_known)

    def test_close_is_checked_and_idempotent(self):
        bench = self.proxy()
        bench.connection.recv.return_value = {"ok": True, "method": "close", "result": None}
        bench.close()
        bench.close()
        bench.connection.send.assert_called_once_with(("close", ()))

    def test_clean_exit_alone_is_not_close_acceptance(self):
        bench = self.proxy()
        bench.worker.is_alive.return_value = False
        with self.assertRaises(AdapterError):
            bench.close()

    def test_actual_spawned_worker_metadata_and_close_need_no_native_dependencies(self):
        bench = RoboDojoProcessSimulator({"repo": "/absent"})
        try:
            diagnostic = bench.diagnostics()
            self.assertTrue(diagnostic["native_application_isolated"])
            self.assertEqual(diagnostic["physics_steps"], 0)
        finally:
            bench.close()

    def test_native_template_relocation_preserves_parameters_and_reference_tree(self):
        with tempfile.TemporaryDirectory(prefix="dojo-path-test-") as temporary:
            root = Path(temporary) / "reference"
            robot = root / "Assets/Robots/x5"
            robot.mkdir(parents=True)
            text = "asset_root: ${ASSETS_PATH}/Assets/Robots/x5\nplanner: {frame_bias: [0.,0.,0.]}\n"
            (robot / "curobo_tmp.yml").write_text(text)
            (robot / "X5A.urdf").write_text("synthetic urdf")
            runtime, evidence = prepare_robot_paths(root, Path(temporary) / "output")
            self.assertEqual((runtime / "x5/curobo.yml").read_text(), text.replace("${ASSETS_PATH}", str(root)))
            self.assertEqual((robot / "curobo_tmp.yml").read_text(), text)
            self.assertFalse((robot / "curobo.yml").exists())
            self.assertTrue((runtime / "x5/X5A.urdf").is_symlink())
            self.assertFalse(evidence["reference_assets_modified"])

    def test_unrecognized_robot_template_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dojo-path-test-") as temporary:
            root = Path(temporary)
            robot = root / "Assets/Robots/x5"
            robot.mkdir(parents=True)
            (robot / "curobo_tmp.yml").write_text("unrecognized template")
            with self.assertRaises(ContractError):
                prepare_robot_paths(root, root / "output")


if __name__ == "__main__":
    unittest.main()
