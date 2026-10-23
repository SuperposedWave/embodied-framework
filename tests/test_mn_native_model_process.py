"""Synthetic model-only lifecycle/transport tests; no weights or simulator."""
from argparse import Namespace
import multiprocessing
import sys
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from embodied_eval.errors import AdapterError, ContractError
from embodied_eval.native.model_catalog import model_options
from embodied_eval.native.model_catalog import readiness
from embodied_eval.native.native_model_process import NativeModelProcess
from embodied_eval.native.native_model_worker import worker_loop, failure
from tests.legacy_support.frozen.mn.registry import default_registry
from embodied_eval.native.robodojo_simulator import launch_application


OPTIONS = {"repo": "/absent", "checkpoint_dir": "/absent", "io_profile": "eef20_arx_absolute",
           "runtime_python": sys.executable}


class NativeModelProcessTests(unittest.TestCase):
    def proxy(self):
        model = NativeModelProcess("openwam", OPTIONS)
        model.connection = Mock()
        model.worker = Mock(pid=123)
        model.worker.poll.return_value = None
        model.worker.wait.return_value = 0
        return model

    def test_factory_is_lazy_and_model_specific(self):
        model = default_registry().model("openwam").factory(OPTIONS, None)
        self.assertIsInstance(model, NativeModelProcess)
        self.assertIsNone(model.worker)
        self.assertFalse(model.diagnostics()["native_worker_started"])
        model.close()

    def test_runtime_cannot_be_applied_to_a_different_profile(self):
        with self.assertRaises(ContractError):
            model_options("openwam", dict(OPTIONS, io_profile="eef10_osc"))

    def test_relative_runtime_path_rejected(self):
        with self.assertRaises(ContractError):
            model_options("openwam", dict(OPTIONS, runtime_python=".venv/bin/python"))

    def test_gpu_without_runtime_rejected(self):
        with self.assertRaises(ContractError):
            model_options("openwam", dict(OPTIONS, runtime_python=None, runtime_gpu=1))

    def test_boolean_gpu_ordinal_rejected(self):
        with self.assertRaises(ContractError):
            model_options("openwam", dict(OPTIONS, runtime_gpu=True))

    def test_readiness_distinguishes_absent_interpreter(self):
        result = readiness(dict(OPTIONS, runtime_python="/absent/bin/python"), True)
        self.assertIn("runtime_python", result["missing_paths"])

    def test_prediction_is_one_private_call_without_environment_object(self):
        model = self.proxy()
        model.connection.recv.return_value = {"ok": True, "method": "predict", "result": "synthetic-public-prediction"}
        self.assertEqual(model.predict("synthetic-public-context"), "synthetic-public-prediction")
        model.connection.send.assert_called_once_with(("predict", ("synthetic-public-context",)))

    def test_lost_prediction_reply_is_not_robot_execution(self):
        model = self.proxy()
        model.connection.recv.side_effect = EOFError()
        with self.assertRaises(AdapterError) as raised:
            model.predict("context")
        self.assertFalse(raised.exception.execution_started)
        self.assertEqual(raised.exception.retryability, "never")
        self.assertTrue(model.failed)

    def test_mismatched_reply_method_rejected(self):
        model = self.proxy()
        model.connection.recv.return_value = {"ok": True, "method": "observe", "result": None}
        with self.assertRaises(AdapterError):
            model.predict("context")

    def test_close_requires_ack_and_clean_exit(self):
        model = self.proxy()
        model.connection.recv.return_value = {"ok": True, "method": "close", "result": None}
        model.close()
        model.close()
        model.connection.send.assert_called_once_with(("close", ()))
        model.connection.close.assert_called_once_with()

    def test_clean_exit_without_close_reply_is_not_success(self):
        model = self.proxy()
        model.worker.poll.return_value = 0
        with self.assertRaises(AdapterError) as raised:
            model.close()
        self.assertEqual(raised.exception.code, "native_model_close_failed")

    def test_no_raw_exception_text_crosses_pipe(self):
        self.assertEqual(failure("predict", RuntimeError("secret-not-for-evidence")),
                         {"ok": False, "method": "predict", "error_type": "RuntimeError"})

    def test_private_worker_preserves_model_owned_cache_order(self):
        parent, child = multiprocessing.Pipe()
        rows = iter(("synthetic-row0", "synthetic-row1"))
        model = SimpleNamespace(predict=lambda context: next(rows), close=lambda: None)
        outcomes = []
        thread = Thread(target=lambda: outcomes.append(worker_loop(child, model)))
        thread.start()
        try:
            for expected in ("synthetic-row0", "synthetic-row1"):
                parent.send(("predict", ("context",)))
                self.assertEqual(parent.recv()["result"], expected)
            parent.send(("close", ()))
            self.assertTrue(parent.recv()["ok"])
        finally:
            thread.join(timeout=5)
            parent.close()
            child.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes, [0])

    def test_actual_child_initialization_does_not_load_model_or_simulator(self):
        model = NativeModelProcess("openwam", OPTIONS)
        try:
            model._start()
            diagnostic = model.diagnostics()
            self.assertTrue(diagnostic["native_model_isolated"])
            self.assertFalse(diagnostic["in_process_native_engine"])
            self.assertEqual(diagnostic["native_generation_requests"], 0)
        finally:
            model.close()

    def test_pinned_dojo_launcher_namespace_and_teardown_options(self):
        sync = SimpleNamespace(add_zero_delay_kit_args=lambda args: setattr(args, "kit_args", "source-render-flags"))
        launcher = Mock(return_value=SimpleNamespace(app="synthetic-app"))
        self.assertEqual(launch_application(sync, launcher, "cuda:0"), "synthetic-app")
        args, kwargs = launcher.call_args
        self.assertIsInstance(args[0], Namespace)
        self.assertEqual(args[0].kit_args, "source-render-flags")
        self.assertEqual(kwargs, {"fast_shutdown": False})


if __name__ == "__main__":
    unittest.main()
