"""CPU-only evidence-safety tests, not real model/simulator acceptance."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import ModuleType, SimpleNamespace

PATH = Path(__file__).resolve().parents[1] / "docs/handoff/probe_mn_components.py"
spec = importlib.util.spec_from_file_location("mn_component_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ComponentProbeSafetyTests(unittest.TestCase):
    def test_arbitrary_exception_message_is_not_serialized(self):
        private_value = "private-test-placeholder-do-not-persist"
        result = probe.public_exception(RuntimeError(private_value))
        self.assertNotIn(private_value, json.dumps(result))
        self.assertEqual(result[0]["error_type"], "RuntimeError")

    def test_cuda_failure_is_classified_without_message(self):
        result = probe.public_exception(RuntimeError("CUDA out of memory: private-test-placeholder"))
        self.assertTrue(result[0]["public_failure_categories"]["cuda_out_of_memory"])
        self.assertNotIn("private-test-placeholder", json.dumps(result))

    def test_normal_distributed_startup_is_not_a_failure(self):
        self.assertFalse(probe.categories("NCCL process group initialized successfully")["distributed_failure"])
        self.assertTrue(probe.categories("NCCL error during collective")["distributed_failure"])

    def test_public_missing_module_is_named(self):
        error = ModuleNotFoundError("private-test-placeholder", name="jsonschema")
        self.assertEqual(probe.public_exception(error)[0]["missing_module"], "jsonschema")

    def test_unsafe_missing_module_text_is_not_named(self):
        error = ModuleNotFoundError("private-test-placeholder", name="https://private-test-placeholder")
        self.assertNotIn("missing_module", probe.public_exception(error)[0])

    def test_nested_exception_classes_preserve_cause_without_text(self):
        original = FileNotFoundError("private-test-placeholder")
        wrapper = RuntimeError("another-private-test-placeholder")
        wrapper.__cause__ = original
        result = probe.public_exception(wrapper)
        self.assertEqual([item["error_type"] for item in result], ["RuntimeError", "FileNotFoundError"])
        self.assertNotIn("private-test-placeholder", json.dumps(result))

    def test_public_contract_or_adapter_code_does_not_expose_message(self):
        from embodied_eval.errors import ContractError, AdapterError
        for error in (ContractError("native_period_changed", "private-test-placeholder", "private-test-placeholder"),
                      AdapterError("execution", "native_period_changed", "private-test-placeholder")):
            with self.subTest(error=type(error).__name__):
                result = probe.public_exception(error)
                self.assertEqual(result[0]["public_error_code"], "native_period_changed")
                self.assertNotIn("private-test-placeholder", json.dumps(result))

    def test_new_evidence_writer_refuses_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            probe.write(path, {"passed": False})
            with self.assertRaises(FileExistsError):
                probe.write(path, {"passed": True})
            self.assertFalse(json.loads(path.read_text())["passed"])

    def test_droid_teardown_does_not_request_process_fast_exit(self):
        from embodied_eval.native import droid_simulator
        app = Mock()
        launcher = Mock(return_value=SimpleNamespace(app=app))
        wrapper = ModuleType("isaaclab.app")
        wrapper.AppLauncher = launcher
        with patch.dict("sys.modules", {"isaaclab.app": wrapper}), patch.object(
                droid_simulator, "native_module", side_effect=RuntimeError("synthetic import stop")):
            with self.assertRaises(RuntimeError):
                droid_simulator.DroidSimulator({"repo": "/absent", "assets_dir": "/absent", "device": "cuda:0"})
        self.assertFalse(launcher.call_args.kwargs["fast_shutdown"])
        self.assertTrue(launcher.call_args.kwargs["enable_cameras"])
        app.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
