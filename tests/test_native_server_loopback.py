"""Synthetic source-factory/transport checks; no model is loaded."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "docs/handoff/serve_openpi_native_loopback.py"
spec = importlib.util.spec_from_file_location("native_loopback_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeLoopbackTest(unittest.TestCase):
    def invoke(self):
        policy = SimpleNamespace(metadata={"source": "native"})
        factory = Mock(return_value=policy)
        server = Mock()
        transport = Mock(return_value=server)
        source = {"Args": lambda **kw: SimpleNamespace(**kw),
                  "Checkpoint": lambda **kw: SimpleNamespace(**kw), "create_policy": factory,
                  "websocket_policy_server": SimpleNamespace(WebsocketPolicyServer=transport)}
        with patch.object(module.sys, "argv", [str(PATH), "--repo", "/fixed/source", "--config-name", "named-weight-rule",
                "--checkpoint", "/fixed/weights", "--port", "8964"]), patch.object(module.sys, "path", list(module.sys.path)), \
                patch.object(module.runpy, "run_path", return_value=source) as run:
            module.main()
        return run, factory, policy, transport, server

    def test_pinned_source_factory_and_named_weight_configuration(self):
        run, factory, _, _, _ = self.invoke()
        run.assert_called_once_with("/fixed/source/scripts/serve_policy.py", run_name="native_policy_entry")
        args = factory.call_args.args[0]
        self.assertEqual((args.policy.config, args.policy.dir, args.port), ("named-weight-rule", "/fixed/weights", 8964))

    def test_server_is_loopback_only(self):
        _, _, policy, transport, _ = self.invoke()
        transport.assert_called_once_with(policy=policy, host="127.0.0.1", port=8964, metadata=policy.metadata)

    def test_original_transport_serves_once(self):
        _, factory, _, _, server = self.invoke()
        factory.assert_called_once()
        server.serve_forever.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
