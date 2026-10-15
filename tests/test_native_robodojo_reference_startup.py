"""Synthetic startup-boundary tests; no GPU, simulator, or real policy."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from native_robodojo_reference_startup import initialize


class NativeRoboDojoReferenceStartup(unittest.TestCase):
    def test_reuses_only_reset_prefix_never_recorded_action_loop(self):
        calls = []
        env = SimpleNamespace(reset=lambda **kw: calls.append(kw))
        source = """def child(args):
    untouched = True
    try:
        proof['initialized'] = True
        env.reset(seed=[options['layout_index']])
        raise AssertionError('recorded action loop must never execute')
    finally:
        raise AssertionError('original cleanup must not close returned environment')
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.py"
            path.write_text(source)
            module = ModuleType("replay_robodojo_native_actions")
            module.__dict__.update(__file__=str(path), env=env, options={"layout_index": 7},
                app="native_app", proof={}, client_calls=["reset"], robot_evidence={"verified": True})
            with patch.dict(sys.modules, {module.__name__: module}):
                app, returned, proof, reset_calls, evidence = initialize(SimpleNamespace())
        self.assertIs(returned, env)
        self.assertEqual(app, "native_app")
        self.assertEqual(calls, [{"seed": [7]}])
        self.assertTrue(proof["initialized"])
        self.assertTrue(proof["startup_prefix_excludes_recorded_action_loop"])
        self.assertEqual(len(proof["startup_helper_sha256"]), 64)
        self.assertEqual(reset_calls, ["reset"])
        self.assertTrue(evidence["verified"])

    def test_rejects_source_without_reviewed_reset_boundary(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "reference.py"
            path.write_text("def child(args):\n    try:\n        env.reset(seed=0)\n    finally:\n        pass\n")
            module = ModuleType("replay_robodojo_native_actions")
            module.__file__ = str(path)
            with patch.dict(sys.modules, {module.__name__: module}):
                with self.assertRaisesRegex(ValueError, "reviewed_native_reset_boundary_missing"):
                    initialize(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
