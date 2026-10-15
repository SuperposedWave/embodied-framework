"""Synthetic private-runtime path checks, not native GPU verification."""
from argparse import Namespace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from compare_native_robodojo_closedloop import ROOT, resolve_runtime_paths


class NativeRoboDojoRuntimePaths(unittest.TestCase):
    def test_all_runtime_paths_absolute_before_changing_child_cwd(self):
        args = Namespace(**{key: Path("outputs") / (key + "-synthetic") for key in
            ("trial", "output", "isaac_libraries", "video_runtime")})
        with patch("os.getcwd", return_value=str(ROOT)):
            resolve_runtime_paths(args)
        for key in vars(args):
            self.assertTrue(getattr(args, key).is_absolute())
            self.assertEqual(getattr(args, key).parent, ROOT / "outputs")

    def test_runtime_path_cannot_escape_output_scope(self):
        args = Namespace(output=ROOT / "outputs/synthetic", trial=ROOT / "outputs/synthetic",
            isaac_libraries=ROOT, video_runtime=ROOT / "outputs/synthetic")
        with self.assertRaises(ValueError):
            resolve_runtime_paths(args)


if __name__ == "__main__":
    unittest.main()
