"""Synthetic checks: diagnostic facts must never become a raw log leak."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('platform_diagnostics', Path(__file__).resolve().parents[1] / 'platform_diagnostics.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
libspec = importlib.util.spec_from_file_location('platform_libraries', Path(__file__).resolve().parents[1] / 'platform_libraries.py')
libraries = importlib.util.module_from_spec(libspec)
libspec.loader.exec_module(libraries)


class DependencyFactsTest(unittest.TestCase):
    def test_library_and_module_names_only(self):
        text = "ImportError: libGL.so.1: cannot open shared object file\nModuleNotFoundError: No module named 'google_crc32c'"
        self.assertEqual(diagnostics.dependency_facts(text), {
            'missing_shared_libraries': ['libGL.so.1'], 'missing_modules': ['google_crc32c'],
            'exception_types': ['ImportError', 'ModuleNotFoundError']})

    def test_no_arbitrary_error_messages_or_secrets(self):
        text = 'RuntimeError: Authorization: Bearer synthetic-private-value\nhttps://example.invalid/?token=synthetic-value'
        self.assertEqual(diagnostics.dependency_facts(text), {
            'missing_shared_libraries': [], 'missing_modules': [], 'exception_types': ['RuntimeError']})

    def test_dynamic_loader_and_deduplication(self):
        text = 'error while loading shared libraries: libgthread-2.0.so.0\nlibgthread-2.0.so.0: cannot open shared object file\nOSError: denied'
        self.assertEqual(diagnostics.dependency_facts(text)['missing_shared_libraries'], ['libgthread-2.0.so.0'])
        self.assertEqual(diagnostics.dependency_facts(text)['exception_types'], ['OSError'])

    def test_runtime_overlay_refuses_driver_and_C_runtime(self):
        for name in ('libcuda.so.1', 'libnvidia-eglcore.so.570', 'libEGL_nvidia.so.0', 'libc.so.6', 'libstdc++.so.6', '../libGL.so.1'):
            self.assertFalse(libraries.allowed(name), name)
        self.assertTrue(libraries.allowed('libGL.so.1'))

    def test_overlay_is_local_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'outputs/private/lib'
            directory.mkdir(parents=True)
            binary = directory / 'libGL.so.1'
            binary.write_bytes(b'synthetic-library-not-an-executable')
            manifest = directory.parent / 'manifest.json'
            manifest.write_text(json.dumps({'libraries': [{'name': binary.name, 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}]}))
            base = {'LD_LIBRARY_PATH': '/existing', 'keep': 'original'}
            with patch.object(libraries, 'ROOT', root):
                result = libraries.environment(base, manifest)
                self.assertEqual(result['LD_LIBRARY_PATH'], str(directory) + ':/existing')
                self.assertEqual(base['LD_LIBRARY_PATH'], '/existing')
                binary.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    libraries.environment(base, manifest)


if __name__ == '__main__':
    unittest.main()
