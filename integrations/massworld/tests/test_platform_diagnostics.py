"""Synthetic checks: diagnostic facts must never become a raw log leak."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('platform_diagnostics', Path(__file__).resolve().parents[1] / 'platform_diagnostics.py')
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


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


if __name__ == '__main__':
    unittest.main()
