"""Synthetic private-array transport tests, not real model acceptance."""
import importlib.util
from pathlib import Path
import pickle
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from native_array_wire import pack, unpack


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
class NativeArrayWire(unittest.TestCase):
    def test_bytes_preserve_dtype_byte_order_shape_and_strided_pixels(self):
        import numpy as np
        for value in (np.arange(42, dtype=">f8").reshape(6, 7),
                np.arange(90, dtype=np.uint8).reshape(5, 6, 3)[::-1, ::-1], np.array([1.01], np.float32)):
            envelope = pickle.loads(pickle.dumps(pack(value)))
            self.assertIsInstance(envelope["data"], bytes)
            actual = unpack(envelope)
            self.assertEqual(actual.dtype, value.dtype)
            self.assertEqual(actual.shape, value.shape)
            self.assertTrue(np.array_equal(actual, value))

    def test_rejects_objects_nonfinite_values_and_malformed_envelopes(self):
        import numpy as np
        for value in (np.array([object()], object), np.array([np.nan]), np.zeros((1,) * 5)):
            with self.assertRaises(ValueError):
                pack(value)
        envelope = pack(np.zeros(3, np.float32))
        for changed in (dict(envelope, data=b"short"), dict(envelope, shape=(-1,)),
                dict(envelope, dtype="O"), dict(envelope, extra=True)):
            with self.assertRaises(ValueError):
                unpack(changed)


if __name__ == "__main__":
    unittest.main()
