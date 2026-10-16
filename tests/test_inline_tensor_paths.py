"""No relaxed validation or aliasing in the inline-pixel copy optimizations."""
from dataclasses import replace
import unittest

from embodied_eval import _legacy_contracts as c
from embodied_eval.codec import decode, json_value, to_data
from embodied_eval.errors import ContractError
from embodied_eval._legacy_validation import validate_tensor


class InlineTensorPathTests(unittest.TestCase):
    def tensor(self, data=None):
        return c.Tensor(dtype="uint8", shape=(1, 2, 3), layout="HWC", data=data or [[[0, 127, 255], [1, 2, 3]]])

    def test_dataclass_instances_still_validate_all_field_types(self):
        with self.assertRaises(ContractError):
            decode(c.Tensor, replace(self.tensor(), layout=42))

    def test_decode_preserves_container_independence(self):
        source = self.tensor()
        result = decode(c.Tensor, source)
        source.data[0][0][0] = 255
        self.assertEqual(result.data[0][0][0], 0)
        self.assertIsNot(source.data[0], result.data[0])

    def test_to_data_also_copies_nested_containers(self):
        source = self.tensor()
        result = to_data(source)
        source.data[0][0][0] = 255
        self.assertEqual(result["data"][0][0][0], 0)

    def test_integer_subclasses_not_silently_json_scalars(self):
        class IntegerSubclass(int):
            pass
        with self.assertRaises(ContractError):
            json_value([IntegerSubclass(1)])

    def test_nested_nonfinite_float_keeps_precise_error_path(self):
        with self.assertRaises(ContractError) as error:
            decode(c.Tensor, replace(self.tensor(), data=[[[0, float("nan"), 2], [1, 2, 3]]]))
        self.assertIn("data[0][0][1]", str(error.exception))

    def test_dtype_range_and_boolean_failures_not_weakened(self):
        for bad in (True, .5, -1, 256):
            with self.assertRaises(ContractError) as error:
                validate_tensor(self.tensor([[[0, bad, 2], [1, 2, 3]]]))
            self.assertEqual(error.exception.issue.code, "tensor_dtype_mismatch")
            self.assertIn("[0][0][1]", error.exception.issue.path)
        validate_tensor(c.Tensor(dtype="bool", shape=(2,), layout="N", data=[True, False]))
        with self.assertRaises(ContractError):
            validate_tensor(c.Tensor(dtype="bool", shape=(2,), layout="N", data=[True, 1]))

    def test_tuple_and_list_subclass_allowed_but_ragged_shape_rejected(self):
        class ListSubclass(list):
            pass
        value = (((0, 1, 2), ListSubclass([3, 4, 5])),)
        validate_tensor(self.tensor(value))
        with self.assertRaises(ContractError) as error:
            validate_tensor(self.tensor([[[0, 1], [3, 4, 5]]]))
        self.assertEqual(error.exception.issue.code, "tensor_shape_mismatch")
