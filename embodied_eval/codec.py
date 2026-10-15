"""Strict dataclass decoding: no silent coercion, ignored keys or NaN."""
from dataclasses import fields, is_dataclass
import json
import math
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

from .errors import ContractError


def _fail(path, message):
    raise ContractError("invalid_type", path, message)


def json_value(value, path="$"):
    if value is None or type(value) in (str, bool, int):
        # JSON scalars are immutable; copying them per image pixel is unnecessary.
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(path, "NaN and infinity are not supported")
        return value
    if isinstance(value, (list, tuple)):
        # Retain a fresh container, but do not recurse or format paths for
        # immutable JSON scalars. Invalid/nested values still take the strict
        # path, including precise locations and finite-float validation.
        return [v if v is None or type(v) in (str, bool, int)
                else json_value(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            _fail(path, "object keys must be strings")
        return {k: json_value(v, f"{path}.{k}") for k, v in value.items()}
    _fail(path, f"expected JSON-compatible data, got {type(value).__name__}")


def decode(cls, value, path="$"):
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, UnionType):
        errors = []
        for variant in args:
            try:
                return decode(variant, value, path)
            except ContractError as error:
                errors.append(error)
        _fail(path, "does not match any declared variant: " + "; ".join(str(e) for e in errors))
    if origin is Literal:
        if not any(type(value) is type(v) and value == v for v in args):
            _fail(path, f"expected one of {args}")
        return value
    if cls is type(None):
        if value is not None:
            _fail(path, "expected null")
        return None
    if cls is object:
        return json_value(value, path)
    if cls in (str, bool, int):
        if type(value) is not cls:
            _fail(path, f"expected {cls.__name__}")
        return value
    if cls is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            _fail(path, "expected finite number")
        return float(value)
    if origin in (tuple, list):
        if not isinstance(value, (list, tuple)):
            _fail(path, "expected array")
        variable = len(args) == 2 and args[1] is Ellipsis
        if not variable and len(args) != len(value):
            _fail(path, f"expected {len(args)} elements")
        result = [decode(args[0] if variable else args[i], v, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(result) if origin is tuple else result
    if origin is dict:
        if type(value) is not dict:
            _fail(path, "expected object")
        return {
            decode(args[0], k, f"{path}.<key>"): decode(args[1], v, f"{path}.{k}")
            for k, v in value.items()
        }
    if is_dataclass(cls):
        if is_dataclass(value):
            # Decode each nested dataclass at its own boundary instead of
            # serializing/copying the entire image graph first. All fields
            # still pass through decode; this is not a trusted-instance bypass.
            value = {f.name: getattr(value, f.name) for f in fields(value)}
        if type(value) is not dict:
            _fail(path, f"expected {cls.__name__} object")
        names = {f.name for f in fields(cls)}
        unknown = set(value) - names
        if unknown:
            raise ContractError("unknown_field", path, f"undeclared fields: {sorted(unknown)}")
        hints = get_type_hints(cls)
        kwargs = {k: decode(hints[k], v, f"{path}.{k}") for k, v in value.items()}
        try:
            return cls(**kwargs)
        except TypeError as error:
            raise ContractError("missing_field", path, str(error)) from error
    _fail(path, f"unsupported contract annotation {cls}")


def to_data(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return json_value(value)
    if is_dataclass(value):
        return {f.name: to_data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: to_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [v if v is None or type(v) in (str, bool, int) else to_data(v) for v in value]
    return json_value(value)


def dumps(value):
    # Revalidate dataclass annotations even for objects constructed directly.
    if is_dataclass(value):
        decode(type(value), to_data(value))
    return json.dumps(to_data(value), ensure_ascii=False, allow_nan=False, indent=2)


def loads(cls, text):
    def reject_constant(value):
        raise ContractError("non_finite", "$", f"unsupported number {value}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError("duplicate_key", "$", f"duplicate object key {key}")
            result[key] = value
        return result
    return decode(cls, json.loads(text, parse_constant=reject_constant, object_pairs_hook=pairs))
