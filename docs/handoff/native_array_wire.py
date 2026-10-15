"""Lossless diagnostic-pipe array envelope across NumPy 1/2 environments.

Avoid NumPy's version-dependent pickle module names. Not a public contract,
Adapter or model transform; byte order, shape, dtype and values stay unchanged.
"""


def pack(value):
    import numpy as np
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or array.ndim > 4 or not np.isfinite(array).all():
        raise ValueError("finite_plain_numerical_array_required")
    return {"dtype": array.dtype.str, "shape": tuple(array.shape), "data": array.tobytes(order="C")}


def unpack(value):
    import math
    import numpy as np
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data"}:
        raise ValueError("explicit_array_envelope_required")
    dtype, shape, data = np.dtype(value["dtype"]), value["shape"], value["data"]
    if (dtype.kind not in "biuf" or not isinstance(shape, (tuple, list)) or len(shape) > 4
            or any(type(n) is not int or n < 0 for n in shape) or not isinstance(data, bytes)
            or len(data) != math.prod(shape) * dtype.itemsize):
        raise ValueError("plain_exact_array_bytes_required")
    array = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError("finite_array_required")
    return array
