"""Historical import alias; implementation is maintained in native.native_benchmarks."""
import sys
from ..native import native_benchmarks as _implementation

sys.modules[__name__] = _implementation
