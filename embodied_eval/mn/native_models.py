"""Historical import alias; implementation is maintained in native.native_models."""
import sys
from ..native import native_models as _implementation

sys.modules[__name__] = _implementation
