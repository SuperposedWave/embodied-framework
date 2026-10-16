"""Historical import alias; implementation is maintained in native.native_model_process."""
import sys
from ..native import native_model_process as _implementation

sys.modules[__name__] = _implementation
