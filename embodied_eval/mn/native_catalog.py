"""Historical import alias; implementation is maintained in native.native_catalog."""
import sys
from ..native import native_catalog as _implementation

sys.modules[__name__] = _implementation
