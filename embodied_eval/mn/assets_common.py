"""Historical import alias; implementation is maintained in native.assets_common."""
import sys
from ..native import assets_common as _implementation

sys.modules[__name__] = _implementation
