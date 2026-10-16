"""Historical import alias; implementation is maintained in native.transforms."""
import sys
from ..native import transforms as _implementation

sys.modules[__name__] = _implementation
