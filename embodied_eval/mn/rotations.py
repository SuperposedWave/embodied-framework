"""Historical import alias; implementation is maintained in native.rotations."""
import sys
from ..native import rotations as _implementation

sys.modules[__name__] = _implementation
