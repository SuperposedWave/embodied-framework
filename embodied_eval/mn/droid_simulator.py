"""Historical import alias; implementation is maintained in native.droid_simulator."""
import sys
from ..native import droid_simulator as _implementation

sys.modules[__name__] = _implementation
