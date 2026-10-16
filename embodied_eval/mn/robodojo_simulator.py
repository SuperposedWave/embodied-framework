"""Historical import alias; implementation is maintained in native.robodojo_simulator."""
import sys
from ..native import robodojo_simulator as _implementation

sys.modules[__name__] = _implementation
