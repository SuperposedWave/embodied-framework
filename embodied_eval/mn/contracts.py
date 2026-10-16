"""Historical import alias; implementation is maintained in native.contracts."""
import sys
from ..native import contracts as _implementation

sys.modules[__name__] = _implementation
