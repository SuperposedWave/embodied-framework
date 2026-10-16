"""Historical import alias; implementation is maintained in native.openwam_arx."""
import sys
from ..native import openwam_arx as _implementation

sys.modules[__name__] = _implementation
