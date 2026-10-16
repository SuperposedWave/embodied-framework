"""Historical import alias; implementation is maintained in native.openwam_dual_pose."""
import sys
from ..native import openwam_dual_pose as _implementation

sys.modules[__name__] = _implementation
