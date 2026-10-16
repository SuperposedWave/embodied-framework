"""Historical import alias; implementation is maintained in native.joint_native."""
import sys
from ..native import joint_native as _implementation

sys.modules[__name__] = _implementation
