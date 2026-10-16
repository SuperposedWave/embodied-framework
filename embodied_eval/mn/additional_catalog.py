"""Historical import alias; implementation is maintained in native.additional_catalog."""
import sys
from ..native import additional_catalog as _implementation

sys.modules[__name__] = _implementation
