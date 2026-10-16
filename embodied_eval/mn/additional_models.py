"""Historical import alias; implementation is maintained in native.additional_models."""
import sys
from ..native import additional_models as _implementation

sys.modules[__name__] = _implementation
