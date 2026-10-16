"""Historical import alias for frozen environment-loader/reference records."""
import sys
from .native import environment_records as _implementation

sys.modules[__name__] = _implementation
