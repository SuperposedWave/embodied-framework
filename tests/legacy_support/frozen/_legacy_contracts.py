"""Historical import alias for frozen environment-loader/reference records."""
import sys
import tests.legacy_support.frozen.environment_records as _implementation

sys.modules[__name__] = _implementation
