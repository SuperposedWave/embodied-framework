"""Private native-codec context conversion, independent of both adapters."""
from ..native import contracts as native
from . import contracts as c
from ..core.validation import require


def native_context(context):
    require(isinstance(context, c.Context), "unsupported_context", "context", "only current public context may cross a runtime boundary")
    return context


def public_context(context):
    require(isinstance(context, c.Context), "unsupported_context", "context", "old records require an explicit offline evidence reader")
    return context
