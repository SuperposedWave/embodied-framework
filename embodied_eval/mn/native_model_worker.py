"""Historical import alias; implementation is maintained in native.native_model_worker."""
import sys
from ..native import native_model_worker as _implementation

sys.modules[__name__] = _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
