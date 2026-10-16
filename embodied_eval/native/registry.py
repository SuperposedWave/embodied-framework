"""Native-only independent factories; registration never loads GPU frameworks."""
from ..core.registry import AdapterRegistry, ModelRegistration, BenchmarkRegistration

_default = None


def default_registry():
    global _default
    if _default is None:
        registry = AdapterRegistry()
        from .native_catalog import register
        from .additional_catalog import register as register_additional
        register(registry)
        register_additional(registry)
        _default = registry
    return _default
