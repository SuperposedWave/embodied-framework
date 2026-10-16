"""Historical 0.2 registry; current native registrations live in native.registry."""
from ..core.registry import AdapterRegistry, ModelRegistration, BenchmarkRegistration


_default = None


def default_registry():
    global _default
    if _default is None:
        registry = AdapterRegistry()
        from .synthetic import register
        register(registry)
        from .native_catalog import register as register_native
        register_native(registry)
        from .additional_catalog import register as register_additional
        register_additional(registry)
        _default = registry
    return _default
