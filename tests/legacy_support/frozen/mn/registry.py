"""Historical 0.2 registry; current native registrations live in native.registry."""
from embodied_eval.core.registry import AdapterRegistry, ModelRegistration, BenchmarkRegistration


_default = None


def default_registry():
    global _default
    if _default is None:
        registry = AdapterRegistry()
        from tests.legacy_support.frozen.mn.synthetic import register
        register(registry)
        from embodied_eval.native.registry import register_components as register_native
        register_native(registry)
        _default = registry
    return _default
