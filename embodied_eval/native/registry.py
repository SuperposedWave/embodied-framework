"""Native-only independent factories; registration never loads GPU frameworks."""
from ..core.registry import AdapterRegistry, ModelRegistration, BenchmarkRegistration

_default = None


def register_components(registry):
    from .model_catalog import register as register_models
    from .additional_catalog import register as register_additional_models
    from .benchmark_catalog import register as register_benchmarks
    from .droid_simulator import register as register_droid
    from .robodojo_simulator import register as register_dojo
    for register in (register_models, register_additional_models, register_benchmarks,
                     register_droid, register_dojo):
        register(registry)


def default_registry():
    global _default
    if _default is None:
        registry = AdapterRegistry()
        register_components(registry)
        _default = registry
    return _default
