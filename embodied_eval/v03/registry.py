"""Independent registrations; native factories are never invoked by describe."""
from ..mn.registry import AdapterRegistry, ModelRegistration, BenchmarkRegistration
from ..mn.registry import default_registry as native_registry
from .models import CanonicalModel, capability
from .benchmarks import CanonicalBenchmark, canonical_space

_default = None


def default_registry():
    global _default
    if _default is None:
        registry = AdapterRegistry()
        originals = native_registry()
        for family in ("openwam", "openpi", "lingbot_vla", "cosmos3", "dreamzero"):
            entry = originals.model(family)
            registry.register_model(family, ModelRegistration(
                describe=lambda o, f=family, e=entry: capability(f, o, e.describe(o)),
                factory=lambda o, out, f=family, e=entry: CanonicalModel(f, o, out, e),
                validate_options=entry.validate_options, verify_assets=entry.verify_assets,
                readiness=entry.readiness, provenance=entry.provenance))
        for family in ("libero", "vlabench", "robotwin", "droid_sim", "robodojo"):
            entry = originals.benchmark(family)
            registry.register_benchmark(family, BenchmarkRegistration(
                describe=lambda o, f=family, e=entry: canonical_space(f, e.describe(o)),
                factory=lambda o, out, f=family, e=entry: CanonicalBenchmark(f, o, out, e),
                validate_options=entry.validate_options, verify_assets=entry.verify_assets,
                readiness=entry.readiness, provenance=entry.provenance, metric_templates=entry.metric_templates))
        from .synthetic import register
        register(registry)
        _default = registry
    return _default
