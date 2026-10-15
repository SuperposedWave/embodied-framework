"""Separate registries; registrations never enumerate compatible pairs."""
from dataclasses import dataclass
from typing import Callable
from ..errors import ContractError
from .contracts import CapabilitySpec, SpaceSpec


@dataclass(frozen=True)
class ModelRegistration:
    describe: Callable[[dict], CapabilitySpec]
    factory: Callable
    validate_options: Callable
    verify_assets: Callable
    readiness: Callable
    provenance: Callable
    synthetic: bool = False


@dataclass(frozen=True)
class BenchmarkRegistration:
    describe: Callable[[dict], SpaceSpec]
    factory: Callable
    validate_options: Callable
    verify_assets: Callable
    readiness: Callable
    provenance: Callable
    metric_templates: Callable
    synthetic: bool = False


class AdapterRegistry:
    def __init__(self):
        self.models, self.benchmarks = {}, {}

    def register_model(self, name, entry):
        self._register(self.models, name, entry)

    def register_benchmark(self, name, entry):
        self._register(self.benchmarks, name, entry)

    @staticmethod
    def _register(table, name, entry):
        if not name or name in table:
            raise ContractError("duplicate_registration", name, "nonempty unique adapter ID required")
        table[name] = entry

    def model(self, name):
        return self._get(self.models, name)

    def benchmark(self, name):
        return self._get(self.benchmarks, name)

    @staticmethod
    def _get(table, name):
        if name not in table:
            raise ContractError("unknown_adapter", name, "adapter is not registered")
        return table[name]


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
