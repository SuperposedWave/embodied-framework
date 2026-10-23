"""Shared official VLABench evaluator with a distinct openpi protocol label."""
from tests.legacy_support.frozen.adapters.vlabench_benchmark import VLABenchBenchmark
import tests.legacy_support.frozen.adapters.openpi_vlabench_spec as openpi_vlabench_spec


class VLABenchOpenpiBenchmark(VLABenchBenchmark):
    space_factory = staticmethod(openpi_vlabench_spec.space)
