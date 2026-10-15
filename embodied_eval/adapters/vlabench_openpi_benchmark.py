"""Shared official VLABench evaluator with a distinct openpi protocol label."""
from .vlabench_benchmark import VLABenchBenchmark
from . import openpi_vlabench_spec


class VLABenchOpenpiBenchmark(VLABenchBenchmark):
    space_factory = staticmethod(openpi_vlabench_spec.space)
