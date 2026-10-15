"""Official openpi LIBERO protocol: ten settle steps, native float64 proprio."""
import numpy as np
from pathlib import Path
from .libero_spatial_benchmark import LiberoSpatialBenchmark
from . import openpi_libero_spec as spec


class LiberoOpenpiBenchmark(LiberoSpatialBenchmark):
    space_factory = staticmethod(spec.space)
    controller_ref = spec.CONTROLLER
    settle_steps = 10
    native_state_dtype = np.float64
    action_dtype = None
    score_revision = "f78abd6"

    def __init__(self, config):
        # The LIBERO package must come from the openpi submodule, in an isolated process.
        from .native_common import native_module
        super().__init__(config)
        try:
            native_module(config.libero_repo, "libero.libero")
        except Exception:
            self.close()
            raise

    def diagnostics(self):
        value = super().diagnostics()
        value["adapter"] = "libero_openpi"
        return value

    def _load_init_states(self):
        # Registered factory preflight has verified this exact local task file
        # against the pinned source blob before unpickling. PyTorch >=2.6 changed
        # its default; retain the old native loader without editing the checkout.
        import torch
        path = (Path(self.config.libero_repo).resolve() / "libero/libero/init_files"
                / self.native_task.problem_folder / self.native_task.init_states_file)
        return torch.load(path, weights_only=False)
