"""Utilities at native adapter boundaries; no simulator or model in the runner."""
import importlib
from pathlib import Path
import sys

from .. import contracts as c
from ..errors import AdapterError


def native_module(repo, module, relative_root=""):
    """Import only from the explicitly configured checkout, rejecting stale imports."""
    root = (Path(repo).resolve() / relative_root).resolve()
    loaded = sys.modules.get(module)
    if loaded is not None and getattr(loaded, "__file__", None):
        if not Path(loaded.__file__).resolve().is_relative_to(root):
            raise AdapterError("infrastructure", "source_import_conflict",
                               f"{module} is already imported from another checkout; use a separate process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    result = importlib.import_module(module)
    if not getattr(result, "__file__", None) or not Path(result.__file__).resolve().is_relative_to(root):
        raise AdapterError("infrastructure", "source_import_conflict", f"{module} is not from the configured checkout")
    return result


def finite_array(value, shape, label, dtype=None):
    import numpy as np
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise AdapterError("model_input", "invalid_native_payload", f"{label} must have finite shape {shape}")
    if dtype is not None:
        array = array.astype(dtype, copy=False)
        if not np.isfinite(array).all():
            raise AdapterError("model_input", "invalid_native_payload", f"{label} overflows the native dtype")
    return array


def rgb_image(observation, key):
    import numpy as np
    sample = observation.sensors[key]
    tensor = sample.payload
    if sample.validity != "valid" or tensor is None or tensor.data is None:
        raise AdapterError("model_input", "missing_inline_image", f"{key} needs a valid inline RGB image")
    array = finite_array(tensor.data, tensor.shape, key)
    if tensor.dtype != "uint8" or tensor.layout != "HWC" or len(tensor.shape) != 3 or tensor.shape[-1] != 3:
        raise AdapterError("model_input", "invalid_image", f"{key} must be uint8 HWC RGB")
    if np.any(array < 0) or np.any(array > 255) or np.any(array != np.floor(array)):
        raise AdapterError("model_input", "invalid_image", f"{key} contains invalid uint8 values")
    return np.ascontiguousarray(array, dtype=np.uint8)


def controller_prediction(request, observation, profile, action_ref, controller_ref, parameters, period, transformations):
    command = c.ControllerCommand(command_id=request.message_id + ":control", target_id="robot0/eef",
                                  controller_profile_ref=controller_ref, parameters=parameters)
    return c.Prediction(context=request, request_id=request.message_id,
        based_on_observation_id=observation.observation_id, profile_ref=profile.profile_id,
        decision=c.ActionSequence(action_spec_ref=action_ref, execution_horizon_steps=1,
            steps=(c.ActionStep(index=0, duration_s=period, commands=(command,)),)),
        diagnostics=c.PredictionDiagnostics(transformations=tuple(transformations), model_forward_count=None))


def check_tick_budget(remaining, period):
    if remaining.env_steps is not None and remaining.env_steps < 1:
        raise AdapterError("execution", "budget_exhausted", "no environment tick remains")
    if remaining.sim_time_s is not None and remaining.sim_time_s + 1e-9 < period:
        raise AdapterError("execution", "budget_exhausted", "no full simulation tick remains")
