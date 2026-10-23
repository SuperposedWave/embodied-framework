"""Source-defined CuRobo/Warp API compatibility at the native sim boundary.

No planning/controller math changes. Fixed CuRobo0.7.8 calls warp.torch but
Warp1.13 exposes the same functions at top-level. The alias exactly matches the
OpenWAM native bootstrap for this shared dependency stack, without making the
LingBot pair import OpenWAM or modifying either upstream checkout.
"""
from types import SimpleNamespace


def prewarm_native_cuda():
    import torch
    import warp
    torch.empty(1, device="cuda")
    from curobo.wrap.reacher.motion_gen import MotionGen
    names = ("from_torch", "to_torch", "dtype_from_torch", "dtype_to_torch", "device_from_torch",
             "device_to_torch", "stream_from_torch", "stream_to_torch")
    if not hasattr(warp, "torch"):
        functions = {name: getattr(warp, name, None) for name in names}
        if functions["device_from_torch"] is None:
            raise ImportError("native Warp Torch interoperability is unavailable")
        warp.torch = SimpleNamespace(**functions)
