"""Native CUDA and SAPIEN render-device creation only, no policy/task/actions."""
import argparse
import ctypes
import faulthandler
import json
from pathlib import Path
import re
import resource

ROOT = Path(__file__).resolve().parents[2]


def main(output):
    output.resolve().relative_to(ROOT / 'outputs')
    output.mkdir(parents=True, exist_ok=False)
    report = {'kind': 'native_render_device_probe_no_policy_or_task', 'libraries': {}, 'passed': False}
    for name in ('libcuda.so.1', 'libEGL_nvidia.so.0', 'libGLX_nvidia.so.0'):
        try:
            ctypes.CDLL(name)
            report['libraries'][name] = True
        except OSError:
            report['libraries'][name] = False
    try:
        import torch
        report['torch_version'] = torch.__version__
        report['cuda_operation_passed'] = (torch.ones(4, device='cuda') + 1).sum().item() == 8
        report['device_name'] = torch.cuda.get_device_name(0)
        import sapien
        # Exact default render constructor called by pinned sapien.Scene().
        render = sapien.render.RenderSystem()
        report['passed'] = True
    except Exception as error:
        report['error_type'] = type(error).__name__
        # This fixed native device constructor takes no credentials or URLs.
        # Keep a short GPU error; redact any unexpected credential marker.
        text = str(error)
        if len(text) <= 400 and not re.search(r'https?://|token|authorization|password|secret|bearer|api.?key', text, re.I):
            report['native_device_error'] = text
    with (output / 'receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output))
