"""Offline dependency probe in the actual task image; no model weights/actions."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
from platform_diagnostics import dependency_facts


def child(args):
    result = {'imports': []}
    for module in args.modules:
        row = {'module': module, 'passed': False}
        try:
            imported = importlib.import_module(module)
            row['passed'] = True
            if module == 'torch':
                row.update(version=imported.__version__, cuda_available=imported.cuda.is_available())
                if args.gpu_probe and imported.cuda.is_available():
                    row['cuda_add_passed'] = (imported.ones(4, device='cuda') + 1).sum().item() == 8
        except Exception as error:
            row.update(error_type=type(error).__name__, **dependency_facts(str(error)))
        result['imports'].append(row)
    print(json.dumps(result), flush=True)
    return 0


def main(args):
    if args.child:
        return child(args)
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    envs = ROOT.parent / 'upstream/envs'
    definitions = {
        'openwam-py310': ['torch', 'cv2', 'scipy', 'transformers'],
        'openpi-stage4': ['torch', 'cv2', 'jax', 'openpi.policies.policy_config'],
        'openpi-vlabench-stage4': ['torch', 'cv2', 'jax', 'openpi.policies.policy_config'],
        'lingbot-stage4': ['torch', 'cv2', 'transformers'],
        'libero': ['google_crc32c', 'cv2', 'mujoco', 'robosuite'],
        'libero-openpi-stage4': ['google_crc32c', 'cv2', 'mujoco', 'robosuite'],
        'vlabench-stage4': ['cv2', 'mujoco', 'dm_control'],
        'robotwin-stage4': ['cv2', 'sapien', 'mplib', 'curobo'],
    }
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(args.overlay.resolve()))),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(key, None)
    if args.library_manifest:
        from platform_libraries import environment as library_environment
        environment = library_environment(environment, args.library_manifest)
    rows = []
    for name, modules in definitions.items():
        command = [str(envs / name / 'bin/python'), __file__, '--child', '--modules', *modules, '--gpu-probe']
        row = {'environment': name}
        try:
            p = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=180)
            row.update(exit_code=p.returncode, stderr_facts=dependency_facts(p.stderr))
            for line in reversed(p.stdout.splitlines()):
                try:
                    value = json.loads(line)
                    if isinstance(value, dict) and 'imports' in value:
                        row.update(value)
                        break
                except ValueError:
                    continue
        except Exception as error:
            row.update(error_type=type(error).__name__, **dependency_facts(str(error)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {'kind': 'offline_task_runtime_import_probe_not_evaluation', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'model_weights_loaded': False, 'actions_executed': False, 'native_logs_saved': False,
              'private_library_overlay_used': args.library_manifest is not None, 'environments': rows}
    with (args.output / 'receipt.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return 0 if all(r.get('imports') and all(x['passed'] for x in r['imports']) for r in rows) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--modules', nargs='+', default=[])
    parser.add_argument('--gpu-probe', action='store_true')
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--library-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    raise SystemExit(main(parser.parse_args()))
