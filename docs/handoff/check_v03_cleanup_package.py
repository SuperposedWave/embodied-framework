"""Check built-package contents and private native-worker startup, without GPU.

Extracts our local wheel into a temporary directory; does not install packages.
The native OpenWAM worker is initialized/closed, never begins an episode and
never loads weights, creates a simulator, calls inference or scores a task.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROBE = '''
import json, pathlib, sys, tempfile
import embodied_eval
from embodied_eval.config import load_config
from embodied_eval.v03.registry import default_registry
from embodied_eval.runner import run
root = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[2]).resolve()
assert pathlib.Path(embodied_eval.__file__).resolve().is_relative_to(root)
registry = default_registry()
assert not any(n.startswith(('embodied_eval.mn', 'embodied_eval._legacy')) for n in sys.modules)
assets = root / 'embodied_eval/native/asset_manifests'
assert len(list(assets.glob('*.json'))) == 6
assert not (root / 'tests').exists()
with tempfile.TemporaryDirectory() as temporary:
    outcome = run(load_config(source/'examples/mn_synthetic_run.json'), pathlib.Path(temporary)/'run')
    assert outcome.exit_code == 0 and outcome.summary['synthetic']
assert not any(n.startswith(('embodied_eval.mn', 'embodied_eval._legacy')) for n in sys.modules)
print(json.dumps(dict(python=sys.version.split()[0], imported_from_unpacked_wheel=True,
    independent_models=len(registry.models), independent_benchmarks=len(registry.benchmarks),
    synthetic_run_exit_code=outcome.exit_code, historical_runtime_modules_loaded=False)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--python310', type=Path, required=True)
    parser.add_argument('--native-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(kind='v03_cleanup_wheel_and_private_worker_startup_no_inference', passed=False,
        at_utc=datetime.now(timezone.utc).isoformat(), new_inference=False, new_simulation=False,
        weights_loaded=False, dependencies_installed=False, native_worker=None, environments=[])
    try:
        result['wheel_sha256'] = hashlib.sha256(args.wheel.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='embodied-v03-wheel-') as temporary:
            unpacked = Path(temporary)
            with ZipFile(args.wheel) as archive:
                names = archive.namelist()
                assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names)
                for required in ('embodied_eval/core/types.py', 'embodied_eval/core/registry.py',
                    'embodied_eval/native/registry.py', 'embodied_eval/native/native_model_worker.py',
                    'embodied_eval/native/environment_records.py', 'embodied_eval/v03/runner.py'):
                    assert required in names, required
                assert len([name for name in names if name.startswith('embodied_eval/native/asset_manifests/')]) == 6
                code_files = [name for name in names if name.startswith('embodied_eval/') and name.endswith('.py')]
                for name in code_files:
                    assert archive.read(name) == (ROOT/name).read_bytes(), name
                result['packaged_python_files_exactly_match_current_source'] = len(code_files)
                archive.extractall(unpacked)
            environment = dict(os.environ, PYTHONPATH=str(unpacked), PYTHONDONTWRITEBYTECODE='1')
            for interpreter in (Path(sys.executable), args.python310):
                # Preserve the venv interpreter path; resolving its symlink
                # would silently select the underlying base environment.
                tested = subprocess.run([str(interpreter.absolute()), '-c', PROBE, str(unpacked), str(ROOT)],
                    cwd=unpacked, env=environment, capture_output=True, text=True, timeout=90)
                assert tested.returncode == 0, tested.stderr
                result['environments'].append(json.loads(tested.stdout))
        from embodied_eval.native.native_model_process import NativeModelProcess
        from embodied_eval.native.contracts import Context
        options = json.loads((ROOT/'examples/v03_openwam_arx_robodojo.json').read_text())['model']['options']
        options['runtime_python'] = str(args.native_python.absolute())
        model = NativeModelProcess('openwam', options)
        try:
            model._start()
            model.end_episode(Context(run_id='private-startup-check', env_id='no-simulator',
                episode_id='not-started', message_id='end-without-begin'))
            diagnostics = model.diagnostics()
            assert diagnostics['engine_in_native_worker'] and diagnostics['native_generation_requests'] == 0
            assert diagnostics['predict_calls'] == 0
            result['native_worker'] = dict(native_python=str(args.native_python.absolute()),
                source_module='embodied_eval.native.native_model_worker', initialized=True,
                private_context_transport_passed=True, episode_started=False, predict_calls=0,
                native_generation_requests=0, model_weights_loaded=False)
        finally:
            model.close()
        assert model.worker.returncode == 0
        result['native_worker'].update(close_acknowledged=True, exit_code=model.worker.returncode)
        result['passed'] = True
    finally:
        with (args.output/'receipt.json').open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
