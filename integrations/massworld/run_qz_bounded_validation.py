"""One platform allocation: direct vs MassWorld bounded real runs, sequentially.

This supervisor owns only the children it starts. It is not a new Runner
timeout/retry protocol, a task-success evaluation, or a qz execution provider.
Native logs remain in RAM; no credentials/environment dump is persisted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/handoff'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def child(args):
    import numpy as np
    from embodied_eval.config import load_config
    random.seed(0)
    np.random.seed(0)
    cfg = load_config(args.config)
    if cfg.benchmark.adapter == 'vlabench':
        sys.path.insert(0, cfg.benchmark.options['repo'])
        os.environ['VLABENCH_ROOT'] = str(Path(cfg.benchmark.options['repo']) / 'VLABench')
        import VLABench.robots
        import VLABench.tasks
        import VLABench.envs
    if args.child == 'direct':
        from embodied_eval.v03.runner import run
        return run(cfg, args.output).exit_code
    from worldevalkit.runner import EvaluationRunner
    runner = EvaluationRunner(execution_backend='embodied-v03',
        backend_config=json.loads(args.config.read_text()), record_dir=args.output,
        output_dir=args.output.parent, reuse_results=False)
    card = runner.run()
    return 0 if card['success'] else 1


def trace_values(root, episode):
    path = root / 'episodes' / episode / 'trace.internal.jsonl'
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row['event'] in ('reset', 'prediction', 'step_result'):
                rows.append(row)
    return rows


def compare(left, right, episode):
    a, b = trace_values(left, episode), trace_values(right, episode)
    initial = a[0] == b[0]
    a_inputs = left / 'episodes' / episode / 'model_input.public.jsonl'
    b_inputs = right / 'episodes' / episode / 'model_input.public.jsonl'
    with a_inputs.open() as f, b_inputs.open() as g:
        first_input_equal = json.loads(next(f)) == json.loads(next(g))
    predictions_a = [x['data']['action'] for x in a if x['event'] == 'prediction']
    predictions_b = [x['data']['action'] for x in b if x['event'] == 'prediction']
    first_difference = next((i + 1 for i, (x, y) in enumerate(zip(predictions_a, predictions_b)) if x != y), None)
    results = [json.loads((p / 'episodes' / episode / 'result.json').read_text()) for p in (left, right)]
    return dict(initial_observation_exact=initial, first_public_model_input_exact=first_input_equal,
                all_recorded_actions_exact=predictions_a == predictions_b,
                first_action_difference=first_difference, action_counts=[len(predictions_a), len(predictions_b)],
                execution_statistics=[r['execution_statistics'] for r in results],
                official_metrics=[r['metrics'] for r in results],
                task_success_not_a_mapping_gate=True,
                fresh_inference_difference_is_not_automatically_an_adapter_bug=True)


def execute(args):
    # The supervisor itself verifies assets (including CRC), before children.
    # Make the pre-staged overlay available to this process as well.
    sys.path.insert(1, str(args.overlay.resolve()))
    from platform_diagnostics import dependency_facts
    from platform_libraries import environment as library_environment
    from embodied_eval.config import load_config
    from embodied_eval.v03.preflight import check
    from probe_mn_components import categories, public_exception
    from run_mn_trials import server_spec, stop_owned
    config = load_config(args.config)
    args.output.resolve().relative_to(ROOT / 'outputs')
    if len(config.episodes) != 1 or config.episodes[0].budgets.env_steps != 4:
        raise ValueError('this reviewed diagnostic requires one episode and four actions')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='qz_allocated_direct_vs_MassWorld_four_action_real_diagnostic', passed=False,
                  created_at_utc=datetime.now(timezone.utc).isoformat(), cases=[],
                  task_success_or_formal_acceptance_claim=False, new_weights=False,
                  automatic_retry=False, native_logs_saved=False,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__).resolve(), ROOT / 'integrations/massworld/src/massworld_embodied/backend.py']})
    samples, stop = [], threading.Event()
    phase = {'name': 'asset_and_runtime_preflight'}
    def sample():
        while not stop.is_set():
            p = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if p.returncode == 0:
                devices = []
                for row in p.stdout.splitlines():
                    index, name, util, memory = [v.strip() for v in row.split(',')]
                    devices.append(dict(index=int(index), name=name, gpu_percent=int(util), memory_used_MiB=int(memory)))
                samples.append(dict(at_utc=datetime.now(timezone.utc).isoformat(), phase=phase['name'], devices=devices))
            stop.wait(2.)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', XLA_PYTHON_CLIENT_MEM_FRACTION='0.6',
        PYTHONPATH=os.pathsep.join((str(ROOT), str(args.overlay.resolve()))),
        QWEN3VL_PATH=str(ROOT.parent.parent.parent / 'Model/Qwen3-VL-4B-Instruct-processor'))
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        environment.pop(name, None)
    try:
        if args.library_manifest:
            environment = library_environment(environment, args.library_manifest)
            report['library_manifest_sha256'] = hashlib.sha256(args.library_manifest.read_bytes()).hexdigest()
        if config.model.adapter == 'openpi':
            if not args.model_cache_manifest:
                raise ValueError('offline OpenPI tokenizer manifest is required')
            from platform_model_cache import environment as cache_environment
            environment = cache_environment(environment, args.model_cache_manifest)
            report['model_cache_manifest_sha256'] = hashlib.sha256(args.model_cache_manifest.read_bytes()).hexdigest()
        command, cwd, client = server_spec(config)
        # Fail cheaply on missing OS/Python dependencies before hashing large
        # checkpoints or launching a model. No weights or actions in this gate.
        report['runtime_dependency_checks'] = []
        simulator_modules = ['cv2', 'sapien', 'mplib'] if config.benchmark.adapter == 'robotwin' else ['cv2', 'mujoco']
        for role, python, modules in [('model', command[0], ['torch', 'cv2']), ('benchmark', str(client), simulator_modules)]:
            result = subprocess.run([python, str(Path(__file__).with_name('probe_platform_runtime.py')), '--child', '--modules', *modules],
                                    env=environment, capture_output=True, text=True, timeout=180)
            dependency = dict(role=role, exit_code=result.returncode, dependency_facts=dependency_facts(result.stderr))
            for line in reversed(result.stdout.splitlines()):
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and 'imports' in data:
                        dependency.update(data)
                        break
                except ValueError:
                    continue
            report['runtime_dependency_checks'].append(dependency)
            if result.returncode != 0 or not dependency.get('imports') or not all(x['passed'] for x in dependency['imports']):
                raise RuntimeError('offline_runtime_dependency_gate_failed')
        preflight = check(config, verify_assets=True)
        write(args.output / 'preflight.json', preflight)
        if not preflight['structurally_compatible'] or not preflight['assets_verified']:
            raise RuntimeError('structure_or_local_asset_verification_failed')
        report['client_python'] = str(client)
        report['model_command'] = command
        report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        for mode in ('direct', 'bridge'):
            phase['name'] = mode + '_server_loading'
            row = dict(mode=mode, passed=False, server_restarted=True)
            server = worker = None
            with ThreadPoolExecutor(max_workers=1) as pool:
                logs = None
                try:
                    with socket.socket() as reservation:
                        reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        reservation.bind(('127.0.0.1', config.model.options['port']))
                    # Preserve the scheduler-assigned GPU mask. Both processes
                    # run inside the same single-card allocation.
                    server = subprocess.Popen(command, cwd=cwd, env=environment,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                    logs = pool.submit(server.communicate)
                    deadline = time.monotonic() + 900
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError('owned_server_start_failed')
                        try:
                            with socket.create_connection(('127.0.0.1', config.model.options['port']), timeout=.5):
                                break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError('diagnostic_server_start_bound')
                            time.sleep(1)
                    phase['name'] = mode + '_evaluation'
                    print(json.dumps(dict(mode=mode, server_ready=True)), flush=True)
                    child_env = dict(environment, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
                    if config.benchmark.adapter == 'robotwin':
                        child_env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(ROOT.parent / 'upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json')
                    child_env['CUDA_HOME'] = str(ROOT.parent / 'upstream/toolchains/cuda-12.1-stage4/toolkit')
                    target = args.output / mode
                    worker = subprocess.Popen([str(client), __file__, '--child', mode, '--config', str(args.config.resolve()),
                        '--output', str(target.resolve())], cwd=ROOT, env=child_env,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                    out, err = worker.communicate(timeout=1200)
                    row.update(exit_code=worker.returncode, failure_categories=categories((out + err).decode(errors='replace')),
                               dependency_facts=dependency_facts((out + err).decode(errors='replace')))
                    backend = target if mode == 'direct' else target / 'backend'
                    summary = backend / 'summary.json'
                    if summary.is_file():
                        value = json.loads(summary.read_text())
                        row.update(synthetic=value['synthetic'], completed=value['completed'], errors=value['errors'])
                        row['passed'] = worker.returncode == 0 and value['errors'] == 0 and not value['synthetic']
                except Exception as error:
                    row['exception_chain'] = public_exception(error)
                    row['execution_state_unknown'] = worker is not None
                finally:
                    stop_owned(worker)
                    stop_owned(server)
                    if logs:
                        out, err = logs.result(timeout=30)
                        row['server_failure_categories'] = categories((out + err).decode(errors='replace'))
                        row['server_dependency_facts'] = dependency_facts((out + err).decode(errors='replace'))
            report['cases'].append(row)
            write(args.output / (mode + '-receipt.json'), row)
            print(json.dumps(row), flush=True)
            if not row['passed']:
                break
        report['passed'] = len(report['cases']) == 2 and all(r['passed'] for r in report['cases'])
        if report['passed']:
            report['comparison'] = compare(args.output / 'direct', args.output / 'bridge/backend', config.episodes[0].episode_id)
            report['strict_reinference_equivalence'] = report['comparison']['all_recorded_actions_exact']
            report['passed'] = report['comparison']['initial_observation_exact'] and report['comparison']['first_public_model_input_exact']
    except Exception as error:
        report['exception_chain'] = public_exception(error)
    finally:
        stop.set()
        monitor.join(timeout=15)
        utilizations = [d['gpu_percent'] for sample in samples for d in sample['devices']]
        mean = sum(utilizations) / len(utilizations) if utilizations else None
        write(args.output / 'gpu-utilization.json', dict(samples=samples, all_card_sample_mean_percent=mean,
            target_percent=20, target_met=mean is not None and mean >= 20,
            extra_occupancy_load_started=False, whole_project_lifetime_average=False))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.output / 'receipt.json', report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--library-manifest', type=Path)
    parser.add_argument('--model-cache-manifest', type=Path)
    parser.add_argument('--child', choices=('direct', 'bridge'))
    args = parser.parse_args()
    if not args.child and args.overlay is None:
        parser.error('--overlay is required for the supervisor')
    raise SystemExit(child(args) if args.child else execute(args))
