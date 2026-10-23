"""Current finite 0.3 real runs, supervised outside Runner; raw logs stay in RAM.

No downloads, native-controller changes, occupancy changes, retries or scoring
changes. Sources, weights and initializations come from completed saved runs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import resource
import subprocess
import sys
import threading
import time
from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_mn_trials import server_spec, stop_owned


def child(args):
    import random
    import numpy as np
    from embodied_eval.config import load_config
    from embodied_eval.v03.runner import run
    cfg = load_config(args.config)
    random.seed(0)
    np.random.seed(0)
    if cfg.benchmark.adapter == 'vlabench':
        sys.path.insert(0, cfg.benchmark.options['repo'])
        os.environ['VLABENCH_ROOT'] = str(Path(cfg.benchmark.options['repo']) / 'VLABench')
        import VLABench.robots
        import VLABench.tasks
        import VLABench.envs
    try:
        from record_current_policy_exchange import record_exchanges
        from record_current_controller_boundary import record_controller
        with record_controller(args.output.parent / 'controller-exchanges'), record_exchanges(args.output.parent / 'model-exchanges'):
            return run(cfg, args.output).exit_code
    except Exception as error:
        write(args.output.parent / 'client-exception.json', {'exception_chain': public_exception(error)})
        return 1


def supervised_side(command, cwd, client, cfgpath, target, args, side, server_env, client_env):
    """Fresh owned policy process per side; no re-send or automatic retry."""
    row = {"side": side, "passed": False, "fresh_owned_server": True}
    server = worker = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        logs = None
        try:
            server = subprocess.Popen(command, cwd=cwd, env=server_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            row["owned_server_pid"] = server.pid
            logs = pool.submit(server.communicate)
            deadline = time.monotonic() + getattr(args, 'startup_timeout_s', 900)
            while True:
                if server.poll() is not None:
                    raise RuntimeError('owned_server_start_failed')
                try:
                    with socket.create_connection(('127.0.0.1', args.port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('diagnostic_startup_bound')
                    time.sleep(1)
            print(json.dumps(dict(case=target.name, side=side, server_ready=True)), flush=True)
            if side == "framework":
                invocation = [str(client), __file__, '--child', '--config', str(cfgpath.resolve()),
                    '--output', str((target / 'framework').resolve())]
            else:
                invocation = [str(client), str(ROOT / 'docs/handoff/compare_current_native.py'),
                    '--config', str(cfgpath.resolve()), '--framework', str((target / 'framework').resolve()),
                    '--output', str((target / 'native').resolve())]
            worker = subprocess.Popen(invocation, cwd=ROOT, env=client_env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            out, err = worker.communicate(timeout=max(1800, getattr(args, 'wall_time_s', 600) + 600))
            row.update(exit_code=worker.returncode, public_failure_categories=categories((out + err).decode(errors='replace')))
            result_path = target / ('framework/summary.json' if side == 'framework' else 'native/native-result.json')
            row['passed'] = worker.returncode == 0 and result_path.is_file()
            if result_path.is_file():
                value = json.loads(result_path.read_text())
                if side == 'framework':
                    row.update(completed=value['completed'], errors=value['errors'], synthetic=value['synthetic'])
                    row['passed'] = row['passed'] and value['errors'] == 0 and value['completed'] == 1 and not value['synthetic']
                    manifest = json.loads((target / 'framework/manifest.json').read_text())
                    episode = manifest['config']['episodes'][0]['episode_id']
                    result = json.loads((target / 'framework/episodes' / episode / 'result.json').read_text())
                    row['execution_statistics'] = result['execution_statistics']
                    row['passed'] = row['passed'] and result['execution_statistics']['env_steps'] > 0
                else:
                    row.update({k: value.get(k) for k in ('completed', 'strict_equal', 'native_actions', 'initial_observation_exact', 'first_divergence')})
                    row['passed'] = row['passed'] and value['completed']
        except Exception as error:
            row['exception_chain'] = public_exception(error)
            row['policy_execution_state'] = 'unknown_no_retry_or_resend' if worker is not None else 'no_policy_action_started'
        finally:
            stop_owned(worker)
            stop_owned(server)
            if logs:
                out, err = logs.result(timeout=30)
                log = (out + err).decode(errors='replace')
                row['server_public_failure_categories'] = categories(log)
                sys.path.insert(0, str(ROOT / 'integrations/massworld'))
                from platform_diagnostics import dependency_facts, import_failure_details
                row['server_dependency_facts'] = dependency_facts(log)
                row['server_import_failure_details'] = import_failure_details(log)
    write(target / (side + '-supervision.json'), row)
    print(json.dumps(dict(case=target.name, **row)), flush=True)
    return row


def execute(args):
    from embodied_eval.codec import to_data
    from embodied_eval.config import load_config
    from embodied_eval.v03.preflight import check
    from embodied_eval.contracts import Budgets
    gate = json.loads(args.mapping_receipt.read_text())
    if not gate.get('passed'):
        raise ValueError('completed_saved_boundary_gate_required')
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    selected = []
    for case in args.cases:
        if case == 'openwam_dual_physical':
            path = ROOT / 'outputs/openwam-dual-pose-live-20261015-run1/recorded-framework/framework/manifest.json'
        else:
            path = ROOT / f'outputs/native-comparison-{case}-20261009-run1/framework/manifest.json'
        if case not in {row['case'] for row in gate['cases']}:
            raise ValueError('selected_saved_run_must_have_passed_boundary_gate')
        # Historical results select comparison evidence, never a retired
        # runtime/configuration. Run only explicit current example files.
        examples = {"openwam_libero": "stage3-openwam-libero-spatial-pilot.json",
            "openpi_libero": "stage4-pi05-libero-spatial.json",
            "openwam_vlabench": "stage4-openwam-vlabench-select-fruit.json",
            "openpi_vlabench": "stage4-pi05-vlabench-select-fruit.json",
            "openwam_robotwin": "stage4-openwam-robotwin-adjust-bottle.json",
            "lingbot_robotwin": "stage4-lingbot-robotwin-adjust-bottle.json",
            "openwam_dual_physical": "mn_openwam_dual_wxyz_encoded_absolute.json"}
        if case not in examples:
            raise ValueError("current_example_required_for_this_supervisor")
        cfg = load_config(ROOT / "examples" / examples[case])
        if args.asset_cache:
            cached = json.loads(args.asset_cache.read_text())
            if not cached.get('passed'):
                raise ValueError('completed_byte_identical_asset_cache_required')
            source = str(Path(cfg.model.options['checkpoint_dir']).resolve())
            cfg = replace(cfg, model=replace(cfg.model, options={**cfg.model.options,
                'checkpoint_dir': cached['checkpoints'][source]}))
        cfg = replace(cfg, run_id='v03-' + case.replace('_', '-'),
            model=replace(cfg.model, options={**cfg.model.options, 'port': args.port}),
            episodes=(replace(cfg.episodes[0], budgets=Budgets(env_steps=4, model_calls=4, wall_time_s=float(args.wall_time_s))),))
        if args.request_timeout_s is not None:
            cfg = replace(cfg, model=replace(cfg.model, options={**cfg.model.options, 'request_timeout_s': args.request_timeout_s}))
        selected.append((case, cfg))
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                devices = [list(map(int, line.split(','))) for line in result.stdout.splitlines()]
                samples.append(dict(at_utc=datetime.now(timezone.utc).isoformat(), devices=devices, all_GPU_mean_percent=sum(v[1] for v in devices) / len(devices)))
            stop.wait(5.)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    report = dict(kind='v03_bounded_real_runs_not_native_reinference_equivalence_or_task_acceptance', passed=False,
        started_at_utc=datetime.now(timezone.utc).isoformat(), formal_acceptance=False, transitions_per_case=4,
        four_action_diagnostic_budget_not_full_task_scores=True, mapping_gate=str(args.mapping_receipt),
        global_proxy_occupancy_driver_controller_weights_or_normalization_changed=False, raw_logs_saved=False,
        fresh_native_reference_requested=args.native_reference, cases=[])
    try:
        for case, cfg in selected:
            target = args.output / case
            target.mkdir()
            cfgpath = target / 'config.json'
            write(cfgpath, to_data(cfg))
            preflight = check(cfg, verify_assets=True)
            write(target / 'preflight.json', preflight)
            if not preflight['structurally_compatible'] or not preflight['assets_verified']:
                row = dict(case=case, passed=False, gate='preflight', policy_execution_state='no_policy_action_started',
                    issues=preflight['issues'], asset_errors=preflight['asset_errors'])
                report['cases'].append(row)
                write(target / 'receipt.json', row)
                print(json.dumps(row), flush=True)
                continue
            command, cwd, client = server_spec(cfg)
            with socket.socket() as port:
                port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                port.bind(('127.0.0.1', args.port))
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                QWEN3VL_PATH=str(WORKSPACE.parent.parent / 'Model/Qwen3-VL-4B-Instruct-processor'))
            env.pop('LD_LIBRARY_PATH', None)
            if cfg.model.adapter == 'openpi':
                sys.path.insert(0, str(ROOT / 'integrations/massworld'))
                from platform_model_cache import environment as cache_environment
                env = cache_environment(env, ROOT / 'outputs/worldeval-openpi-public-cache-20261019-run1/manifest.json')
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                env.pop(key, None)
            server_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.model_gpu), XLA_PYTHON_CLIENT_MEM_FRACTION='0.6')
            client_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.simulator_gpu), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                CUDA_HOME=str(WORKSPACE / 'upstream/toolchains/cuda-12.1-stage4/toolkit'))
            if cfg.benchmark.adapter == 'robotwin':
                client_env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(WORKSPACE / 'upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json')
            row = dict(case=case, passed=False, model_gpu=args.model_gpu, simulator_gpu=args.simulator_gpu,
                client_python=str(client), model_command=command)
            framework = supervised_side(command, cwd, client, cfgpath, target, args, 'framework', server_env, client_env)
            row.update(passed=framework['passed'], framework=framework)
            if framework['passed'] and args.native_reference:
                native = supervised_side(command, cwd, client, cfgpath, target, args, 'native', server_env, client_env)
                row.update(native=native, passed=native['passed'] and framework['passed'])
            report['cases'].append(row)
            write(target / 'receipt.json', row)
            print(json.dumps(row), flush=True)
        report['passed'] = all(row['passed'] for row in report['cases'])
    except Exception as error:
        report['supervisor_exception_chain'] = public_exception(error)
    finally:
        stop.set()
        monitor.join(timeout=30)
        mean = sum(row['all_GPU_mean_percent'] for row in samples) / len(samples) if samples else None
        write(args.output / 'gpu-utilization.json', dict(samples=samples, aggregate_mean_percent=mean,
            target_percent=20, target_met=mean is not None and mean >= 20,
            includes_preexisting_occupancy_not_evaluation_work=True, whole_project_lifetime_average=False))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(args.output / 'receipt.json', report)
    return 0 if report['passed'] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--mapping-receipt', type=Path)
    parser.add_argument('--cases', nargs='+', default=['openwam_libero', 'openpi_libero', 'openwam_vlabench', 'openpi_vlabench', 'openwam_robotwin', 'lingbot_robotwin', 'openwam_dual_physical'])
    parser.add_argument('--model-gpu', type=int, default=6)
    parser.add_argument('--simulator-gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=8963)
    parser.add_argument('--native-reference', action='store_true', help='Fresh independent fixed-source reference per current framework run')
    parser.add_argument('--asset-cache', type=Path, help='Completed byte-identical local checkpoint-cache receipt; no tensor or normalization changes')
    parser.add_argument('--request-timeout-s', type=float, help='Explicit existing RPC wait limit for cold-start diagnostics only; defaults remain unchanged')
    parser.add_argument('--wall-time-s', type=int, default=600, help='Existing cooperative episode diagnostic budget; does not change action/physics timing')
    parser.add_argument('--startup-timeout-s', type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.wall_time_s <= 3600 or not 1 <= args.startup_timeout_s <= 3600 or args.request_timeout_s is not None and not 1 <= args.request_timeout_s <= 1800:
        raise ValueError('explicit_bounded_diagnostic_waits_required')
    return child(args) if args.child else execute(args)


if __name__ == '__main__':
    raise SystemExit(main())
