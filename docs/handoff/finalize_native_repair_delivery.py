"""Bind explicit completed diagnostic evidence, without model or scene calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    delivery.relative_to(ROOT / 'outputs')
    args.output.resolve().relative_to(ROOT / 'docs/handoff')
    report = read(delivery / 'receipt.json')
    assert report['completed_pairs'] == 7 and report['strict_pairs'] == 3
    assert report['primary_actual_policy_actions'] == 56
    assert report['supplementary_actual_policy_actions'] == 8
    gate_path = ROOT / 'outputs/native-repair-evidence-gate-20261023-final1/receipt.json'
    gate = read(gate_path)
    assert gate['passed'] and gate['bridge_tests']['tests'] == 34
    for name, expected in gate['production_runtime_sha256'].items():
        assert digest(ROOT / name) == expected, name
    previous_path = ROOT / 'docs/handoff/current-only-execution-20261023-index.json'
    previous = read(previous_path)
    assert digest(ROOT / previous['report']) == previous['report_sha256']
    original_index = subprocess.check_output(['git', 'show', '2d1ba7e:docs/handoff/current-only-execution-20261023-index.json'], cwd=ROOT)
    assert previous_path.read_bytes() == original_index
    evidence = {}

    def bind(path):
        path = path.absolute()
        path.resolve().relative_to(ROOT)
        key = str(path.relative_to(ROOT))
        evidence[key] = {'sha256': digest(path), 'bytes': path.stat().st_size}

    for suffix in ('receipt.json', 'native-environment-repair-and-comparison.html',
                   'preview-lifecycle.json', 'http-verification.json', 'video-http-verification.json'):
        bind(delivery / suffix)
    http = read(delivery / 'http-verification.json')
    video_http = read(delivery / 'video-http-verification.json')
    assert http['passed'] and video_http['passed'] and video_http['video_count'] == 21
    bind(gate_path)
    for folder in ('native-env-repair-20261023-run1', 'native-env-links-after-20261023-run1',
                   'native-runtime-readiness-20261023-run2', 'native-render-readiness-20261023-run1',
                   'native-repair-package-worker-20261023-run1',
                   'native-local-checkpoint-cache-20261023-run1', 'native-local-lingbot-cache-20261023-run1',
                   'native-code-prefetch-20261023-run1'):
        bind(ROOT / 'outputs' / folder / 'receipt.json')
    for folder in ('native-repair-baseline-20261023-final3', 'native-repair-baseline-py310-20261023-final3'):
        path = ROOT / 'outputs' / folder / 'baseline-check.json'
        baseline = read(path)
        assert baseline['passed'] and baseline['checks'][0]['tests_run'] == 688
        assert baseline['checks'][0]['skips'] == 0
        assert [row['actual_exit_code'] for row in baseline['checks'][1:]] == [0, 0, 0, 1, 2]
        bind(path)
        bind(path.parent / 'unittest.txt')
    for case in report['cases']:
        directory = ROOT / case['directory']
        assert case['framework_completed'] and case['native_completed']
        assert case['framework_statistics']['env_steps'] == case['native_actions'] == 4
        for suffix in ('framework/manifest.json', 'framework/summary.json', 'native/native-result.json'):
            bind(directory / suffix)
        manifest = read(directory / 'framework/manifest.json')
        episode = manifest['config']['episodes'][0]['episode_id']
        for name in ('result.json', 'model_input.public.jsonl', 'trace.internal.jsonl'):
            bind(directory / 'framework/episodes' / episode / name)
        for folder in ('model-exchanges', 'controller-exchanges', 'native/native-exchanges'):
            if (directory / folder).is_dir():
                for path in sorted((directory / folder).iterdir()):
                    if path.is_file():
                        bind(path)
        video_root = ROOT / 'outputs/current-native-videos-20261023' / case['id']
        videos = read(video_root / 'receipt.json')
        assert len(videos['videos']) == 3 and videos['additional_model_or_environment_calls'] == 0
        bind(video_root / 'receipt.json')
        for video in videos['videos']:
            path = video_root / video['file']
            assert digest(path) == video['sha256']
            bind(path)
    supplement = ROOT / 'outputs/current-controller-boundary-20261023-run1/openwam_robotwin'
    bind(supplement / 'native/native-result.json')
    bind(supplement / 'controller-exchanges/receipt.json')
    bind(supplement / 'framework/summary.json')
    for path in sorted((ROOT / 'outputs/current-wire-diagnostics-20261023').glob('*.json')):
        bind(path)
    gpu_path = ROOT / 'outputs/native-env-repair-20261023-run1/gpu-utilization.json'
    gpu = read(gpu_path)
    bind(gpu_path)
    libs = ROOT / 'outputs/worldeval-platform-assets-20261019/LIBERO'
    assert subprocess.check_output(['git', '-C', str(libs), 'rev-parse', 'HEAD'], text=True).strip() == '8f1084e3132a39270c3a13ebe37270a43ece2a01'
    subprocess.run(['git', '-C', str(libs), 'diff', '--quiet', 'HEAD'], check=True)
    resource = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader,nounits'], text=True)
    devices = [dict(zip(('index', 'name', 'memory_MiB', 'driver_version'), [v.strip() for v in line.split(',')])) for line in resource.splitlines()]
    disks = {name: dict(zip(('total', 'used', 'free'), shutil.disk_usage(path))) for name, path in (('shared_workspace', ROOT), ('local_tmp', Path('/tmp')))}
    receipt = {
        'kind': 'native_environment_repair_and_seven_current_real_comparison_delivery_not_stage_acceptance',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'branch': git('branch', '--show-current'),
        'implementation_source_commit': git('rev-parse', 'HEAD'), 'public_contract': '0.3',
        'report': report['report'], 'report_sha256': digest(ROOT / report['report']),
        'evidence': evidence, 'cases': report['cases'], 'completed_primary_pairs': 7, 'strict_primary_pairs': 3,
        'primary_policy_execution_actions': 56, 'supplementary_policy_execution_actions': 8,
        'total_policy_execution_actions': 64, 'real_GPU_forward_count': None,
        'videos': 21, 'repaired_interpreter_links': 9, 'available_environment_interpreters': 18,
        'dependency_environments_passed': 14, 'empty_training_scaffold_not_claimed_ready': 'pi05-arx-mn',
        'main_tests_each_Python': 688, 'original_tests_retained': 676, 'new_synthetic_diagnostic_tests': 12,
        'tested_Python_versions': ['3.13.5', '3.10.21'], 'bridge_synthetic_tests': 34, 'skips': 0,
        'CLI_exit_codes': [0, 0, 0, 1, 2], 'production_runtime_python_sha256_unchanged': gate['production_runtime_sha256'],
        'previous_delivery_index_and_report_unchanged': True,
        'same_historical_archived_Python_binary_and_package_metadata': True,
        'shared_assets_originals_or_upstream_sources_modified': False,
        'clean_same_revision_LIBERO_copy_used_without_reverting_original_dirty_source': True,
        'weights_normalization_controller_scoring_or_public_protocol_changed': False,
        'same_bytes_local_weight_cache_files': 38, 'same_bytes_local_weight_cache_bytes': 100023996010,
        'resources': {'system': platform.system(), 'architecture': platform.machine(), 'kernel': platform.release(), 'GPUs': devices, 'disk_bytes': disks},
        'GPU_sampling_window': {'start': gpu['samples'][0]['at_utc'], 'end': gpu['samples'][-1]['at_utc'],
            'samples': len(gpu['samples']), 'mean_percent': gpu['aggregate_mean_percent'], 'target_percent': 20,
            'target_met': gpu['target_met'], 'includes_preexisting_occupancy_not_evaluation_work': True,
            'whole_project_lifetime_average': False},
        'qz_authentication': 'expired_invalid_grant_user_login_required', 'new_platform_jobs': 0,
        'new_MassWorld_bridge_real_runs': 0, 'live_loaded_weight_tensor_attestation': False,
        'unique_root_cause_of_remaining_strict_differences_proven': False,
        'actual_controller_arguments_newly_captured': True,
        'full_task_or_multi_seed_success_rate_evaluation': False,
        'DreamZero_inference_or_simulation_started': False, 'DreamZero_hardware_deferred_to_user': True,
        'formal_stage4_acceptance': False, 'contract_frozen': False,
        'HTTP_verified': True, 'HTTP_allowed_files': http['exposed_file_count'],
        'all_21_videos_byte_ranges_verified': True, 'preview_bind': '127.0.0.1',
        'preview_port': read(delivery / 'preview-lifecycle.json')['port'], 'remote_browser_access_verified': False,
        'remaining': ['four primary strict differences remain', 'fresh ARX/Cosmos and full task suites not executed this round',
            'weight action physical periods and universal benchmark prefix permissions still unverified',
            'platform H100 GLX rendering gap not fixed by local RTX validation',
            'DreamZero two 80GB H100s deferred', 'stage4 acceptance and contract freeze await user'],
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'evidence_files': len(evidence), 'pairs': 7, 'strict_pairs': 3, 'actions': 64, 'videos': 21}))


if __name__ == '__main__':
    main()
