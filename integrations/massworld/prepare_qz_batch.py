"""Prepare verified OFFLINE job requests, never submit or download anything."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--clean-libero', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=['openwam_libero','openpi_libero','openwam_vlabench','openpi_vlabench','openwam_robotwin','lingbot_robotwin','openwam_dual_physical'])
    args = parser.parse_args()
    from embodied_eval.v03.migration import migrate
    from embodied_eval.v03.preflight import check
    from embodied_eval.codec import to_data
    from embodied_eval.contracts import Budgets
    gate = json.loads((ROOT / 'outputs/massworld-saved-boundaries-20261019-run1/receipt.json').read_text())
    assert gate['passed']
    selection = json.loads(args.selection.read_text())
    cases = args.cases or ['openwam_libero', 'openpi_libero', 'openwam_vlabench', 'openpi_vlabench',
             'openwam_robotwin', 'lingbot_robotwin', 'openwam_dual_physical']
    assert len(cases) * selection['gpu_count_per_job'] <= selection['active_plus_queued_GPU_limit'] <= 32
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    def prepare(case):
        assert case in {x['case'] for x in gate['cases']}
        if case == 'openwam_dual_physical':
            source = ROOT / 'outputs/openwam-dual-pose-live-20261015-run1/recorded-framework/framework/manifest.json'
        else:
            source = ROOT / ('outputs/native-comparison-' + case + '-20261009-run1/framework/manifest.json')
        cfg, _ = migrate(json.loads(source.read_text())['config'])
        name = 'worldeval-' + case.replace('_', '-') + '-' + stamp
        folder = args.output / case
        folder.mkdir()
        benchmark = cfg.benchmark
        if benchmark.adapter == 'libero' and Path(benchmark.options['repo']).resolve() == ROOT.parent / 'upstream/LIBERO':
            benchmark = replace(benchmark, options={**benchmark.options, 'repo': str(args.clean_libero.resolve())})
        cfg = replace(cfg, run_id=name, model=replace(cfg.model, options={**cfg.model.options, 'port': 38977}),
            benchmark=benchmark, episodes=(replace(cfg.episodes[0], budgets=Budgets(env_steps=4, model_calls=4, wall_time_s=600.)),))
        write(folder / 'config.json', to_data(cfg))
        checked = check(cfg, verify_assets=True)
        write(folder / 'preflight-local.json', checked)
        row = dict(case=case, name=name, source_manifest=str(source), source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   config=str((folder / 'config.json').resolve()), ready=checked['structurally_compatible'] and checked['assets_verified'],
                   issues=checked['issues'], asset_errors=checked['asset_errors'])
        if not row['ready']:
            print(json.dumps(row), flush=True)
            return row
        python = str(ROOT.parent / 'upstream/envs/libero/bin/python')
        invocation = [python, 'integrations/massworld/run_qz_bounded_validation.py', '--config', row['config'],
                      '--overlay', str(args.overlay.resolve()), '--output', str((folder / 'runtime').resolve())]
        command = 'cd ' + shlex.quote(str(ROOT)) + ' && '
        version = {'openpi': '3.11', 'lingbot_vla': '3.12'}.get(cfg.model.adapter)
        if version:
            command += shlex.join([python, 'integrations/massworld/bootstrap_platform_runtime.py',
                '--manifest', str(args.runtime_manifest.resolve()), '--version', version]) + ' && '
        command += shlex.join(invocation)
        request = dict(name=name,workspace_id=selection['workspace_id'],project_id=selection['project_id'],
            logic_compute_group_id=selection['logic_compute_group_id'],framework='PyTorch',command=command,
            framework_config=[dict(image=selection['image_address'],instance_count=1,shm_gi=16,spec_id=selection['spec_id'])],
            auto_fault_tolerance=False,fault_tolerance_max_retry=0,fault_tolerance_retry_interval_sec=0,
            max_running_time_ms=selection['job_max_running_time_ms'],reserve_on_fail_ms='0',reserve_on_success_ms='0',
            enable_notification=False,enable_troubleshoot=False,
            envs=[dict(name='PYTHONDONTWRITEBYTECODE',value='1'),dict(name='PYTHONUNBUFFERED',value='1'),dict(name='WORLDEVAL_PLATFORM_TASK',value='1'),
                  dict(name='HF_HUB_OFFLINE',value='1'),dict(name='TRANSFORMERS_OFFLINE',value='1')],
            description='worldeval: '+case+'，直接／MassWorld各4动作；固定已有权重与协议，离线运行，无重试，非正式验收。')
        write(folder / 'job-request.json',request)
        row.update(request=str((folder / 'job-request.json').resolve()),gpus=selection['gpu_count_per_job'])
        print(json.dumps(dict(case=case,ready=True,name=name,gpus=row['gpus'])),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(prepare,cases))
    index = dict(kind='prepared_not_submitted_offline_qz_batch',created_at_utc=datetime.now(timezone.utc).isoformat(),
                 gpu_limit=32,maximum_planned_concurrent_gpus=sum(r.get('gpus',0) for r in rows),selection=selection,cases=rows)
    write(args.output/'prepared-batch.json',index)
    print(json.dumps(dict(ready=sum(x['ready'] for x in rows),total=len(rows),index=str(args.output/'prepared-batch.json'))))


if __name__ == '__main__':
    main()
