"""Index completed independent diagnostic runs without rewriting their evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--gpu-windows', type=Path, nargs='+', required=True)
    parser.add_argument('--port-check', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    rows, attempts, sources, samples = {}, [], {}, []
    for run in args.runs:
        path = run / 'receipt.json'
        receipt = json.loads(path.read_text())
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        attempts.append(dict(receipt=str(path), supervisor_passed=receipt['passed'], cases=[row['case'] for row in receipt['cases']]))
        for row in receipt['cases']:
            if row['case'] in rows:
                raise ValueError('do_not_hide_duplicate_or_failed_attempts')
            row = dict(row, output_directory=str((run / row['case']).resolve()))
            target = Path(row['output_directory']) / 'framework'
            assert row['passed'] and (target / 'summary.json').is_file()
            manifest = json.loads((target / 'manifest.json').read_text())
            summary = json.loads((target / 'summary.json').read_text())
            assert manifest['config']['contract_version'] == summary['contract_version'] == '0.3'
            assert not manifest['synthetic'] and summary['errors'] == 0 and summary['completed'] == 1
            task = manifest['tasks'][0]['episode_id']
            result = json.loads((target / f'episodes/{task}/result.json').read_text())
            assert result['execution_statistics']['env_steps'] == 4
            row['execution_statistics'] = result['execution_statistics']
            row['metrics'] = result['metrics']
            for path in (target / 'manifest.json', target / 'summary.json', target / f'episodes/{task}/result.json'):
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows[row['case']] = row
    assert len(rows) == 7
    for path in args.gpu_windows:
        value = json.loads(path.read_text())
        samples.extend(value['samples'])
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Nonoverlapping global windows only; per-run parallel samples are NOT
    # pooled, which would overweight periods when more supervisors were alive.
    times = [row['at_utc'] for row in samples]
    assert times == sorted(times) and len(set(times)) == len(times)
    mean = sum(row['all_GPU_mean_percent'] for row in samples) / len(samples)
    port = json.loads(args.port_check.read_text())
    assert port['passed']
    report = dict(kind='v03_seven_bounded_real_case_index_not_stage_acceptance', passed=True,
        at_utc=datetime.now(timezone.utc).isoformat(), cases=list(rows.values()), attempts=attempts,
        four_actions_per_case_not_full_task_success_rate=True, current_protocol='0.3',
        sequential_supervisor_first_case_completed_then_port_rebind_failed=True,
        first_supervisor_failure_is_not_erased=True, local_port_recovery_test=str(args.port_check),
        remaining_cases_use_independent_ports=True, source_evidence_sha256=sources,
        global_proxy_occupancy_weights_normalization_controller_or_scoring_changed=False,
        formal_acceptance=False)
    for filename, value in (('receipt.json', report), ('gpu-utilization.json', dict(samples=samples, aggregate_mean_percent=mean,
            target_percent=20, target_met=mean >= 20, nonoverlapping_windows=True,
            includes_preexisting_occupancy_not_evaluation_work=True, whole_project_lifetime_average=False))):
        with (args.output / filename).open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(dict(passed=True, real_cases=7, actions=28, GPU_mean_percent=mean)))


if __name__ == '__main__':
    main()
