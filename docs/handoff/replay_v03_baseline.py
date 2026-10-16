"""Current 0.3 regression; old tool/discrete CLI fixtures are explicitly retired."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-tests', type=int, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(kind='current_v03_software_and_synthetic_CLI_regression', passed=False,
        checked_at_utc=datetime.now(timezone.utc).isoformat(), current_contract='0.3',
        legacy_443_test_coverage_retained_with_frozen_reference_helpers=True,
        retired_tool_runner_owned_chunk_and_discrete_CLI_are_not_claimed_current_features=True,
        new_real_inference_or_simulation=False, checks=[])
    def command(command, filename):
        result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)
        with (args.output / filename).open('x') as stream:
            stream.write(result.stdout + result.stderr)
        return result
    try:
        tests = command(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'unittest.txt')
        match = re.search(r'Ran (\d+) tests?', tests.stdout + tests.stderr)
        count = int(match.group(1)) if match else 0
        assert tests.returncode == 0 and count == args.expected_tests
        assert not re.search(r'skipped=\d+', tests.stdout + tests.stderr)
        receipt['checks'].append(dict(case='unittest', exit_code=tests.returncode, tests_run=count, skips=0))
        cases = [('single', 0), ('cache', 0), ('native_timing', 0), ('score_error', 1), ('rejected', 2)]
        for case, expected in cases:
            target = args.output / case
            result = command(['-m', 'embodied_eval', 'run', '--config', str(ROOT / f'examples/synthetic_{case}_run.json'), '--output', str(target)], case + '.txt')
            assert result.returncode == expected, (case, result.returncode)
            item = dict(case=case, actual_exit_code=result.returncode, expected_exit_code=expected)
            if expected == 2:
                assert not target.exists()
            else:
                summary = json.loads((target / 'summary.json').read_text())
                assert summary['synthetic'] and summary['contract_version'] == '0.3'
                ep = summary['episodes'][0]['episode_id']
                record = json.loads((target / f'episodes/{ep}/result.json').read_text())
                item['execution_statistics'] = record['execution_statistics']
                assert record['execution_statistics']['env_steps'] == 6
                assert record['execution_statistics']['model_forward_count'] == (2 if case == 'cache' else 6)
                assert record['metrics']['success']['valid'] == (expected == 0)
                if case == 'native_timing':
                    assert record['execution_statistics']['physics_steps'] == 33
            receipt['checks'].append(item)
        receipt['passed'] = True
    finally:
        with (args.output / 'baseline-check.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    print(json.dumps(receipt))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
