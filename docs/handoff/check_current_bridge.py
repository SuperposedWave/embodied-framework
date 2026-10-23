"""Persist current MassWorld bridge tests and five synthetic CLI receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(kind='current_massworld_bridge_software_and_synthetic_CLI', passed=False,
        at_utc=datetime.now(timezone.utc).isoformat(), new_GPU_inference=False,
        new_simulation=False, formal_acceptance=False, CLI=[])
    def command(argv, name):
        proc = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True)
        (args.output / (name + '.txt')).write_text(proc.stdout + proc.stderr)
        return proc
    try:
        tests = command(['-m', 'unittest', 'discover', '-s', 'integrations/massworld/tests', '-v'], 'unittest')
        count = int(re.search(r'Ran (\d+) tests', tests.stdout + tests.stderr)[1])
        assert tests.returncode == 0 and count == 34 and 'skipped=' not in tests.stderr
        receipt.update(tests=count, skips=0)
        for case, expected in [('single', 0), ('cache', 0), ('native_timing', 0), ('score_error', 1), ('rejected', 2)]:
            target = args.output / case
            proc = command(['-m', 'worldevalkit.cli', 'run', '--execution-backend', 'embodied-v03',
                '--backend-config', 'examples/synthetic_' + case + '_run.json', '--output-dir', str(target)], case)
            assert proc.returncode == expected, (case, proc.returncode)
            row = dict(case=case, expected=expected, actual=proc.returncode)
            if expected != 2:
                results = list(target.rglob('backend-result.json'))
                assert len(results) == 1
                evidence = json.loads(results[0].read_text())['evidence']['execution_plan']
                assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
                row['execution_plan_link_and_hash_verified'] = True
            receipt['CLI'].append(row)
        receipt['passed'] = True
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
