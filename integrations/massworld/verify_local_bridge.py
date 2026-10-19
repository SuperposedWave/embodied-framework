"""Persist local software checks; all robot runs here are explicitly SYNTHETIC."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--patched', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / 'outputs')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(kind='MassWorld_local_bridge_software_not_real_inference', passed=False,
                   created_at_utc=datetime.now(timezone.utc).isoformat(), commands=[],
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   new_GPU_inference=False, new_simulation=False, stage4_formally_accepted=False)
    def command(name, argv, env=None):
        print('START ' + name, flush=True)
        proc = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        (output / (name + '.txt')).write_text(log, encoding='utf-8')
        match = re.search(r'Ran (\d+) tests?', log)
        skipped = re.search(r'skipped=(\d+)', log)
        item = dict(name=name, argv=argv, exit_code=proc.returncode,
                    log=name + '.txt', log_sha256=hashlib.sha256(log.encode()).hexdigest())
        if match:
            item.update(tests=int(match.group(1)), skipped=int(skipped.group(1)) if skipped else 0)
            item['failed_test_ids'] = re.findall(r'^(?:FAIL|ERROR): ([^\n]+)', log, re.MULTILINE)
        receipt['commands'].append(item)
        print(json.dumps(item), flush=True)
        return proc, item
    try:
        for label, workspace in [('massworld-original', args.original), ('massworld-patched', args.patched)]:
            env = dict(os.environ, PYTHONPATH=str(workspace.resolve() / 'src'))
            proc, item = command(label, ['-m', 'unittest', 'discover', '-s', str(workspace.resolve() / 'tests'), '-v'], env)
            # Classify exact pre-existing skips/failures rather than relabel them passes.
        original, patched = receipt['commands'][:2]
        assert original['tests'] == patched['tests']
        assert original['failed_test_ids'] == patched['failed_test_ids']
        assert original['skipped'] == patched['skipped']
        assert original['exit_code'] == patched['exit_code']
        receipt['MassWorld_original_and_patched_test_outcomes_match'] = True
        receipt['MassWorld_all_discovered_tests_passed'] = original['exit_code'] == patched['exit_code'] == 0
        proc, item = command('bridge-tests', ['-m', 'unittest', 'discover', '-s', 'integrations/massworld/tests', '-v'])
        assert proc.returncode == 0 and item['skipped'] == 0
        proc, _ = command('backend-baseline', ['docs/handoff/replay_v03_baseline.py', '--output', str(output / 'backend-baseline'), '--expected-tests', '515'])
        assert proc.returncode == 0
        # Durable CLI receipts, not only temporary unit-test outputs.
        proc, _ = command('catalog', ['-m', 'worldevalkit.cli', 'catalog', 'backend', '--execution-backend', 'embodied-v03'])
        assert proc.returncode == 0
        (output / 'catalog.json').write_text(proc.stdout)
        cli = []
        for case, expected in [('single', 0), ('cache', 0), ('native_timing', 0), ('score_error', 1), ('rejected', 2)]:
            proc, _ = command('bridge-cli-' + case, ['-m', 'worldevalkit.cli', 'run', '--execution-backend', 'embodied-v03',
                '--backend-config', 'examples/synthetic_' + case + '_run.json', '--output-dir', str(output / ('cli-' + case))])
            assert proc.returncode == expected
            cli.append(dict(case=case, expected=expected, actual=proc.returncode))
        receipt['bridge_CLI'] = cli
        # The original backend source must not be changed by this bridge.
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD', 'embodied_eval'], cwd=ROOT, text=True).splitlines()
        for relative in paths:
            original_bytes = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=ROOT)
            assert (ROOT / relative).read_bytes() == original_bytes
        receipt['original_EmbodiedEval_source_files_unchanged'] = len(paths)
        receipt['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'integrations/massworld').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and not any(s.endswith('.egg-info') for s in p.parts)}
        receipt['passed'] = True
    finally:
        (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': receipt['passed'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
