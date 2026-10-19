"""Supplement original/patched CPU wire-protocol tests in an isolated compatible Python.

The suite uses fake services/simulators, not real model inference or task scores.
No dependency or source changes are made by this verifier.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--patched', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(kind='compatible_optional_client_CPU_tests_not_model_or_simulator_validation',
                   passed=False, created_at_utc=datetime.now(timezone.utc).isoformat(), checks=[],
                   real_inference=False, real_simulation=False)
    try:
        for label, root in [('original', args.original.resolve()), ('patched', args.patched.resolve())]:
            env = dict(os.environ, PYTHONPATH=str(root / 'src'))
            for pattern in ('test_gpt_droid.py', 'test_droid_sim.py', 'test_file_policy.py'):
                # Keep the venv executable path: resolve() follows its symlink
                # to the base interpreter and silently loses the venv packages.
                proc = subprocess.run([str(args.python.absolute()), '-m', 'unittest', 'discover',
                    '-s', str(root / 'tests'), '-p', pattern, '-v'], env=env, capture_output=True, text=True)
                text = proc.stdout + proc.stderr
                name = label + '-' + pattern[:-3] + '.txt'
                (args.output / name).write_text(text)
                count = re.search(r'Ran (\d+) tests?', text)
                row = dict(source=label, suite=pattern, exit_code=proc.returncode,
                           tests=int(count.group(1)) if count else 0, log=name,
                           skipped=re.findall(r'\.\.\. skipped [^\n]+', text),
                           log_sha256=hashlib.sha256(text.encode()).hexdigest())
                receipt['checks'].append(row)
                print(json.dumps(row), flush=True)
                assert proc.returncode == 0 and not row['skipped']
        probe = "import sys,json; from importlib.metadata import version; print(json.dumps({'python':sys.version.split()[0], 'packages':{n:version(n) for n in ['numpy','Pillow','openpi-client','jsonschema','PyYAML','python-dotenv','websockets','av','imageio-ffmpeg']}}))"
        receipt['environment'] = json.loads(subprocess.check_output([str(args.python.absolute()), '-c', probe], text=True))
        receipt['passed'] = True
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
