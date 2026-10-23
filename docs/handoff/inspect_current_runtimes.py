"""Read-only interpreter-link inventory; not a dependency/GPU readiness test."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for directory in sorted((ROOT.parent / 'upstream/envs').iterdir()):
        executable = directory / 'bin/python'
        if not executable.exists() and not executable.is_symlink():
            continue
        rows.append({'environment': directory.name, 'python': str(executable.absolute()),
            'symlink_target': str(executable.readlink()) if executable.is_symlink() else None,
            'executable_available': executable.is_file() and os.access(executable, os.X_OK),
            'dependency_imports_verified': False, 'GPU_runtime_verified': False})
    receipt = {'kind': 'read_only_native_interpreter_links_not_adapter_or_GPU_acceptance',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'environments': rows,
        'missing_interpreters': [r['environment'] for r in rows if not r['executable_available']],
        'external_environment_modified': False, 'proxy_modified': False, 'weights_loaded': False}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'environments': len(rows), 'missing_interpreters': receipt['missing_interpreters']}))


if __name__ == '__main__':
    main()
