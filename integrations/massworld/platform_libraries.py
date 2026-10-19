"""Stage portable user-space dependencies, never CUDA/driver or C-runtime libs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {'libc.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1', 'libm.so.6',
            'libgcc_s.so.1', 'libstdc++.so.6', 'ld-linux-x86-64.so.2'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed(name):
    return (bool(re.fullmatch(r'[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*', name))
            and name not in EXCLUDED and not name.startswith(('libcuda', 'libnvidia'))
            and '_nvidia' not in name)


def environment(base, manifest):
    manifest = Path(manifest).resolve()
    manifest.relative_to(ROOT / 'outputs')
    record = json.loads(manifest.read_text())
    directory = manifest.parent / 'lib'
    for item in record['libraries']:
        if not allowed(item['name']):
            raise ValueError('driver/C-runtime library overlay is forbidden')
        path = directory / item['name']
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != item['sha256']:
            raise ValueError('private runtime library checksum mismatch')
    result = dict(base)
    result['LD_LIBRARY_PATH'] = str(directory) + (':' + result['LD_LIBRARY_PATH'] if result.get('LD_LIBRARY_PATH') else '')
    return result


def prepare(output):
    output.resolve().relative_to(ROOT / 'outputs')
    output.mkdir(parents=True, exist_ok=False)
    libraries = output / 'lib'
    libraries.mkdir()
    cache = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, check=True).stdout
    paths = {}
    for name, path in re.findall(r'^\s*(\S+) \([^\n]*x86-64[^\n]*\) => (\S+)', cache, re.M):
        paths.setdefault(name, Path(path))
    old = ROOT / 'outputs/mn-runtime-isaac-private-libs-20261008-run1/library-root/usr/lib/x86_64-linux-gnu'
    paths.setdefault('libGLU.so.1', old / 'libGLU.so.1.3.1')
    paths.setdefault('libOpenGL.so.0', old / 'libOpenGL.so.0.0.0')
    pending = ['libGL.so.1', 'libGLX.so.0', 'libGLdispatch.so.0', 'libEGL.so.1',
               'libglib-2.0.so.0', 'libgthread-2.0.so.0', 'libGLU.so.1', 'libOpenGL.so.0']
    records = {}
    while pending:
        name = pending.pop()
        if name in records or not allowed(name):
            continue
        source = paths[name].resolve(strict=True)
        headers = subprocess.run(['readelf', '-d', str(source)], capture_output=True, text=True, check=True).stdout
        needed = re.findall(r'\(NEEDED\).*\[([^\]]+)\]', headers)
        target = libraries / name
        shutil.copyfile(source, target)
        records[name] = {'name': name, 'source': str(source), 'sha256': digest(target), 'needed': needed}
        pending.extend(needed)
    record = {'kind': 'offline_user_space_library_overlay_not_driver_or_system_install',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'network_downloads': False,
              'global_or_native_environment_modified': False, 'libraries': list(records.values())}
    with (output / 'manifest.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({'libraries': len(result['libraries']), 'manifest': str(args.output / 'manifest.json')}))
