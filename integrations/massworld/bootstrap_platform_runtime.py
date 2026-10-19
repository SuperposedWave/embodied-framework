"""Offline restore of an already-verified Python runtime in a new task container.

Never rewrites a shared venv or a historical absolute interpreter path. Existing
container runtimes must match the staged binary; they are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

DESTINATION = Path('/root/.local/share/uv/python')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def bootstrap(manifest, version):
    record = json.loads(Path(manifest).read_text())['runtimes'][version]
    binary = DESTINATION / record['binary_relative']
    if binary.exists():
        if digest(binary) != record['binary_sha256']:
            raise ValueError('existing container interpreter does not match; refusing overwrite')
        return {'version': version, 'restored': False, 'binary_verified': True}
    if os.environ.get('WORLDEVAL_PLATFORM_TASK') != '1':
        raise RuntimeError('runtime restoration is restricted to the explicit platform task')
    archive = Path(record['archive'])
    if digest(archive) != record['archive_sha256']:
        raise ValueError('staged interpreter archive SHA256 mismatch')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in record['top_level_names']:
                raise ValueError('unexpected runtime archive member')
            if member.issym() or member.islnk():
                target = Path(member.linkname)
                resolved = target if target.is_absolute() else (DESTINATION / member.name).parent / target
                if not resolved.resolve().is_relative_to(DESTINATION):
                    raise ValueError('runtime link leaves the container interpreter directory')
        if any((DESTINATION / name).exists() for name in record['top_level_names']):
            raise FileExistsError('partial interpreter directory already exists; refusing overwrite')
        source.extractall(DESTINATION)
    if digest(binary) != record['binary_sha256']:
        raise ValueError('restored interpreter SHA256 mismatch')
    return {'version': version, 'restored': True, 'binary_verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--version', required=True, choices=('3.11', '3.12'))
    args = parser.parse_args()
    print(json.dumps(bootstrap(args.manifest, args.version)))
