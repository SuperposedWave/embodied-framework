"""Build a fresh, pinned MassWorld test workspace; never edit the supplied ZIP.

No downloads, dependency installation, setup commands or model execution.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import zipfile

ZIP_SHA256 = 'e64747418982c8787645925b86f78f120024bea3fbe1dc84496e3ac80966034a'
PREFIX = 'MassWorld-main/WorldEvalKit/'


def prepare(archive, output, patched=True):
    archive, output = Path(archive), Path(output).resolve()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ZIP_SHA256:
        raise ValueError('archive SHA256 does not match reviewed MassWorld snapshot')
    if output.exists():
        raise FileExistsError('fresh output directory required')
    root = Path(__file__).resolve().parent
    inventory = {}
    with zipfile.ZipFile(archive) as source:
        selected = [i for i in source.infolist() if i.filename.startswith(PREFIX) and not i.is_dir()]
        paths = []
        for info in selected:
            relative = PurePosixPath(info.filename[len(PREFIX):])
            if relative.is_absolute() or '..' in relative.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('unsafe archive entry')
            if str(relative) in paths:
                raise ValueError('duplicate archive entry')
            paths.append(str(relative))
        output.mkdir(parents=True, exist_ok=False)
        checkout = output / 'WorldEvalKit'
        for info, relative in zip(selected, paths):
            data = source.read(info)
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            inventory[relative] = hashlib.sha256(data).hexdigest()
        # Retain the containing repository's separate license, too.
        (output / 'MassWorld-LICENSE').write_bytes(source.read('MassWorld-main/LICENSE'))
    patch = root / 'patches' / 'execution-backend.patch'
    if patched and patch.exists():
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=checkout, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=checkout, check=True)
    for src in sorted((root / 'worldevalkit-overlay').glob('*.py')) if patched else ():
        shutil.copyfile(src, checkout / 'src/worldevalkit' / src.name)
    receipt = dict(archive_sha256=ZIP_SHA256, original_files=inventory,
                   archive_modified=False, executable_attachment_instructions_followed=False,
                   patch_applied=patched and patch.exists(), workspace=str(checkout))
    (output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return checkout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', action='store_true', help='extract exact original for regression comparison')
    args = parser.parse_args()
    print(prepare(args.archive, args.output, patched=not args.original))
