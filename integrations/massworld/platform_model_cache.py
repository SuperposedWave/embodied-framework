"""Stage only the known public OpenPI tokenizer, not a user's full cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RELATIVE = Path('big_vision/paligemma_tokenizer.model')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(base, manifest):
    manifest = Path(manifest).resolve()
    manifest.relative_to(ROOT / 'outputs')
    record = json.loads(manifest.read_text())
    if record['relative'] != str(RELATIVE):
        raise ValueError('only the reviewed public tokenizer cache is supported')
    path = manifest.parent / RELATIVE
    if not path.resolve().is_relative_to(manifest.parent) or digest(path) != record['sha256']:
        raise ValueError('offline tokenizer checksum mismatch')
    return dict(base, OPENPI_DATA_HOME=str(manifest.parent))


def prepare(output):
    output.resolve().relative_to(ROOT / 'outputs')
    output.mkdir(parents=True, exist_ok=False)
    source = Path('/root/.cache/openpi') / RELATIVE
    target = output / RELATIVE
    target.parent.mkdir()
    shutil.copyfile(source, target)
    record = {'kind': 'existing_public_tokenizer_copy_not_new_weights', 'relative': str(RELATIVE),
              'source': str(source), 'upstream_url': 'gs://big_vision/paligemma_tokenizer.model',
              'sha256': digest(target), 'source_sha256': digest(source), 'size': target.stat().st_size,
              'network_downloaded': False, 'credentials_or_other_cache_files_copied': False}
    with (output / 'manifest.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
