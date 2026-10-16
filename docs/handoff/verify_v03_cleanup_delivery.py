"""Final source-bound cleanup verification, including selected local HTTP files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, build_opener
from serve_report import ROOT, catalog


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--html-validator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/'outputs')
    index = json.loads(args.index.read_text())
    result = dict(kind='v03_internal_cleanup_final_verification_not_stage_acceptance', passed=False,
        at_utc=datetime.now(timezone.utc).isoformat(), stage4_accepted=False, contract_frozen=False,
        new_inference=False, new_simulation=False, desktop_forwarding_verified=False)
    def read(key):
        return json.loads((ROOT/index[key]).read_text())
    def hashes_unchanged(table):
        for path, expected in table.items():
            resolved = Path(path) if Path(path).is_absolute() else ROOT/path
            assert sha(resolved) == expected, path
    try:
        audit, mapping, package = read('source_audit'), read('saved_boundary'), read('package_check')
        assert audit['passed'] and audit['public_wire_changed'] is False
        hashes_unchanged(audit['framework_source_sha256'])
        hashes_unchanged(audit['test_source_sha256'])
        hashes_unchanged(audit['protected_historical_files'])
        assert mapping['passed'] and len(mapping['cases']) == 11
        assert sum(r['observations'] for r in mapping['cases']) == 55
        assert sum(r['predictions'] for r in mapping['cases']) == 44
        assert not mapping['new_GPU_inference'] and not mapping['new_simulation']
        hashes_unchanged(mapping['framework_source_sha256'])
        hashes_unchanged(mapping['protected_sha256'])
        current = {str(p.relative_to(ROOT/'embodied_eval')):sha(p)
            for p in sorted((ROOT/'embodied_eval').rglob('*.py'))}
        for key in ('baseline', 'second_environment_baseline'):
            baseline = read(key)
            assert baseline['passed'] and baseline['checks'][0]['tests_run'] == 515
            assert baseline['checks'][0]['skips'] == 0
            assert [row['actual_exit_code'] for row in baseline['checks'][1:]] == [0, 0, 0, 1, 2]
            for row in baseline['checks'][1:]:
                if row['actual_exit_code'] != 2:
                    manifest = json.loads((ROOT/Path(index[key]).parent/row['case']/'manifest.json').read_text())
                    assert manifest['source_sha256'] == current, row['case']
        assert package['passed'] and not package['weights_loaded'] and not package['dependencies_installed']
        assert package['packaged_python_files_exactly_match_current_source'] == len(current)
        assert sha(ROOT/index['wheel']) == package['wheel_sha256']
        assert [row['python'] for row in package['environments']] == ['3.13.5', '3.10.21']
        assert package['native_worker']['close_acknowledged'] and package['native_worker']['exit_code'] == 0
        assert package['native_worker']['predict_calls'] == 0
        report = ROOT/index['report']
        validation = subprocess.run([sys.executable, str(args.html_validator), str(report)],
            capture_output=True, text=True, cwd=ROOT)
        assert validation.returncode == 0, validation.stdout + validation.stderr
        with (args.output.parent/'report-validation.txt').open('x') as stream:
            stream.write(validation.stdout + validation.stderr)
        route, routes, digests = catalog(report)
        for path in index.get('preview_additional_reports', []):
            _, selected, selected_hashes = catalog(ROOT/path)
            for shared in routes.keys() & selected.keys():
                assert routes[shared] == selected[shared], shared
            routes.update(selected)
            digests.update(selected_hashes)
        opener = build_opener(ProxyHandler({}))
        base = 'http://127.0.0.1:' + str(index['preview_port'])
        http = []
        for path, (payload, _) in routes.items():
            with opener.open(base+quote(path, safe='/'), timeout=20) as response:
                assert response.status == 200 and response.read() == payload
                assert 'nginx' not in response.headers.get('Server', '').lower()
            http.append(dict(path=path, status=200, sha256=digests[path]))
        with opener.open(base+'/', timeout=20) as response:
            assert response.read() == routes[route][0]
        try:
            opener.open(base+'/.git/config', timeout=20)
            raise AssertionError('workspace route must not be exposed')
        except HTTPError as error:
            assert error.code == 404
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        assert subprocess.check_output(['git', 'rev-parse', index['delivery_tag']], cwd=ROOT, text=True).strip() == commit
        assert subprocess.run(['git', 'diff', '--quiet', index['source_commit'], '--', 'embodied_eval', 'tests', 'pyproject.toml'], cwd=ROOT).returncode == 0
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
        result.update(passed=True, source_commit=index['source_commit'], delivery_commit=commit,
            delivery_tag=index['delivery_tag'], tests_each_environment=515, test_skips=0,
            CLI=[0, 0, 0, 1, 2], saved_real_observations=55, saved_real_actions=44,
            protected_historical_files=len(audit['protected_historical_files']),
            packaged_source_files=len(current), report_sha256=sha(report), HTML_validation_passed=True,
            HTTP=http, loopback_root_redirect_passed=True, unknown_route_denied=True,
            preview_bind='127.0.0.1', preview_port=index['preview_port'])
    finally:
        with args.output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'HTTP'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
