"""Final local HTTP, source, real/synthetic evidence and immutable-history gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, build_opener
from serve_report import ROOT, catalog


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8 << 20),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    index = read(args.index)
    result = dict(kind='v03_final_delivery_verification_not_stage_acceptance', passed=False,
        at_utc=datetime.now(timezone.utc).isoformat(), stage4_accepted=False, contract_frozen=False)
    try:
        assert not index['pair_specific_binding_required']
        for key in ('baseline', 'second_environment_baseline'):
            base = read(ROOT/index[key])
            assert base['passed'] and base['checks'][0]['tests_run'] == 495 and base['checks'][0]['skips'] == 0
            assert [r['actual_exit_code'] for r in base['checks'] if 'actual_exit_code' in r] == [0,0,0,1,2]
        mapping = read(ROOT/index['saved_boundary'])
        assert mapping['passed'] and len(mapping['cases']) == 11
        assert sum(c['observations'] for c in mapping['cases']) == 55
        assert sum(c['predictions'] for c in mapping['cases']) == 44
        for path, expected in mapping['protected_sha256'].items():
            assert sha(Path(path)) == expected, path
        for path, expected in mapping['framework_source_sha256'].items():
            assert sha(ROOT/path) == expected, path
        inv = read(ROOT/index['metadata_and_history'])
        assert inv['passed'] and inv['metadata_cells'] == 77 and inv['structurally_compatible_cells'] == 20
        for row in inv['historical_integrity']['historical_stage2_sha256_files']:
            assert row['unchanged'] and (ROOT/row['path']).is_file()
        for name, expected in inv['preserved_contract_and_proposal_sha256'].items():
            assert sha(ROOT/name) == expected
        for name, expected in inv['preserved_legacy_example_sha256'].items():
            assert sha(ROOT/'tests/fixtures/legacy-configs'/name) == expected
        real = read(ROOT/index['bounded_real_index'])
        patch = read(ROOT/index['post_runtime_validation_patch_audit'])
        assert patch['passed'] and patch['all_other_production_AST_nodes_equal']
        assert real['passed'] and len(real['cases']) == 7
        for row in real['cases']:
            assert row['passed'] and row['errors'] == 0 and not row['synthetic']
            assert row['execution_statistics']['env_steps'] == 4
            assert row['metrics']['success']['valid'] and row['metrics']['success']['value'] is False
            manifest = read(Path(row['output_directory'])/'framework/manifest.json')
            current = {str(p.relative_to(ROOT/'embodied_eval')):sha(p) for p in (ROOT/'embodied_eval').rglob('*.py')}
            differences = [name for name in current if current[name] != manifest['source_sha256'].get(name)]
            assert differences == ['v03/validation.py']
            assert manifest['source_sha256']['v03/validation.py'] == patch['before_sha256']
            assert current['v03/validation.py'] == patch['after_sha256']
        for path, expected in real['source_evidence_sha256'].items():
            assert sha(Path(path)) == expected
        gpu = read(ROOT/index['GPU_evidence'])
        assert gpu['target_met'] and gpu['includes_preexisting_occupancy_not_evaluation_work']
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert subprocess.check_output(['git','rev-parse',index['delivery_engineering_tag']],cwd=ROOT,text=True).strip() == commit
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
        assert subprocess.run(['git','diff','--quiet',index['final_runtime_source_commit'],'--','embodied_eval'],cwd=ROOT).returncode == 0
        assert subprocess.run(['git','diff','--quiet',index['final_test_source_commit'],'--','tests'],cwd=ROOT).returncode == 0
        bundle = ROOT/index['source_bundle']
        assert subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,capture_output=True).returncode == 0
        route, routes, hashes = catalog(ROOT/index['report'])
        opener = build_opener(ProxyHandler({}))
        base_url = 'http://127.0.0.1:' + str(index['preview_port'])
        http = []
        for path, (payload, _) in routes.items():
            with opener.open(base_url+quote(path,safe='/'),timeout=20) as response:
                assert response.status == 200 and response.read() == payload
                assert 'nginx' not in response.headers.get('Server','').lower()
                http.append(dict(path=path,status=200,sha256=hashes[path]))
        with opener.open(base_url+'/',timeout=20) as response:
            assert response.read() == routes[route][0]
        try:
            opener.open(base_url+'/.git/config',timeout=10)
            raise AssertionError('must_not_expose_git_or_workspace')
        except HTTPError as error:
            assert error.code == 404
        result.update(passed=True, final_commit=commit, engineering_tag=index['delivery_engineering_tag'],
            production_source_unchanged_since_final_runtime_source_commit=True,
            test_source_unchanged_since_final_test_source_commit=True,
            real_recordings_precede_validated_field_name_only_patch=True,
            source_bundle_sha256=sha(bundle), report_sha256=sha(ROOT/index['report']),
            HTTP=http, loopback_root_redirect_passed=True, unknown_route_denied=True,
            desktop_forwarding_verified=False, new_external_upload=False,
            software_tests_each_environment=495, software_environments=2, skips=0, CLI=[0,0,0,1,2],
            saved_real_observations=55,saved_real_actions=44, saved_history_files_rehashed=len(mapping['protected_sha256']),
            new_real_cases=7,new_real_actions=28,GPU_sample_mean_percent=gpu['aggregate_mean_percent'])
    finally:
        with args.output.open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
            stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key!='HTTP'},ensure_ascii=False))


if __name__ == '__main__':
    main()
