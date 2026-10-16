"""Prove the post-runtime patch only tightens public field-name validation.

No new model inference/simulation is claimed. Existing real results retain the
exact original source hashes; current full tests and saved-data gates are separate.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
FILE = 'embodied_eval/v03/validation.py'
BEFORE = 'cdd1ddbed8eac87ee23a6bcf4afca4c42703ee94'


def git_source(commit):
    return subprocess.check_output(['git','show',commit+':'+FILE],cwd=ROOT,text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    before, after = git_source(BEFORE),(ROOT/FILE).read_text()
    oldtree,newtree = ast.parse(before),ast.parse(after)
    oldbase = next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='state_base')
    newbase = next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='state_base')
    assert ast.dump(oldbase) == ast.dump(ast.parse('def state_base(key):\n return key.removeprefix("left_").removeprefix("right_")').body[0])
    assert ast.dump(newbase) == ast.dump(ast.parse('def state_base(key):\n return key[5:] if key.startswith("left_") else key[6:] if key.startswith("right_") else key').body[0])
    oldns,newns = {},{}
    for node,ns in ((oldbase,oldns),(newbase,newns)):
        exec(compile(ast.Module(body=[node],type_ignores=[]),FILE,'exec'),ns)
    def normalize(tree):
        tree.body = [n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='state_base')]
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_space')
        loop = next(n for n in fn.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='space.sensors.items()')
        # Keep the unchanged provenance guard; only canonical-name construction
        # and its name/modality require() guard are the allowed sensor delta.
        assert ast.unparse(loop.body[-1]).startswith('require(bool(spec.native_source)')
        loop.body = [loop.body[-1]]
        return ast.dump(tree)
    assert normalize(oldtree) == normalize(newtree)
    changed = subprocess.check_output(['git','diff','--name-only',BEFORE,'--','embodied_eval'],cwd=ROOT,text=True).splitlines()
    assert changed == [FILE],changed
    from embodied_eval.codec import decode
    from embodied_eval.v03.contracts import SpaceSpec
    from embodied_eval.v03.validation import validate_space,CAMERAS
    real = json.loads(args.real.read_text())
    rows = []
    before_sha = hashlib.sha256(before.encode()).hexdigest()
    after_sha = hashlib.sha256(after.encode()).hexdigest()
    for item in real['cases']:
        manifest = json.loads((Path(item['output_directory'])/'framework/manifest.json').read_text())
        space = validate_space(decode(SpaceSpec,manifest['space']))
        for key in space.state_fields:
            assert oldns['state_base'](key) == newns['state_base'](key)
        for key,sensor in space.sensors.items():
            role=key
            for suffix in ('_depth','_point_cloud','_audio','_tactile'):
                role=role.removesuffix(suffix)
            assert role in CAMERAS and role==sensor.role
            assert key == sensor.role + ('' if sensor.modality=='rgb' else '_'+sensor.modality)
        current = {str(p.relative_to(ROOT/'embodied_eval')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'embodied_eval').rglob('*.py')}
        prior = manifest['source_sha256']
        assert prior['v03/validation.py'] == before_sha
        assert set(current) == set(prior)
        assert [name for name in current if current[name]!=prior[name]] == ['v03/validation.py']
        rows.append(dict(case=item['case'],all_recorded_public_field_names_have_identical_acceptance=True,
            all_Model_Benchmark_Adapter_Runner_and_numeric_conversion_source_hashes_exact=True,
            real_run_before_validation_only_patch_not_rerun=True))
    result = dict(kind='post_real_run_validation_only_source_delta_audit_not_fresh_inference',passed=True,
        at_utc=datetime.now(timezone.utc).isoformat(),before_commit=BEFORE,
        after_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        changed_production_file=FILE,before_sha256=before_sha,after_sha256=after_sha,
        state_base_only_one_side_prefix=True,sensor_name_requires_matching_modality=True,
        all_other_production_AST_nodes_equal=True,real_cases=rows,
        new_GPU_inference=False,new_simulation=False,formal_acceptance=False)
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
