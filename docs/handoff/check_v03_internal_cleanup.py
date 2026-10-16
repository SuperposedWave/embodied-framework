"""Audit code movement, unchanged public records and protected historical files.

This is a source/serialization audit, not GPU inference or simulation.
AST means parsed Python syntax: import locations and module descriptions may
move, but native computation/control/scoring statements must remain identical.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
MOVED = ('additional_catalog', 'additional_models', 'assets', 'assets_common',
    'droid_simulator', 'joint_native', 'native_benchmarks', 'native_catalog',
    'native_model_process', 'native_model_worker', 'native_models', 'openwam_arx',
    'openwam_dual_pose', 'robodojo_simulator', 'rotations')
LEAVES = ('Transform', 'Pose', 'Twist', 'FrameSpec', 'TimedTransform', 'TensorSpec',
    'DataRef', 'Tensor', 'SensorSpec', 'JointSpec', 'EndEffectorSpec', 'GripperSpec',
    'RobotSpec', 'ControllerSpec', 'ActionSpec', 'Budgets', 'PublicTask', 'SensorSample',
    'JointState', 'GripperState', 'EndEffectorState', 'RobotState', 'JointTarget',
    'PoseTarget', 'PoseDelta', 'TwistCommand', 'GripperCommand', 'ControllerCommand',
    'ExecutionFeedback', 'Metric', 'ExecutionStatistics')
HELPERS = ('require', 'quaternion', 'pose', 'unique', 'check_schema', 'schema_value',
    'target_table', 'validate_tensor', 'shape_matches', 'validate_gripper')
ACTIVE_LOADERS = ('libero_spatial_benchmark', 'vlabench_benchmark', 'robotwin_benchmark',
    'openwam_libero_spec', 'openpi_libero_spec', 'vlabench_spec', 'robotwin_spec',
    'native_common', 'runtime_options')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def previous(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ROOT, text=True)


def read(path):
    return (ROOT / path).read_text()


class Computation(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Constant(self, node):
        if node.value == 'embodied_eval.mn.native_model_worker':
            return ast.copy_location(ast.Constant('embodied_eval.native.native_model_worker'), node)
        return node

    def visit_Module(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            node.body.pop(0)
        return self.generic_visit(node)


class PhysicalNames(ast.NodeTransformer):
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'leaf':
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return self.generic_visit(node)


def dump(node):
    return ast.dump(node, include_attributes=False)


def definitions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', default='pre-v03-internal-cleanup-20261016')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(exist_ok=False, parents=True)
    result = dict(kind='v03_internal_cleanup_source_and_history_audit', passed=False,
        at_utc=datetime.now(timezone.utc).isoformat(), before_ref=args.before,
        new_inference=False, new_simulation=False, public_wire_changed=False,
        stage4_accepted=False, contract_frozen=False, checks=[])
    try:
        for name in (*MOVED,):
            before = ast.parse(previous(args.before, f'embodied_eval/mn/{name}.py'))
            after = ast.parse(read(f'embodied_eval/native/{name}.py'))
            assert dump(Computation().visit(before)) == dump(Computation().visit(after)), name
            result['checks'].append(dict(kind='native_computation_AST_unchanged', name=name))
        for name in ACTIVE_LOADERS:
            path = f'embodied_eval/adapters/{name}.py'
            assert dump(Computation().visit(ast.parse(previous(args.before, path)))) == dump(
                Computation().visit(ast.parse(read(path)))), name
            result['checks'].append(dict(kind='environment_loader_or_spec_AST_unchanged', name=name))
        for path in ('embodied_eval/v03/runner.py', 'embodied_eval/v03/transforms.py', 'embodied_eval/geometry.py'):
            assert dump(Computation().visit(ast.parse(previous(args.before, path)))) == dump(
                Computation().visit(ast.parse(read(path)))), path
            result['checks'].append(dict(kind='runner_or_coordinate_math_AST_unchanged', name=path))
        old_leaf = definitions(previous(args.before, 'embodied_eval/_legacy_contracts.py'))
        new_leaf = definitions(read('embodied_eval/core/types.py'))
        for name in LEAVES:
            assert dump(old_leaf[name]) == dump(new_leaf[name]), name
        result['shared_physical_classes_unchanged'] = len(LEAVES)
        old_helpers = definitions(previous(args.before, 'embodied_eval/_legacy_validation.py'))
        new_helpers = definitions(read('embodied_eval/core/validation.py'))
        for name in HELPERS:
            assert dump(old_helpers[name]) == dump(new_helpers[name]), name
        before_episode, after_episode = old_helpers['same_episode'], new_helpers['same_episode']
        for node in (before_episode, after_episode):
            for arg in node.args.args:
                arg.annotation = None
        assert dump(before_episode) == dump(after_episode)
        result['shared_validation_helpers_unchanged'] = len(HELPERS) + 1
        old_transform = definitions(previous(args.before, 'embodied_eval/mn/transforms.py'))['transform_between']
        new_transform = definitions(read('embodied_eval/core/transforms.py'))['transform_between']
        assert dump(old_transform) == dump(new_transform)
        result['frame_path_math_unchanged'] = True
        old_public = definitions(previous(args.before, 'embodied_eval/v03/contracts.py'))
        new_public = definitions(read('embodied_eval/v03/contracts.py'))
        new_public['Action'] = new_leaf['Action']
        public_count = 0
        for name, node in old_public.items():
            if isinstance(node, ast.ClassDef):
                assert dump(PhysicalNames().visit(node)) == dump(PhysicalNames().visit(new_public[name])), name
                public_count += 1
        result['public_v03_classes_fields_defaults_annotations_unchanged'] = public_count
        old_private = definitions(previous(args.before, 'embodied_eval/mn/contracts.py'))
        new_private = definitions(read('embodied_eval/native/contracts.py'))
        new_private['Action'] = new_leaf['Action']
        for name, node in old_private.items():
            assert dump(PhysicalNames().visit(node)) == dump(PhysicalNames().visit(new_private[name])), name
        result['private_native_record_fields_unchanged'] = len(old_private)
        env_records = definitions(read('embodied_eval/native/environment_records.py'))
        for name, node in old_leaf.items():
            if name not in LEAVES:
                assert dump(node) == dump(env_records[name]), name
        result['frozen_environment_record_fields_unchanged'] = len(env_records)
        manifests = []
        for path in sorted((ROOT / 'embodied_eval/native/asset_manifests').glob('*.json')):
            expected = previous(args.before, f'embodied_eval/mn/asset_manifests/{path.name}').encode()
            assert path.read_bytes() == expected, path.name
            manifests.append(dict(name=path.name, sha256=sha(path)))
        assert len(manifests) == 6
        result['unchanged_asset_manifests'] = manifests
        # Previously published hashes are authoritative; never regenerate them
        # from the current file and call that a historical comparison.
        index = json.loads(read('docs/handoff/v03-delivery-index.json'))
        historical = json.loads(read(index['final_verification']))
        protected = {}
        for row in historical['HTTP']:
            name = unquote(row['path']).lstrip('/')
            if name.startswith('outputs/'):
                protected[name] = row['sha256']
        protected[index['report']] = historical['report_sha256']
        protected[index['source_bundle']] = historical['source_bundle_sha256']
        saved = json.loads(read(index['saved_boundary']))
        protected.update(saved['protected_sha256'])
        inv = json.loads(read(index['metadata_and_history']))
        for name, expected in inv['preserved_contract_and_proposal_sha256'].items():
            if name != 'docs/contract-v0.3.md':
                protected[name] = expected
        for name, expected in inv['preserved_legacy_example_sha256'].items():
            protected['tests/fixtures/legacy-configs/' + name] = expected
        for path, expected in protected.items():
            target = Path(path) if Path(path).is_absolute() else ROOT / path
            assert sha(target) == expected, path
        result['protected_historical_files'] = protected
        result['historical_hashes_unchanged'] = True
        result['framework_source_sha256'] = {str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / 'embodied_eval').rglob('*.py'))}
        result['test_source_sha256'] = {str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / 'tests').rglob('*.py'))}
        result['before_commit'] = subprocess.check_output(['git', 'rev-parse', args.before], cwd=ROOT, text=True).strip()
        result['git_commit_at_start'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        result['passed'] = True
    finally:
        with (args.output / 'receipt.json').open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items()
        if k not in ('framework_source_sha256', 'test_source_sha256', 'protected_historical_files', 'checks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
