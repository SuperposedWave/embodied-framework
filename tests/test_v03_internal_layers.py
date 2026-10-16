"""Synthetic import, shared-record and private-boundary regression gates."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from embodied_eval import contracts as public
from embodied_eval.core import types as physical
from embodied_eval.core.validation import validate_structure
from embodied_eval.errors import ContractError
from embodied_eval.native import contracts as private
from embodied_eval.v03 import contracts as current
from embodied_eval.v03.benchmarks import CanonicalBenchmark
from embodied_eval.v03.models import CanonicalModel
from embodied_eval.v03.synthetic import space
from embodied_eval.v03.validation import validate_space

ROOT = Path(__file__).resolve().parents[1]
MOVED = (
    'contracts', 'transforms', 'additional_catalog', 'additional_models', 'assets',
    'assets_common', 'droid_simulator', 'joint_native', 'native_benchmarks',
    'native_catalog', 'native_model_process', 'native_model_worker', 'native_models',
    'openwam_arx', 'openwam_dual_pose', 'robodojo_simulator', 'rotations',
)


class InternalLayerTests(unittest.TestCase):
    def subprocess_check(self, body):
        result = subprocess.run([sys.executable, '-c', body], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def action(self):
        return physical.Action(commands=(physical.JointTarget(command_id='j', target_id='robot0/j0',
            mode='position', value=.25, unit='rad'),), native_dtype='float64')

    def test_current_registration_does_not_import_historical_modules(self):
        self.subprocess_check("""
import sys
from embodied_eval.v03.registry import default_registry
registry = default_registry()
assert len(registry.models) == len(registry.benchmarks) == 7
assert not any(n.startswith(('embodied_eval.mn', 'embodied_eval._legacy')) for n in sys.modules)
assert not any(n in sys.modules for n in ('numpy', 'torch', 'jax', 'sapien', 'mujoco', 'isaacsim'))
""")

    def test_public_preflight_can_block_all_old_imports(self):
        self.subprocess_check("""
import importlib.abc, sys
class NoOldImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('embodied_eval.mn', 'embodied_eval._legacy')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoOldImports())
from embodied_eval.config import load_config
from embodied_eval.v03.preflight import check
for path in ('examples/mn_synthetic_run.json', 'examples/mn_openwam_dual_wxyz_encoded_absolute.json'):
    assert check(load_config(path))['structurally_compatible']
""")

    def test_synthetic_run_does_not_use_old_runner_or_pair_factories(self):
        self.subprocess_check("""
import importlib.abc, pathlib, sys, tempfile
class NoOldImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('embodied_eval.mn', 'embodied_eval._legacy', 'embodied_eval.stage4_pairs', 'embodied_eval.robotwin_pairs')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoOldImports())
from embodied_eval.config import load_config
from embodied_eval.runner import run
with tempfile.TemporaryDirectory() as temporary:
    result = run(load_config('examples/mn_synthetic_run.json'), pathlib.Path(temporary) / 'run')
    assert result.exit_code == 0 and result.summary['contract_version'] == '0.3'
""")

    def test_shared_physical_types_are_one_definition(self):
        old = importlib.import_module('embodied_eval._legacy_contracts')
        for name in ('Transform', 'Pose', 'Tensor', 'Budgets', 'SensorSample', 'JointTarget',
                     'PoseTarget', 'ControllerCommand', 'ExecutionFeedback', 'Metric'):
            with self.subTest(name=name):
                self.assertIs(getattr(public, name), getattr(physical, name))
                self.assertIs(getattr(old, name), getattr(physical, name))

    def test_action_is_shared_without_rebuilding(self):
        self.assertIs(current.Action, physical.Action)
        self.assertIs(private.Action, physical.Action)

    def test_physical_module_has_no_protocol_message_envelopes(self):
        for name in ('Context', 'SpaceSpec', 'CapabilitySpec', 'BindingConfig', 'Prediction', 'TaskSpec'):
            self.assertFalse(hasattr(physical, name), name)

    def test_native_registry_has_no_synthetic_or_pair_registrations(self):
        from embodied_eval.native.registry import default_registry
        registry = default_registry()
        self.assertEqual(set(registry.models), {'openwam', 'openpi', 'lingbot_vla', 'cosmos3', 'dreamzero'})
        self.assertEqual(set(registry.benchmarks), {'libero', 'vlabench', 'robotwin', 'droid_sim', 'robodojo'})
        self.assertTrue(all(not e.synthetic for e in (*registry.models.values(), *registry.benchmarks.values())))

    def test_compatibility_aliases_reference_the_same_module(self):
        for name in MOVED:
            with self.subTest(name=name):
                self.assertIs(importlib.import_module('embodied_eval.mn.' + name),
                              importlib.import_module('embodied_eval.native.' + name))

    def test_historical_monkeypatch_reaches_current_native_engine(self):
        import embodied_eval.native.native_models as active
        sentinel = object()
        with patch('embodied_eval.mn.native_models.native_module', sentinel):
            self.assertIs(active.native_module, sentinel)

    def test_old_pickled_class_paths_still_resolve(self):
        self.assertIs(pickle.loads(b'cembodied_eval.mn.contracts\nAction\n.'), physical.Action)
        self.assertIs(pickle.loads(b'cembodied_eval._legacy_contracts\nPose\n.'), physical.Pose)

    def test_shared_action_can_roundtrip_through_a_fresh_process(self):
        action = self.action()
        result = subprocess.run([sys.executable, '-c',
            'import pickle,sys; x=pickle.load(sys.stdin.buffer); from embodied_eval.core.types import Action; '
            'assert type(x) is Action; pickle.dump(x,sys.stdout.buffer)'], input=pickle.dumps(action),
            capture_output=True, cwd=ROOT, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(pickle.loads(result.stdout), action)

    def test_shared_action_json_fields_are_unchanged(self):
        from embodied_eval.codec import decode, to_data
        expected = dict(commands=[dict(command_id='j', target_id='robot0/j0', mode='position',
            value=.25, unit='rad', kind='joint_target')], acquire_observation_after=True,
            native_protocol=None, native_dtype='float64', source_euler_xyz_rad={})
        self.assertEqual(to_data(self.action()), expected)
        self.assertEqual(decode(current.Action, expected), self.action())

    def test_model_boundary_keeps_action_object_and_precision(self):
        action = self.action()
        adapter = CanonicalModel.__new__(CanonicalModel)
        adapter.engine = Mock()
        context = current.Context(run_id='r', env_id='e', episode_id='ep', message_id='m')
        adapter.engine.predict.return_value = private.Prediction(context=private.Context(
            run_id='r', env_id='e', episode_id='ep', message_id='m'),
            based_on_observation_id='o', action=action, model_forward_count=0)
        result = adapter.predict(context)
        self.assertIs(result.action, action)
        self.assertIs(result.context, context)
        self.assertEqual(result.model_forward_count, 0)
        self.assertEqual(result.action.native_dtype, 'float64')

    def test_environment_boundary_does_not_reconstruct_private_action(self):
        adapter = CanonicalBenchmark.__new__(CanonicalBenchmark)
        adapter.engine = Mock()
        adapter.family, adapter.native_space = 'synthetic', None
        adapter.space, adapter.latest = space({}), None
        context = current.Context(run_id='r', env_id='e', episode_id='ep', message_id='m')
        action = self.action()
        reply = Mock(terminated=False, truncated=False)
        adapter.engine.advance.return_value = reply
        with patch('embodied_eval.v03.benchmarks.canonical_observation', return_value='next-observation'), \
             patch.object(private, 'Action', side_effect=AssertionError('unnecessary action rebuild')):
            result = adapter.advance(action, context, physical.Budgets(env_steps=1))
        passed_action = adapter.engine.advance.call_args.args[0]
        self.assertIs(type(passed_action), physical.Action)
        self.assertEqual(passed_action, action)
        self.assertEqual(result.observation, 'next-observation')

    def test_public_geometry_validation_does_not_build_an_old_space(self):
        from embodied_eval.native import environment_records
        current_space = space({})
        with patch.object(environment_records, 'SpaceSpec', side_effect=AssertionError('old space reconstruction')):
            self.assertIs(validate_space(current_space), current_space)

    def test_shared_structure_rejects_cycles(self):
        current_space = space({})
        frames = dict(current_space.frames, a=physical.FrameSpec(parent_id='a',
            static_transform=physical.Transform(translation_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.))))
        with self.assertRaises(ContractError) as raised:
            validate_structure(replace(current_space, frames=frames))
        self.assertEqual(raised.exception.issue.code, 'frame_cycle')

    def test_shared_structure_rejects_nonunit_quaternions(self):
        frames = dict(space({}).frames, a=physical.FrameSpec(parent_id='world',
            static_transform=physical.Transform(translation_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 2.))))
        with self.assertRaises(ContractError) as raised:
            validate_space(replace(space({}), frames=frames))
        self.assertEqual(raised.exception.issue.code, 'invalid_rotation')

    def test_native_asset_manifests_match_the_frozen_copies(self):
        paths = sorted((ROOT / 'embodied_eval/native/asset_manifests').glob('*.json'))
        self.assertEqual(len(paths), 6)
        for path in paths:
            with self.subTest(name=path.name):
                self.assertEqual(path.read_bytes(), (ROOT / 'embodied_eval/mn/asset_manifests' / path.name).read_bytes())
                self.assertTrue(json.loads(path.read_text())['objects'])

    def test_new_native_worker_is_an_importable_module(self):
        self.subprocess_check("""
import sys
from embodied_eval.native.native_model_worker import METHODS, worker_loop
assert callable(worker_loop) and 'predict' in METHODS
assert not any(n.startswith('embodied_eval.mn') for n in sys.modules)
""")

    def test_current_config_still_rejects_old_versions(self):
        from embodied_eval.codec import decode
        from embodied_eval.config import RunConfig
        raw = json.loads((ROOT / 'examples/mn_synthetic_run.json').read_text())
        for version in ('0.1', '0.2'):
            with self.subTest(version=version), self.assertRaises(ContractError):
                decode(RunConfig, dict(raw, contract_version=version))
