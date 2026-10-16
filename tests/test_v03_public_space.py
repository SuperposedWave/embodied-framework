"""Canonical 0.3 architecture/negative gates. All generated values are synthetic."""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from embodied_eval import contracts as leaf
from embodied_eval.codec import decode, to_data
from embodied_eval.errors import ContractError
from embodied_eval.config import AdapterConfig, EpisodeConfig, RunConfig, load_config
from embodied_eval.runner import run
from embodied_eval.v03 import contracts as c
from embodied_eval.v03.registry import AdapterRegistry, default_registry
from embodied_eval.v03.synthetic import register, space, capability, SyntheticModel, SyntheticBenchmark
from embodied_eval.v03.validation import validate_space, validate_observation, validate_prediction, validate_step
from embodied_eval.v03.matching import resolve, compatibility, model_input
from embodied_eval.v03.preflight import check
from embodied_eval.v03.models import private_input, profile_paths, private_space
from embodied_eval.v03.benchmarks import canonical_space, canonical_observation
from embodied_eval.v03.transforms import inputs_for_model, action_for_environment
from embodied_eval.v03.migration import migrate
from embodied_eval.mn.registry import default_registry as old_registry
from embodied_eval.mn.binding import resolve as old_resolve, model_input as old_input
from embodied_eval.mn import contracts as old
from tests.v03_helpers import ROOT, cases, observation


def config(model='synthetic_policy_a', benchmark='synthetic_simulator_a', options=None):
    return RunConfig(run_id='canonical-test', model=AdapterConfig(adapter=model), benchmark=AdapterConfig(adapter=benchmark, options=options or {}),
        episodes=(EpisodeConfig(episode_id='ep-0', budgets=leaf.Budgets(env_steps=6, model_calls=6)),))


class CanonicalArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.registry = AdapterRegistry()
        register(self.registry)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def execute(self, cfg, name='run'):
        return run(cfg, self.root / name, self.registry)

    def test_public_api_is_03(self):
        import embodied_eval
        from embodied_eval.interfaces import ModelAdapter
        self.assertEqual(embodied_eval.CONTRACT_VERSION, '0.3')
        self.assertEqual(leaf.CONTRACT_VERSION, '0.3')
        self.assertIs(leaf.Context, c.Context)
        self.assertEqual(ModelAdapter.__module__, 'embodied_eval.v03.interfaces')

    def test_two_by_two_all_four_crosses(self):
        for i, m in enumerate(self.registry.models):
            for j, b in enumerate(self.registry.benchmarks):
                outcome = self.execute(config(m, b), f'cross-{i}-{j}')
                self.assertEqual(outcome.exit_code, 0, outcome.summary)
                manifest = json.loads((outcome.output / 'manifest.json').read_text())
                self.assertEqual(set(manifest['space']['sensors']), {'head'})
                inputs = [json.loads(line) for line in (outcome.output / 'episodes/ep-0/model_input.public.jsonl').read_text().splitlines()]
                self.assertTrue(all(set(row['sensors']) == {'head'} and row['context']['contract_version'] == '0.3' for row in inputs))
                self.assertNotIn('binding', manifest['config'])
                self.assertTrue(outcome.summary['synthetic'])

    def test_third_model_and_environment_need_only_registration(self):
        self.registry.register_model('third-model', self.registry.model('synthetic_policy_b'))
        self.registry.register_benchmark('third-env', self.registry.benchmark('synthetic_simulator_b'))
        self.assertEqual(self.execute(config('third-model', 'third-env')).exit_code, 0)

    def test_native_camera_names_do_not_escape(self):
        result = self.execute(config(options={'native_camera_name': 'completely_different_rgb'}))
        manifest = json.loads((result.output / 'manifest.json').read_text())
        self.assertEqual(set(manifest['space']['sensors']), {'head'})
        self.assertEqual(manifest['space']['sensors']['head']['native_source']['field'], 'completely_different_rgb')

    def test_cache_and_second_episode_reset(self):
        cfg = replace(config(), model=AdapterConfig(adapter='synthetic_policy_a', options={'cache_rows': 3}))
        cfg = replace(cfg, episodes=cfg.episodes + (replace(cfg.episodes[0], episode_id='ep-1'),))
        result = self.execute(cfg)
        for episode in ('ep-0', 'ep-1'):
            diag = json.loads((result.output / f'episodes/{episode}/diagnostics.internal.json').read_text())['model']
            self.assertEqual((diag['calls'], diag['forwards'], diag['observe_count']), (6, 2, 6))
            self.assertEqual(diag['events'][0]['cache_at_begin'], 0)

    def test_variable_duration_records_all_three_clocks(self):
        result = self.execute(config(options={'timing_mode': 'measured_native'}))
        stats = result.results[0]['execution_statistics']
        self.assertEqual(stats['env_steps'], 6)
        self.assertEqual(stats['physics_steps'], 33)
        self.assertAlmostEqual(stats['sim_time_s'], .132)

    def test_native_hard_time_budget_is_rejected(self):
        cfg = config(options={'timing_mode': 'measured_native'})
        cfg = replace(cfg, episodes=(replace(cfg.episodes[0], budgets=leaf.Budgets(sim_time_s=1.)),))
        self.assertIn('unsupported_native_budget', [i['code'] for i in check(cfg, self.registry)['issues']])

    def test_preflight_never_constructs_factories(self):
        with patch.object(SyntheticModel, '__init__', side_effect=AssertionError('GPU model')), patch.object(SyntheticBenchmark, '__init__', side_effect=AssertionError('simulator')):
            result = check(config(), self.registry, verify_assets=True)
        self.assertTrue(result['structurally_compatible'])
        self.assertTrue(result['assets_verified'])
        self.assertEqual(result['real_validation'], 'not_performed_by_preflight')

    def test_custom_registry_cannot_bypass_config_validation(self):
        for cfg in (replace(config(), run_id='../unsafe'), replace(config(), episodes=()),
                    replace(config(), episodes=(EpisodeConfig(episode_id='ep-0', seed=-1),))):
            with self.assertRaises(ContractError):
                check(cfg, self.registry)

    def test_existing_output_is_not_overwritten(self):
        self.execute(config())
        with self.assertRaises(ContractError):
            self.execute(config())

    def test_score_error_is_error_not_success(self):
        result = self.execute(config(options={'fault': 'score_error'}))
        self.assertEqual(result.exit_code, 1)
        self.assertFalse(result.results[0]['metrics']['success'].valid)

    def test_control_mode_does_not_guess(self):
        self.assertFalse(check(replace(config(), control_mode='wrong'), self.registry)['structurally_compatible'])
        s = space({})
        s = replace(s, action_specs={**s.action_specs, 'other': s.action_specs['joint-position@1']}, action_joint_modes={**s.action_joint_modes, 'other': ('position',)})
        self.assertIn('ambiguous_control_mode', [i.code for i in compatibility(s, capability({}))[0].issues])


class CanonicalValidationTests(unittest.TestCase):
    def setUp(self):
        self.space, self.cap = space({}), capability({})

    def issue(self, s=None, cap=None):
        return {i.code for i in compatibility(s or self.space, cap or self.cap)[0].issues}

    def test_no_native_camera_key_accepted(self):
        s = replace(self.space, sensors={'agentview_image': self.space.sensors['head']})
        with self.assertRaises(ContractError):
            validate_space(s)

    def test_wrong_role_is_not_a_public_rename(self):
        s = replace(self.space, sensors={'head': replace(self.space.sensors['head'], role='agentview')})
        with self.assertRaises(ContractError):
            validate_space(s)
        for key in ('head_depth', 'head_point_cloud_depth', 'head_audio_depth'):
            with self.assertRaises(ContractError):
                validate_space(replace(self.space, sensors={key: self.space.sensors['head']}, view_aliases={'main': key}))
        depth = replace(self.space.sensors['head'], modality='depth', unit='m', color_order=None,
            tensor=leaf.TensorSpec(dtype='float32', shape=(2, 2), layout='HW'))
        validate_space(replace(self.space, sensors={'head_depth': depth}, view_aliases={'main': 'head_depth'}))

    def test_missing_head_does_not_use_wrist(self):
        s = replace(self.space, sensors={'wrist': replace(self.space.sensors['head'], role='wrist')}, view_aliases={'main': 'wrist'})
        self.assertIn('missing_required_input', self.issue(s))

    def test_main_requires_explicit_environment_alias(self):
        self.assertIn('missing_view', self.issue(replace(self.space, view_aliases={}), capability({}, True)))

    def test_alias_may_not_point_to_missing_sensor(self):
        with self.assertRaises(ContractError):
            validate_space(replace(self.space, view_aliases={'main': 'missing'}))

    def test_optional_input_cannot_hide_malformed_path(self):
        cap = replace(self.cap, inputs=(replace(self.cap.inputs[0], public_path='wrong.camera', required=False),))
        self.assertIn('invalid_public_path', self.issue(cap=cap))

    def test_optional_missing_view_is_permitted(self):
        cap = replace(self.cap, inputs=(replace(self.cap.inputs[0], public_path='views.main', required=False),))
        self.assertNotIn('missing_view', self.issue(replace(self.space, view_aliases={}), cap))

    def test_wrong_tensor_or_action_kind_fails(self):
        cap = replace(self.cap, inputs=(replace(self.cap.inputs[0], tensor=leaf.TensorSpec(dtype='float32', shape=(2, 2, 3), layout='HWC')),))
        self.assertIn('tensor_spec_mismatch', self.issue(cap=cap))
        self.assertIn('unsupported_action', self.issue(cap=replace(self.cap, output=replace(self.cap.output, command_kinds=('twist_command',)))))

    def test_training_identity_is_warning_not_a_gate(self):
        report = compatibility(self.space, self.cap)[0]
        self.assertFalse(report.issues)
        self.assertTrue(report.warnings)

    def test_dynamic_instruction_does_not_expose_task_internal(self):
        bench = SyntheticBenchmark({})
        task = bench.enumerate_tasks(config().episodes)[0]
        ctx = c.Context(run_id='r', env_id='e', episode_id='ep-0', message_id='start')
        obs = replace(bench.reset(task, ctx), public_instruction='new public instruction')
        view = model_input(obs, task, self.cap, resolve(self.space, self.cap), self.space)
        self.assertEqual(view.task_public.instruction, 'new public instruction')
        self.assertFalse(hasattr(view, 'task_internal'))

    def test_stale_image_rejected(self):
        bench = SyntheticBenchmark({})
        task = bench.enumerate_tasks(config().episodes)[0]
        ctx = c.Context(run_id='r', env_id='e', episode_id='ep-0', message_id='start')
        obs = replace(bench.reset(task, ctx), sim_time_s=2.)
        cap = replace(self.cap, inputs=(replace(self.cap.inputs[0], max_age_s=.1),))
        with self.assertRaises(ContractError):
            model_input(obs, task, cap, resolve(self.space, cap), self.space)

    def test_duplicate_feedback_rejected(self):
        bench = SyntheticBenchmark({})
        task = bench.enumerate_tasks(config().episodes)[0]
        ctx = c.Context(run_id='r', env_id='e', episode_id='ep-0', message_id='a')
        obs = bench.reset(task, ctx)
        cmd = leaf.JointTarget(command_id='j', target_id='robot0/j0', mode='position', value=0., unit='rad')
        action = c.Action(commands=(cmd,))
        reply = bench.advance(action, ctx, task.budgets)
        reply = replace(reply, execution_feedback=replace(reply.execution_feedback, command_ids=('j', 'j')))
        with self.assertRaises(ContractError):
            validate_step(reply, obs, ctx, action, self.space)


class NativeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = cases()

    def test_all_eleven_profiles_seven_protocols(self):
        profiles, protocols = set(), set()
        for name, cfg in self.cases.items():
            with self.subTest(case=name):
                checked = check(cfg)
                self.assertTrue(checked['structurally_compatible'], checked['issues'])
                self.assertNotIn('binding', to_data(cfg))
                profiles.add(cfg.model.options['io_profile'])
                protocols.add((cfg.benchmark.adapter, cfg.benchmark.options['protocol']))
        self.assertEqual(len(profiles), 11)
        self.assertEqual(len(protocols), 7)

    def test_all_native_private_inputs_are_value_exact(self):
        for name, cfg in self.cases.items():
            with self.subTest(case=name):
                model = old_registry().model(cfg.model.adapter)
                bench = old_registry().benchmark(cfg.benchmark.adapter)
                original, raw_space = model.describe(cfg.model.options), bench.describe(cfg.benchmark.options)
                raw_obs = observation(raw_space)
                s = validate_space(canonical_space(cfg.benchmark.adapter, raw_space))
                obs = validate_observation(canonical_observation(cfg.benchmark.adapter, raw_space, raw_obs, s), s)
                cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
                task = c.TaskSpec(task_id='synthetic', episode_id='ep-0', public=leaf.PublicTask(instruction='test'), budgets=leaf.Budgets(env_steps=1), initialization_ref='synthetic', evaluation_config_ref='synthetic')
                view = model_input(obs, task, cap, resolve(s, cap, cfg.control_mode), s)
                private = private_input(view, original, profile_paths(cfg.model.options['io_profile'], original))
                for req in original.inputs:
                    if req.kind == 'task':
                        continue
                    self.assertIn(req.key, private.sensors if req.kind == 'sensor' else private.states)
                    public_path = profile_paths(cfg.model.options['io_profile'], original).get(req.key)
                    if req.key == 'drive_targets':
                        self.assertEqual(private.states[req.key].payload, raw_obs.states['joint_drive_targets'].payload)
                    else:
                        sample = (private.sensors if req.kind == 'sensor' else private.states)[req.key]
                        canonical = (view.sensors if req.kind == 'sensor' else view.states)[public_path.split('.')[1]]
                        self.assertEqual(sample.payload, canonical.payload)
                        self.assertEqual(sample.captured_at_s, canonical.captured_at_s)

    def rw(self):
        cfg = self.cases['stage4-lingbot-robotwin-adjust-bottle']
        old_space = old_registry().benchmark('robotwin').describe(cfg.benchmark.options)
        return cfg, old_space, canonical_space('robotwin', old_space)

    def test_both_six_joint_targets_are_not_measured_positions(self):
        cfg, native_space, s = self.rw()
        self.assertEqual(len(s.state_fields['left_joint_drive_targets'].joint_ids), 6)
        self.assertEqual(len(s.state_fields['right_joint_drive_targets'].joint_ids), 6)
        self.assertNotIn('joint_drive_targets', s.state_fields)
        req = c.InputSpec(key='q', kind='state', public_path='states.left_joint_drive_targets', semantic='joint_position', provenance='measurement')
        cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
        report = compatibility(s, replace(cap, inputs=(req,)), cfg.control_mode)[0]
        self.assertIn('state_provenance_mismatch', [i.code for i in report.issues])

    def test_inconsistent_mixed_gripper_targets_refuse_conversion(self):
        _, native_space, s = self.rw()
        obs = observation(native_space)
        sample = obs.states['left_gripper_open_fraction']
        obs.states['left_gripper_open_fraction'] = replace(sample, payload=replace(sample.payload, data=[.5]))
        with self.assertRaises(ContractError):
            canonical_observation('robotwin', native_space, obs, s)

    def test_missing_right_hand_is_never_silently_dropped(self):
        cfg, _, s = self.rw()
        cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
        s = replace(s, state_fields={k: v for k, v in s.state_fields.items() if not k.startswith('right_')})
        self.assertIn('missing_required_input', [i.code for i in compatibility(s, cap, cfg.control_mode)[0].issues])

    def test_joint_order_mismatch_rejected(self):
        cfg, _, s = self.rw()
        cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
        spec = s.state_fields['left_joint_drive_targets']
        s = replace(s, state_fields={**s.state_fields, 'left_joint_drive_targets': replace(spec, joint_ids=tuple(reversed(spec.joint_ids)))})
        self.assertIn('joint_order_mismatch', [i.code for i in compatibility(s, cap, cfg.control_mode)[0].issues])

    def test_libero_position_orientation_reference_sites_stay_distinct(self):
        cfg = self.cases['stage3-openwam-libero-spatial-pilot']
        s = default_registry().benchmark('libero').describe(cfg.benchmark.options)
        self.assertNotEqual(s.state_fields['eef_position'].site_id, s.state_fields['eef_quaternion'].site_id)
        self.assertIsNone(s.robots['robot0'].end_effectors['eef'].control_site_id)

    def test_unknown_base_transform_not_an_identity(self):
        _, _, s = self.rw()
        self.assertIsNone(s.frames['robot0/base'].parent_id)
        cap = default_registry().model('lingbot_vla').describe(self.rw()[0].model.options)
        req = c.InputSpec(key='p', public_path='states.left_eef_position', kind='state', frame_id='robot0/base')
        self.assertIn('disconnected_frames', [i.code for i in compatibility(s, replace(cap, inputs=(req,)))[0].issues])

    def test_droid_keeps_stable_camera_identity_and_profile_order(self):
        cosmos, dream = self.cases['cosmos3_droid_sim'], self.cases['dreamzero_droid_sim']
        for cfg, right in ((cosmos, 'external_1'), (dream, 'external_0')):
            cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
            self.assertEqual(next(r.public_path for r in cap.inputs if r.key == 'exterior_right'), 'sensors.' + right)
            s = default_registry().benchmark('droid_sim').describe(cfg.benchmark.options)
            self.assertEqual(set(s.sensors), {'external_0', 'external_1', 'wrist'})
            self.assertFalse(s.view_aliases)


class CoordinateTests(unittest.TestCase):
    def pose_space(self, quaternion=(0., 0., 0., 1.)):
        site = c.ReferenceSite(robot_id='robot0', kind='control_target', origin='synthetic known TCP')
        s = space({})
        robot = replace(s.robots['robot0'], end_effectors={'eef': c.EndEffectorSpec(frame_id='local', tcp_id='tcp', control_site_id='robot0/eef_site')})
        return replace(s, frames={**s.frames, 'local': leaf.FrameSpec(parent_id='world', static_transform=leaf.Transform(translation_m=(1., 2., 3.), quaternion_xyzw=quaternion))}, robots={'robot0': robot}, reference_sites={'robot0/eef_site': site})

    def test_identity_xyz90_xyz180_general_and_sign(self):
        rotations = [(0., 0., 0., 1.)]
        for axis in range(3):
            for angle in (math.pi / 2, math.pi):
                q = [0., 0., 0., math.cos(angle / 2)]
                q[axis] = math.sin(angle / 2)
                rotations.append(tuple(q))
        norm = math.sqrt(.7**2 + .2**2 + .3**2 + .4**2)
        rotations.append(tuple(v / norm for v in (.2, -.3, .4, .7)))
        from embodied_eval.geometry import compose
        for q in rotations:
            for sign in (1., -1.):
                s = self.pose_space(q)
                command = leaf.PoseTarget(command_id='p', target_id='robot0/eef', pose=leaf.Pose(position_m=(.2, .3, .4), quaternion_xyzw=tuple(sign * v for v in q), frame_id='world'))
                action = c.Action(commands=(command,), source_euler_xyz_rad={})
                out = action_for_environment(action, s, None)
                back = compose(s.frames['local'].static_transform, leaf.Transform(translation_m=out.commands[0].pose.position_m, quaternion_xyzw=out.commands[0].pose.quaternion_xyzw))
                self.assertLess(max(abs(a - b) for a, b in zip(back.translation_m, command.pose.position_m)), 1e-12)
                self.assertAlmostEqual(abs(sum(a * b for a, b in zip(back.quaternion_xyzw, command.pose.quaternion_xyzw))), 1.)

    def test_old_frame_euler_hint_cleared(self):
        s = self.pose_space()
        command = leaf.PoseTarget(command_id='p', target_id='robot0/eef', pose=leaf.Pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.), frame_id='world'))
        result = action_for_environment(c.Action(commands=(command,), source_euler_xyz_rad={'robot0/eef': (0., 0., 0.)}), s, None)
        self.assertFalse(result.source_euler_xyz_rad)

    def test_missing_dynamic_transform_is_rejected(self):
        s = self.pose_space()
        s = replace(s, frames={**s.frames, 'local': leaf.FrameSpec(parent_id='world', dynamic=True)})
        command = leaf.PoseTarget(command_id='p', target_id='robot0/eef', pose=leaf.Pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.), frame_id='world'))
        from types import SimpleNamespace
        with self.assertRaises(ContractError):
            action_for_environment(c.Action(commands=(command,)), s, SimpleNamespace(frame_transforms={}))


class CurrentConfigTests(unittest.TestCase):
    def test_unknown_legacy_fields_not_dropped_by_migration(self):
        raw = json.loads((ROOT / 'tests/fixtures/legacy-configs/stage4-pi05-libero-spatial.json').read_text())
        for section in (None, 'model', 'benchmark', 'profile'):
            modified = copy.deepcopy(raw)
            (modified if section is None else modified[section])['unknown_important_setting'] = True
            with self.assertRaises(ContractError):
                migrate(modified)

    def test_nondefault_retired_legacy_fault_not_silently_discarded(self):
        raw = json.loads((ROOT / 'tests/fixtures/legacy-configs/stage4-pi05-libero-spatial.json').read_text())
        raw['model']['fault'] = 'service_error'
        with self.assertRaises(ContractError):
            migrate(raw)

    def test_new_preflight_does_not_import_GPU_or_simulator_modules(self):
        code = "from embodied_eval.config import load_config; from embodied_eval.v03.preflight import check; import sys; check(load_config('examples/mn_openwam_dual_wxyz_encoded_absolute.json')); assert not any(n.split('.')[0] in ('torch','jax','sapien','mujoco','isaacsim','isaaclab') for n in sys.modules)"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_versions_missing_unknown_and_old_refused(self):
        raw = to_data(config())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'config.json'
            for version in (None, '0.1', '0.2', '99'):
                item = dict(raw)
                if version is None:
                    item.pop('contract_version')
                else:
                    item['contract_version'] = version
                p.write_text(json.dumps(item))
                with self.assertRaises(ContractError):
                    load_config(p)

    def test_binding_and_unknown_fields_not_silently_ignored(self):
        for extra in ({'binding': {}}, {'extra': 1}):
            with self.assertRaises(ContractError):
                decode(RunConfig, to_data(config()) | extra)

    def test_duplicate_json_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'config.json'
            for data in ('{"contract_version":"0.3","contract_version":"0.3"}', '{"a":NaN}'):
                p.write_text(data)
                with self.assertRaises(ContractError):
                    load_config(p)

    def test_native_legacy_config_migration_preserves_asset_paths(self):
        for path in (ROOT / 'tests/fixtures/legacy-configs').glob('stage*.json'):
            raw = json.loads(path.read_text())
            cfg, receipt = migrate(raw)
            self.assertEqual(cfg.model.options['checkpoint_dir'], raw['model']['checkpoint_dir'])
            self.assertEqual(cfg.episodes, tuple(decode(EpisodeConfig, item) for item in raw['episodes']))
            self.assertFalse(hasattr(cfg, 'binding'))
            self.assertEqual(receipt['real_validation'], 'not_performed')

    def test_retired_tool_mock_not_claimed_equivalent(self):
        with self.assertRaises(ContractError):
            migrate(json.loads((ROOT / 'tests/fixtures/legacy-configs/mock_tool_run.json').read_text()))

    def test_migration_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'existing.json'
            p.write_text('keep')
            result = subprocess.run([sys.executable, '-m', 'embodied_eval', 'migrate', '--config', str(ROOT / 'tests/fixtures/legacy-configs/stage4-pi05-libero-spatial.json'), '--output', str(p)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(p.read_text(), 'keep')


class StateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.space = replace(space({}), reference_sites={'robot0/orientation': c.ReferenceSite(robot_id='robot0', kind='measured_orientation', origin='synthetic hand body')},
            state_fields={'eef_quaternion': c.StateSpec(semantic='eef_orientation', unit='quaternion_xyzw', frame_id='world',
                tensor=leaf.TensorSpec(dtype='float64', shape=(4,), layout='vector'), origin='synthetic orientation',
                site_id='robot0/orientation', native_source={'synthetic': True})})
        self.sample = c.StateSample(captured_at_s=0., payload=leaf.Tensor(dtype='float64', shape=(4,), layout='vector', data=[0., 0., 0., 1.]))
        self.context = c.Context(run_id='r', env_id='e', episode_id='ep-0', message_id='obs')
        self.obs = c.Observation(context=self.context, observation_id='obs', spec_ref=self.space.spec_id, env_step=0, sim_time_s=0., sensors={}, states={'eef_quaternion': self.sample})

    def test_nonunit_and_zero_quaternions_rejected(self):
        for data in ([0., 0., 0., 0.], [0., 0., 0., 2.], [float('nan'), 0., 0., 1.]):
            with self.assertRaises(ContractError):
                validate_observation(replace(self.obs, states={'eef_quaternion': replace(self.sample, payload=replace(self.sample.payload, data=data))}), self.space)

    def test_invalid_state_must_have_reason_and_no_payload(self):
        for sample in (replace(self.sample, validity='invalid'), replace(self.sample, validity='invalid', invalid_reason='unavailable')):
            with self.assertRaises(ContractError):
                validate_observation(replace(self.obs, states={'eef_quaternion': sample}), self.space)
        validate_observation(replace(self.obs, states={'eef_quaternion': replace(self.sample, validity='invalid', invalid_reason='unavailable', payload=None)}), self.space)

    def test_orientation_frame_and_site_required(self):
        for spec in (replace(self.space.state_fields['eef_quaternion'], frame_id=None), replace(self.space.state_fields['eef_quaternion'], site_id=None)):
            with self.assertRaises(ContractError):
                validate_space(replace(self.space, state_fields={'eef_quaternion': spec}))
        for key in ('left_right_eef_quaternion', 'right_left_eef_quaternion', 'left_left_eef_quaternion'):
            with self.assertRaises(ContractError):
                validate_space(replace(self.space, state_fields={key: self.space.state_fields['eef_quaternion']}))

    def test_wrong_state_provenance_rejected(self):
        with self.assertRaises(ContractError):
            validate_observation(replace(self.obs, states={'eef_quaternion': replace(self.sample, provenance='controller_target')}), self.space)

    def test_wrong_context_version_and_future_time_rejected(self):
        for raw in (to_data(self.obs) | {'context': to_data(self.context) | {'contract_version': '0.2'}},
                    to_data(self.obs) | {'states': {'eef_quaternion': to_data(self.sample) | {'captured_at_s': 1.}}}):
            with self.assertRaises(ContractError):
                validate_observation(decode(c.Observation, raw), self.space)

    def test_required_invalid_state_cannot_reach_native_model(self):
        cap = replace(capability({}), inputs=(c.InputSpec(key='q', public_path='states.eef_quaternion', kind='state'),))
        task = c.TaskSpec(task_id='t', episode_id='ep-0', public=leaf.PublicTask(instruction='test'), budgets=leaf.Budgets(env_steps=1), initialization_ref='synthetic', evaluation_config_ref='synthetic')
        obs = replace(self.obs, states={'eef_quaternion': replace(self.sample, validity='invalid', payload=None, invalid_reason='lost')})
        with self.assertRaises(ContractError):
            model_input(obs, task, cap, resolve(self.space, cap), self.space)

    def test_input_coordinate_conversion_keeps_canonical_key_and_metadata(self):
        s = replace(self.space, frames={**self.space.frames, 'local': leaf.FrameSpec(parent_id='world', static_transform=leaf.Transform(translation_m=(1., 2., 3.), quaternion_xyzw=(0., 0., 1., 0.)))})
        cap = replace(capability({}), inputs=(c.InputSpec(key='private_q_slot', public_path='states.eef_quaternion', kind='state', frame_id='local'),))
        view = c.ModelInput(context=self.context, observation_id='obs', sim_time_s=0., task_public=leaf.PublicTask(instruction='t'), sensors={}, states=self.obs.states, state_specs=s.state_fields)
        moved = inputs_for_model(view, cap, s)
        self.assertEqual(set(moved.states), {'eef_quaternion'})
        self.assertEqual(moved.state_specs['eef_quaternion'].frame_id, 'local')
        self.assertEqual(moved.states['eef_quaternion'].captured_at_s, 0.)
        self.assertAlmostEqual(abs(moved.states['eef_quaternion'].payload.data[2]), 1.)
