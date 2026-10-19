"""SYNTHETIC integration checks against the actual patched MassWorld runtime."""
import contextlib
import copy
from dataclasses import replace
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch

from embodied_eval.codec import decode, to_data
from embodied_eval.errors import AdapterError
from embodied_eval.v03.config import RunConfig
from embodied_eval.v03.registry import default_registry
from embodied_eval.v03.runner import run
from embodied_eval.v03.synthetic import SyntheticBenchmark
from massworld_embodied import EmbodiedBackend
from worldevalkit.cli import main
from worldevalkit.execution_backends import backend_catalog, backend_type, load_config
from worldevalkit.runner import EvaluationRunner

ROOT = Path(__file__).resolve().parents[3]


def config():
    return json.loads((ROOT / 'examples/synthetic_single_run.json').read_text())


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, raw=None, **kwargs):
        return EvaluationRunner(execution_backend='embodied-v03', backend_config=raw or config(),
                                output_dir=self.root, **kwargs)

    def test_entry_point_discovery_and_live_catalog(self):
        self.assertIs(backend_type('embodied-v03'), EmbodiedBackend)
        catalog = backend_catalog('embodied-v03')
        registry = default_registry()
        self.assertEqual({x['id'] for x in catalog['models']}, set(registry.models))
        self.assertEqual({x['id'] for x in catalog['benchmarks']}, set(registry.benchmarks))
        self.assertFalse(catalog['pair_whitelist'])

    def test_four_cross_combinations(self):
        for model in ('synthetic_policy_a', 'synthetic_policy_b'):
            for bench in ('synthetic_simulator_a', 'synthetic_simulator_b'):
                raw = config()
                raw['model']['adapter'], raw['benchmark']['adapter'] = model, bench
                with self.subTest(model=model, benchmark=bench):
                    runner = self.runner(raw)
                    self.assertTrue(runner.plan(write=False).runnable)
                    card = runner.run()
                    self.assertTrue(card['success'])
                    sample = json.loads((runner.run_dir / 'result.json').read_text())['samples'][0]
                    self.assertEqual(sample['execution_statistics']['env_steps'], 6)
                    self.assertIs(sample['metrics']['success']['value'], False)
                    self.assertEqual(card['metrics'], {})
                    self.assertTrue(card['synthetic'])

    def test_new_independent_registrations_need_no_runner_or_other_adapter_changes(self):
        registry = copy.deepcopy(default_registry())
        registry.register_model('third-policy', registry.model('synthetic_policy_a'))
        registry.register_benchmark('third-sim', registry.benchmark('synthetic_simulator_a'))
        for m, b in [('third-policy', 'synthetic_simulator_b'), ('synthetic_policy_b', 'third-sim')]:
            raw = config()
            raw['model']['adapter'], raw['benchmark']['adapter'] = m, b
            factory = lambda cfg: EmbodiedBackend(cfg, registry)
            with patch('worldevalkit.execution_backends.backend_type', return_value=factory):
                self.assertTrue(self.runner(raw).run()['success'])
        self.assertIn('third-policy', [x['id'] for x in EmbodiedBackend.catalog(registry)['models']])

    def test_preflight_never_constructs_model_or_benchmark(self):
        registry = copy.deepcopy(default_registry())
        def forbidden(*args):
            raise AssertionError('factory was called by preflight')
        for table in (registry.models, registry.benchmarks):
            for name, entry in table.items():
                table[name] = replace(entry, factory=forbidden)
        backend = EmbodiedBackend(config(), registry)
        self.assertTrue(backend.preflight(True)['assets_verified'])
        with patch('worldevalkit.execution_backends.backend_type', return_value=lambda cfg: backend):
            runner = self.runner()
            self.assertTrue(runner.plan(write=False).runnable)
            self.assertTrue(runner.doctor()['ok'])
            self.assertTrue(runner.setup()['ready'])
        self.assertFalse(list(self.root.iterdir()))

    def test_direct_and_bridged_boundaries_exact(self):
        for name in ('single', 'cache', 'native_timing'):
            raw = json.loads((ROOT / f'examples/synthetic_{name}_run.json').read_text())
            direct = self.root / ('direct-' + name)
            run(decode(RunConfig, raw), direct)
            outer = self.runner(raw)
            self.assertTrue(outer.run()['success'])
            for filename in ('task.internal.json', 'model_input.public.jsonl', 'trace.internal.jsonl',
                             'result.json', 'diagnostics.internal.json'):
                relative = Path('episodes/episode-0') / filename
                self.assertEqual((direct / relative).read_bytes(), (outer.run_dir / 'backend' / relative).read_bytes(), filename)
            manifest = json.loads((outer.run_dir / 'backend/manifest.json').read_text())
            self.assertEqual(manifest['config'], raw_with_defaults(raw))

    def test_cache_and_native_timing_counters_preserved(self):
        for name, forwards, physics in [('cache', 2, None), ('native_timing', 6, 33)]:
            raw = json.loads((ROOT / f'examples/synthetic_{name}_run.json').read_text())
            runner = self.runner(raw)
            runner.run()
            stats = json.loads((runner.run_dir / 'result.json').read_text())['samples'][0]['execution_statistics']
            self.assertEqual(stats['model_forward_count'], forwards)
            self.assertEqual(stats['physics_steps'], physics)
            self.assertEqual(stats['sim_time_s'], .132 if physics else .3)

    def test_score_error_is_failure_not_zero_metric(self):
        raw = json.loads((ROOT / 'examples/synthetic_score_error_run.json').read_text())
        runner = self.runner(raw)
        card = runner.run()
        self.assertFalse(card['success'])
        self.assertTrue(card['execution_state_known'])
        self.assertEqual(card['stages']['backend-run']['attempts'], 1)
        sample = json.loads((runner.run_dir / 'result.json').read_text())['samples'][0]
        self.assertIsNone(sample['metrics']['success']['value'])
        self.assertFalse(sample['metrics']['success']['valid'])
        self.assertEqual(sample['metrics']['success']['unavailable_reason'], 'scorer_failed')
        self.assertEqual(card['metrics'], {})

    def test_execution_unknown_is_preserved_without_retry(self):
        class Disconnected(SyntheticBenchmark):
            calls = 0
            def advance(self, *args):
                type(self).calls += 1
                raise AdapterError('execution', 'disconnected', 'synthetic disconnect', execution_started=True, state_known=False)
        registry = copy.deepcopy(default_registry())
        registry.benchmarks['synthetic_simulator_a'] = replace(registry.benchmark('synthetic_simulator_a'), factory=Disconnected)
        with patch('worldevalkit.execution_backends.backend_type', return_value=lambda cfg: EmbodiedBackend(cfg, registry)):
            runner = self.runner()
            card = runner.run()
        self.assertFalse(card['success'])
        self.assertFalse(card['execution_state_known'])
        self.assertEqual(Disconnected.calls, 1)
        sample = json.loads((runner.run_dir / 'result.json').read_text())['samples'][0]
        self.assertEqual(sample['termination_reason'], 'execution_state_unknown')
        self.assertIsNone(sample['metrics']['success']['value'])

    def test_outer_cache_and_resume_disabled_and_fresh_runs(self):
        runner = self.runner(reuse_results=True)
        plan = runner.plan(write=False)
        self.assertFalse(plan.plan.stages[0].cache.enabled)
        self.assertEqual(plan.plan.stages[0].retry.attempts, 1)
        runner.run()
        before = runner.run_dir
        runner.run()
        self.assertNotEqual(before, runner.run_dir)
        for root in (before, runner.run_dir):
            provenance = json.loads((root / 'evidence/provenance.json').read_text())
            self.assertFalse(provenance['resume'])
            stage = json.loads((root / 'scorecard.json').read_text())['stages']['backend-run']
            self.assertEqual(stage['status'], 'completed')
            self.assertEqual(stage['attempts'], 1)

    def test_existing_run_directory_refused(self):
        runner = self.runner(record_dir=self.root / 'fixed')
        runner.run()
        manifest = (runner.run_dir / 'backend/manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            runner.run()
        self.assertEqual(manifest, (runner.run_dir / 'backend/manifest.json').read_bytes())

    def test_structure_rejection_never_runs_factories(self):
        raw = json.loads((ROOT / 'examples/synthetic_rejected_run.json').read_text())
        runner = self.runner(raw)
        self.assertEqual(runner.plan(write=False).status, 'not_applicable')
        with self.assertRaises(RuntimeError):
            runner.run()
        self.assertFalse(list(self.root.iterdir()))

    def test_no_mixed_model_selection_or_options(self):
        for kwargs in ({'model': 'openwam'}, {'benchmark': 'libero'}, {'parameters': {'seed': 8}}, {'endpoint': 'robot'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.runner(**kwargs)
        with self.assertRaises(ValueError):
            EvaluationRunner(model='mock', benchmark='mock', backend_config=config())

    def test_duplicate_or_missing_backend_entry_point_rejected(self):
        with patch('worldevalkit.execution_backends.entry_points', return_value=[]), self.assertRaises(ValueError):
            backend_type('embodied-v03')
        entry = type('Entry', (), {'name': 'duplicate'})()
        with patch('worldevalkit.execution_backends.entry_points', return_value=[entry, entry]), self.assertRaises(ValueError):
            backend_type('duplicate')

    def test_old_contract_and_unknown_fields_rejected(self):
        for key, value in [('contract_version', '0.2'), ('binding', {})]:
            raw = config()
            raw[key] = value
            with self.assertRaises(ValueError):
                self.runner(raw)
        raw = config()
        del raw['contract_version']
        with self.assertRaises(ValueError):
            self.runner(raw)

    def test_secrets_rejected_recursively_before_writing(self):
        raw = config()
        raw['model']['options']['nested'] = [{'api_key': 'synthetic-secret-do-not-copy'}]
        with self.assertRaises(ValueError) as caught:
            self.runner(raw)
        self.assertNotIn('synthetic-secret-do-not-copy', str(caught.exception))
        self.assertFalse(list(self.root.iterdir()))

    def test_strict_json_duplicate_and_non_finite(self):
        for index, raw in enumerate(['{"x":1,"x":2}', '{"x":NaN}']):
            path = self.root / str(index)
            path.write_text(raw)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_evidence_hashes_are_full_sha256(self):
        runner = self.runner()
        runner.run()
        evidence = json.loads((runner.run_dir / 'backend-result.json').read_text())['evidence']
        for record in evidence.values():
            self.assertEqual(record['sha256'], hashlib.sha256(Path(record['path']).read_bytes()).hexdigest())

    def test_numeric_projection_keeps_validity_units_and_sources(self):
        from embodied_eval.core.types import Metric
        class MetricsBenchmark(SyntheticBenchmark):
            def score(self, context):
                return {**super().score(context),
                        'distance': Metric(value=1.25, valid=True, unit='m', source='synthetic-distance'),
                        'unavailable': Metric(value=None, valid=False, unit='m', source='synthetic-distance', unavailable_reason='not_measured')}
        registry = copy.deepcopy(default_registry())
        registry.benchmarks['synthetic_simulator_a'] = replace(registry.benchmark('synthetic_simulator_a'), factory=MetricsBenchmark)
        with patch('worldevalkit.execution_backends.backend_type', return_value=lambda cfg: EmbodiedBackend(cfg, registry)):
            runner = self.runner()
            card = runner.run()
        self.assertEqual(card['metrics'], {'episode-0/distance': 1.25})
        metrics = json.loads((runner.run_dir / 'result.json').read_text())['samples'][0]['metrics']
        self.assertEqual(metrics['distance']['unit'], 'm')
        self.assertEqual(metrics['distance']['source'], 'synthetic-distance')
        self.assertIsNone(metrics['unavailable']['value'])
        self.assertIs(metrics['success']['value'], False)

    def test_failure_stops_later_episodes_and_lists_unattempted(self):
        raw = json.loads((ROOT / 'examples/synthetic_score_error_run.json').read_text())
        raw['episodes'].append({**raw['episodes'][0], 'episode_id': 'episode-1'})
        runner = self.runner(raw)
        self.assertFalse(runner.run()['success'])
        result = json.loads((runner.run_dir / 'backend-result.json').read_text())
        self.assertEqual(result['metadata']['unattempted_episode_ids'], ['episode-1'])
        self.assertEqual(result['metadata']['planned'], 2)
        self.assertEqual(len(result['samples']), 1)

    def test_unreported_startup_error_is_not_assumed_state_known(self):
        backend = EmbodiedBackend(config())
        with patch.object(backend, 'execute', side_effect=RuntimeError('synthetic startup failure')) as execute:
            with patch('worldevalkit.execution_backends.backend_type', return_value=lambda cfg: backend):
                runner = self.runner()
                card = runner.run()
        self.assertFalse(card['success'])
        self.assertIsNone(card['execution_state_known'])
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(card['stages']['backend-run']['attempts'], 1)

    def test_missing_assets_block_readiness_but_not_mislabeled_as_structure_error(self):
        registry = copy.deepcopy(default_registry())
        def missing(options):
            raise FileNotFoundError('synthetic missing file')
        registry.models['synthetic_policy_a'] = replace(registry.model('synthetic_policy_a'), verify_assets=missing)
        with patch('worldevalkit.execution_backends.backend_type', return_value=lambda cfg: EmbodiedBackend(cfg, registry)):
            runner = self.runner()
            plan = runner.plan(write=False)
            self.assertEqual(plan.status, 'blocked')
            self.assertTrue(plan.plan.metadata['preflight']['structurally_compatible'])
            self.assertFalse(runner.doctor()['ok'])
        self.assertFalse(list(self.root.iterdir()))

    def test_assets_alone_never_mark_real_runtime_ready(self):
        registry = copy.deepcopy(default_registry())
        registry.models['synthetic_policy_a'] = replace(registry.model('synthetic_policy_a'),
            readiness=lambda o: {'local_paths_available': True, 'dependencies_verified': False})
        with patch('worldevalkit.execution_backends.backend_type', return_value=lambda cfg: EmbodiedBackend(cfg, registry)):
            runner = self.runner()
            decision = runner.plan(write=False)
            self.assertTrue(decision.plan.metadata['preflight']['assets_verified'])
            self.assertEqual(decision.status, 'needs_setup')
            self.assertIn('runtime_dependencies_unverified', decision.reasons)
            self.assertFalse(runner.doctor()['ok'])

    def test_cli_catalog_and_five_exit_codes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['catalog', 'backend', '--execution-backend', 'embodied-v03']), 0)
        for case, expected in [('single', 0), ('cache', 0), ('native_timing', 0), ('score_error', 1), ('rejected', 2)]:
            with self.subTest(case=case), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(['run', '--execution-backend', 'embodied-v03', '--backend-config',
                             str(ROOT / f'examples/synthetic_{case}_run.json'), '--output-dir', str(self.root / case)])
                self.assertEqual(code, expected)

    def preparation_module(self):
        spec = importlib.util.spec_from_file_location('massworld_preparation', ROOT / 'integrations/massworld/prepare_workspace.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_preparation_rejects_unreviewed_zip_before_writing(self):
        archive = self.root / 'unreviewed.zip'
        archive.write_bytes(b'not the reviewed source')
        with self.assertRaises(ValueError):
            self.preparation_module().prepare(archive, self.root / 'output')
        self.assertFalse((self.root / 'output').exists())

    def test_preparation_preserves_source_and_refuses_existing_directory(self):
        module = self.preparation_module()
        archive = self.root / 'fixture.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr(module.PREFIX + 'src/example.txt', 'synthetic source')
            z.writestr(module.PREFIX + 'LICENSE', 'synthetic inner license')
            z.writestr('MassWorld-main/LICENSE', 'synthetic containing license')
        source = archive.read_bytes()
        with patch.object(module, 'ZIP_SHA256', hashlib.sha256(source).hexdigest()):
            checkout = module.prepare(archive, self.root / 'output', patched=False)
            self.assertEqual((checkout / 'src/example.txt').read_text(), 'synthetic source')
            self.assertEqual((checkout / 'LICENSE').read_text(), 'synthetic inner license')
            self.assertEqual((checkout.parent / 'MassWorld-LICENSE').read_text(), 'synthetic containing license')
            with self.assertRaises(FileExistsError):
                module.prepare(archive, self.root / 'output', patched=False)
        self.assertEqual(archive.read_bytes(), source)

    def test_preparation_rejects_traversal_symlinks_and_duplicate_paths(self):
        module = self.preparation_module()
        for kind in ('traversal', 'symlink', 'duplicate'):
            archive = self.root / (kind + '.zip')
            info = zipfile.ZipInfo(module.PREFIX + ('../outside.txt' if kind == 'traversal' else 'source.txt'))
            if kind == 'symlink':
                info.external_attr = 0o120777 << 16
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr(info, b'synthetic')
                    if kind == 'duplicate':
                        z.writestr(info, b'duplicate')
            with patch.object(module, 'ZIP_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest()):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    module.prepare(archive, self.root / ('out-' + kind), patched=False)
            self.assertFalse((self.root / ('out-' + kind)).exists())


def raw_with_defaults(raw):
    return to_data(decode(RunConfig, raw))


if __name__ == '__main__':
    unittest.main()
