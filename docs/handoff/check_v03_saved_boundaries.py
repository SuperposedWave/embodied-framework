"""Read saved real traces through current 0.3 boundaries; never infer/simulate.

This checks EXACT input tensors and action execution values, not new task scores.
The first bounded transitions of each saved run are read, without rewriting it.
"""
import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transitions', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.transitions <= 400:
        raise ValueError('bounded_positive_transition_count_required')
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    from build_native_comparison_report import cases
    from embodied_eval.codec import decode, to_data
    from embodied_eval.mn import contracts as old
    from embodied_eval.mn.binding import model_input as old_input
    from embodied_eval.v03 import contracts as c
    from embodied_eval.v03.migration import migrate
    from embodied_eval.v03.registry import default_registry
    from embodied_eval.v03.benchmarks import canonical_space, canonical_observation
    from embodied_eval.v03.models import private_input, profile_paths
    from embodied_eval.v03.matching import resolve, model_input
    from embodied_eval.v03.validation import validate_space, validate_observation, validate_prediction, validate_step
    from embodied_eval.v03.transforms import action_for_environment
    from embodied_eval.v03.context import public_context
    items = cases()
    items.append({'id': 'openwam_dual_physical', 'framework': ROOT / 'outputs/openwam-dual-pose-live-20261015-run1/recorded-framework/framework'})
    report = dict(kind='v03_saved_real_boundary_values_not_new_inference', at_utc=datetime.now(timezone.utc).isoformat(),
        passed=False, new_GPU_inference=False, new_simulation=False, formal_acceptance=False, cases=[], protected_sha256={})
    report['framework_source_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / 'embodied_eval').rglob('*.py'))}
    report['git_commit_at_start'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    try:
        for item in items:
            manifest_path = item['framework'] / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            assert manifest['synthetic'] is False
            cfg, receipt = migrate(manifest['config'])
            s = validate_space(canonical_space(cfg.benchmark.adapter, decode(old.SpaceSpec, manifest['space'])))
            cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
            original = decode(old.CapabilitySpec, manifest['capability'])
            profile = resolve(s, cap, cfg.control_mode)
            old_profile = decode(old.BoundProfile, manifest['profile'])
            task = decode(old.TaskSpec, manifest['tasks'][0])
            public_task = c.TaskSpec(**{f.name: getattr(task, f.name) for f in fields(c.TaskSpec) if f.name != 'contract_version'})
            trace = item['framework'] / f'episodes/{task.episode_id}/trace.internal.jsonl'
            for path in (manifest_path, trace):
                report['protected_sha256'][str(path)] = digest(path)
            previous = raw_previous = pending = None
            row = dict(case=item['id'], manifest=str(manifest_path), observations=0, predictions=0, transitions=0,
                canonical_sensors=list(s.sensors), canonical_states=list(s.state_fields), input_tensors_and_acquisition_times_exact=True,
                native_execution_action_values_precision_and_acquisition_flag_exact=True, mapping_only_not_task_success=True)
            with trace.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    kind, data = event['event'], event['data']
                    if kind in ('reset', 'step_result'):
                        raw = decode(old.Observation, data if kind == 'reset' else data['observation'])
                        obs = validate_observation(canonical_observation(cfg.benchmark.adapter, decode(old.SpaceSpec, manifest['space']), raw, s), s)
                        view = model_input(obs, public_task, cap, profile, s)
                        actual = private_input(view, original, profile_paths(cfg.model.options['io_profile'], original))
                        expected = old_input(raw, task, original, old_profile)
                        assert set(actual.sensors) == set(expected.sensors)
                        assert actual.states == expected.states
                        assert actual.task_public == expected.task_public
                        assert actual.frame_transforms == expected.frame_transforms
                        for key in actual.sensors:
                            a, b = actual.sensors[key], expected.sensors[key]
                            assert a.payload == b.payload and a.captured_at_s == b.captured_at_s
                            assert a.validity == b.validity and a.provenance == b.provenance
                        row['observations'] += 1
                        if kind == 'step_result':
                            reply = c.StepResult(observation=obs, execution_feedback=decode(old.ExecutionFeedback, data['execution_feedback']),
                                terminated=data['terminated'], truncated=data['truncated'])
                            validate_step(reply, previous, pending.context, pending.action, s)
                            row['transitions'] += 1
                        previous, raw_previous = obs, raw
                        if row['transitions'] >= args.transitions:
                            break
                    elif kind == 'prediction':
                        raw_pred = decode(old.Prediction, data)
                        action = c.Action(**{f.name: getattr(raw_pred.action, f.name) for f in fields(c.Action)})
                        pending = c.Prediction(context=public_context(raw_pred.context), action=action,
                            based_on_observation_id=raw_pred.based_on_observation_id, model_forward_count=raw_pred.model_forward_count,
                            transformations=raw_pred.transformations)
                        validate_prediction(pending, pending.context, previous, s, cap, profile)
                        executed = action_for_environment(action, s, previous)
                        assert to_data(executed) == to_data(raw_pred.action)
                        row['predictions'] += 1
            assert row['transitions'] == args.transitions, (item['id'], row)
            report['cases'].append(row)
            print(json.dumps(row), flush=True)
        assert all(digest(Path(path)) == sha for path, sha in report['protected_sha256'].items())
        assert all(digest(ROOT / path) == sha for path, sha in report['framework_source_sha256'].items())
        report['historical_hashes_unchanged'] = True
        report['passed'] = True
    finally:
        with (args.output / 'receipt.json').open('x') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
