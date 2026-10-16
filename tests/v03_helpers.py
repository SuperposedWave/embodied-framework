"""Synthetic values and pinned metadata ONLY, never robot performance evidence."""
from dataclasses import replace
import json
from pathlib import Path
import runpy
from embodied_eval import _legacy_contracts as leaf
from embodied_eval.codec import decode, to_data
from embodied_eval.mn import contracts as native
from embodied_eval.v03.migration import migrate

ROOT = Path(__file__).resolve().parents[1]


def cases():
    result = {}
    for path in sorted((ROOT / 'tests/fixtures/legacy-configs').glob('stage*.json')):
        result[path.stem] = migrate(json.loads(path.read_text()))[0]
    extensions = runpy.run_path(str(ROOT / 'docs/handoff/prepare_mn_extension.py'))['configurations']()
    for key, cfg in extensions.items():
        result[key] = migrate(to_data(cfg))[0]
    path = ROOT / 'examples/mn_openwam_dual_wxyz_encoded_absolute.json'
    result['openwam_dual_physical'] = decode(__import__('embodied_eval.v03.config', fromlist=['RunConfig']).RunConfig, json.loads(path.read_text()))
    return result


def observation(space):
    def tensor(spec, data):
        return leaf.Tensor(dtype=spec.dtype, shape=spec.shape, layout=spec.layout, data=data)
    sensors = {}
    for key, spec in space.sensors.items():
        actual = replace(spec.tensor, shape=tuple(n if n > 0 else 2 for n in spec.tensor.shape))
        h, w, _ = actual.shape
        data = [[[0, 1, 2]] * w] * h
        sensors[key] = leaf.SensorSample(captured_at_s=0., frame_id=spec.frame_id, payload=tensor(actual, data), provenance='sensor', validity='valid')
    states = {}
    for key, spec in space.state_fields.items():
        data = [0.] * spec.tensor.shape[0]
        if spec.semantic.endswith('orientation'):
            data[-1] = 1.
        states[key] = native.StateSample(captured_at_s=0., payload=tensor(spec.tensor, data), provenance='controller_target' if 'target' in spec.semantic else 'measurement')
    context = native.Context(run_id='synthetic-values', env_id='env-0', episode_id='ep-0', message_id='obs-0')
    return native.Observation(context=context, observation_id='obs-0', spec_ref=space.spec_id, env_step=0, sim_time_s=0., sensors=sensors, states=states)
