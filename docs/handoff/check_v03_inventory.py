"""Current independent 11x7 metadata grid; NOT a runtime pairing whitelist."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    from embodied_eval.codec import to_data
    from embodied_eval.config import load_config
    from embodied_eval.v03.registry import default_registry
    from embodied_eval.v03.validation import validate_space
    from embodied_eval.v03.matching import compatibility
    from prepare_mn_extension import historical_integrity
    registry = default_registry()
    models, benchmarks, sources = {}, {}, {}
    for path in sorted((ROOT / 'examples').glob('*.json')):
        if path.name == 'contract_examples.json' or path.name.startswith('synthetic_') or path.name == 'mn_synthetic_run.json':
            continue
        cfg = load_config(path)
        assert cfg.contract_version == '0.3'
        mk = cfg.model.adapter + '/' + cfg.model.options['io_profile']
        bk = cfg.benchmark.adapter + '/' + cfg.benchmark.options['protocol']
        models[mk] = (cfg.model, registry.model(cfg.model.adapter).describe(cfg.model.options))
        benchmarks[bk] = (cfg.benchmark, validate_space(registry.benchmark(cfg.benchmark.adapter).describe(cfg.benchmark.options)))
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(models) == 11 and len(benchmarks) == 7
    rows = []
    for mk, (_, cap) in models.items():
        for bk, (_, space) in benchmarks.items():
            report = compatibility(space, cap)[0]
            rows.append(dict(model=mk, benchmark=bk, structurally_compatible=not report.issues,
                issues=[to_data(issue) for issue in report.issues], warnings=list(report.warnings),
                structure_is_not_asset_or_real_validation=True))
    forbidden = [name for name in sys.modules if name.split('.')[0] in ('torch', 'jax', 'mujoco', 'sapien', 'isaacsim', 'isaaclab')]
    assert not forbidden
    old_examples = {}
    for path in (ROOT / 'tests/fixtures/legacy-configs').glob('*.json'):
        old = subprocess.check_output(['git', 'show', 'pre-v03-20261016:examples/' + path.name], cwd=ROOT)
        assert old == path.read_bytes()
        old_examples[path.name] = hashlib.sha256(old).hexdigest()
    protected_contracts = {}
    for name in ('docs/contract-v0.1.md', 'docs/contract-v0.2.md', 'docs/public-space-fields-review-20261016.html'):
        path = ROOT / name
        old = subprocess.check_output(['git', 'show', 'pre-v03-20261016:' + name], cwd=ROOT)
        assert old == path.read_bytes()
        protected_contracts[name] = hashlib.sha256(old).hexdigest()
    result = dict(kind='v03_current_metadata_and_history_integrity', at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        passed=True, current_contract='0.3', model_profiles=len(models), environment_protocols=len(benchmarks),
        metadata_cells=len(rows), structurally_compatible_cells=sum(row['structurally_compatible'] for row in rows),
        no_pair_specific_binding_required=True, runtime_pair_whitelist=False, no_GPU_modules_imported=True,
        models={k:to_data(v[1]) for k,v in models.items()}, benchmarks={k:to_data(v[1]) for k,v in benchmarks.items()},
        rows=rows, current_example_sha256=sources, preserved_legacy_example_sha256=old_examples,
        preserved_contract_and_proposal_sha256=protected_contracts, historical_integrity=historical_integrity(),
        new_real_inference=False, formal_acceptance=False)
    with (args.output / 'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({key:result[key] for key in ('passed', 'model_profiles', 'environment_protocols', 'metadata_cells', 'structurally_compatible_cells')}))


if __name__ == '__main__':
    main()
