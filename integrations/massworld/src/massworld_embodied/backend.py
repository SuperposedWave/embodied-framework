"""Keep public semantics, inference/cache ordering and scoring inside EmbodiedEval."""
from pathlib import Path
import hashlib
import math

from embodied_eval.codec import decode, to_data
from embodied_eval.v03.config import RunConfig, validate_config
from embodied_eval.v03.preflight import check
from embodied_eval.v03.registry import default_registry
from embodied_eval.v03.runner import run


class EmbodiedBackend:
    def __init__(self, config, registry=None):
        self.registry = registry or default_registry()
        if type(config) is not dict or config.get('contract_version') != '0.3':
            raise ValueError('backend requires an explicit public contract_version=0.3')
        # No environment/model selection is repeated in the outer framework.
        self.config = validate_config(decode(RunConfig, config), self.registry)

    @classmethod
    def catalog(cls, registry=None):
        registry = registry or default_registry()
        def entries(table):
            return [{"id": key, "synthetic": entry.synthetic} for key, entry in sorted(table.items())]
        return {
            "backend": "embodied-v03", "public_contract": "0.3",
            "models": entries(registry.models), "benchmarks": entries(registry.benchmarks),
            "pair_whitelist": False,
            "io_profiles_and_environment_protocols": "selected independently in each adapter options; validated by preflight",
            "structural_compatibility_is_not_asset_readiness_or_real_validation": True,
        }

    def preflight(self, verify_assets=False):
        return check(self.config, self.registry, verify_assets=verify_assets)

    def execute(self, output):
        output = Path(output)
        outcome = run(self.config, output, self.registry)
        episodes = to_data(outcome.results)
        rows = {row["episode_id"]: row for row in outcome.summary["episodes"]}
        samples = []
        numeric = {}
        for result in episodes:
            row = rows[result["episode_id"]]
            # Raw bool/null validity, source, unit and reasons are kept intact.
            sample = {**result, "errors": row["errors"],
                      "execution_statistics_complete": row["execution_statistics_complete"]}
            samples.append(sample)
            for name, metric in result["metrics"].items():
                value = metric["value"]
                if metric["valid"] and type(value) in (int, float) and math.isfinite(value):
                    numeric[result["episode_id"] + "/" + name] = value
        manifest = output / "manifest.json"
        summary = output / "summary.json"
        return {
            "success": outcome.exit_code == 0,
            "exit_code": outcome.exit_code,
            "execution_state_known": all(row["execution_statistics_complete"] for row in rows.values()),
            "numeric_metrics": numeric, "samples": samples,
            "metadata": {
                "backend": "embodied-v03", "contract_version": "0.3",
                "synthetic": outcome.summary["synthetic"],
                "success_meaning": "framework completed without errors, not task success",
                "planned": outcome.summary["planned"], "completed": outcome.summary["completed"],
                "errors": outcome.summary["errors"],
                "provenance": outcome.summary["provenance"],
                "compatibility_warnings": outcome.summary["compatibility_warnings"],
                "unattempted_episode_ids": [
                    e.episode_id for e in self.config.episodes if e.episode_id not in rows
                ],
            },
            "evidence": {
                "manifest": {"path": str(manifest.resolve()), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
                "summary": {"path": str(summary.resolve()), "sha256": hashlib.sha256(summary.read_bytes()).hexdigest()},
            },
        }
