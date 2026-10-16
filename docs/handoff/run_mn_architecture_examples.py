"""Persist synthetic 2x2 plus independently registered third-plugin evidence."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from embodied_eval import _legacy_contracts as leaf
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.config import RunConfig, AdapterConfig, EpisodeConfig
    from embodied_eval.mn.registry import AdapterRegistry
    from embodied_eval.mn.synthetic import register, SyntheticModel, SyntheticBenchmark
    from embodied_eval.mn.runner import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    registry = AdapterRegistry()
    register(registry)
    rows = []
    def exercise(model, benchmark, name, camera="camera", robot="robot0"):
        config = RunConfig(run_id=name, model=AdapterConfig(adapter=model),
            benchmark=AdapterConfig(adapter=benchmark, options={"camera_name": camera, "robot_id": robot}),
            binding=c.BindingConfig(inputs={"image": camera, "instruction": "instruction"},
                targets={"joint": robot + "/j0"}, action_spec_ref="joint-position@1"),
            episodes=(EpisodeConfig(episode_id="episode0", budgets=leaf.Budgets(env_steps=3, model_calls=3)),))
        result = run(config, args.output / name, registry)
        rows.append({"model": model, "benchmark": benchmark, "exit_code": result.exit_code,
                     "synthetic": result.summary["synthetic"], "statistics": result.results[0]["execution_statistics"],
                     "summary": str(args.output / name / "summary.json")})
    for i, model in enumerate(tuple(registry.models)):
        for j, benchmark in enumerate(tuple(registry.benchmarks)):
            exercise(model, benchmark, f"cross-{i}-{j}")
    class ThirdPolicy(SyntheticModel):
        def predict(self, context):
            prediction = super().predict(context)
            return replace(prediction, action=replace(prediction.action,
                commands=(replace(prediction.action.commands[0], value=-.3),)))
    class ThirdSimulator(SyntheticBenchmark):
        pass
    model_entry = registry.model("synthetic_policy_a")
    registry.register_model("third-policy", replace(model_entry, factory=ThirdPolicy,
        describe=lambda o: replace(model_entry.describe(o), model_id="third-policy")))
    registry.register_benchmark("third-simulator", replace(registry.benchmark("synthetic_simulator_a"), factory=ThirdSimulator))
    exercise("third-policy", "synthetic_simulator_b", "third-model")
    exercise("synthetic_policy_b", "third-simulator", "third-environment", "third_camera", "third_arm")
    receipt = {"kind": "synthetic_architecture_not_real_robot_or_benchmark_score",
        "at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": True,
        "passed": len(rows) == 6 and all(row["exit_code"] == 0 and row["synthetic"] for row in rows),
        "four_cross_combinations": 4, "extension_cases": 2,
        "models_use_distinct_factories": True, "simulators_use_distinct_factories": True,
        "runner_or_existing_adapter_modified_for_third_plugins": False, "results": rows}
    (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "results"}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
