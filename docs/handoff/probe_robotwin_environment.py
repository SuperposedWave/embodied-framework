"""Actual native expert-valid setup/render/public mapping; no policy or score."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIGS = {"lingbot_robotwin": "stage4-lingbot-robotwin-adjust-bottle.json",
           "openwam_robotwin": "stage4-openwam-robotwin-adjust-bottle.json"}


def run(pair, output):
    from embodied_eval.config import load_config, make_adapters, make_profile
    from embodied_eval.codec import to_data
    from embodied_eval import _legacy_contracts as c
    from embodied_eval.compatibility import check_compatibility
    from embodied_eval._legacy_validation import validate_observation
    from embodied_eval.visibility import build_model_input
    config = load_config(ROOT / "examples" / CONFIGS[pair])
    model, benchmark = make_adapters(config, output / "not-a-framework-run")
    try:
        space, cap = benchmark.describe(), model.describe()
        profile = make_profile(config, space, cap)
        check_compatibility(space, cap, profile).require_compatible()
        task = benchmark.enumerate_tasks({"episodes": to_data(config.episodes)})[0]
        context = c.Context(run_id="robotwin-native-environment-probe", env_id="env0",
                            episode_id=task.episode_id, message_id="reset")
        observation = benchmark.reset(task, context)
        validate_observation(observation, space)
        public_input = build_model_input(task, observation, profile)
        native = model.native_input(public_input)
        with (output / "public-model-input.json").open("x") as stream:
            json.dump(to_data(public_input), stream)
        return {"passed": True, "pair": pair, "native_expert_valid_setup": True,
            "real_render_performed": True, "public_mapping_validated": True,
            "physics_period_s": space.physics_period_s,
            "native_camera_shapes": {key: list(value["rgb"].shape) for key, value in native["observation"].items()},
            "native_joint_drive_target_shape": list(native["joint_action"]["vector"].shape),
            "policy_actions": 0, "model_request_sent": False,
            "expert_actions_are_initialization_not_policy_score": True,
            "benchmark_diagnostics": benchmark.diagnostics()}
    finally:
        try:
            model.close()
        finally:
            benchmark.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args.pair, args.output)
    except Exception as error:
        chain = []
        while error is not None and len(chain) < 6:
            chain.append({"type": type(error).__name__, "missing_module": getattr(error, "name", None),
                          "code": getattr(error, "code", None)})
            error = error.__cause__
        report = {"passed": False, "pair": args.pair, "gate": "native_environment", "error_chain": chain}
    report.update(kind="real_robotwin_initialization_render_mapping_not_policy_or_score",
        checked_at_utc=datetime.now(timezone.utc).isoformat(), scoring_performed=False, formal_acceptance=False)
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
