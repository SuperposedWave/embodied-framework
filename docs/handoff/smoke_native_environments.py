"""Bounded native reset/render probes; no model, policy action or score claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def child(pair):
    from embodied_eval.config import load_config
    from embodied_eval.codec import to_data
    from embodied_eval.validation import validate_observation
    from embodied_eval import contracts as c
    config = load_config(ROOT / "examples" / {
        "openwam_vlabench": "stage4-openwam-vlabench-select-fruit.json",
        "openpi_vlabench": "stage4-pi05-vlabench-select-fruit.json",
        "openpi_libero": "stage4-pi05-libero-spatial.json"}[pair])
    if pair == "openpi_libero":
        from embodied_eval.stage4_pairs import verify
        verify(config)  # Verify trusted local initial-state pickle before load.
        from embodied_eval.adapters.libero_openpi_benchmark import LiberoOpenpiBenchmark
        benchmark = LiberoOpenpiBenchmark(config.benchmark)
    elif pair == "openpi_vlabench":
        from embodied_eval.adapters.vlabench_openpi_benchmark import VLABenchOpenpiBenchmark
        benchmark = VLABenchOpenpiBenchmark(config.benchmark)
    else:
        from embodied_eval.adapters.vlabench_benchmark import VLABenchBenchmark
        benchmark = VLABenchBenchmark(config.benchmark)
    try:
        space = benchmark.describe()
        tasks = benchmark.enumerate_tasks({"episodes": to_data(config.episodes)})
        observation = benchmark.reset(tasks[0], c.Context(run_id="native-reset-smoke", env_id="env0",
                                           episode_id=tasks[0].episode_id, message_id="reset"))
        validate_observation(observation, space)
        print("PROBE_RECEIPT " + json.dumps({"pair": pair, "passed": True, "synthetic": False,
              "task_id": tasks[0].task_id, "protocol_version": space.protocol_version,
              "control_period_s": space.control_period_s, "sensors": {key: sample.payload.shape for key, sample in observation.sensors.items()},
              "env_steps": 0, "policy_queries": 0, "closed_loop": False, "scoring_performed": False}))
    finally:
        benchmark.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", choices=("openwam_vlabench", "openpi_vlabench", "openpi_libero"))
    args = parser.parse_args()
    if args.child:
        try:
            child(args.child)
            return 0
        except Exception as error:
            chain = []
            cause = error
            while cause is not None and len(chain) < 6:
                chain.append({"type": type(cause).__name__,
                              "missing_module": cause.name if isinstance(cause, ModuleNotFoundError) else None,
                              "missing_file": cause.filename if isinstance(cause, FileNotFoundError) else None})
                cause = cause.__cause__
            print("PROBE_RECEIPT " + json.dumps({"pair": args.child, "passed": False,
                  "error_type": type(error).__name__, "error_code": getattr(error, "code", None),
                  "missing_module": error.name if isinstance(error, ModuleNotFoundError) else None,
                  "error_chain": chain,
                  "closed_loop": False, "scoring_performed": False}))
            return 1
    if args.output is None:
        parser.error("--output required")
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for pair, env in (("openwam_vlabench", "vlabench-stage4"), ("openpi_vlabench", "vlabench-stage4"), ("openpi_libero", "libero-openpi-stage4")):
        try:
            result = subprocess.run([str(WORKSPACE / "upstream/envs" / env / "bin/python"), __file__, "--child", pair],
                                    cwd=ROOT, capture_output=True, text=True, timeout=180)
            # Logs contain local public source diagnostics only; no credentials are read.
            (args.output / (pair + ".txt")).write_text(result.stdout + result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith("PROBE_RECEIPT ")]
            value = json.loads(lines[-1].removeprefix("PROBE_RECEIPT ")) if lines else {"pair": pair, "passed": False, "error_type": "ProcessFailure"}
            value["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            value = {"pair": pair, "passed": False, "error_type": "BoundedProbeTimeout", "closed_loop": False}
        results.append(value)
        print(json.dumps(value), flush=True)
    report = {"kind": "native_environment_reset_render_only_not_closed_loop_or_score",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": all(r["passed"] for r in results), "results": results}
    (args.output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
