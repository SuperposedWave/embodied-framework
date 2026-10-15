"""Independent native VLABench replay of recorded world targets, not fresh inference.

Use native load_env, quaternion/IK/step/scoring directly. Compare every raw state
and the three recorded camera streams. Never invoke BenchmarkAdapter.advance.
Instruction sampling is not rerun: historical Python random state was not saved.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIGS = {"openwam_vlabench": "stage4-openwam-vlabench-select-fruit.json",
           "openpi_vlabench": "stage4-pi05-vlabench-select-fruit.json"}
CAMERAS = (("forward_rgb", 2), ("franka_wrist_rgb", 3), ("right_rgb", 0))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def snapshot(recorded, native, robot_frame):
    ext = recorded["extensions"]["vlabench.native_proprio.v1"]
    fields = {}
    for name, current in (("ee_state_world_wxyz", native["ee_state"]), ("robot_base_world_m", robot_frame)):
        left, right = np.asarray(ext[name]), np.asarray(current)
        fields[name] = {"exact_equal": np.array_equal(left, right),
                        "max_abs_diff": float(np.max(np.abs(left - right)))}
    for name, index in CAMERAS:
        left = np.asarray(recorded["sensors"][name]["payload"]["data"], dtype=np.uint8)
        right = native["rgb"][index]
        fields[name] = {"exact_equal": np.array_equal(left, right),
                        "different_scalar_values": int(np.count_nonzero(left != right)),
                        "native_pixel_sha256": hashlib.sha256(right.tobytes()).hexdigest()}
    return fields


def run(args):
    from embodied_eval.config import load_config
    from embodied_eval.codec import to_data
    from embodied_eval.stage4_pairs import verify
    config = load_config(ROOT / "examples" / CONFIGS[args.pair])
    verify(config)
    manifest_path = args.framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["synthetic"] or manifest["config"] != to_data(config):
        raise ValueError("real_fixed_pair_manifest_required")
    package = Path(config.benchmark.vlabench_repo) / "VLABench"
    os.environ.update(VLABENCH_ROOT=str(package), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    sys.path.insert(0, config.benchmark.vlabench_repo)
    import VLABench.robots
    import VLABench.tasks
    from VLABench.envs import load_env
    from VLABench.utils.utils import euler_to_quaternion
    frozen = json.loads((package / "configs/evaluation/tracks/track_1_in_distribution.json").read_text())["select_fruit"][0]
    env = load_env("select_fruit", episode_config=copy.deepcopy(frozen), random_init=False, eval=False, run_mode="eval")
    rows = []
    try:
        env.reset()  # Official Evaluator's second reset, including native settle behavior.
        raw = env.get_observation(require_pcd=False)
        period = float(env.control_timestep())
        elapsed = 0.0
        pending, done, steps = None, False, 0
        episode_dir = args.framework / "episodes" / config.episodes[0].episode_id
        trace_path = episode_dir / "trace.internal.jsonl"
        with trace_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "prediction":
                    if pending is not None:
                        raise ValueError("overlapping_recorded_commands")
                    pending = event["data"]["decision"]["steps"][0]["commands"][0]["parameters"]
                elif event["event"] in ("reset", "step_result"):
                    if event["event"] == "step_result":
                        if pending is None or done:
                            raise ValueError("invalid_replay_sequence")
                        position = np.asarray(pending["position_world_m"])
                        quaternion = euler_to_quaternion(*pending["euler_xyz_rad"])
                        _, joints = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=position, quat=quaternion)
                        before = float(env.physics.time())
                        native_step = env.step(np.concatenate([joints, pending["finger_targets_m"]]))
                        elapsed += float(env.physics.time()) - before
                        done = bool(native_step.last())
                        steps += 1
                        if done != event["data"]["terminated"]:
                            raise ValueError("native_termination_mismatch")
                        pending = None
                        raw = env.get_observation(require_pcd=False)
                    observation = event["data"] if event["event"] == "reset" else event["data"]["observation"]
                    fields = snapshot(observation, raw, env.get_robot_frame_position())
                    fields["sim_time_s"] = {"exact_equal": math.isclose(
                        elapsed, observation["sim_time_s"], rel_tol=0, abs_tol=1e-8),
                        "max_abs_diff": abs(elapsed - observation["sim_time_s"])}
                    rows.append({"env_step": steps, "fields": fields})
                    if len(rows) % 25 == 0:
                        print(f"native replay compared {len(rows)} snapshots", flush=True)
        if pending is not None:
            raise ValueError("unexecuted_recorded_command")
        result_path = episode_dir / "result.json"
        result = json.loads(result_path.read_text())
        native_metrics = {"success": done, "progress_score": float(env.get_task_progress())}
        try:
            native_metrics["intention_score"] = float(env.get_intention_score(threshold=0.1))
        except KeyError:
            native_metrics["intention_score"] = None
        metric_checks = {}
        for key, value in native_metrics.items():
            expected = result["metrics"][key]
            metric_checks[key] = ((value is None or isinstance(value, float) and not math.isfinite(value))
                                 and not expected["valid"] or value == expected["value"] and expected["valid"])
        first_divergence = next((row["env_step"] for row in rows
            if not all(field["exact_equal"] for field in row["fields"].values())), None)
        passed = (steps == result["execution_statistics"]["env_steps"] and len(rows) == steps + 1
                  and first_divergence is None and all(metric_checks.values()))
        return {"kind": "independent_native_recorded_action_replay_not_fresh_policy_rollout",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "pair": args.pair, "passed": passed,
            "synthetic": False, "model_called": False, "formal_acceptance": False,
            "framework_actions": result["execution_statistics"]["env_steps"], "replayed_actions": steps,
            "compared_snapshots": len(rows), "control_period_s": period, "elapsed_sim_time_s": elapsed,
            "native_metrics": native_metrics, "metric_checks": metric_checks, "first_divergence": first_divergence,
            "instruction_sampling_reproduced": False,
            "instruction_note": "Historical Python RNG state was not saved; replay compares physical observations and scores, not sampled prompts or fresh predictions",
            "manifest_sha256": digest(manifest_path), "trace_sha256": digest(trace_path),
            "result_sha256": digest(result_path), "rows": rows}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args)
    except Exception as error:
        report = {"kind": "independent_native_action_replay_failed", "passed": False,
                  "checked_at_utc": datetime.now(timezone.utc).isoformat(), "pair": args.pair,
                  "error_type": type(error).__name__, "error_code": getattr(error, "code", None),
                  "formal_acceptance": False}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
