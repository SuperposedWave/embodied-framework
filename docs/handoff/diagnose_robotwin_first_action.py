"""Observe one recorded real RoboTwin pose action without fresh policy inference.

Native mode calls the independent fixed-source replay, bounded to its first
action. Adapter mode observes the same action through the independent Benchmark
component. Observers only read native arguments/results; no planner seeds,
physics settings, score rules or tolerances are changed. Native logs stay in
memory. This is a diagnostic, never a replacement for full trajectory replay.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from importlib import metadata
from itertools import islice
import json
import os
from pathlib import Path
import resource
import sys

from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
sys.path.insert(0, str(ROOT))


def child(args):
    import hashlib
    import numpy as np
    from embodied_eval.codec import decode, to_data
    from embodied_eval.core import types as leaf
    from embodied_eval.native import contracts as c
    from tests.legacy_support.frozen.mn.config import RunConfig, validate_config
    from embodied_eval.native.registry import default_registry
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.adapters.native_robotwin_runtime import prewarm_native_cuda
    import replay_mn_native_actions as replay
    source = args.framework.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    config = validate_config(decode(RunConfig, manifest["config"]))
    if manifest["synthetic"] or config.benchmark.adapter != "robotwin" or config.benchmark.options["protocol"] != "sealed-instruction":
        raise ValueError("fixed real sealed-instruction RoboTwin trial required")
    verified = default_registry().benchmark("robotwin").verify_assets(config.benchmark.options)
    episode = source / "episodes" / config.episodes[0].episode_id
    prefix = list(islice(replay.trace(episode / "trace.internal.jsonl"), 2))
    if len(prefix) != 2 or prefix[0][1] is not None or prefix[1][1] is None:
        raise ValueError("recorded reset and first completed action required")
    action = decode(c.Action, prefix[1][1])
    if action.native_protocol != "dual-eef16-legacy-quaternion@1":
        raise ValueError("historical native pose-extension diagnostic only")
    prewarm_native_cuda()
    # Native envs package reads immutable relative assets during import. Use
    # the same owned symlink layout as replay, never chdir into the reference.
    bootstrap = args.output / "native-import-runtime"
    bootstrap.mkdir()
    for name in ("envs", "assets", "task_config", "description"):
        (bootstrap / name).symlink_to(Path(config.benchmark.options["repo"]) / name, target_is_directory=True)
    previous = Path.cwd()
    try:
        os.chdir(bootstrap)
        robot_class = native_module(config.benchmark.options["repo"], "envs.robot.robot").Robot
    finally:
        os.chdir(previous)
    captured, active = [], [False]

    def array(value):
        data = np.asarray(value)
        return {"dtype": str(data.dtype), "shape": list(data.shape), "values": data.tolist(),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest()}

    def state(robot, side):
        entity = getattr(robot, side + "_entity")
        return {"qpos": array(entity.get_qpos()),
                "qvel": array(entity.get_qvel()) if hasattr(entity, "get_qvel") else None}

    @contextmanager
    def observers():
        originals = {}
        for side in ("left", "right"):
            name = side + "_plan_path"
            original = getattr(robot_class, name)
            originals[name] = original
            def plan(robot, *values, _original=original, _side=side, **kwargs):
                entry = {"operation": _side + "_plan_path", "target_pose": array(values[0]),
                         "initial_articulation": state(robot, _side)} if active[0] else None
                result = _original(robot, *values, **kwargs)
                if entry is not None:
                    entry["result"] = {key: array(value) if isinstance(value, np.ndarray) else value for key, value in result.items()}
                    captured.append(entry)
                return result
            setattr(robot_class, name, plan)
        for name in ("set_arm_joints", "set_gripper"):
            original = getattr(robot_class, name)
            originals[name] = original
            def command(robot, *values, _original=original, _name=name, **kwargs):
                if active[0]:
                    captured.append({"operation": _name, "arguments": [array(value) if not isinstance(value, str) else value for value in values],
                                     "keyword_arguments": {key: array(value) if not isinstance(value, str) else value for key, value in kwargs.items()}})
                return _original(robot, *values, **kwargs)
            setattr(robot_class, name, command)
        try:
            yield
        finally:
            for name, original in originals.items():
                setattr(robot_class, name, original)

    result = {"kind": "recorded_real_robotwin_first_pose_action_observation_not_full_replay",
              "mode": args.mode, "synthetic": False, "fresh_model_inference": False, "formal_acceptance": False,
              "completed": False, "source_assets_verified": verified,
              "manifest_sha256": replay.digest(source / "manifest.json"), "recorded_action": prefix[1][1]}
    benchmark = None
    try:
        with observers():
            if args.mode == "native":
                original_trace = replay.trace
                def first_action(path):
                    for observation, recorded, reply in islice(original_trace(path), 2):
                        active[0] = recorded is not None
                        yield observation, recorded, reply
                        active[0] = False
                replay.trace = first_action
                expected = {"execution_statistics": {"env_steps": 1,
                    "physics_steps": prefix[1][2]["execution_feedback"]["consumed_physics_steps"]},
                    "metrics": {"success": {"value": False, "valid": True}}}
                result["prefix_comparison"] = replay.replay_robotwin(config, episode, expected, args.output)
            else:
                from embodied_eval.native.native_benchmarks import NativeSimulator
                benchmark = NativeSimulator("robotwin", config.benchmark.options, args.output / "diagnostic")
                benchmark.describe()  # Runner negotiates actual native timing before enumerating.
                task = benchmark.enumerate_tasks(config.episodes)[0]
                context = c.Context(run_id="first-pose-action-diagnostic", env_id="env0", episode_id=task.episode_id, message_id="reset")
                initial = benchmark.reset(task, context)
                expected_states = set(prefix[0][0]["states"])
                if not expected_states <= set(initial.states):
                    raise ValueError("historical native state field missing")
                result["additional_measured_state_fields_not_in_historical_trace"] = {
                    name: to_data(sample) for name, sample in initial.states.items() if name not in expected_states}
                initial_fields = replay.snapshot(prefix[0][0],
                    {name: sample.payload.data for name, sample in initial.states.items() if name in expected_states},
                    {name: sample.payload.data for name, sample in initial.sensors.items()}, 0.)
                active[0] = True
                reply = benchmark.advance(action, replace(context, message_id="action1"), leaf.Budgets(env_steps=2))
                active[0] = False
                current = reply.observation
                fields = replay.snapshot(prefix[1][0],
                    {name: sample.payload.data for name, sample in current.states.items() if name in expected_states},
                    {name: sample.payload.data for name, sample in current.sensors.items()}, current.sim_time_s)
                result.update(prefix_comparison={"rows": [{"env_step": 0, "fields": initial_fields}, {"env_step": 1, "fields": fields}],
                    "native_physx_steps": reply.execution_feedback.consumed_physics_steps},
                    execution_feedback=to_data(reply.execution_feedback), benchmark_diagnostics=benchmark.diagnostics())
            result.update(completed=True, captured_operations=captured,
                native_source_sha256=replay.digest(Path(config.benchmark.options["repo"]) / "envs/robot/robot.py"))
            result["package_versions"] = {name: metadata.version(name) for name in ("torch", "numpy", "sapien", "warp-lang")}
    except Exception as error:
        result.update(exception_chain=public_exception(error), captured_operations=captured)
    finally:
        if benchmark is not None:
            try:
                benchmark.close()
            except Exception as error:
                result.update(completed=False, close_exception_chain=public_exception(error))
    write(args.output / "component-result.json", result)
    return 0 if result["completed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "adapter"), default="native")
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.child:
        try:
            return child(args)
        except Exception as error:
            write(args.output / "early-child-failure.json", {"completed": False, "exception_chain": public_exception(error)})
            return 1
    args.output.relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    # Installation helpers use Python3.11 tomllib; the native RoboTwin client
    # is Python3.10 and must not import unrelated deployment dependencies.
    from prepare_robodojo_environment import bounded
    if not 0 <= args.gpu < 8:
        raise ValueError("existing GPU ordinal required")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    command = [str(WORKSPACE / "upstream/envs/robotwin-stage4/bin/python"), __file__, "--child", "--framework", str(args.framework.resolve()),
               "--output", str(args.output), "--mode", args.mode, "--gpu", str(args.gpu)]
    code, out, err = bounded(command, environment, 900)
    path = args.output / "component-result.json"
    receipt = {"kind": "supervised_recorded_robotwin_first_action_diagnostic", "mode": args.mode,
        "exit_code": code, "completed": code == 0 and path.is_file() and json.loads(path.read_text())["completed"],
        "gpu": args.gpu, "public_log_categories": categories((out + err).decode(errors="replace")), "raw_logs_saved": False,
        "formal_acceptance": False, "planner_or_physics_settings_changed": False, "fresh_model_inference": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit("Retired historical execution tool. Use current 0.3 run/check or run_v03_bounded.py; historical references are test-only.")
