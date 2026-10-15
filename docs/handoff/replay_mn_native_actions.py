"""Direct native replay of 0.2 recorded actions, without either Adapter.

No fresh inference, pair registry/factory, Benchmark.advance, IK fallback or
modified scoring. Source/assets are verified per component before native load.
RoboTwin counts completed PhysX steps independently of the production meter.
Differences are retained, not replaced by a task-success acceptance criterion.
"""
import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def trace(path):
    pending = None
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "prediction":
                if pending is not None:
                    raise ValueError("overlapping_recorded_predictions")
                pending = event["data"]["action"]
            elif event["event"] == "reset":
                yield event["data"], None, None
            elif event["event"] == "step_result":
                if pending is None:
                    raise ValueError("step_without_prediction")
                yield event["data"]["observation"], pending, event["data"]
                pending = None
    if pending is not None:
        raise ValueError("unexecuted_recorded_action")


def field(left, right):
    a, b = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    return {"exact_equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if a.shape == b.shape and a.size else None}


def snapshot(observation, values, images, acquired_at=None, state_acquired_at=None):
    result = {}
    if set(values) != set(observation["states"]):
        raise ValueError("public_state_inventory_changed")
    if set(images) != set(observation["sensors"]):
        raise ValueError("public_camera_inventory_changed")
    for name, value in values.items():
        sample = observation["states"][name]
        result[name] = field(sample["payload"]["data"], value)
        timestamp = (state_acquired_at or {}).get(name, acquired_at)
        if timestamp is not None:
            result[name + ":acquisition_time"] = field(sample["captured_at_s"], timestamp)
    for name, value in images.items():
        sample = observation["sensors"][name]
        left = np.asarray(sample["payload"]["data"], np.uint8)
        right = np.asarray(value, np.uint8)
        result[name] = {"exact_equal": bool(np.array_equal(left, right)),
                        "different_scalar_values": int(np.count_nonzero(left != right)),
                        "native_pixel_sha256": hashlib.sha256(right.tobytes()).hexdigest()}
        if acquired_at is not None:
            result[name + ":acquisition_time"] = field(sample["captured_at_s"], acquired_at)
    return result


def summarize(rows, steps, result, metrics, **details):
    checks = {key: (value == result["metrics"][key]["value"] and result["metrics"][key]["valid"])
              if value is not None else not result["metrics"][key]["valid"] for key, value in metrics.items()}
    divergence = next((r["env_step"] for r in rows if not all(v["exact_equal"] for v in r["fields"].values())), None)
    return {"passed": steps == result["execution_statistics"]["env_steps"] and len(rows) == steps + 1
            and divergence is None and all(checks.values()), "replayed_actions": steps,
            "framework_actions": result["execution_statistics"]["env_steps"],
            "compared_snapshots": len(rows), "first_divergence": divergence,
            "native_metrics": metrics, "metric_checks": checks, "rows": rows, **details}


def replay_libero(config, episode_dir, result, output):
    import torch
    import yaml
    options = config.benchmark.options
    package = Path(options["repo"]) / "libero/libero"
    with tempfile.TemporaryDirectory(prefix="mn-direct-libero-") as setup:
        paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                 "init_states": str(package / "init_files"), "datasets": str(package.parent.parent / "datasets"),
                 "assets": str(package / "assets")}
        (Path(setup) / "config.yaml").write_text(yaml.safe_dump(paths))
        os.environ.update(LIBERO_CONFIG_PATH=setup, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        sys.path.insert(0, options["repo"])
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        seed = config.episodes[0].seed
        np.random.seed(seed)
        task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
        states = torch.load(package / "init_files" / task.problem_folder / task.init_states_file, weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=str(package / "bddl_files" / task.problem_folder / task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        settle = 30 if options["protocol"] == "spatial-30settle" else 10
        try:
            env.seed(seed)
            env.reset()
            raw = env.set_init_state(states[0])
            for _ in range(settle):
                raw, _, _, _ = env.step([0.] * 6 + [-1.])
            rows, steps, done = [], 0, False
            for observation, action, reply in trace(episode_dir / "trace.internal.jsonl"):
                if action is not None:
                    if done:
                        break
                    row = np.asarray(action["commands"][0]["parameters"]["osc_action7"], dtype=action["native_dtype"])
                    raw, _, done, _ = env.step(row)
                    steps += 1
                state_dtype = np.float32 if options["protocol"] == "spatial-30settle" else np.float64
                values = {"eef_position": np.asarray(raw["robot0_eef_pos"], state_dtype),
                          "eef_quaternion": np.asarray(raw["robot0_eef_quat"], state_dtype),
                          "gripper_positions": np.asarray(raw["robot0_gripper_qpos"], state_dtype)}
                images = {key: raw[key] for key in ("agentview_image", "robot0_eye_in_hand_image")}
                fields = snapshot(observation, values, images, steps * .05)
                fields["env_step"] = field(observation["env_step"], steps)
                fields["sim_time_s"] = {"exact_equal": math.isclose(observation["sim_time_s"], steps * .05, abs_tol=1e-9)}
                if reply is not None:
                    fields["official_terminated"] = field(reply["terminated"], bool(done))
                    fields["consumed_env_steps"] = field(reply["execution_feedback"]["consumed_env_steps"], 1)
                rows.append({"env_step": steps, "fields": fields})
            return summarize(rows, steps, result, {"success": bool(done)}, native_settle_steps=settle)
        finally:
            env.close()


def replay_vlabench(config, episode_dir, result, output):
    options = config.benchmark.options
    package = Path(options["repo"]) / "VLABench"
    os.environ.update(VLABENCH_ROOT=str(package), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    sys.path.insert(0, options["repo"])
    import VLABench.robots
    import VLABench.tasks
    from VLABench.envs import load_env
    from VLABench.utils.utils import euler_to_quaternion
    from scipy.spatial.transform import Rotation
    frozen = json.loads((package / "configs/evaluation/tracks/track_1_in_distribution.json").read_text())["select_fruit"][0]
    env = load_env("select_fruit", episode_config=copy.deepcopy(frozen), random_init=False, eval=False, run_mode="eval")
    try:
        env.reset()
        raw = env.get_observation(require_pcd=False)
        rows, steps, elapsed, done = [], 0, 0., False
        for observation, action, reply in trace(episode_dir / "trace.internal.jsonl"):
            if action is not None:
                if done:
                    break
                commands = {cmd["kind"]: cmd for cmd in action["commands"]}
                pose, grip = commands["pose_target"], commands["gripper_command"]
                if pose["pose"]["frame_id"] != "world" or grip["mode"] != "width_m":
                    raise ValueError("unsupported_native_replay_semantics")
                euler = action["source_euler_xyz_rad"].get(pose["target_id"])
                if euler is None:
                    euler = Rotation.from_quat(pose["pose"]["quaternion_xyzw"]).as_euler("xyz")
                _, joints = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=np.asarray(pose["pose"]["position_m"]),
                                                         quat=euler_to_quaternion(*euler))
                before = float(env.physics.time())
                native = env.step(np.concatenate([joints, [grip["value"] / 2., grip["value"] / 2.]]))
                elapsed += float(env.physics.time()) - before
                steps += 1
                done = bool(native.last())
                raw = env.get_observation(require_pcd=False)
            ee = np.asarray(raw["ee_state"], np.float64)
            values = {"eef_position": ee[:3], "eef_quaternion": ee[[4, 5, 6, 3]],
                      "gripper_open_fraction": [1. - ee[-1]], "robot_base_position": env.get_robot_frame_position()}
            images = {name: raw["rgb"][index] for name, index in (("forward_rgb", 2), ("franka_wrist_rgb", 3), ("right_rgb", 0))}
            fields = snapshot(observation, values, images)
            fields["env_step"] = field(observation["env_step"], steps)
            fields["sim_time_s"] = {"exact_equal": math.isclose(elapsed, observation["sim_time_s"], rel_tol=0, abs_tol=1e-8)}
            if reply is not None:
                fields["official_terminated"] = field(reply["terminated"], done)
                fields["consumed_env_steps"] = field(reply["execution_feedback"]["consumed_env_steps"], 1)
            rows.append({"env_step": steps, "fields": fields})
            if len(rows) % 25 == 0:
                print(f"direct native replay: {len(rows)} snapshots", flush=True)
        metrics = {"success": done, "progress_score": float(env.get_task_progress())}
        try:
            intention = float(env.get_intention_score(threshold=.1))
            metrics["intention_score"] = intention if math.isfinite(intention) else None
        except KeyError:
            metrics["intention_score"] = None
        return summarize(rows, steps, result, metrics, elapsed_sim_time_s=elapsed,
                         instruction_sampling_reproduced=False,
                         instruction_note="Recorded public text is fixed; independent Python instruction RNG is not replayed. No model is called.")
    finally:
        env.close()


def robotwin_action_row(action):
    """Independent saved-evidence reader; never call the production packer.

    Preserve the historical native/joint routes and recognize the new physical
    dual-pose route explicitly. Unknown or incomplete commands must not fall
    through to a joint-array interpretation.
    """
    commands = {cmd["target_id"].split("/", 1)[1]: cmd for cmd in action["commands"]}
    if len(commands) != len(action["commands"]):
        raise ValueError("duplicate_recorded_target")
    if action["native_protocol"] == "dual-eef16-legacy-quaternion@1":
        parameters = action["commands"][0]["parameters"]
        row = np.asarray(parameters["native_row"], dtype=parameters["native_dtype"])
        mode, size = "ee", 16
    elif any(cmd["kind"] == "pose_target" for cmd in commands.values()):
        from compare_native_robotwin_closedloop import recorded_dual_pose_row
        return recorded_dual_pose_row(action), "ee"
    else:
        expected = {side + f"_j{i}" for side in ("left", "right") for i in range(6)} | {"left_grip", "right_grip"}
        if set(commands) != expected or any(commands[side + f"_j{i}"]["kind"] != "joint_target"
                for side in ("left", "right") for i in range(6)):
            raise ValueError("complete_recorded_joint14_required")
        row = np.asarray([v for side in ("left", "right") for v in
            [*[commands[side + f"_j{i}"]["value"] for i in range(6)],
             commands[side + "_grip"]["parameters"]["requested_open_fraction"]
             if commands[side + "_grip"]["kind"] == "controller_command" else commands[side + "_grip"]["value"]]],
            dtype=action["native_dtype"])
        mode, size = "qpos", 14
    if row.shape != (size,) or not np.isfinite(row).all():
        raise ValueError("finite_recorded_native_row_required")
    return row, mode


def replay_robotwin(config, episode_dir, result, output, observer=None):
    import torch
    import yaml
    from replay_robotwin_native_actions import observe_physx, seed_native
    from embodied_eval.adapters.native_robotwin_runtime import prewarm_native_cuda
    prewarm_native_cuda()  # CUDA initialization utility only, not execution/counter.
    options = config.benchmark.options
    source = Path(options["repo"])
    task = json.loads((episode_dir / "task.internal.json").read_text())
    seed = task["evaluation_private"]["selected_native_seed"]
    runtime = output / "native-runtime"
    runtime.mkdir()
    for name in ("envs", "assets", "task_config", "description"):
        (runtime / name).symlink_to(source / name, target_is_directory=True)
    old_cwd = Path.cwd()
    os.chdir(runtime)
    env = None
    try:
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(source / "description/utils"))
        from generate_episode_instructions import generate_episode_descriptions
        spec = importlib.util.spec_from_file_location("mn_native_replay_env", source / "script/eval_policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = yaml.safe_load((source / "task_config/demo_clean.yml").read_text())
        args.update(task_name="adjust_bottle", task_config="demo_clean", policy_name="embodied_eval_native",
                    ckpt_setting=options["protocol"], eval_mode=True, eval_video_log=False, collect_data=False)
        robots = yaml.safe_load((source / "task_config/_embodiment_config.yml").read_text())
        robot_file = robots["aloha-agilex"]["file_path"]
        args.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
                    left_embodiment_config=module.get_embodiment_config(robot_file),
                    right_embodiment_config=module.get_embodiment_config(robot_file))
        camera = module.get_camera_config(args["camera"]["head_camera_type"])
        args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
        env = module.class_decorator("adjust_bottle")
        history, info = [], None
        for attempt in task["evaluation_private"]["seed_attempts"]:
            seed_native(attempt["seed"])
            error_type, expert = None, None
            try:
                env.setup_demo(now_ep_num=0, seed=attempt["seed"], is_test=True, **copy.deepcopy(args))
                expert = env.play_once()
                env.close_env()
                valid = bool(env.plan_success and env.check_success())
            except Exception as error:
                error_type, valid = type(error).__name__, False
                env.close_env()
            history.append({"seed": attempt["seed"], "valid": valid, "error_type": error_type})
            if valid != attempt["valid"] or error_type != attempt.get("error_type"):
                raise ValueError("recorded_seed_search_differs")
            if valid:
                info = expert
        if info is None:
            raise ValueError("no_reproducible_expert_seed")
        seed_native(seed)
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(args))
        # The sealed protocol starts a distinct seed domain for text selection;
        # preserve the simulation RNG after sampling, as the production port does.
        outside = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                   [state.clone() for state in torch.cuda.get_rng_state_all()])
        if options["protocol"] == "sealed-instruction":
            seed_native(seed)
        descriptions = generate_episode_descriptions("adjust_bottle", [info["info"]], 100)
        instruction = str(np.random.choice(descriptions[0][options["instruction_type"]]))
        if options["protocol"] == "sealed-instruction":
            random.setstate(outside[0]); np.random.set_state(outside[1]); torch.set_rng_state(outside[2])
            torch.cuda.set_rng_state_all(outside[3])
        if instruction != task["public"]["instruction"]:
            raise ValueError("native_instruction_generation_differs")
        env.set_instruction(instruction=instruction)
        raw = copy.deepcopy(env.get_obs())
        period = float(env.scene.get_timestep())
        rows, steps, acquired = [], 0, 0.
        with observe_physx(env.scene.physx_system) as counter, (observer.watch(env) if observer else nullcontext()):
            for observation, action, reply in trace(episode_dir / "trace.internal.jsonl"):
                fields = {}
                if action is not None:
                    if env.eval_success:
                        break
                    row, mode = robotwin_action_row(action)
                    if observer:
                        observer.begin_action(steps + 1)
                    before = counter["completed"]
                    env.take_action(row, action_type=mode)
                    steps += 1
                    delta = counter["completed"] - before
                    feedback = reply["execution_feedback"]
                    fields["completed_physx_steps"] = field(delta, feedback["consumed_physics_steps"])
                    fields["native_action_counter"] = field(env.take_action_cnt, steps)
                    fields["elapsed_sim_time_s"] = {"exact_equal": math.isclose(delta * period, feedback["elapsed_sim_time_s"], abs_tol=1e-12)}
                    fields["official_terminated"] = field(bool(env.eval_success), reply["terminated"])
                    final = env.eval_success or env.take_action_cnt >= env.step_lim
                    if action["acquire_observation_after"] and not final and steps < task["budgets"]["env_steps"]:
                        raw = copy.deepcopy(env.get_obs())
                        acquired = counter["completed"] * period
                values = {"joint_drive_targets": raw["joint_action"]["vector"]}
                for side in ("left", "right"):
                    ee = np.asarray(raw["endpose"][side + "_endpose"], np.float64)
                    values.update({side + "_eef_position": ee[:3], side + "_eef_quaternion": ee[[4, 5, 6, 3]],
                                   side + "_gripper_open_fraction": raw["endpose"][side + "_gripper"]})
                    # Newer public recordings include actual arm qpos. Query
                    # the same native accessor only when that field exists;
                    # do not change old traces or substitute drive targets.
                    key = side + "_joint_positions"
                    if key in observation["states"]:
                        values[key] = np.asarray(getattr(env.robot, "get_" + side + "_arm_real_jointState")(), np.float64)[:6]
                images = {name: raw["observation"][name]["rgb"] for name in ("head_camera", "left_camera", "right_camera")}
                measured_times = {side + "_joint_positions": counter["completed"] * period for side in ("left", "right")
                                  if side + "_joint_positions" in observation["states"]}
                fields.update(snapshot(observation, values, images, acquired, measured_times))
                fields["cumulative_sim_time_s"] = {"exact_equal": math.isclose(counter["completed"] * period, observation["sim_time_s"], abs_tol=1e-12)}
                rows.append({"env_step": steps, "fields": fields})
                if len(rows) % 25 == 0:
                    print(f"direct native RoboTwin replay: {len(rows)} snapshots", flush=True)
            report = summarize(rows, steps, result, {"success": bool(env.eval_success)},
                               native_physx_steps=counter["completed"], framework_physics_steps=result["execution_statistics"]["physics_steps"],
                               physics_counter_backend=type(env.scene.physx_system).__name__,
                               instruction_generation_exact=True, replayed_seed_attempts=history)
            report["passed"] &= counter["completed"] == result["execution_statistics"]["physics_steps"]
            return report
    finally:
        if env is not None:
            env.close_env()
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, target = args.framework.resolve(), args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    report = {"kind": "mn_independent_native_recorded_action_replay_not_fresh_model",
              "contract_version": "0.2", "synthetic": False, "fresh_model_inference": False,
              "formal_acceptance": False, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False}
    try:
        from embodied_eval.codec import decode
        from embodied_eval.mn.config import RunConfig, validate_config
        from embodied_eval.mn.registry import default_registry
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["synthetic"] or manifest["config"]["contract_version"] != "0.2":
            raise ValueError("real_0.2_manifest_required")
        config = validate_config(decode(RunConfig, manifest["config"]))
        registry = default_registry()
        registry.model(config.model.adapter).verify_assets(config.model.options)
        registry.benchmark(config.benchmark.adapter).verify_assets(config.benchmark.options)
        episode_dir = source / "episodes" / config.episodes[0].episode_id
        result = json.loads((episode_dir / "result.json").read_text())
        if result["run_status"] != "completed":
            raise ValueError("completed_source_run_required")
        handler = {"libero": replay_libero, "vlabench": replay_vlabench, "robotwin": replay_robotwin}[config.benchmark.adapter]
        report.update(handler(config, episode_dir, result, target))
        report.update(benchmark=config.benchmark.adapter, protocol=config.benchmark.options["protocol"],
                      manifest_sha256=digest(manifest_path), trace_sha256=digest(episode_dir / "trace.internal.jsonl"),
                      result_sha256=digest(episode_dir / "result.json"))
    except Exception as error:
        report.update(error_type=type(error).__name__, error_code=getattr(error, "code", None),
                      error_message=str(error)[:240] if isinstance(error, ValueError) else None)
        import traceback
        traceback.print_exc()
    (target / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
