"""Direct native RoboTwin checks with an independent PhysX-level counter.

No ModelAdapter or BenchmarkAdapter reset/advance/score is called. Observe the
completed native physics-system step, rather than the framework's scene proxy.
Default mode replays recorded actions with the native chunk camera cadence.
Optional fresh LingBot mode independently calls the official native client;
its receipt kind and fresh_model_inference flag distinguish the two checks.
"""
import argparse
from contextlib import contextmanager
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

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIGS = {"lingbot_robotwin": "stage4-lingbot-robotwin-adjust-bottle.json",
           "openwam_robotwin": "stage4-openwam-robotwin-adjust-bottle.json"}
CAMERAS = ("head_camera", "left_camera", "right_camera")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


@contextmanager
def observe_physx(system):
    """Only this process/class binding is changed; call original native step."""
    system_class = type(system)
    original = system_class.step
    counter = {"completed": 0}
    def observed(current, *args, **kwargs):
        result = original(current, *args, **kwargs)
        if current is system:
            counter["completed"] += 1
        return result
    system_class.step = observed
    try:
        yield counter
    finally:
        system_class.step = original


def seed_native(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compare(recorded, native):
    ext = recorded["extensions"]["robotwin.native_proprio.v1"]
    fields = {}
    values = {"joint_drive_targets_and_grippers": native["joint_action"]["vector"]}
    for side in ("left", "right"):
        values[side + "_endpose_world_wxyz"] = native["endpose"][side + "_endpose"]
        values[side + "_gripper_open_fraction"] = native["endpose"][side + "_gripper"]
    for key, value in values.items():
        # Native grippers may be scalar or a one-element array; the reviewed
        # public mapping takes that one value, without changing its precision.
        left, right = np.asarray(ext[key]).reshape(-1), np.asarray(value).reshape(-1)
        fields[key] = {"exact_equal": np.array_equal(left, right),
                       "max_abs_diff": float(np.max(np.abs(left - right)))}
    for camera in CAMERAS:
        left = np.asarray(recorded["sensors"][camera]["payload"]["data"], dtype=np.uint8)
        right = native["observation"][camera]["rgb"]
        fields[camera] = {"exact_equal": np.array_equal(left, right),
            "different_scalar_values": int(np.count_nonzero(left != right)),
            "native_pixel_sha256": hashlib.sha256(right.tobytes()).hexdigest()}
    return fields


def native_trace(path):
    pending = None
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "prediction":
                pending = event["data"]["decision"]["commands"][0]["parameters"]
            elif event["event"] == "reset":
                yield event["data"], None, None
            elif event["event"] == "step_result":
                yield event["data"]["observation"], pending, event["data"]["execution_feedback"]
                pending = None


def fresh_lingbot_run(args, config, env, raw, instruction, period, episode_dir, manifest_path, task_path, seed):
    """Native client and full returned chunks, not either framework adapter."""
    from collections import deque
    sys.path.insert(0, config.model.lingbot_repo)
    from deploy.websocket_client_policy import WebsocketClientPolicy
    trace_path = episode_dir / "trace.internal.jsonl"
    result_path = episode_dir / "result.json"
    result = json.loads(result_path.read_text())
    source = iter(native_trace(trace_path))
    first, _, _ = next(source)
    rows = [{"env_step": 0, "fields": compare(first, raw), "action_exact": True}]
    client = WebsocketClientPolicy("127.0.0.1", config.model.port)
    requests = steps = 0
    input_check = None
    acquired_at = 0.
    cache = deque()
    try:
        client.infer({"reset": True, "robo_name": "robotwin", "path_to_pi_model": None})
        with observe_physx(env.scene.physx_system) as counter:
            while steps < 400 and not env.eval_success:
                if not cache:
                    payload = {
                        "observation.images.cam_high": raw["observation"]["head_camera"]["rgb"],
                        "observation.images.cam_left_wrist": raw["observation"]["left_camera"]["rgb"],
                        "observation.images.cam_right_wrist": raw["observation"]["right_camera"]["rgb"],
                        "observation.state": raw["joint_action"]["vector"], "task": env.get_instruction()}
                    if requests == 0:
                        # Diagnostic comparison only: do not use the Adapter's
                        # payload or prediction to control this native rollout.
                        from embodied_eval.core import types as c
                        from embodied_eval.codec import decode
                        from embodied_eval.adapters.robotwin_model import RobotwinModel
                        from deploy import msgpack_numpy
                        with (episode_dir / "model_input.public.jsonl").open() as inputs:
                            public = decode(c.ModelInput, json.loads(inputs.readline()))
                        adapter = RobotwinModel(config.model)
                        mapped = adapter.bridge.native_input(adapter.native_input(public), instruction)
                        native_wire = msgpack_numpy.unpackb(client._packer.pack(payload))
                        adapter_wire = msgpack_numpy.unpackb(msgpack_numpy.packb(mapped))
                        checks = {}
                        for key, value in native_wire.items():
                            other = adapter_wire[key]
                            if isinstance(value, str):
                                checks[key] = {"exact_equal": value == other}
                            else:
                                checks[key] = {"exact_equal": bool(np.array_equal(value, other)),
                                    "dtype_equal": value.dtype == other.dtype, "native_dtype": str(value.dtype),
                                    "adapter_dtype": str(other.dtype), "shape": list(value.shape),
                                    "native_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                        input_check = {"passed": all(v["exact_equal"] and v.get("dtype_equal", True)
                                                     for v in checks.values()), "fields": checks}
                    actions = np.asarray(client.infer(payload)["action"])
                    requests += 1
                    if actions.ndim == 1:
                        actions = actions[None, :]
                    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 14 or not np.isfinite(actions).all():
                        raise ValueError("invalid_fresh_native_joint14_chunk")
                    cache.extend(actions)
                action = cache.popleft()
                before = counter["completed"]
                env.take_action(action, action_type="qpos")
                steps += 1
                delta = counter["completed"] - before
                final = env.eval_success or env.take_action_cnt >= env.step_lim or steps >= 400
                if not cache and not final:
                    raw = copy.deepcopy(env.get_obs())
                    acquired_at = counter["completed"] * period
                recorded = next(source, None)
                fields = compare(recorded[0], raw) if recorded else {}
                if recorded:
                    fields["native_physx_delta"] = {"exact_equal": delta == recorded[2]["consumed_physics_steps"]}
                    fields["captured_at_sim_time_s"] = {"exact_equal": acquired_at ==
                        recorded[0]["extensions"]["robotwin.native_proprio.v1"]["captured_at_sim_time_s"]}
                    fields["cumulative_sim_time_s"] = {"exact_equal": counter["completed"] * period == recorded[0]["sim_time_s"]}
                same_action = bool(recorded and np.array_equal(action, np.asarray(
                    recorded[1]["native_row"], dtype=recorded[1]["native_dtype"])))
                rows.append({"env_step": steps, "action_exact": same_action, "native_action": action.tolist(),
                             "recorded_snapshot_available": recorded is not None, "fields": fields})
                if steps % 25 == 0:
                    print(f"fresh native LingBot policy compared {steps} actions", flush=True)
            def same(row):
                return row["action_exact"] and all(v["exact_equal"] for v in row["fields"].values())
            passed = (steps == result["execution_statistics"]["env_steps"] and next(source, None) is None
                and counter["completed"] == result["execution_statistics"]["physics_steps"]
                and bool(env.eval_success) == result["metrics"]["success"]["value"] and all(same(row) for row in rows))
            return {"kind": "fresh_independent_native_policy_environment_rollout_comparison",
                "pair": "lingbot_robotwin", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "passed": passed, "synthetic": False, "fresh_model_inference": True,
                "recorded_actions_used_to_control_native_environment": False, "formal_acceptance": False,
                "protocol_version": "lingbot-seeded-instruction-port-1", "native_seed": seed,
                "native_policy_actions": steps, "native_policy_requests": requests,
                "native_physx_steps": counter["completed"], "native_sim_time_s": counter["completed"] * period,
                "compared_snapshots": len(rows), "native_success": bool(env.eval_success),
                "first_divergence": next((r["env_step"] for r in rows if not same(r)), None),
                "native_client": "deploy.websocket_client_policy.WebsocketClientPolicy",
                "source_native_model_seed": 42, "instruction_generation_exact": instruction == json.loads(task_path.read_text())["public"]["instruction"],
                "first_native_input_mapping_and_codec": input_check,
                "manifest_sha256": digest(manifest_path), "task_sha256": digest(task_path),
                "trace_sha256": digest(trace_path), "result_sha256": digest(result_path), "rows": rows}
    finally:
        client._ws.close()


def run(args):
    from embodied_eval.config import load_config
    from embodied_eval.codec import to_data
    from embodied_eval.robotwin_pairs import verify
    config = load_config(ROOT / "examples" / CONFIGS[args.pair])
    verify(config)
    manifest_path = args.framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["synthetic"] or manifest["config"] != to_data(config):
        raise ValueError("expected_real_fixed_robotwin_configuration")
    episode_dir = args.framework / "episodes" / config.episodes[0].episode_id
    result_path = episode_dir / "result.json"
    result = json.loads(result_path.read_text())
    if result["run_status"] != "completed":
        raise ValueError("completed_source_trial_required")
    task_path = episode_dir / "task.internal.json"
    task = json.loads(task_path.read_text())
    seed = task["evaluation_private"]["selected_native_seed"]
    source = Path(config.benchmark.robotwin_repo)
    runtime = args.output / "native-runtime"
    runtime.mkdir(exist_ok=False)
    for name in ("envs", "assets", "task_config", "description"):
        (runtime / name).symlink_to(source / name, target_is_directory=True)
    old_cwd = Path.cwd()
    os.chdir(runtime)
    env = None
    try:
        import torch
        torch.empty(1, device="cuda")
        if args.pair == "openwam_robotwin":
            sys.path.insert(0, config.model.openwam_repo)
            from benchmarks.robotwin.eval_policy_wrapper import _prewarm_cuda_for_curobo, _patch_warp_torch_namespace
            _prewarm_cuda_for_curobo()
            _patch_warp_torch_namespace()
        else:
            from embodied_eval.adapters.native_robotwin_runtime import prewarm_native_cuda
            prewarm_native_cuda()
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(source / "description/utils"))
        import envs
        from generate_episode_instructions import generate_episode_descriptions
        module_spec = importlib.util.spec_from_file_location("native_robotwin_replay_eval", source / "script/eval_policy.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        native_args = yaml.safe_load((source / "task_config/demo_clean.yml").read_text())
        native_args.update(task_name="adjust_bottle", task_config="demo_clean", policy_name="embodied_eval_native",
            ckpt_setting=args.pair, eval_mode=True, eval_video_log=False, collect_data=False)
        embodiments = yaml.safe_load((source / "task_config/_embodiment_config.yml").read_text())
        robot_file = embodiments["aloha-agilex"]["file_path"]
        native_args.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
            left_embodiment_config=module.get_embodiment_config(robot_file),
            right_embodiment_config=module.get_embodiment_config(robot_file))
        camera = module.get_camera_config(native_args["camera"]["head_camera_type"])
        native_args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
        env = module.class_decorator("adjust_bottle")
        # Recreate the recorded rejected candidates too: native planner caches
        # and other process-local state need not be covered by RNG seeding.
        # Do not silently force a rejected candidate or pick a different seed.
        history = task["evaluation_private"]["seed_attempts"] if args.reproduce_seed_search else [{"seed": seed, "valid": True}]
        replayed_seed_attempts = []
        info = None
        for attempt in history:
            seed_native(attempt["seed"])
            expert_info = None
            error_type = None
            try:
                env.setup_demo(now_ep_num=0, seed=attempt["seed"], is_test=True, **copy.deepcopy(native_args))
                expert_info = env.play_once()
                env.close_env()
                valid = bool(env.plan_success and env.check_success())
            except Exception as error:
                error_type = type(error).__name__
                env.close_env()
                valid = False
            replayed_seed_attempts.append({"seed": attempt["seed"], "valid": valid, "error_type": error_type})
            if valid != attempt["valid"] or error_type != attempt.get("error_type"):
                raise ValueError("recorded_native_seed_search_differs")
            if valid:
                info = expert_info
        if info is None or history[-1]["seed"] != seed:
            raise ValueError("recorded_expert_valid_seed_not_reproducible")
        seed_native(seed)
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(native_args))
        if args.pair == "openwam_robotwin":
            from benchmarks.robotwin.eval_manifest import choose_instructions
            instruction = choose_instructions(generate_episode_descriptions, task="adjust_bottle",
                candidate_seed=seed, episode_info=info["info"])["unseen"]
        else:
            descriptions = generate_episode_descriptions("adjust_bottle", [info["info"]], 100)
            instruction = str(np.random.choice(descriptions[0]["unseen"]))
        if instruction != task["public"]["instruction"]:
            raise ValueError("native_instruction_generation_differs")
        env.set_instruction(instruction=instruction)
        raw = copy.deepcopy(env.get_obs())
        period = float(env.scene.get_timestep())
        if args.fresh_native_lingbot:
            receipt = fresh_lingbot_run(args, config, env, raw, instruction, period, episode_dir,
                                       manifest_path, task_path, seed)
            receipt.update(recorded_seed_search_reproduced=args.reproduce_seed_search,
                           replayed_seed_attempts=replayed_seed_attempts)
            return receipt
        steps = 0
        acquired_at = 0.
        pending = None
        rows = []
        trace_path = episode_dir / "trace.internal.jsonl"
        action_type = "qpos" if args.pair == "lingbot_robotwin" else "ee"
        meter = None
        meter_checks = []
        if args.meter_cross_check:
            from embodied_eval.core import types as c
            from embodied_eval.adapters.native_robotwin_executor import MeasuredRobotwinExecutor
            meter = MeasuredRobotwinExecutor(env, benchmark_revision=manifest["space"]["revision"], action_type=action_type)
        with observe_physx(env.scene.physx_system) as counter, trace_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "prediction":
                    if pending is not None:
                        raise ValueError("overlapping_native_replay_commands")
                    pending = event["data"]["decision"]["commands"][0]["parameters"]
                elif event["event"] in ("reset", "step_result"):
                    fields = {}
                    if event["event"] == "step_result":
                        if pending is None:
                            raise ValueError("invalid_native_replay_sequence")
                        if env.eval_success:
                            # Keep the actual first divergence and partial PhysX
                            # counter instead of discarding them in a generic error.
                            break
                        before = counter["completed"]
                        row = np.asarray(pending["native_row"], dtype=pending["native_dtype"])
                        measured = (meter.execute(row, c.Budgets(env_steps=400-steps)) if meter is not None
                                    else None)
                        if meter is None:
                            env.take_action(row, action_type=action_type)
                        steps += 1
                        feedback = event["data"]["execution_feedback"]
                        delta = counter["completed"] - before
                        if measured is not None:
                            meter_checks.append({"env_step": steps, "native_physx_delta": delta,
                                "framework_meter_delta": measured.consumed_physics_steps,
                                "exact_equal": delta == measured.consumed_physics_steps})
                        fields["completed_physx_steps"] = {"exact_equal": delta == feedback["consumed_physics_steps"],
                            "native": delta, "framework": feedback["consumed_physics_steps"]}
                        fields["native_action_counter"] = {"exact_equal": env.take_action_cnt == steps,
                            "native": env.take_action_cnt, "framework": steps}
                        fields["elapsed_sim_time_s"] = {"exact_equal": math.isclose(delta * period,
                            feedback["elapsed_sim_time_s"], rel_tol=0, abs_tol=1e-12)}
                        fields["official_terminated"] = {"exact_equal": bool(env.eval_success) == event["data"]["terminated"]}
                        final = env.eval_success or env.take_action_cnt >= env.step_lim
                        if pending["acquire_observation_after"] and not final and steps < task["budgets"]["env_steps"]:
                            raw = copy.deepcopy(env.get_obs())
                            acquired_at = counter["completed"] * period
                        pending = None
                    observation = event["data"] if event["event"] == "reset" else event["data"]["observation"]
                    fields.update(compare(observation, raw))
                    fields["captured_at_sim_time_s"] = {"exact_equal": acquired_at ==
                        observation["extensions"]["robotwin.native_proprio.v1"]["captured_at_sim_time_s"]}
                    fields["cumulative_sim_time_s"] = {"exact_equal": math.isclose(
                        counter["completed"] * period, observation["sim_time_s"], rel_tol=0, abs_tol=1e-12)}
                    rows.append({"env_step": steps, "fields": fields})
                    if len(rows) % 25 == 0:
                        print(f"native replay compared {len(rows)} snapshots", flush=True)
        statistics = result["execution_statistics"]
        first_divergence = next((r["env_step"] for r in rows
            if not all(v["exact_equal"] for v in r["fields"].values())), None)
        passed = (pending is None and steps == statistics["env_steps"] and len(rows) == steps + 1
            and counter["completed"] == statistics["physics_steps"] and first_divergence is None
            and bool(env.eval_success) == result["metrics"]["success"]["value"])
        cross_check_passed = (meter is not None and steps > 0 and len(meter_checks) == steps
            and all(row["exact_equal"] for row in meter_checks)
            and meter.physics_steps == counter["completed"])
        return {"kind": "independent_native_robotwin_recorded_action_replay_not_fresh_inference",
            "passed": passed, "pair": args.pair, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic": False, "formal_acceptance": False, "model_called": False,
            "native_seed": seed, "instruction_generation_exact": True,
            "recorded_seed_search_reproduced": args.reproduce_seed_search,
            "replayed_seed_attempts": replayed_seed_attempts,
            "replayed_actions": steps, "compared_snapshots": len(rows), "first_divergence": first_divergence,
            "physics_counter_backend": type(env.scene.physx_system).__name__,
            "framework_physics_steps": statistics["physics_steps"], "native_physx_steps": counter["completed"],
            "native_sim_time_s": counter["completed"] * period, "native_success": bool(env.eval_success),
            "native_terminated_before_source_action": pending is not None,
            "framework_execution_meter_used_this_check": meter is not None,
            "simultaneous_meter_cross_check_passed": cross_check_passed if meter is not None else None,
            "framework_meter_physics_steps_this_check": meter.physics_steps if meter is not None else None,
            "meter_cross_check_rows": meter_checks,
            "manifest_sha256": digest(manifest_path), "task_sha256": digest(task_path),
            "trace_sha256": digest(trace_path), "result_sha256": digest(result_path),
            "observer_is_process_local_calls_original_method_and_restored": True, "rows": rows}
    finally:
        try:
            if env is not None:
                env.close_env()
        finally:
            os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meter-cross-check", action="store_true",
                        help="Also compare production scene meter with PhysX observer in this same actual execution; separate from source-trial reproduction")
    parser.add_argument("--reproduce-seed-search", action="store_true",
                        help="Recreate every recorded rejected expert candidate before the selected seed; reject any changed native outcome")
    parser.add_argument("--fresh-native-lingbot", action="store_true",
                        help="Independent native environment and official client, freshly infer full joint14 chunks from an externally supervised matching server")
    args = parser.parse_args()
    if args.fresh_native_lingbot and (args.pair != "lingbot_robotwin" or args.meter_cross_check):
        parser.error("fresh native option is only for LingBot and cannot reuse the framework meter")
    args.framework = args.framework.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args)
    except Exception as error:
        chain = []
        while error is not None and len(chain) < 6:
            chain.append({"type": type(error).__name__, "code": getattr(error, "code", None),
                          "missing_module": getattr(error, "name", None)})
            error = error.__cause__ or error.__context__
        report = {"kind": "independent_native_robotwin_action_replay_failed", "passed": False,
                  "checked_at_utc": datetime.now(timezone.utc).isoformat(), "pair": args.pair,
                  "error_chain": chain, "formal_acceptance": False}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "meter_cross_check_rows")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
