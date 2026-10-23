"""Fresh OpenWAM/LingBot-VLA source-component RoboTwin reference and video.

Native environment setup, policy transport/conversions, action execution and
PhysX counter are independent of Model/Benchmark Adapter execution methods.
"""
import argparse
from collections import deque
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

from probe_mn_components import ROOT, write, public_exception
sys.path.insert(0, str(ROOT))


def recorded_dual_pose_row(action):
    """Independent evidence reader, not the production Benchmark conversion."""
    import numpy as np
    commands = {cmd["target_id"].split("/",1)[1]:cmd for cmd in action["commands"]}
    if len(action["commands"]) != 4 or set(commands) != {"left_eef","right_eef","left_grip","right_grip"}:
        raise ValueError("complete_recorded_dual_pose_required")
    if action["native_protocol"] != "gripper-open-fraction-interpolate-then-clamp@1":
        raise ValueError("declared_recorded_native_gripper_protocol_required")
    values=[]
    for side in ("left","right"):
        pose,grip=commands[side+"_eef"],commands[side+"_grip"]
        if (pose["kind"]!="pose_target" or pose["pose"]["frame_id"]!="world" or grip["kind"]!="controller_command"
                or grip["controller_profile_ref"]!="gripper-open-fraction-raw-request@1"):
            raise ValueError("world_pose_and_native_gripper_request_required")
        q=pose["pose"]["quaternion_xyzw"]
        if len(q) != 4 or len(pose["pose"]["position_m"]) != 3:
            raise ValueError("physical_pose_dimensions_required")
        values.extend((*pose["pose"]["position_m"],q[3],q[0],q[1],q[2],grip["parameters"]["requested_open_fraction"]))
    row=np.asarray(values,dtype=action["native_dtype"])
    if row.shape!=(16,) or not np.isfinite(row).all() or not np.array_equal(np.asarray(values),row):
        raise ValueError("lossless_recorded_native_row_required")
    return row


def run(args):
    import numpy as np
    import torch
    import yaml
    from comparison_video import Frames
    from compare_pi05_robotwin_closedloop import recorded_joint_row
    from replay_mn_native_actions import trace, snapshot, field, digest
    from replay_robotwin_native_actions import observe_physx, seed_native
    from embodied_eval.native.model_assets import verify_model
    from embodied_eval.native.benchmark_assets import verify_benchmark
    manifest = json.loads((args.framework / "manifest.json").read_text())
    config = json.loads(args.config.read_text())
    if manifest["synthetic"] or manifest["config"] != config or config["benchmark"]["adapter"] != "robotwin":
        raise ValueError("matching_real_RoboTwin_configuration_required")
    profile = config["model"]["options"]["io_profile"]
    if profile not in ("joint14", "eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute"):
        raise ValueError("fixed_OpenWAM_or_LingBot_profile_required")
    is_wam = profile in ("eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute")
    physical_profile = profile == "eef20_dual_wxyz_encoded_absolute"
    assets = {"model": verify_model(config["model"]["adapter"], config["model"]["options"]), "benchmark": verify_benchmark("robotwin", config["benchmark"]["options"])}
    if assets != manifest["provenance"]["verified_assets"]:
        raise ValueError("same_component_source_and_assets_required")
    episode = args.framework / "episodes" / config["episodes"][0]["episode_id"]
    task = json.loads((episode / "task.internal.json").read_text())
    result = json.loads((episode / "result.json").read_text())
    options = config["benchmark"]["options"]
    source = Path(options["repo"])
    model_source = Path(config["model"]["options"]["repo"])
    report = {"kind": "fresh_fixed_source_native_component_robotwin_closedloop", "synthetic": False,
        "completed": False, "strict_equal": False, "fresh_native_policy_inference": True,
        "recorded_actions_control_native_environment": False, "Adapter_or_Runner_execution_called": False,
        "verified_assets": assets, "framework_manifest_sha256": digest(args.framework / "manifest.json"),
        "formal_acceptance": False, "rows": [], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework_uses_explicit_weight_encoding_and_public_physical_poses": physical_profile,
        "native_reference_retains_fixed_source_input_decoder_controller": True}
    runtime = args.output / "native-runtime"
    runtime.mkdir()
    for name in ("envs", "assets", "task_config", "description"):
        (runtime / name).symlink_to(source / name, target_is_directory=True)
    old_cwd = Path.cwd()
    os.chdir(runtime)
    env = client = frames = None
    try:
        torch.empty(1, device="cuda")
        import warp
        from types import SimpleNamespace
        from curobo.wrap.reacher.motion_gen import MotionGen
        if not hasattr(warp, "torch"):
            names = ("from_torch", "to_torch", "dtype_from_torch", "dtype_to_torch", "device_from_torch", "device_to_torch", "stream_from_torch", "stream_to_torch")
            warp.torch = SimpleNamespace(**{name: getattr(warp, name, None) for name in names})
        sys.path[:0] = [str(source), str(source / "description/utils"), str(model_source)]
        from generate_episode_instructions import generate_episode_descriptions
        spec = importlib.util.spec_from_file_location("fresh_native_rw_reference", source / "script/eval_policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        setup = yaml.safe_load((source / "task_config/demo_clean.yml").read_text())
        setup.update(task_name="adjust_bottle", task_config="demo_clean", policy_name="embodied_eval_native",
            ckpt_setting="openwam_robotwin" if is_wam else "lingbot_robotwin", eval_mode=True, eval_video_log=False, collect_data=False)
        robots = yaml.safe_load((source / "task_config/_embodiment_config.yml").read_text())
        robot_file = robots["aloha-agilex"]["file_path"]
        setup.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
            left_embodiment_config=module.get_embodiment_config(robot_file), right_embodiment_config=module.get_embodiment_config(robot_file))
        camera = module.get_camera_config(setup["camera"]["head_camera_type"])
        setup.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
        env = module.class_decorator("adjust_bottle")
        history, info = [], None
        for attempt in task["evaluation_private"]["seed_attempts"]:
            seed_native(attempt["seed"])
            error_type, expert = None, None
            try:
                env.setup_demo(now_ep_num=0, seed=attempt["seed"], is_test=True, **copy.deepcopy(setup))
                expert = env.play_once(); env.close_env()
                valid = bool(env.plan_success and env.check_success())
            except Exception as error:
                error_type, valid = type(error).__name__, False
                env.close_env()
            history.append({"seed": attempt["seed"], "valid": valid, "error_type": error_type})
            if valid != attempt["valid"] or error_type != attempt.get("error_type"):
                raise ValueError("native_expert_seed_search_differs")
            if valid:
                info = expert
        seed = task["evaluation_private"]["selected_native_seed"]
        if info is None or history[-1]["seed"] != seed:
            raise ValueError("matching_native_expert_valid_seed_required")
        seed_native(seed)
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(setup))
        if options["protocol"] == "sealed-instruction":
            from benchmarks.robotwin.eval_manifest import choose_instructions
            instruction = choose_instructions(generate_episode_descriptions, task="adjust_bottle", candidate_seed=seed, episode_info=info["info"])[options["instruction_type"]]
        else:
            descriptions = generate_episode_descriptions("adjust_bottle", [info["info"]], 100)
            instruction = str(np.random.choice(descriptions[0][options["instruction_type"]]))
        if instruction != task["public"]["instruction"]:
            raise ValueError("native_public_instruction_differs")
        env.set_instruction(instruction=instruction)
        raw = copy.deepcopy(env.get_obs())
        period = float(env.scene.get_timestep())
        port = config["model"]["options"]["port"]
        if is_wam:
            from benchmarks.utils import WSPolicyClient, build_payload, encode_numpy_b64
            from benchmarks.utils.action_conversion import robotwin_endpose_to_eef20d, eef20d_to_ee16d
            from benchmarks.robotwin.prompt_template import format_prompt_for_inference
            client = WSPolicyClient(f"ws://127.0.0.1:{port}")
            if client.ping().get("type") != "pong" or client.reset().get("type") != "reset_ack":
                raise ValueError("native_OpenWAM_liveness_or_reset_failed")
        else:
            from deploy.websocket_client_policy import WebsocketClientPolicy
            from deploy import msgpack_numpy
            client = WebsocketClientPolicy("127.0.0.1", port)
            client.infer({"reset": True, "robo_name": "robotwin", "path_to_pi_model": None})
        cameras = ("head_camera", "left_camera", "right_camera")
        frames = Frames(args.output / "frames", cameras)
        exchanges = args.output / "native-exchanges"
        exchanges.mkdir()
        old = iter(trace(episode / "trace.internal.jsonl"))
        reference = next(old)
        steps, requests, acquired, cache = 0, 0, 0., deque()
        def save(ref, counter, action=None, delta=None):
            values = {"joint_drive_targets": raw["joint_action"]["vector"]}
            for side in ("left", "right"):
                ee = np.asarray(raw["endpose"][side + "_endpose"], np.float64)
                joints = np.asarray(getattr(env.robot, "get_" + side + "_arm_real_jointState")(), np.float64)
                values.update({side + "_eef_position": ee[:3], side + "_eef_quaternion": ee[[4, 5, 6, 3]],
                    side + "_gripper_open_fraction": np.asarray([raw["endpose"][side + "_gripper"]]).reshape(-1),
                    side + "_joint_positions": joints[:6]})
            images = {name: raw["observation"][name]["rgb"] for name in cameras}
            clock = counter["completed"] * period
            frames.append(images, steps, clock, {name: acquired for name in cameras})
            fields = snapshot(ref[0], values, images) if ref else {}
            if ref:
                fields["sim_time_s"] = field(clock, ref[0]["sim_time_s"])
                if action is not None:
                    old_action = recorded_dual_pose_row(ref[1]) if physical_profile else np.asarray(ref[1]["commands"][0]["parameters"]["native_row"], dtype=ref[1]["native_dtype"]) if is_wam else recorded_joint_row(ref[1])
                    fields["action"] = {**field(old_action, action), "dtype_equal": old_action.dtype == action.dtype}
                    fields["physics_delta"] = field(delta, ref[2]["execution_feedback"]["consumed_physics_steps"])
                    fields["success"] = field(bool(env.eval_success), ref[2]["terminated"])
            report["rows"].append({"env_step": steps, "reference_available": ref is not None, "fields": fields,
                "native_action": action.tolist() if action is not None else None, "native_dtype": str(action.dtype) if action is not None else None,
                "native_sim_time_s": clock, "native_physics_delta": delta})
        with observe_physx(env.scene.physx_system) as counter:
            save(reference, counter)
            cap = min(env.step_lim, task["budgets"]["env_steps"])
            while steps < cap and not env.eval_success:
                if not cache:
                    if is_wam:
                        state = robotwin_endpose_to_eef20d(raw["endpose"]["left_endpose"], raw["endpose"]["right_endpose"],
                            raw["endpose"]["left_gripper"], raw["endpose"]["right_gripper"])
                        payload = build_payload(**{slot: encode_numpy_b64(raw["observation"][name]["rgb"]) for slot, name in
                            (("head", "head_camera"), ("left_wrist", "left_camera"), ("right_wrist", "right_camera"))},
                            prompt=format_prompt_for_inference(instruction), state=np.asarray(state, np.float32).tolist())
                        write(exchanges / f"request-{requests + 1:04d}.json", payload)
                        reply = client.predict_once(payload)
                        native = np.asarray(reply["action"], np.float32)
                        write(exchanges / f"action-{requests + 1:04d}.json", {"action": native.tolist()})
                        cache.append(eef20d_to_ee16d(native))
                    else:
                        payload = {**{key: raw["observation"][name]["rgb"] for key, name in
                            (("observation.images.cam_high", "head_camera"), ("observation.images.cam_left_wrist", "left_camera"), ("observation.images.cam_right_wrist", "right_camera"))},
                            "observation.state": np.asarray(raw["joint_action"]["vector"], np.float64), "task": env.get_instruction()}
                        with (exchanges / f"request-{requests + 1:04d}.msgpack").open("xb") as stream:
                            stream.write(msgpack_numpy.packb(payload))
                        reply = client.infer(payload)
                        actions = np.asarray(reply["action"])
                        if actions.ndim == 1:
                            actions = actions[None, :]
                        if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
                            raise ValueError("finite_native_joint14_chunk_required")
                        with (exchanges / f"action-{requests + 1:04d}.msgpack").open("xb") as stream:
                            stream.write(msgpack_numpy.packb({"action": actions}))
                        cache.extend(actions)
                    requests += 1
                action = cache.popleft()
                before = counter["completed"]
                env.take_action(action, action_type="ee" if is_wam else "qpos")
                steps += 1
                final = bool(env.eval_success) or env.take_action_cnt >= env.step_lim or steps >= cap
                if not cache and not final:
                    raw = copy.deepcopy(env.get_obs())
                    acquired = counter["completed"] * period
                reference = next(old, None)
                save(reference, counter, action, counter["completed"] - before)
                if steps % 25 == 0:
                    print(f"independent native RoboTwin {profile}: {steps} actions", flush=True)
            same = lambda row: row["reference_available"] and all(value["exact_equal"] and value.get("dtype_equal", True) for value in row["fields"].values())
            report.update(completed=True, native_actions=steps, native_requests=requests, native_success=bool(env.eval_success),
                native_physics_steps=counter["completed"], native_sim_time_s=counter["completed"] * period,
                native_seed=seed, replayed_seed_attempts=history, instruction=instruction, action_cap=cap,
                framework_metrics=result["metrics"], initial_observation_exact=same(report["rows"][0]),
                first_divergence=next((row["env_step"] for row in report["rows"] if not same(row)), None),
                strict_equal=steps == result["execution_statistics"]["env_steps"] and next(old, None) is None and all(same(row) for row in report["rows"]),
                additional_video_get_obs_render_or_simulation_calls=0, physics_counter_backend=type(env.scene.physx_system).__name__)
    except Exception as error:
        report["exception_chain"] = public_exception(error)
    finally:
        if frames is not None and frames.rows:
            frames.close()
        if client is not None:
            (client.close if is_wam else client._ws.close)()
        if env is not None:
            env.close_env()
        os.chdir(old_cwd)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "native-result.json", report)
    print(json.dumps({key: report.get(key) for key in ("completed", "strict_equal", "native_actions", "native_success", "initial_observation_exact", "first_divergence")}), flush=True)
    return 0 if report["completed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup-seed", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
