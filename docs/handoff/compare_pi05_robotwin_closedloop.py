"""Fresh ARX pi05/RoboTwin native-component reference with passive video.

This is NOT the unmodified RoboTwin pi05 entrypoint: that entrypoint uses a
different model state definition. The reference matches the existing named
ARX/measured-joints protocol, calls pinned pure policy functions, the native
client and native take_action directly. No Adapter/Runner execution is used.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from typing import Any

from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_cosmos_droid_bounded import stop_owned


def recorded_joint_row(action):
    import numpy as np
    commands = {item["target_id"].split("/", 1)[1]: item for item in action["commands"]}
    expected = {side + key for side in ("left", "right") for key in ("_j0", "_j1", "_j2", "_j3", "_j4", "_j5", "_grip")}
    if set(commands) != expected or len(action["commands"]) != 14:
        raise ValueError("complete_unique_dual_arm_commands_required")
    return np.asarray([value for side in ("left", "right") for value in
        [*[commands[side + f"_j{i}"]["value"] for i in range(6)],
        commands[side + "_grip"]["parameters"]["requested_open_fraction"]]], dtype=action["native_dtype"])


def child(args):
    import numpy as np
    import torch
    import yaml
    from comparison_video import Frames
    from check_mn_extension_native_mappings import functions, same
    from replay_mn_native_actions import trace, snapshot, field, digest
    from replay_robotwin_native_actions import observe_physx, seed_native
    from embodied_eval.mn.assets import verify_model, verify_benchmark
    manifest_path = args.framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    if manifest["synthetic"] or config["contract_version"] != "0.2" or config["model"]["adapter"] != "openpi" or config["model"]["options"]["io_profile"] != "arx_joint14" or config["benchmark"]["adapter"] != "robotwin":
        raise ValueError("real_reviewed_pi05_ARX_RoboTwin_trial_required")
    assets = {"model": verify_model("openpi", config["model"]["options"]),
        "benchmark": verify_benchmark("robotwin", config["benchmark"]["options"])}
    if assets != manifest["provenance"]["verified_assets"]:
        raise ValueError("matching_framework_source_and_asset_fingerprints_required")
    episode = args.framework / "episodes" / config["episodes"][0]["episode_id"]
    task = json.loads((episode / "task.internal.json").read_text())
    result = json.loads((episode / "result.json").read_text())
    options = config["benchmark"]["options"]
    if options["protocol"] != "seeded-instruction" or options["task_name"] != "adjust_bottle" or options["task_config"] != "demo_clean" or task["budgets"]["env_steps"] != 400:
        raise ValueError("reviewed_fixed_native_400_action_protocol_required")
    repo = Path(config["model"]["options"]["repo"])
    scope = {"np": np, "Any": Any}
    hashes = functions(repo / "utils/process_data.py", ("_validate_config", "_get_state_keys", "_ensure_valid_state_array", "pack_robot_state", "unpack_robot_state"), scope)
    hashes.update(functions(repo / "policy/Pi_05/model.py", ("encode_obs", "extract_image", "ensure_chw_uint8"), scope))
    codec_root = repo / "policy/Pi_05/openpi/packages/openpi-client/src"
    sys.path.insert(0, str(codec_root))
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi_client import msgpack_numpy
    source = Path(options["repo"])
    native_runtime = args.output / "native-runtime"
    native_runtime.mkdir(exist_ok=False)
    for name in ("envs", "assets", "task_config", "description"):
        (native_runtime / name).symlink_to(source / name, target_is_directory=True)
    old_cwd = Path.cwd()
    os.chdir(native_runtime)
    env = client = frames = None
    report = {"kind": "fresh_same_protocol_native_component_closedloop_not_unmodified_official_entrypoint",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": False,
        "fresh_native_policy_inference": True, "recorded_actions_control_native_environment": False,
        "Adapter_or_Runner_execution_called": False, "official_pi05_state_protocol_substituted": False,
        "reference_protocol": "ARX joint14; measured six joints plus controller gripper target per arm",
        "source_function_sha256": hashes, "verified_assets": assets,
        "framework_manifest_sha256": digest(manifest_path), "framework_result_sha256": digest(episode / "result.json"),
        "framework_trace_sha256": digest(episode / "trace.internal.jsonl"), "formal_acceptance": False,
        "completed": False, "strict_equal": False, "rows": []}
    try:
        # Same dependency bootstrap as the fixed source wrapper: no planning or
        # controller implementation is imported from either Adapter.
        torch.empty(1, device="cuda")
        import warp
        from types import SimpleNamespace
        from curobo.wrap.reacher.motion_gen import MotionGen
        if not hasattr(warp, "torch"):
            names = ("from_torch", "to_torch", "dtype_from_torch", "dtype_to_torch", "device_from_torch", "device_to_torch", "stream_from_torch", "stream_to_torch")
            warp.torch = SimpleNamespace(**{name: getattr(warp, name, None) for name in names})
        sys.path[:0] = [str(source), str(source / "description/utils")]
        from generate_episode_instructions import generate_episode_descriptions
        spec = importlib.util.spec_from_file_location("pi05_rw_reference_environment", source / "script/eval_policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        setup = yaml.safe_load((source / "task_config/demo_clean.yml").read_text())
        setup.update(task_name="adjust_bottle", task_config="demo_clean", policy_name="embodied_eval_native",
            ckpt_setting="lingbot_robotwin", eval_mode=True, eval_video_log=False, collect_data=False)
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
                expert = env.play_once()
                env.close_env()
                valid = bool(env.plan_success and env.check_success())
            except Exception as error:
                error_type, valid = type(error).__name__, False
                env.close_env()
            history.append({"seed": attempt["seed"], "valid": valid, "error_type": error_type})
            if valid != attempt["valid"] or error_type != attempt.get("error_type"):
                raise ValueError("recorded_expert_valid_seed_search_not_reproduced")
            if valid:
                info = expert
        seed = task["evaluation_private"]["selected_native_seed"]
        if info is None or history[-1]["seed"] != seed:
            raise ValueError("recorded_expert_valid_seed_required")
        seed_native(seed)
        env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(setup))
        descriptions = generate_episode_descriptions("adjust_bottle", [info["info"]], 100)
        instruction = str(np.random.choice(descriptions[0][options["instruction_type"]]))
        if instruction != task["public"]["instruction"] or env.step_lim != 400:
            raise ValueError("same_instruction_and_native_limit_required")
        env.set_instruction(instruction=instruction)
        raw = copy.deepcopy(env.get_obs())
        period = float(env.scene.get_timestep())
        client = WebsocketClientPolicy("127.0.0.1", args.port)
        cameras = ("head_camera", "left_camera", "right_camera")
        frames = Frames(args.output / "frames", cameras)
        exchanges = args.output / "native-exchanges"
        exchanges.mkdir()
        cache, requests, steps, acquired = deque(), 0, 0, 0.
        recorded = iter(trace(episode / "trace.internal.jsonl"))
        reference = next(recorded)
        dims = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        def measured():
            return {side: np.asarray(getattr(env.robot, "get_" + side + "_arm_real_jointState")(), np.float64) for side in ("left", "right")}
        def input_payload(joints):
            native = {"vision": {slot: {"color": raw["observation"][name]["rgb"]} for slot, name in
                (("cam_head", "head_camera"), ("cam_left_wrist", "left_camera"), ("cam_right_wrist", "right_camera"))},
                "state": {side + suffix: joints[side][:6] if suffix == "_arm_joint_state" else np.asarray([raw["endpose"][side + "_gripper"]]).reshape(-1)
                    for side in ("left", "right") for suffix in ("_arm_joint_state", "_ee_joint_state")}, "instruction": instruction}
            return scope["encode_obs"](native, "joint", dims)
        def record_snapshot(reference, joints, counter, delta=None, action=None):
            values = {"joint_drive_targets": raw["joint_action"]["vector"]}
            for side in ("left", "right"):
                ee = np.asarray(raw["endpose"][side + "_endpose"], np.float64)
                values.update({side + "_eef_position": ee[:3], side + "_eef_quaternion": ee[[4, 5, 6, 3]],
                    side + "_gripper_open_fraction": np.asarray([raw["endpose"][side + "_gripper"]]).reshape(-1),
                    side + "_joint_positions": joints[side][:6]})
            images = {name: raw["observation"][name]["rgb"] for name in cameras}
            clock = counter["completed"] * period
            frames.append(images, steps, clock, {name: acquired for name in cameras})
            fields = snapshot(reference[0], values, images) if reference else {}
            if reference:
                fields["sim_time_s"] = field(clock, reference[0]["sim_time_s"])
                fields["camera_acquisition_time"] = field(acquired, reference[0]["sensors"]["head_camera"]["captured_at_s"])
                if delta is not None:
                    fields["physics_steps"] = field(delta, reference[2]["execution_feedback"]["consumed_physics_steps"])
                    fields["official_terminated"] = field(bool(env.eval_success), reference[2]["terminated"])
            entry = {"env_step": steps, "native_sim_time_s": clock, "native_physics_delta": delta,
                "recorded_snapshot_available": reference is not None, "fields": fields}
            if action is not None:
                old_action = recorded_joint_row(reference[1]) if reference else None
                entry.update(action_exact=bool(old_action is not None and old_action.dtype == action.dtype and np.array_equal(old_action, action)),
                    action_max_abs_diff=float(np.max(np.abs(old_action - action))) if old_action is not None else None,
                    native_action=action.tolist(), native_dtype=str(action.dtype))
            report["rows"].append(entry)
        with observe_physx(env.scene.physx_system) as counter:
            joints = measured()
            record_snapshot(reference, joints, counter)
            while steps < 400 and not env.eval_success:
                if not cache:
                    payload = input_payload(joints)
                    stem = f"exchange-{requests + 1:04d}"
                    with (exchanges / (stem + "-request.msgpack")).open("xb") as stream:
                        stream.write(msgpack_numpy.packb(payload))
                    response = client.infer(payload)
                    actions = np.asarray(response["actions"])
                    if actions.shape != (50, 14) or not np.isfinite(actions).all():
                        raise ValueError("fixed_native_50_by_14_chunk_required")
                    with (exchanges / (stem + "-actions.msgpack")).open("xb") as stream:
                        stream.write(msgpack_numpy.packb({"actions": actions}))
                    # Execute the pinned native unpacking, then concatenate in
                    # the order native take_action documents; no Model decode.
                    decoded = scope["unpack_robot_state"](actions, "joint", dims, source_type="obs")
                    cache.extend(np.concatenate([item[name] for name in ("left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state", "right_ee_joint_state")]) for item in decoded)
                    requests += 1
                action = cache.popleft()
                before = counter["completed"]
                env.take_action(action, action_type="qpos")
                steps += 1
                if env.take_action_cnt != steps:
                    raise ValueError("native_action_counter_differs")
                final = bool(env.eval_success) or env.take_action_cnt >= env.step_lim or steps >= 400
                if not final:
                    raw = copy.deepcopy(env.get_obs())
                    acquired = counter["completed"] * period
                joints = measured()
                reference = next(recorded, None)
                record_snapshot(reference, joints, counter, counter["completed"] - before, action)
                if steps % 25 == 0:
                    print(f"native reference completed {steps} actions", flush=True)
            remaining_reference = next(recorded, None) is not None
            equal = lambda row: row["recorded_snapshot_available"] and row.get("action_exact", True) and all(value["exact_equal"] for value in row["fields"].values())
            report.update(completed=True, native_policy_actions=steps, native_policy_requests=requests,
                native_physics_steps=counter["completed"], native_sim_time_s=counter["completed"] * period,
                native_success=bool(env.eval_success), native_action_limit_reached=env.take_action_cnt >= env.step_lim,
                framework_actions=result["execution_statistics"]["env_steps"], framework_success=result["metrics"]["success"],
                native_seed=seed, replayed_seed_attempts=history, instruction=instruction,
                first_divergence=next((row["env_step"] for row in report["rows"] if not equal(row)), None),
                initial_observation_exact=equal(report["rows"][0]),
                strict_equal=not remaining_reference and steps == result["execution_statistics"]["env_steps"] and all(equal(row) for row in report["rows"]),
                cache_rows_unexecuted=len(cache), physics_counter_backend=type(env.scene.physx_system).__name__,
                native_client="openpi_client.websocket_client_policy.WebsocketClientPolicy",
                server_identity_verified_by_handshake=False, additional_video_camera_or_physics_calls=0)
            # Native and framework start requests are separately captured. This
            # equality is stronger than matching only a scene seed number.
            old_exchange = args.framework.parent / "native-exchanges/exchange-0001-request.msgpack"
            if old_exchange.is_file():
                old_payload = msgpack_numpy.unpackb(old_exchange.read_bytes())
                native_payload = msgpack_numpy.unpackb((exchanges / "exchange-0001-request.msgpack").read_bytes())
                report["initial_native_model_payload_exact"] = same(old_payload, native_payload)
    except Exception as error:
        report["exception_chain"] = public_exception(error)
    finally:
        if frames is not None and frames.rows:
            frames.close()
        if client is not None:
            client._ws.close()
        if env is not None:
            env.close_env()
        os.chdir(old_cwd)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "native-result.json", report)
    print(json.dumps({key: report.get(key) for key in ("completed", "strict_equal", "native_policy_actions", "native_physics_steps", "native_success", "first_divergence")}), flush=True)
    return 0 if report["completed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8949)
    parser.add_argument("--model-gpu", type=int, default=5)
    parser.add_argument("--simulator-gpu", type=int, default=7)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.framework, args.output = args.framework.resolve(), args.output.resolve()
    args.framework.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    if args.child:
        return child(args)
    if args.model_gpu == args.simulator_gpu or not (0 <= args.model_gpu <= 7 and 0 <= args.simulator_gpu <= 7 and 1 <= args.port <= 65535):
        raise ValueError("distinct_local_gpu_and_local_port_required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.framework / "manifest.json").read_text())
    checkpoint = manifest["config"]["model"]["options"]["checkpoint_dir"]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    model_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.model_gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.6")
    sim_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.simulator_gpu), CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"))
    server_command = [str(WORKSPACE / "upstream/envs/pi05-arx-mn-inference/bin/python"), str(ROOT / "docs/handoff/serve_pi05_arx_native.py"),
        "--checkpoint", checkpoint, "--port", str(args.port)]
    client_command = [str(WORKSPACE / "upstream/envs/robotwin-stage4/bin/python"), __file__, "--child", "--framework", str(args.framework),
        "--output", str(args.output), "--port", str(args.port)]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    receipt = {"kind": "owned_native_closedloop_supervision_not_stage_acceptance", "completed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "model_gpu": args.model_gpu,
        "simulator_gpu": args.simulator_gpu, "port": args.port, "native_raw_logs_or_credentials_saved": False,
        "own_fresh_server_not_reused": True, "global_proxy_driver_or_occupancy_changed": False}
    hardware = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    receipt["current_gpu_hardware"] = hardware.stdout.strip().splitlines()
    server = worker = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        logs = None
        try:
            server = subprocess.Popen(server_command, cwd=ROOT, env=model_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            logs = pool.submit(server.communicate)
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("owned_native_server_start_failed")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("diagnostic_server_start_bound")
                    time.sleep(1)
            worker = subprocess.Popen(client_command, cwd=ROOT, env=sim_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            out, err = worker.communicate(timeout=7800)
            receipt.update(client_exit_code=worker.returncode, client_public_failure_categories=categories((out + err).decode(errors="replace")))
            native = args.output / "native-result.json"
            if native.is_file():
                result = json.loads(native.read_text())
                receipt.update(completed=worker.returncode == 0 and result["completed"], strict_equal=result["strict_equal"],
                    native_success=result.get("native_success"), native_actions=result.get("native_policy_actions"))
        except Exception as error:
            receipt["exception_chain"] = public_exception(error)
        finally:
            stop_owned(worker)
            stop_owned(server)
            if logs is not None:
                out, err = logs.result(timeout=15)
                receipt["server_public_failure_categories"] = categories((out + err).decode(errors="replace"))
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("completed", "strict_equal", "native_success", "native_actions")}), flush=True)
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
