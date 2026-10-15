"""Fresh native-source RoboDojo policy closedloop with consumed-frame videos.

Matching ARX pi05/OpenWAM weights, native factory/control/reward, no production
Adapter/Runner execution. Native model calls and saved-action replay are not
interchanged. Only this diagnostic's children and private pipe are owned.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
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
sys.path.insert(0, str(ROOT))


class NativeCameraBufferError(BaseException):
    """Do not let upstream's broad Exception handler swallow an unsafe buffer."""


def resolve_runtime_paths(args):
    """A child changes cwd: private native runtime paths must be absolute."""
    for name in ("output", "trial", "isaac_libraries", "video_runtime"):
        path = getattr(args, name).resolve()
        path.relative_to(ROOT / "outputs")
        setattr(args, name, path)


def child(args):
    def phase(name):
        with (args.output / "native-lifecycle.jsonl").open("a") as stream:
            stream.write(json.dumps({"phase": name, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    phase("child_entered")
    import numpy as np
    from argparse import Namespace
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.mn.assets_common import _revision, _hf_objects
    from embodied_eval.mn.robodojo_simulator import REVISION, prepare_robot_paths
    from embodied_eval.mn.native_catalog import PROFILES
    from replay_robodojo_native_actions import native_action
    from replay_mn_native_actions import trace, snapshot, field
    manifest = json.loads((args.trial / "framework/manifest.json").read_text())
    config = manifest["config"]
    if manifest["synthetic"] or config["benchmark"]["adapter"] != "robodojo":
        raise ValueError("real_named_RoboDojo_trial_required")
    options = config["benchmark"]["options"]
    if options["protocol"] != "stack-bowls-arx-x5-native-v1":
        raise ValueError("fixed_native_ARX_protocol_required")
    is_wam = config["model"]["adapter"] == "openwam"
    repo, reference = Path(options["repo"]), Path(config["model"]["options"]["repo"])
    _revision(repo, REVISION); _revision(reference, PROFILES["arx_joint14"][2])
    assets = _hf_objects(repo.resolve(), ROOT / "embodied_eval/mn/asset_manifests/robodojo.json")
    model_proof = json.loads((args.output / "model-assets-verified.json").read_text())
    if model_proof["model_config_sha256"] != hashlib.sha256(json.dumps(config["model"], sort_keys=True).encode()).hexdigest():
        raise ValueError("matching_before_native_load_model_verification_required")
    model_assets = model_proof["verification"]
    phase("assets_verified_before_load")
    # Both sources contain a top-level ``utils`` package. Native RoboDojo
    # initialization must resolve its own package first, just like the prior
    # independent native replay. Model helpers below are read by exact path.
    sys.path[:0] = [str(repo), str(reference)]
    # Kit has its own argument parser. The diagnostic's --port/--model-gpu
    # etc. are already parsed and must not become Kit startup arguments.
    # Native application settings continue to come from the fixed Namespace.
    sys.argv = [sys.argv[0]]
    # Same startup flags used by the existing Benchmark worker. They only
    # permit this root-owned Kit process and suppress its unsolicited log file.
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    from isaaclab.app import AppLauncher
    sync = native_module(str(repo), "env.camera_manager.capture.render_sync")
    launch = Namespace(kit_args="", headless=True, enable_cameras=True, device="cuda:0")
    sync.add_zero_delay_kit_args(launch)
    app = AppLauncher(launch, fast_shutdown=False).app
    phase("native_application_started")
    env = client = worker = frames = connection = None
    report = {"kind": "fresh_native_component_RoboDojo_policy_closedloop", "completed": False, "strict_equal": False,
        "synthetic": False, "fresh_native_policy_inference": False, "recorded_actions_control_native_environment": False,
        "production_Model_or_BenchmarkAdapter_instantiated": False, "native_factory_created": False, "close_completed": False,
        "verified_model_assets": model_assets, "verified_before_native_load_environment_assets": assets,
        "physical_render_gpu": args.gpu, "model_gpu": args.model_gpu, "formal_acceptance": False, "rows": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    with ExitStack() as stack:
        pool = None
        logs = None
        try:
            from omegaconf import OmegaConf
            loaders = native_module(str(repo), "utils.load_file")
            pipeline = native_module(str(repo), "utils.pipeline_utils")
            robot_paths, robot_evidence = prepare_robot_paths(repo, args.output)
            pipeline.ROBOTS_PATH = str(robot_paths)
            read = lambda relative: loaders.load_yaml(str(repo / relative))
            eval_cfg = read("env_cfg/arx_x5.yml")
            eval_cfg.update(task_name="stack_bowls", num_envs=1, seed=0, eval_num=1, physx_monitor_enabled=False,
                eval_batch=False, policy_name="native_component_reference", description={})
            cfg = OmegaConf.create({"sim": read("env_cfg/sim/sim_config.yml"), "scene": read("env_cfg/scene/default.yml"),
                "camera": read("env_cfg/camera/camera_config.yml"), "robot": read("env_cfg/robot/dual_x5.yml"),
                "task_env": read("task/RoboDojo/config/stack_bowls.yml"), "eval_cfg": eval_cfg,
                "deploy_cfg": {"policy_name": "native_component_reference", "port": args.port, "host": "127.0.0.1"}})
            cfg = pipeline.process_randomization(cfg); cfg, _ = pipeline.process_config(cfg, task_name="stack_bowls")
            cfg.sim.scene.num_envs = 1; cfg.sim.seed = [0]
            cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
            phase("native_configuration_processed")
            if cfg.sim.get("device", "cpu") != "cpu":
                raise ValueError("fixed_native_CPU_physics_required")
            exchanges = args.output / "native-exchanges"
            exchanges.mkdir()
            cache, requests = deque(), 0
            if is_wam:
                def start_native_model():
                    nonlocal connection, worker, logs
                    from multiprocessing import Pipe
                    connection, inherited = Pipe(duplex=True)
                    worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.model_gpu))
                    worker = subprocess.Popen([str(WORKSPACE / "upstream/envs/openwam_arx-mn-native/bin/python"),
                        str(ROOT / "docs/handoff/native_openwam_arx_worker.py"), "--pipe-fd", str(inherited.fileno()), "--output", str(exchanges)],
                        cwd=ROOT, env=worker_env, pass_fds=(inherited.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    inherited.close(); logs = pool.submit(worker.communicate)
                    connection.send({key: value for key, value in config["model"]["options"].items() if key not in ("runtime_python", "runtime_gpu")})
                    if not connection.poll(900) or not connection.recv().get("ok"):
                        raise RuntimeError("native_source_model_loading_failed")
                    phase("native_model_worker_ready")
                def infer(raw):
                    public = {"vision": {name: {"color": pack(raw["vision"][name]["color"])} for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                        "state": {side + key: pack(raw["state"][side + key]) for side in ("left", "right") for key in ("_ee_pose", "_ee_joint_state")},
                        "instruction": raw["instruction"]}
                    connection.send(("infer", public))
                    if not connection.poll(900):
                        raise TimeoutError("diagnostic_native_generation_bound")
                    reply = connection.recv()
                    if not reply.get("ok"):
                        raise RuntimeError("native_source_model_generation_failed")
                    return [{key: unpack(value) for key, value in row.items()} for row in reply["actions"]]
            # Native factory expects a policy reset endpoint. Its substitute is
            # an actual native-component policy bridge; the closedloop queries
            # above call real inference, never a reset-only mock policy result.
            module = native_module(str(repo), "src.eval_client.eval_env")
            reset_calls = []
            class NativeComponentPolicyBridge:
                def __init__(self, **kwargs):
                    pass
                def call(self, func_name, **kwargs):
                    if func_name != "reset" or kwargs:
                        raise ValueError("unexpected_native_factory_policy_call")
                    cache.clear(); reset_calls.append("reset")
            module.WsModelClient = NativeComponentPolicyBridge
            # Observe only the native image kernel boundary, not its locals or
            # image bytes. Forward all original arguments unchanged. This
            # pinpoints a startup stall without changing rendering/physics.
            def observed_launch(*values, **keywords):
                kernel = keywords.get("kernel", values[0] if values else None)
                if str(getattr(kernel, "key", "")).startswith("reshape_tiled_image"):
                    camera_launch_count[0] += 1
                    inputs = keywords.get("inputs", [])
                    with (args.output / "native-camera-kernels.jsonl").open("a") as stream:
                        stream.write(json.dumps({"phase": "before_image_kernel", "call": camera_launch_count[0],
                            "at_utc": datetime.now(timezone.utc).isoformat(), "dim": keywords.get("dim"),
                            "device": str(keywords.get("device")), "buffers": [
                                {"shape": list(getattr(value, "shape", [])), "device": str(getattr(value, "device", None)),
                                 "dtype": str(getattr(value, "dtype", None))} for value in inputs[:2]],
                            "dimensions_channels_tiles_offset": [int(value) for value in inputs[2:]]}) + "\n")
                    if inputs and any(int(size) == 0 for size in inputs[0].shape):
                        # Native launch would dereference an empty CPU buffer.
                        # Stop before it, never substitute pixels, warm-render
                        # or silently skip a native initialization observation.
                        raise NativeCameraBufferError("native_camera_returned_empty_buffer_before_first_action")
                result = original_launch(*values, **keywords)
                if str(getattr(kernel, "key", "")).startswith("reshape_tiled_image"):
                    with (args.output / "native-camera-kernels.jsonl").open("a") as stream:
                        stream.write(json.dumps({"phase": "image_kernel_returned", "call": camera_launch_count[0],
                            "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                return result
            phase("before_native_factory")
            env = module.create_eval_env(cfg, app)
            phase("native_factory_returned")
            import warp as wp
            original_launch = wp.launch
            camera_launch_count = [0]
            wp.launch = observed_launch
            report["native_factory_created"] = True
            env.reset(seed=[options["layout_index"]])
            phase("native_reset_returned")
            env.run_reward(); env.get_score()
            if env.episode_nums != 1 or env.unstable_envs or not env.reward_manager.check_list[0]:
                raise ValueError("native_layout_and_registered_official_checks_required")
            report.update(robot_path_evidence=robot_evidence, resolved_native_simulation_config=OmegaConf.to_container(cfg.sim, resolve=True),
                native_reward_registration={"groups": len(env.reward_manager.check_list[0]), "stages": len(env.reward_manager.score_list[0])})
            physical_steps = [0]
            original_step = env.sim.unwrapped.sim.step
            def counted_step(*values, **keywords):
                value = original_step(*values, **keywords)
                physical_steps[0] += 1
                return value
            env.sim.unwrapped.sim.step = counted_step
            episode = args.trial / "framework/episodes/ep-0"
            result = json.loads((episode / "result.json").read_text())
            old = iter(trace(episode / "trace.internal.jsonl"))
            reference_row = next(old)
            raw = env.get_obs()
            phase("initial_native_observation_returned")
            from comparison_video import Frames
            from native_array_wire import pack, unpack
            cameras = tuple(reference_row[0]["sensors"])
            frames = Frames(args.output / "frames", cameras)
            steps, total = 0, 0
            def save(ref, action=None, delta=None):
                values = {}
                for side in ("left", "right"):
                    pose = np.asarray(raw["state"][side + "_ee_pose"], np.float64)
                    values.update({side + "_joint_positions": np.asarray(raw["state"][side + "_arm_joint_state"], np.float64),
                        side + "_eef_position": pose[:3], side + "_eef_quaternion": pose[[4, 5, 6, 3]],
                        side + "_gripper_target": np.asarray(raw["state"][side + "_ee_joint_state"], np.float64)})
                images = {name: raw["vision"][name]["color"] for name in cameras}
                elapsed = total * env.dt
                frames.append(images, steps, elapsed)
                fields = snapshot(ref[0], values, images) if ref else {}
                if ref:
                    fields["simulation_time"] = field(elapsed, ref[0]["sim_time_s"])
                    fields["public_instruction"] = {"exact_equal": str(raw["instruction"]) == ref[0]["public_instruction"]}
                    if action is not None:
                        recorded = native_action(ref[1])
                        fields["action"] = {"exact_equal": set(recorded) == set(action) and all(recorded[key].dtype == action[key].dtype and np.array_equal(recorded[key], action[key]) for key in recorded)}
                        fields["physics_delta"] = field(delta, ref[2]["execution_feedback"]["consumed_physics_steps"])
                        fields["terminated"] = field(bool(env.end_flag[0] and env.success[0]), ref[2]["terminated"])
                report["rows"].append({"env_step": steps, "native_sim_time_s": elapsed, "native_physics_delta": delta,
                    "reference_available": ref is not None, "fields": fields,
                    "native_action": {key: {"values": value.tolist(), "dtype": str(value.dtype)} for key, value in action.items()} if action is not None else None})
            save(reference_row)
            phase("initial_consumed_frame_recorded")
            if is_wam:
                # Match Runner's observed order: benchmark reset/snapshot first,
                # then the lazy native model load. Do not initialize the tiled
                # CUDA camera while a new model context is loading on that card.
                report["model_loading_order"] = "after_native_reset_and_initial_observation_like_framework"
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=1))
                start_native_model()
            else:
                from check_mn_extension_native_mappings import functions
                scope = {"np": np, "Any": Any}
                functions(reference / "utils/process_data.py", ("_validate_config", "_get_state_keys", "_ensure_valid_state_array", "pack_robot_state", "unpack_robot_state"), scope)
                functions(reference / "policy/Pi_05/model.py", ("encode_obs", "extract_image", "ensure_chw_uint8"), scope)
                sys.path.insert(0, str(reference / "policy/Pi_05/openpi/packages/openpi-client/src"))
                from openpi_client.websocket_client_policy import WebsocketClientPolicy
                from openpi_client import msgpack_numpy
                client = WebsocketClientPolicy("127.0.0.1", args.port)
                def infer(raw):
                    payload = scope["encode_obs"](raw, "joint", {"arm_dim": [6, 6], "ee_dim": [1, 1]})
                    with (exchanges / f"request-{requests + 1:04d}.msgpack").open("xb") as stream:
                        stream.write(msgpack_numpy.packb(payload))
                    actions = np.asarray(client.infer(payload)["actions"])
                    if actions.shape != (50, 14) or not np.isfinite(actions).all():
                        raise ValueError("native_ARX_50_by_14_chunk_required")
                    with (exchanges / f"actions-{requests + 1:04d}.msgpack").open("xb") as stream:
                        stream.write(msgpack_numpy.packb({"actions": actions}))
                    return scope["unpack_robot_state"](actions, "joint", {"arm_dim": [6, 6], "ee_dim": [1, 1]}, source_type="obs")
            cap = config["episodes"][0]["budgets"]["env_steps"]
            wall = config["episodes"][0]["budgets"]["wall_time_s"]
            started = time.monotonic()
            while steps < cap and not env.end_flag[0]:
                if wall is not None and time.monotonic() - started >= wall:
                    report["cooperative_wall_budget_reached"] = True
                    break
                if not cache:
                    cache.extend(infer(raw)); requests += 1
                action = cache.popleft()
                before, before_action = physical_steps[0], env.take_action_cnt[0]
                env.take_action(action)
                consumed = physical_steps[0] - before
                if env.take_action_cnt[0] != before_action + 1:
                    raise ValueError("native_action_counter_mismatch")
                steps += 1; total += consumed
                raw = env.get_obs()
                reference_row = next(old, None)
                save(reference_row, action, consumed)
                if steps % 25 == 0:
                    print(f"fresh native RoboDojo: {steps} actions", flush=True)
            same = lambda row: row["reference_available"] and all(value["exact_equal"] for value in row["fields"].values())
            success = bool(env.end_flag[0] and env.success[0])
            report.update(completed=True, fresh_native_policy_inference=requests > 0, native_actions=steps, native_requests=requests, native_physics_steps=total,
                native_sim_time_s=total * env.dt, native_metrics={"success": success, "score": 100. if success else float(env.reward_manager.get_score()[0])},
                framework_metrics=result["metrics"], initial_observation_exact=same(report["rows"][0]),
                first_divergence=next((row["env_step"] for row in report["rows"] if not same(row)), None),
                strict_equal=steps == result["execution_statistics"]["env_steps"] and next(old, None) is None and all(same(row) for row in report["rows"]),
                additional_video_get_obs_render_or_physics_calls=0, native_reset_calls=reset_calls,
                action_cap=cap, cooperative_wall_budget_s=wall)
            if connection is not None:
                connection.send(("close", None))
                if not connection.poll(30) or not connection.recv().get("ok"):
                    raise RuntimeError("native_model_close_ack_missing")
                worker.wait(timeout=30)
                if worker.returncode != 0:
                    raise RuntimeError("native_model_close_exit_failed")
        except (Exception, NativeCameraBufferError) as error:
            report["exception_chain"] = public_exception(error)
            if isinstance(error, NativeCameraBufferError):
                report["diagnostic_safety_refusal"] = "empty_native_camera_buffer; no policy action executed or image substituted"
            report["completed"] = False
        finally:
            if frames is not None and frames.rows:
                frames.close()
            if client is not None:
                client._ws.close()
            if connection is not None:
                connection.close()
            # This child belongs to our simulator process group so a parent
            # diagnostic deadline cannot orphan a separate GPU worker group.
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill(); worker.wait(timeout=10)
            if logs is not None:
                out, err = logs.result(timeout=15)
                report["native_model_public_failure_categories"] = categories((out + err).decode(errors="replace"))
            try:
                if env is not None:
                    env.close()
                app.close()
                report["close_completed"] = True
            except Exception as error:
                report["close_exception_chain"] = public_exception(error)
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            phase("native_cleanup_finished")
            write(args.output / "native-result.json", report)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if report["completed"] and report["close_completed"] else 1)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--video-runtime", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-gpu", type=int, default=2)
    parser.add_argument("--port", type=int, default=8956)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--reference-startup", action="store_true", help="Reuse only the independently working native-factory reset prefix, never its saved-action loop")
    parser.add_argument("--minimal-reference-startup", action="store_true", help="Initialize through a minimal native entry before importing policy comparison code")
    args = parser.parse_args()
    resolve_runtime_paths(args)
    if args.child:
        try:
            if args.reference_startup:
                from native_robodojo_reference_rollout import child as reference_child
                return reference_child(args)
            return child(args)
        except Exception as error:
            write(args.output / "native-result.json", {"completed": False, "exception_chain": public_exception(error)})
            os._exit(1)
    if not args.accept_runtime_eula or not (0 <= args.gpu <= 7 and 0 <= args.model_gpu <= 7 and 1 <= args.port <= 65535):
        raise ValueError("project_EULA_and_explicit_local_resources_required")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.minimal_reference_startup:
        write(args.output / "native-launch-config.json", {"model_gpu": args.model_gpu, "port": args.port,
            "video_runtime": str(args.video_runtime.resolve().relative_to(ROOT))})
    libraries = json.loads((args.isaac_libraries / "receipt.json").read_text())
    if not libraries["passed"] or any(hashlib.sha256((args.isaac_libraries / row["file"]).read_bytes()).hexdigest() != row["sha256"] for row in libraries["libraries"]):
        raise ValueError("verified_private_shared_libraries_required")
    video = json.loads((args.video_runtime / "receipt.json").read_text())
    if not video["passed"] or hashlib.sha256(Path(video["binary"]).read_bytes()).hexdigest() != video["binary_sha256"]:
        raise ValueError("verified_private_FFmpeg_required")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        PATH=video["private_bin"] + os.pathsep + os.environ.get("PATH", ""))
    if args.minimal_reference_startup:
        environment.update(PYTHONUNBUFFERED="1", ROBODOJO_RUN_ID="recorded-native-action-diagnostic")
    # Match the existing successful Benchmark worker's inherited network
    # environment. Only loopback is excluded, not all Kit/asset requests.
    for key in ("NO_PROXY", "no_proxy"):
        existing = environment.get(key, "")
        environment[key] = existing + ("," if existing else "") + "127.0.0.1,localhost"
    private = args.isaac_libraries / "library-root/usr/lib/x86_64-linux-gnu"
    if not private.is_absolute() or not private.is_dir():
        raise ValueError("existing_absolute_private_graphics_library_directory_required")
    environment["LD_LIBRARY_PATH"] = str(private) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    write(args.output / "private-runtime-paths.json", {"private_library_directory": str(private),
        "absolute_and_exists_before_child_changes_directory": True, "child_working_directory": str(args.output),
        "global_environment_changed": False})
    config = json.loads((args.trial / "framework/manifest.json").read_text())["config"]
    from embodied_eval.mn.assets import verify_model
    write(args.output / "model-assets-verified.json", {
        "model_config_sha256": hashlib.sha256(json.dumps(config["model"], sort_keys=True).encode()).hexdigest(),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification": verify_model(config["model"]["adapter"], config["model"]["options"])})
    is_wam = config["model"]["adapter"] == "openwam"
    server = worker = None
    receipt = {"kind": "owned_fresh_native_RoboDojo_comparison_supervision", "completed": False, "model_gpu": args.model_gpu,
        "render_gpu": args.gpu, "started_at_utc": datetime.now(timezone.utc).isoformat(), "raw_logs_or_credentials_saved": False,
        "global_environment_driver_occupancy_or_source_changed": False, "formal_acceptance": False}
    with ThreadPoolExecutor(max_workers=1) as pool:
        logs = None
        try:
            if not is_wam:
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", args.port))
                model_env = dict(environment, CUDA_VISIBLE_DEVICES=str(args.model_gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.6")
                server = subprocess.Popen([str(WORKSPACE / "upstream/envs/pi05-arx-mn-inference/bin/python"),
                    str(ROOT / "docs/handoff/serve_pi05_arx_native.py"), "--port", str(args.port), "--checkpoint", config["model"]["options"]["checkpoint_dir"]],
                    cwd=ROOT, env=model_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                logs = pool.submit(server.communicate)
                deadline = time.monotonic() + 900
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("native_policy_server_start_failed")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError("diagnostic_startup_bound")
                        time.sleep(1)
            invocation = [str(WORKSPACE / "upstream/envs/robodojo-mn-native/bin/python"), __file__, "--child", "--trial", str(args.trial), "--output", str(args.output),
                "--isaac-libraries", str(args.isaac_libraries), "--video-runtime", str(args.video_runtime), "--gpu", str(args.gpu), "--model-gpu", str(args.model_gpu), "--port", str(args.port)]
            if args.reference_startup:
                invocation.append("--reference-startup")
            if args.minimal_reference_startup:
                invocation = [str(WORKSPACE / "upstream/envs/robodojo-mn-native/bin/python"),
                    str(ROOT / "docs/handoff/native_robodojo_reference_entry.py"), "--child", "--trial", str(args.trial),
                    "--output", str(args.output), "--isaac-libraries", str(args.isaac_libraries), "--gpu", str(args.gpu), "--capture-native-images"]
            worker = subprocess.Popen(invocation, cwd=args.output, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            out, err = worker.communicate(timeout=15000)
            receipt.update(child_exit_code=worker.returncode, public_failure_categories=categories((out + err).decode(errors="replace")))
            if (args.output / "native-result.json").is_file():
                result = json.loads((args.output / "native-result.json").read_text())
                receipt.update(completed=worker.returncode == 0 and result["completed"] and result["close_completed"],
                    native_actions=result.get("native_actions"), native_metrics=result.get("native_metrics"), strict_equal=result.get("strict_equal"))
        except Exception as error:
            receipt["exception_chain"] = public_exception(error)
        finally:
            stop_owned(worker); stop_owned(server)
            if logs is not None:
                out, err = logs.result(timeout=15)
                receipt["model_public_failure_categories"] = categories((out + err).decode(errors="replace"))
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("completed", "native_actions", "native_metrics", "strict_equal")}), flush=True)
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
