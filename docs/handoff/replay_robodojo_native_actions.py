"""Independent recorded-action RoboDojo replay through the fixed native factory.

No Model/Benchmark Adapter runs or policy inference. The only native policy
client replacement is a reset-only diagnostic placeholder; its calls are
recorded. Native reset, controls, scoring and observations run unchanged.
Only this standalone diagnostic's child has a deadline; Runner is untouched.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_cosmos_droid_bounded import stop_owned

sys.path.insert(0, str(ROOT))


def native_action(action):
    """Direct SI typed-command reconstruction, no Benchmark conversion calls."""
    import numpy as np
    commands = {row["target_id"]: row for row in action["commands"]}
    if len(commands) != len(action["commands"]):
        raise ValueError("duplicate_recorded_action_target")
    def lossless_array(values):
        original = np.asarray(values)
        cast = original.astype(action["native_dtype"])
        if not np.isfinite(cast).all() or not np.array_equal(original, cast):
            raise ValueError("recorded_native_dtype_loss")
        return cast
    row, used = {}, set()
    for side in ("left", "right"):
        pose_target = "robot0/" + side + "_eef"
        if pose_target in commands:
            pose = commands[pose_target]["pose"]
            if commands[pose_target]["kind"] != "pose_target" or pose["frame_id"] != "world":
                raise ValueError("recorded_absolute_scene_pose_required")
            q = pose["quaternion_xyzw"]
            row[side + "_ee_pose"] = lossless_array([*pose["position_m"], q[3], *q[:3]])
            used.add(pose_target)
        else:
            values = []
            for index in range(6):
                key = f"robot0/{side}_j{index}"
                command = commands[key]
                if (command["kind"], command["mode"], command["unit"]) != ("joint_target", "position", "rad"):
                    raise ValueError("recorded_absolute_radian_joints_required")
                values.append(command["value"])
                used.add(key)
            row[side + "_arm_joint_state"] = lossless_array(values)
        grip_key = "robot0/" + side + "_grip"
        grip = commands[grip_key]
        if grip["kind"] == "controller_command":
            from embodied_eval.mn.joint_native import RAW_GRIPPER_PROTOCOL, RAW_GRIPPER_CONTROLLER
            if action["native_protocol"] != RAW_GRIPPER_PROTOCOL or grip["controller_profile_ref"] != RAW_GRIPPER_CONTROLLER:
                raise ValueError("matching_source_raw_gripper_required")
            opening = grip["parameters"]["requested_open_fraction"]
        elif grip["kind"] == "gripper_command" and grip["mode"] == "open_fraction":
            opening = grip["value"]
        else:
            raise ValueError("recorded_open_fraction_required")
        row[side + "_ee_joint_state"] = lossless_array([opening])
        used.add(grip_key)
    if used != set(commands) or any(not np.isfinite(value).all() for value in row.values()):
        raise ValueError("incomplete_or_nonfinite_recorded_action")
    return row


def child(args):
    import numpy as np
    from argparse import Namespace
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.mn.assets_common import _revision, _hf_objects
    from embodied_eval.mn.robodojo_simulator import REVISION, prepare_robot_paths
    from embodied_eval.mn.native_catalog import PROFILES
    from replay_mn_native_actions import trace, snapshot, field, summarize
    manifest = json.loads((args.trial / "framework/manifest.json").read_text())
    if manifest["synthetic"] or manifest["config"]["benchmark"]["adapter"] != "robodojo":
        raise ValueError("real_source_protocol_robodojo_trial_required")
    options = manifest["config"]["benchmark"]["options"]
    if options["protocol"] != "stack-bowls-arx-x5-native-v1":
        raise ValueError("fixed_native_protocol_required")
    repo = Path(options["repo"])
    reference = WORKSPACE / "upstream/xpolicy-reference-mn"
    _revision(repo, REVISION)
    _revision(reference, PROFILES["arx_joint14"][2])
    verified_assets = _hf_objects(repo.resolve(), ROOT / "embodied_eval/mn/asset_manifests/robodojo.json")
    sys.path.insert(0, str(reference))
    sys.path.insert(0, str(repo))
    from isaaclab.app import AppLauncher
    sync = native_module(str(repo), "env.camera_manager.capture.render_sync")
    launch = Namespace(kit_args="", headless=True, enable_cameras=True, device="cuda:0")
    sync.add_zero_delay_kit_args(launch)
    app = AppLauncher(launch, fast_shutdown=False).app
    env = None
    proof = {"passed": False, "native_factory_created": False, "close_completed": False,
        "synthetic": False, "fresh_policy_inference": False, "formal_acceptance": False,
        "model_or_benchmark_adapter_runtime_used": False, "physics_gpu_override": False,
        "native_video_writers_disabled": False, "physical_render_gpu": args.gpu,
        "verified_before_native_load_asset_hashes": verified_assets}
    try:
        from omegaconf import OmegaConf
        loaders = native_module(str(repo), "utils.load_file")
        pipeline = native_module(str(repo), "utils.pipeline_utils")
        robot_paths, robot_evidence = prepare_robot_paths(repo, args.output)
        pipeline.ROBOTS_PATH = str(robot_paths)
        read = lambda relative: loaders.load_yaml(str(repo / relative))
        eval_cfg = read("env_cfg/arx_x5.yml")
        eval_cfg.update(task_name="stack_bowls", num_envs=1, seed=0, eval_num=1,
            physx_monitor_enabled=False, eval_batch=False, policy_name="recorded_action_diagnostic", description={})
        cfg = OmegaConf.create({"sim": read("env_cfg/sim/sim_config.yml"), "scene": read("env_cfg/scene/default.yml"),
            "camera": read("env_cfg/camera/camera_config.yml"), "robot": read("env_cfg/robot/dual_x5.yml"),
            "task_env": read("task/RoboDojo/config/stack_bowls.yml"), "eval_cfg": eval_cfg,
            "deploy_cfg": {"policy_name": "recorded_action_diagnostic", "port": 1, "host": "127.0.0.1"}})
        cfg = pipeline.process_randomization(cfg)
        cfg, _ = pipeline.process_config(cfg, task_name="stack_bowls")
        cfg.sim.scene.num_envs = 1
        cfg.sim.seed = [0]
        cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
        if cfg.sim.get("device", "cpu") != "cpu":
            raise ValueError("fixed_native_cpu_physics_required")
        module = native_module(str(repo), "src.eval_client.eval_env")
        client_calls = []
        class ResetOnlyRecordedActionClient:
            def __init__(self, **kwargs):
                pass
            def call(self, func_name, **kwargs):
                if func_name != "reset" or kwargs:
                    raise ValueError("recorded_replay_must_not_infer")
                client_calls.append("reset")
        module.WsModelClient = ResetOnlyRecordedActionClient
        env = module.create_eval_env(cfg, app)
        proof["native_factory_created"] = True
        # Native BaseEnv creates its gym simulation during reset, not in the
        # task constructor. Observe only action physics after that native reset.
        env.reset(seed=[options["layout_index"]])
        original_step = env.sim.unwrapped.sim.step
        physical_steps = [0]
        def counted_step(*values, **keywords):
            result = original_step(*values, **keywords)
            physical_steps[0] += 1
            return result
        env.sim.unwrapped.sim.step = counted_step
        env.run_reward()
        env.get_score()
        if env.episode_nums != 1 or env.unstable_envs or not env.reward_manager.check_list[0]:
            raise ValueError("native_layout_or_registered_checks_required")
        proof["native_reward_registration"] = {"groups": len(env.reward_manager.check_list[0]),
            "score_stages": len(env.reward_manager.score_list[0]), "mode": env.reward_manager.score_meta[0]["mode"]}
        proof["resolved_native_simulation_config"] = OmegaConf.to_container(cfg.sim, resolve=True)
        if args.capture_native_images:
            proof["native_image_captures"] = []
            proof["native_camera_extrinsics"] = {env.camera_manager.camera_names[0][index]:
                np.asarray(env.camera_manager.get_camera_extrinsics(index, 0)).tolist()
                for index in range(env.camera_manager.num_cams)}
            import carb
            settings = carb.settings.get_settings()
            keys = ("/rtx/post/aa/op", "/rtx/ambientOcclusion/enabled", "/rtx/directLighting/enabled",
                "/rtx/indirectDiffuse/enabled", "/rtx/raytracing/fractionalCutoutOpacity", "/app/renderer/waitIdle", "/app/hydraEngine/waitIdle")
            proof["allowlisted_render_settings"] = {key: settings.get(key) for key in keys}
        episode = args.trial / "framework/episodes/ep-0"
        result = json.loads((episode / "result.json").read_text())
        rows, executed, action_physics, elapsed = [], 0, 0, 0.
        for observation, action, reply in trace(episode / "trace.internal.jsonl"):
            consumed = 0
            if action is not None:
                if env.end_flag[0]:
                    break
                before, before_action = physical_steps[0], env.take_action_cnt[0]
                env.take_action(native_action(action))
                consumed = physical_steps[0] - before
                if env.take_action_cnt[0] != before_action + 1:
                    raise ValueError("native_action_not_executed")
                executed += 1
                action_physics += consumed
                elapsed = action_physics * env.dt
            raw = env.get_obs()
            values = {}
            for side in ("left", "right"):
                pose = np.asarray(raw["state"][side + "_ee_pose"], np.float64)
                values.update({side + "_joint_positions": np.asarray(raw["state"][side + "_arm_joint_state"], np.float64),
                    side + "_eef_position": pose[:3], side + "_eef_quaternion": pose[[4, 5, 6, 3]],
                    side + "_gripper_target": np.asarray(raw["state"][side + "_ee_joint_state"], np.float64)})
            images = {key: raw["vision"][key]["color"] for key in observation["sensors"]}
            if args.capture_native_images:
                capture = args.output / f"native-images-{executed:04d}.npz"
                np.savez_compressed(capture, **{name: np.asarray(value) for name, value in images.items()})
                proof["native_image_captures"].append({"env_step": executed, "file": capture.name,
                    "sha256": hashlib.sha256(capture.read_bytes()).hexdigest()})
            fields = snapshot(observation, values, images, elapsed)
            fields["env_step"] = field(observation["env_step"], executed)
            fields["sim_time_s"] = field(observation["sim_time_s"], elapsed)
            fields["public_instruction"] = {"exact_equal": observation["public_instruction"] == str(raw["instruction"])}
            if reply:
                fields["physics_steps"] = field(reply["execution_feedback"]["consumed_physics_steps"], consumed)
                fields["elapsed_sim_time_s"] = field(reply["execution_feedback"]["elapsed_sim_time_s"], consumed * env.dt)
                fields["consumed_env_steps"] = field(reply["execution_feedback"]["consumed_env_steps"], 1)
                fields["terminated"] = field(reply["terminated"], bool(env.end_flag[0] and env.success[0]))
                fields["truncated"] = field(reply["truncated"], bool(env.end_flag[0] and not env.success[0]))
            rows.append({"env_step": executed, "physics_steps": consumed, "fields": fields})
        success = bool(env.end_flag[0] and env.success[0])
        metrics = {"success": success, "score": 100. if success else float(env.reward_manager.get_score()[0])}
        proof.update(summarize(rows, executed, result, metrics), independent_observed_physics_steps=action_physics,
            physics_steps_equal=action_physics == result["execution_statistics"]["physics_steps"],
            native_source_factory_sha256=hashlib.sha256((repo / "src/eval_client/eval_env.py").read_bytes()).hexdigest(),
            reset_only_native_client_calls=client_calls, robot_path_evidence=robot_evidence)
        proof["passed"] = proof["passed"] and proof["physics_steps_equal"] and client_calls == ["reset"]
    except Exception as error:
        proof["passed"] = False
        proof["exception_chain"] = public_exception(error)
    finally:
        try:
            if env is not None:
                env.close()
            app.close()
            proof["close_completed"] = True
        except Exception as error:
            proof["passed"] = False
            proof["close_exception_chain"] = public_exception(error)
        write(args.output / "native-result.json", proof)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if proof["passed"] and proof["close_completed"] else 1)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--video-runtime", type=Path,
        help="Verified private FFmpeg directory; native video writer remains unchanged")
    parser.add_argument("--capture-native-images", action="store_true",
        help="Save unchanged native RGB for exact repeated-render diagnostics, no image processing")
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output, args.trial, args.isaac_libraries = args.output.resolve(), args.trial.resolve(), args.isaac_libraries.resolve()
    for path in (args.output, args.trial, args.isaac_libraries):
        path.relative_to(ROOT / "outputs")
    if args.child:
        try:
            return child(args)
        except Exception as error:
            write(args.output / "native-result.json", {"passed": False, "exception_chain": public_exception(error)})
            os._exit(1)
    if not args.accept_runtime_eula or not 0 <= args.gpu <= 7:
        raise ValueError("project-only EULA and GPU ordinal required")
    args.output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((args.isaac_libraries / "receipt.json").read_text())
    if not preparation["passed"] or any(hashlib.sha256((args.isaac_libraries / row["file"]).read_bytes()).hexdigest() != row["sha256"] for row in preparation["libraries"]):
        raise ValueError("verified_private_libraries_required")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", ROBODOJO_RUN_ID="recorded-native-action-diagnostic")
    video_evidence = None
    if args.video_runtime is not None:
        video = args.video_runtime.resolve()
        video.relative_to(ROOT / "outputs")
        video_evidence = json.loads((video / "receipt.json").read_text())
        binary = video / "bin/ffmpeg"
        if (not video_evidence["passed"] or video_evidence["global_path_modified"]
                or binary.resolve() != Path(video_evidence["binary"]).resolve()
                or hashlib.sha256(binary.read_bytes()).hexdigest() != video_evidence["binary_sha256"]):
            raise ValueError("verified_private_native_video_binary_required")
        environment["PATH"] = str(video / "bin") + os.pathsep + environment.get("PATH", "")
    directory = args.isaac_libraries / "library-root/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = str(directory) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    command = [str(WORKSPACE / "upstream/envs/robodojo-mn-native/bin/python"), __file__, "--child", "--trial", str(args.trial),
        "--output", str(args.output), "--isaac-libraries", str(args.isaac_libraries), "--gpu", str(args.gpu)]
    if args.capture_native_images:
        command.append("--capture-native-images")
    process = None
    receipt = {"kind": "independent_recorded_native_robodojo_action_replay_not_reinference", "passed": False,
        "trial": str(args.trial.relative_to(ROOT)), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_logs_saved": False, "global_proxy_or_shared_driver_modified": False, "runner_unchanged": True}
    if video_evidence is not None:
        receipt["private_video_runtime"] = {"receipt": str(video.relative_to(ROOT) / "receipt.json"),
            "binary_sha256": video_evidence["binary_sha256"], "ffmpeg_version": video_evidence["ffmpeg_version"]}
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30)
    selected = [line.split(",") for line in inventory.stdout.splitlines() if int(line.split(",")[0]) == args.gpu]
    if len(selected) != 1:
        raise ValueError("one_selected_physical_render_gpu_required")
    receipt["render_gpu_identity"] = {"index": args.gpu, "uuid": selected[0][1].strip(),
        "name": selected[0][2].strip(), "driver": selected[0][3].strip()}
    try:
        process = subprocess.Popen(command, cwd=args.output, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        out, err = process.communicate(timeout=1800)
        receipt.update(child_exit_code=process.returncode, public_log_categories=categories((out + err).decode(errors="replace")))
        result_path = args.output / "native-result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            receipt.update(passed=process.returncode == 0 and result["passed"] and result["close_completed"],
                result=str(result_path.relative_to(ROOT)))
    except Exception as error:
        receipt["exception_chain"] = public_exception(error)
    finally:
        stop_owned(process)
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("passed", "child_exit_code", "result")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
