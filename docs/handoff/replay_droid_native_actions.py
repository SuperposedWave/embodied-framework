"""Independent native DROID recorded-action replay, not fresh model inference.

Uses the fixed sim_evals registered environment and native double reset,
controller and physics. No Model/Benchmark Adapter runtime or Runner executes.
Private GL libraries and local USD relocation are deployment-only. Native logs
stay in memory. Absence of an official task-success scorer stays explicit.
"""
import argparse
from copy import deepcopy
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


def native_action(action):
    import numpy as np
    commands = {item["target_id"]: item for item in action["commands"]}
    expected = {f"robot0/j{index}" for index in range(7)} | {"robot0/grip"}
    if len(commands) != len(action["commands"]) or set(commands) != expected or action["native_protocol"] is not None:
        raise ValueError("exact_canonical_seven_joints_and_gripper_required")
    values = []
    for index in range(7):
        item = commands[f"robot0/j{index}"]
        if (item["kind"], item["mode"], item["unit"]) != ("joint_target", "position", "rad"):
            raise ValueError("absolute_joint_radians_required")
        values.append(item["value"])
    gripper = commands["robot0/grip"]
    if gripper["kind"] != "gripper_command" or gripper["mode"] != "open_fraction" or not 0 <= gripper["value"] <= 1:
        raise ValueError("canonical_gripper_opening_required")
    values.append(1. - gripper["value"])
    original = np.asarray(values)
    converted = original.astype(action["native_dtype"] or "float64")
    if not np.isfinite(converted).all() or not np.array_equal(original, converted):
        raise ValueError("recorded_native_action_dtype_loss")
    return converted


def child(args):
    import numpy as np
    import torch
    from embodied_eval.native.assets_common import _revision, _hf_objects
    from replay_mn_native_actions import trace, snapshot, field, summarize
    manifest = json.loads((args.trial / "framework/manifest.json").read_text())
    if manifest["synthetic"] or manifest["config"]["benchmark"]["adapter"] != "droid_sim":
        raise ValueError("recorded_real_droid_trial_required")
    config = manifest["config"]
    options = config["benchmark"]["options"]
    repo, assets = Path(options["repo"]), Path(options["assets_dir"])
    revision = "3a6b0e8177f696947f49c082728f10ef5133ede1"
    _revision(repo, revision)
    verified = _hf_objects(assets.resolve(), ROOT / "embodied_eval/native/asset_manifests/droid_sim.json")
    if options["scene"] != 1 or options["protocol"] != "droid-three-scenes-native-v1" or len(config["episodes"]) != 1:
        raise ValueError("reviewed_scene_one_single_episode_protocol_required")
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", fast_shutdown=False).app
    env = None
    proof = {"passed": False, "close_completed": False, "synthetic": False, "fresh_model_inference": False,
        "formal_acceptance": False, "model_or_benchmark_adapter_runtime_used": False,
        "official_task_success_scorer_available": False, "source_revision": revision,
        "fresh_policy_inference": False, "native_factory_created": False,
        "verified_asset_hashes": verified, "native_scoring_rule_added": False, "physical_gpu": args.gpu}
    try:
        sys.path.insert(0, str(repo / "src"))
        import sim_evals.environments
        import sim_evals.environments.droid_environment as module
        import gymnasium as gym
        from isaaclab_tasks.utils import parse_env_cfg
        module.DATA_PATH = assets.resolve()
        cfg = parse_env_cfg("DROID", device="cuda:0", num_envs=1, use_fabric=True)
        cfg.scene.robot = deepcopy(cfg.scene.robot)
        cfg.scene.robot.spawn.usd_path = str(assets / "franka_robotiq_2f_85_flattened.usd")
        cfg.set_scene(1)
        env = gym.make("DROID", cfg=cfg)
        proof["native_factory_created"] = True
        native = env.unwrapped
        if cfg.decimation != 8 or cfg.sim.dt != 1 / 120:
            raise ValueError("fixed_native_droid_timing_required")
        env.reset(seed=config["episodes"][0]["seed"])
        raw, _ = env.reset()
        proof["native_double_reset_completed"] = True
        proof["resolved_native_simulation_config"] = {"dt": cfg.sim.dt, "decimation": cfg.decimation,
            "render_interval": cfg.sim.render_interval, "device": cfg.sim.device,
            "gravity": list(cfg.sim.gravity), "use_fabric": cfg.sim.use_fabric,
            "seed": cfg.seed, "scene": 1, "num_envs": cfg.scene.num_envs}
        if args.capture_native_images:
            proof["native_image_captures"] = []
            proof["native_camera_extrinsics"] = {name: {key: getattr(native.scene[name].data, key).detach().cpu().numpy().tolist()
                for key in ("pos_w", "quat_w_ros", "intrinsic_matrices")}
                for name in ("external_cam", "external_cam_2", "wrist_cam")}
            import carb
            settings = carb.settings.get_settings()
            keys = ("/rtx/post/aa/op", "/rtx/ambientOcclusion/enabled", "/rtx/directLighting/enabled",
                "/rtx/indirectDiffuse/enabled", "/rtx/raytracing/fractionalCutoutOpacity", "/app/renderer/waitIdle", "/app/hydraEngine/waitIdle")
            proof["allowlisted_render_settings"] = {key: settings.get(key) for key in keys}
        # Observe actual SimulationContext calls rather than reuse its own
        # public counter or the Benchmark's production meter.
        original_step = native.sim.step
        physical_steps = [0]
        def counted_step(*values, **keywords):
            reply = original_step(*values, **keywords)
            physical_steps[0] += 1
            return reply
        native.sim.step = counted_step
        rows, executed, total, recorded_total = [], 0, 0, 0
        episode = args.trial / "framework/episodes/ep-0"
        result = json.loads((episode / "result.json").read_text())
        for observation, action, reply in trace(episode / "trace.internal.jsonl"):
            consumed = 0
            if action is not None:
                before = physical_steps[0]
                vector = torch.as_tensor(native_action(action), device=native.device)[None]
                raw, _, terminated, truncated, _ = env.step(vector)
                consumed = physical_steps[0] - before
                total += consumed
                executed += 1
            policy = raw["policy"]
            as_numpy = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            values = {"arm_joint_positions": np.asarray(as_numpy(policy["arm_joint_pos"]).reshape(-1), np.float64),
                "gripper_open_fraction": 1. - np.asarray(as_numpy(policy["gripper_pos"]).reshape(-1), np.float64)}
            images = {name: as_numpy(policy[name])[0] for name in observation["sensors"]}
            if args.capture_native_images:
                capture = args.output / f"native-images-{executed:04d}.npz"
                np.savez_compressed(capture, **images)
                proof["native_image_captures"].append({"env_step": executed, "file": capture.name,
                    "sha256": hashlib.sha256(capture.read_bytes()).hexdigest()})
            fields = snapshot(observation, values, images, executed / 15)
            fields["env_step"] = field(observation["env_step"], executed)
            fields["sim_time_s"] = field(observation["sim_time_s"], executed / 15)
            fields["public_instruction"] = {"exact_equal": observation["public_instruction"] == "put the cube in the bowl"}
            if reply is not None:
                recorded_total += reply["execution_feedback"]["consumed_physics_steps"]
                fields["physics_steps"] = field(reply["execution_feedback"]["consumed_physics_steps"], consumed)
                fields["elapsed_sim_time_s"] = field(reply["execution_feedback"]["elapsed_sim_time_s"], 1 / 15)
                fields["terminated"] = field(reply["terminated"], bool(terminated.item()))
                fields["truncated"] = field(reply["truncated"], bool(truncated.item()))
            rows.append({"env_step": executed, "physics_steps": consumed, "fields": fields})
        proof.update(summarize(rows, executed, result, {"success": None}),
            independent_observed_physics_steps=total, recorded_feedback_physics_steps=recorded_total,
            physics_steps_equal=total == recorded_total,
            native_environment_source_sha256=hashlib.sha256((repo / "src/sim_evals/environments/droid_environment.py").read_bytes()).hexdigest())
        proof["passed"] = proof["passed"] and proof["physics_steps_equal"]
        proof["native_source_factory_sha256"] = proof["native_environment_source_sha256"]
    except Exception as error:
        proof["exception_chain"] = public_exception(error)
    finally:
        try:
            if env is not None:
                env.close()
            app.close()
            proof["close_completed"] = True
        except Exception as error:
            proof.update(passed=False, close_exception_chain=public_exception(error))
        write(args.output / "native-result.json", proof)
    os._exit(0 if proof["passed"] and proof["close_completed"] else 1)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--capture-native-images", action="store_true",
        help="Save unchanged native RGB and read-only camera/render metadata")
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
        raise ValueError("project_only_EULA_and_explicit_physical_GPU_required")
    args.output.mkdir(parents=True, exist_ok=False)
    libraries = json.loads((args.isaac_libraries / "receipt.json").read_text())
    if not libraries["passed"] or any(hashlib.sha256((args.isaac_libraries / item["file"]).read_bytes()).hexdigest() != item["sha256"] for item in libraries["libraries"]):
        raise ValueError("verified_private_isaac_libraries_required")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    directory = args.isaac_libraries / "library-root/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = str(directory) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    command = [str(WORKSPACE / "upstream/envs/droid_sim-mn-native/bin/python"), __file__, "--child", "--trial", str(args.trial),
        "--output", str(args.output), "--isaac-libraries", str(args.isaac_libraries), "--gpu", str(args.gpu)]
    if args.capture_native_images:
        command.append("--capture-native-images")
    process = None
    receipt = {"kind": "independent_native_droid_recorded_action_replay_not_reinference", "passed": False,
        "trial": str(args.trial.relative_to(ROOT)), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_logs_saved": False, "global_proxy_or_shared_driver_modified": False, "runner_unchanged": True}
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
        path = args.output / "native-result.json"
        if path.is_file():
            result = json.loads(path.read_text())
            receipt.update(passed=process.returncode == 0 and result["passed"] and result["close_completed"], result=str(path.relative_to(ROOT)))
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
