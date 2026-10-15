"""Fresh Cosmos native-component/DROID closedloop, at the recorded budget.

No production Adapter or Runner executes on this native side. The fixed RoboLab
packing/decoding methods, real native policy service, and sim_evals factory are
used directly. Saved actions are comparison evidence, never simulator inputs.
Logs and credentials are not persisted. No retries or invented task scoring.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
from uuid import uuid4

from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from run_cosmos_droid_bounded import stop_owned
sys.path.insert(0, str(ROOT))


def native_state(value, width):
    """One declared environment: preserve all native float32 measurements."""
    import numpy as np
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.shape not in ((width,), (1, width)) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("complete_finite_single_environment_float32_state_required")
    return array.reshape(width)


def child(args):
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from comparison_video import Frames
    from check_mn_extension_native_mappings import native_class
    from embodied_eval.mn.assets_common import _revision, _hf_objects
    from embodied_eval.mn.additional_catalog import ROBOLAB_REVISION, OPENPI_REVISION, verify_policy
    from embodied_eval.adapters.native_common import native_module
    from replay_mn_native_actions import trace, snapshot, field
    from replay_droid_native_actions import native_action
    manifest = json.loads((args.trial / "framework/manifest.json").read_text())
    config = manifest["config"]
    if manifest["synthetic"] or config["model"]["adapter"] != "cosmos3" or config["benchmark"]["adapter"] != "droid_sim":
        raise ValueError("real_Cosmos_DROID_named_trial_required")
    opts, model_opts = config["benchmark"]["options"], config["model"]["options"]
    if opts["protocol"] != "droid-three-scenes-native-v1" or opts["scene"] != 1 or len(config["episodes"]) != 1:
        raise ValueError("fixed_scene_one_single_episode_required")
    repo, assets = Path(opts["repo"]), Path(opts["assets_dir"])
    _revision(repo, "3a6b0e8177f696947f49c082728f10ef5133ede1")
    _revision(model_opts["client_repo"], ROBOLAB_REVISION)
    _revision(model_opts["codec_repo"], OPENPI_REVISION)
    env_assets = _hf_objects(assets.resolve(), ROOT / "embodied_eval/mn/asset_manifests/droid_sim.json")
    model_assets = verify_policy("cosmos3", model_opts)
    source = Path(model_opts["client_repo"]) / "policies/cosmos3/client.py"
    oracle = native_class(source, "Cosmos3Client", ("_pack_request", "_unpack_response", "_postprocess_chunk"),
        {"np": np, "torch": torch, "F": functional})
    oracle._image_h, oracle._image_w = 360, 640
    images = native_module(model_opts["client_repo"], "robolab.core.utils.image_utils")
    transport_module = native_module(model_opts["client_repo"], "robolab.eval.websocket_transport")
    # Deliberately do not read COSMOS3_API_TOKEN; this is our owned loopback
    # server with no authentication headers. One request, no native retry loop.
    peer = transport_module.MsgPackWebSocketTransport(f"ws://127.0.0.1:{args.port}", api_token=None,
        connect_kwargs={"ping_interval": None, "ping_timeout": None}, metadata_timeout=30.)
    peer.connect()
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", fast_shutdown=False).app
    env = frames = None
    report = {"kind": "fresh_native_component_Cosmos_DROID_closedloop_at_recorded_budget", "completed": False,
        "strict_equal": False, "close_completed": False, "synthetic": False, "formal_acceptance": False,
        "fresh_policy_inference": True, "recorded_actions_control_native_environment": False,
        "production_Model_or_BenchmarkAdapter_instantiated": False, "official_task_success_scorer_available": False,
        "native_scoring_rule_added": False, "verified_environment_assets_before_load": env_assets,
        "verified_model_assets_before_load": model_assets, "native_client_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "render_gpu": args.gpu, "model_gpu": args.model_gpu, "rows": [], "additional_video_get_obs_render_or_physics_calls": 0,
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
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
        native = env.unwrapped
        if cfg.decimation != 8 or cfg.sim.dt != 1 / 120:
            raise ValueError("fixed_native_DROID_timing_required")
        env.reset(seed=config["episodes"][0]["seed"])
        raw, _ = env.reset()
        report["native_double_reset_completed"] = True
        report["resolved_native_simulation_config"] = {"dt": cfg.sim.dt, "decimation": cfg.decimation,
            "render_interval": cfg.sim.render_interval, "device": cfg.sim.device, "gravity": list(cfg.sim.gravity),
            "use_fabric": cfg.sim.use_fabric, "seed": cfg.seed, "scene": 1, "num_envs": cfg.scene.num_envs}
        physical_steps = [0]
        original_step = native.sim.step
        def counted_step(*values, **keywords):
            value = original_step(*values, **keywords)
            physical_steps[0] += 1
            return value
        native.sim.step = counted_step
        episode = args.trial / "framework/episodes/ep-0"
        result = json.loads((episode / "result.json").read_text())
        previous = iter(trace(episode / "trace.internal.jsonl"))
        reference = next(previous)
        cameras = tuple(reference[0]["sensors"])
        frames = Frames(args.output / "frames", cameras)
        exchanges = args.output / "native-exchanges"
        exchanges.mkdir()
        cache, requests, steps, total = deque(), 0, 0, 0
        session_id = str(uuid4())
        report["policy_session"] = {"kind": "fresh_unique_uuid", "same_UUID_as_historical_framework": False,
            "reviewed_Cosmos_service_reads_session_id": False, "session_id": session_id}
        as_numpy = lambda value: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        def save(ref, action=None, consumed=None, terminated=None, truncated=None):
            policy = raw["policy"]
            values = {"arm_joint_positions": np.asarray(as_numpy(policy["arm_joint_pos"]).reshape(-1), np.float64),
                "gripper_open_fraction": 1. - np.asarray(as_numpy(policy["gripper_pos"]).reshape(-1), np.float64)}
            rgb = {name: as_numpy(policy[name])[0] for name in cameras}
            frames.append(rgb, steps, total * cfg.sim.dt)
            fields = snapshot(ref[0], values, rgb, steps / 15) if ref else {}
            if ref:
                fields["public_instruction"] = {"exact_equal": ref[0]["public_instruction"] == "put the cube in the bowl"}
                fields["simulation_time"] = field(steps / 15, ref[0]["sim_time_s"])
                if action is not None:
                    recorded = native_action(ref[1])
                    fields["action"] = {"exact_equal": recorded.dtype == action.dtype and np.array_equal(recorded, action),
                        "max_abs_error": float(np.max(np.abs(recorded.astype(np.float64) - action.astype(np.float64))))}
                    fields["physics_delta"] = field(consumed, ref[2]["execution_feedback"]["consumed_physics_steps"])
                    fields["terminated"] = field(terminated, ref[2]["terminated"])
                    fields["truncated"] = field(truncated, ref[2]["truncated"])
            report["rows"].append({"env_step": steps, "native_sim_time_s": total * cfg.sim.dt,
                "reference_available": ref is not None, "fields": fields,
                "native_action": None if action is None else {"values": action.tolist(), "dtype": str(action.dtype)}})
        save(reference)
        budgets = config["episodes"][0]["budgets"]
        cap, horizon = budgets["env_steps"], model_opts.get("open_loop_horizon", 32)
        started = time.monotonic()
        for _ in range(cap):
            if budgets["wall_time_s"] is not None and time.monotonic() - started >= budgets["wall_time_s"]:
                report["cooperative_wall_budget_reached"] = True
                break
            if not cache:
                policy = raw["policy"]
                rgb = {slot: images.resize_with_pad(as_numpy(policy[name])[0], 360, 640)
                    for slot, name in (("left_image", "external_cam"), ("right_image", "external_cam_2"), ("wrist_image", "wrist_cam"))}
                payload = oracle._pack_request(dict(rgb,
                    joint_position=native_state(policy["arm_joint_pos"], 7),
                    gripper_position=native_state(policy["gripper_pos"], 1), session_id=session_id), "put the cube in the bowl")
                # Allowlisted numerical payload only; never metadata/headers.
                with (exchanges / f"request-{requests + 1:04d}.msgpack").open("xb") as stream:
                    stream.write(peer.packer.pack(payload))
                chunk = oracle._unpack_response(peer.request(payload, timeout=300.))
                if chunk.ndim != 2 or chunk.shape[1] != 8 or chunk.shape[0] < horizon or chunk.dtype not in (np.float32, np.float64) or not np.isfinite(chunk).all():
                    raise ValueError("finite_native_joint8_chunk_required")
                with (exchanges / f"actions-{requests + 1:04d}.msgpack").open("xb") as stream:
                    stream.write(peer.packer.pack({"actions": chunk}))
                cache.extend(oracle._postprocess_chunk(chunk[:horizon])); requests += 1
            action = cache.popleft()
            before = physical_steps[0]
            raw, _, terminated, truncated, _ = env.step(torch.as_tensor(action, device=native.device)[None])
            consumed = physical_steps[0] - before
            total += consumed; steps += 1
            reference = next(previous, None)
            save(reference, action, consumed, bool(terminated.item()), bool(truncated.item()))
            if bool(terminated.item()) or bool(truncated.item()):
                break
        same = lambda row: row["reference_available"] and all(v["exact_equal"] for v in row["fields"].values())
        report.update(completed=True, native_actions=steps, native_requests=requests, native_physics_steps=total,
            native_sim_time_s=total * cfg.sim.dt, native_metrics={"success": None}, framework_metrics=result["metrics"],
            action_budget=cap, full_task_trial=False, cache_rows_remaining=len(cache),
            initial_observation_exact=same(report["rows"][0]),
            first_divergence=next((r["env_step"] for r in report["rows"] if not same(r)), None),
            strict_equal=steps == result["execution_statistics"]["env_steps"] and next(previous, None) is None and all(same(r) for r in report["rows"]))
    except Exception as error:
        report["exception_chain"] = public_exception(error)
    finally:
        if frames is not None and frames.rows:
            frames.close()
        peer.close()
        try:
            if env is not None:
                env.close()
            app.close()
            report["close_completed"] = True
        except Exception as error:
            report["close_exception_chain"] = public_exception(error)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "native-result.json", report)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0 if report["completed"] and report["close_completed"] else 1)


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "output", "isaac-libraries", "vae"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-gpu", type=int, default=4)
    parser.add_argument("--port", type=int, default=8958)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--pair", action="store_true", help="New framework and native runs on identical selected GPUs")
    args = parser.parse_args()
    for key in ("trial", "output", "isaac_libraries", "vae"):
        value = getattr(args, key).resolve(); value.relative_to(ROOT / "outputs"); setattr(args, key, value)
    if args.child:
        try:
            return child(args)
        except Exception as error:
            write(args.output / "native-result.json", {"completed": False, "exception_chain": public_exception(error)})
            os._exit(1)
    if not args.accept_runtime_eula or not (0 <= args.gpu <= 7 and 0 <= args.model_gpu <= 7 and 1 <= args.port <= 65535):
        raise ValueError("explicit_project_EULA_and_resources_required")
    args.output.mkdir(parents=True, exist_ok=False)
    prep = json.loads((args.isaac_libraries / "receipt.json").read_text())
    if not prep["passed"] or any(hashlib.sha256((args.isaac_libraries / row["file"]).read_bytes()).hexdigest() != row["sha256"] for row in prep["libraries"]):
        raise ValueError("verified_private_GL_libraries_required")
    if not json.loads((args.vae / "receipt.json").read_text())["passed"]:
        raise ValueError("verified_fixed_native_VAE_required")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", args.port))
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)  # Owned loopback children only, never global proxy.
    private = args.isaac_libraries / "library-root/usr/lib/x86_64-linux-gnu"
    sim_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES",
        LD_LIBRARY_PATH=str(private) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""))
    libraries = WORKSPACE / "upstream/envs/cosmos3-mn-native/lib/python3.13/site-packages/nvidia"
    model_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.model_gpu), CUDA_HOME=str(libraries), NVTE_CUDA_INCLUDE_DIR=str(libraries / "cuda_runtime/include"))
    if args.pair:
        cfg = json.loads((args.trial / "framework/manifest.json").read_text())["config"]
        if cfg["model"]["adapter"] != "cosmos3" or cfg["benchmark"]["adapter"] != "droid_sim":
            raise ValueError("matching_existing_Cosmos_DROID_config_required")
        cfg["run_id"] += "-same-condition-20261009"
        cfg["model"]["options"]["port"] = args.port
        config_path = args.output / "config.json"
        write(config_path, cfg)
    receipt = {"kind": "owned_fresh_native_Cosmos_DROID_supervision", "completed": False, "formal_acceptance": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "raw_logs_or_credentials_saved": False,
        "global_proxy_or_driver_or_occupancy_changed": False,
        "both_framework_and_native_new_this_round": args.pair, "sides": [],
        "model_gpu": args.model_gpu, "simulator_gpu": args.gpu,
        "same_native_service_config_and_child_environments_both_sides": args.pair}
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=30)
    receipt["GPU_identities"] = [{"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(), "driver": row[3].strip()}
        for line in inventory.stdout.splitlines() if (row := line.split(",")) and int(row[0]) in (args.gpu, args.model_gpu)]
    native_dir = args.output / "native" if args.pair else args.output
    if args.pair:
        native_dir.mkdir()
    for side in (("framework", "native") if args.pair else ("native",)):
        server_dir = args.output / (side + "-server"); server_dir.mkdir()
        server_cmd = [str(WORKSPACE / "upstream/envs/cosmos3-mn-native/bin/python"), str(ROOT / "docs/handoff/probe_mn_components.py"),
            "--child", "--component", "cosmos3", "--mode", "cosmos-server", "--cosmos-native-vae", str(args.vae), "--port", str(args.port), "--output", str(server_dir)]
        if side == "framework":
            client_cmd = [str(WORKSPACE / "upstream/envs/droid_sim-mn-native/bin/python"), str(ROOT / "docs/handoff/run_cosmos_droid_bounded.py"),
                "--child", str(config_path), "--output", str(args.output / "framework"), "--vae", str(args.vae), "--isaac-libraries", str(args.isaac_libraries)]
        else:
            client_cmd = [str(WORKSPACE / "upstream/envs/droid_sim-mn-native/bin/python"), __file__, "--child", "--trial", str(args.output if args.pair else args.trial),
                "--output", str(native_dir), "--isaac-libraries", str(args.isaac_libraries), "--vae", str(args.vae),
                "--gpu", str(args.gpu), "--model-gpu", str(args.model_gpu), "--port", str(args.port)]
        server = client = None
        with ThreadPoolExecutor(max_workers=1) as pool:
            logs = None
            try:
                server = subprocess.Popen(server_cmd, cwd=ROOT, env=model_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                logs = pool.submit(server.communicate)
                deadline = time.monotonic() + 900
                while not (server_dir / "server-ready.json").is_file():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("native_server_loading_failed_or_diagnostic_bound")
                    time.sleep(1)
                while True:
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=.5):
                            break
                    except OSError:
                        if server.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("native_policy_socket_not_ready")
                        time.sleep(1)
                client = subprocess.Popen(client_cmd, cwd=args.output, env=sim_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                out, err = client.communicate(timeout=1800)
                receipt["sides"].append({"side": side, "exit_code": client.returncode, "public_failure_categories": categories((out + err).decode(errors="replace"))})
                if client.returncode != 0:
                    raise RuntimeError("owned_comparison_side_incomplete")
                print(f"{side} Cosmos/DROID complete", flush=True)
            except Exception as error:
                receipt["exception_chain"] = public_exception(error)
                break
            finally:
                stop_owned(client); stop_owned(server)
                if logs is not None:
                    out, err = logs.result(timeout=15)
                    receipt.setdefault("model_public_failure_categories", []).append(categories((out + err).decode(errors="replace")))
    if (native_dir / "native-result.json").is_file():
        result = json.loads((native_dir / "native-result.json").read_text())
        receipt.update(completed=len(receipt["sides"]) == (2 if args.pair else 1) and all(row["exit_code"] == 0 for row in receipt["sides"]) and result["completed"] and result["close_completed"],
            native_actions=result.get("native_actions"), strict_equal=result.get("strict_equal"))
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("completed", "native_actions", "strict_equal")}), flush=True)
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
