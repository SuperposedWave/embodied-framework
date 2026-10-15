"""Fresh policy actions after the independently working native source reset.

Native factory/controller/scorer run directly; saved actions are comparison
only. This is not an old pair Adapter or a recorded-action policy replay.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories


def child(args, initialized=None):
    def phase(name):
        with (args.output / "native-lifecycle.jsonl").open("a") as stream:
            stream.write(json.dumps({"phase": name, "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    phase("before_independent_reference_startup")
    manifest = json.loads((args.trial / "framework/manifest.json").read_text())
    config = manifest["config"]
    source = Path(config["model"]["options"]["repo"])
    is_wam = config["model"]["adapter"] == "openwam"
    app = env = frames = worker = connection = client = None
    report = {"kind": "fresh_native_component_RoboDojo_policy_closedloop", "completed": False,
        "strict_equal": False, "synthetic": False, "formal_acceptance": False, "close_completed": False,
        "fresh_native_policy_inference": False, "recorded_actions_control_native_environment": False,
        "production_Model_or_BenchmarkAdapter_instantiated": False, "rows": [], "model_gpu": args.model_gpu,
        "physical_render_gpu": args.gpu, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    logs = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            from native_robodojo_reference_startup import initialize
            app, env, proof, reset_calls, robot_evidence = initialized if initialized is not None else initialize(args)
            phase("independent_native_reset_returned")
            report.update(native_factory_created=True, startup_reference_proof=proof, robot_path_evidence=robot_evidence,
                verified_model_assets=json.loads((args.output / "model-assets-verified.json").read_text())["verification"])
            import numpy as np
            from comparison_video import Frames
            from native_array_wire import pack, unpack
            from replay_mn_native_actions import trace, snapshot, field
            from replay_robodojo_native_actions import native_action
            from check_mn_extension_native_mappings import functions
            from omegaconf import OmegaConf
            env.run_reward(); env.get_score()
            if env.episode_nums != 1 or env.unstable_envs or not env.reward_manager.check_list[0]:
                raise ValueError("native_layout_and_registered_checks_required")
            report.update(resolved_native_simulation_config=OmegaConf.to_container(env.sim_config, resolve=True),
                native_reward_registration={"groups": len(env.reward_manager.check_list[0]), "stages": len(env.reward_manager.score_list[0])})
            counted = [0]
            original_step = env.sim.unwrapped.sim.step
            def counted_step(*values, **keywords):
                value = original_step(*values, **keywords); counted[0] += 1; return value
            env.sim.unwrapped.sim.step = counted_step
            episode = args.trial / "framework/episodes/ep-0"
            result = json.loads((episode / "result.json").read_text())
            saved = iter(trace(episode / "trace.internal.jsonl")); ref = next(saved)
            raw = env.get_obs()
            cameras = tuple(ref[0]["sensors"])
            frames = Frames(args.output / "frames", cameras)
            cache, requests, steps, total = deque(), 0, 0, 0
            def save(reference, action=None, consumed=None):
                values = {}
                for side in ("left", "right"):
                    pose = np.asarray(raw["state"][side + "_ee_pose"], np.float64)
                    values.update({side + "_joint_positions": np.asarray(raw["state"][side + "_arm_joint_state"], np.float64),
                        side + "_eef_position": pose[:3], side + "_eef_quaternion": pose[[4, 5, 6, 3]],
                        side + "_gripper_target": np.asarray(raw["state"][side + "_ee_joint_state"], np.float64)})
                images = {name: raw["vision"][name]["color"] for name in cameras}
                clock = total * env.dt; frames.append(images, steps, clock)
                fields = snapshot(reference[0], values, images) if reference else {}
                if reference:
                    fields["simulation_time"] = field(clock, reference[0]["sim_time_s"])
                    fields["public_instruction"] = {"exact_equal": str(raw["instruction"]) == reference[0]["public_instruction"]}
                    if action is not None:
                        recorded = native_action(reference[1])
                        fields["action"] = {"exact_equal": set(recorded) == set(action) and all(recorded[k].dtype == action[k].dtype and np.array_equal(recorded[k], action[k]) for k in recorded)}
                        fields["physics_delta"] = field(consumed, reference[2]["execution_feedback"]["consumed_physics_steps"])
                        fields["terminated"] = field(bool(env.end_flag[0] and env.success[0]), reference[2]["terminated"])
                report["rows"].append({"env_step": steps, "reference_available": reference is not None,
                    "native_sim_time_s": clock, "native_physics_delta": consumed, "fields": fields,
                    "native_action": None if action is None else {k: {"values": v.tolist(), "dtype": str(v.dtype)} for k, v in action.items()}})
            save(ref); phase("initial_consumed_frame_recorded")
            exchanges = args.output / "native-exchanges"; exchanges.mkdir()
            if is_wam:
                from multiprocessing import Pipe
                connection, inherited = Pipe(duplex=True)
                worker = subprocess.Popen([str(WORKSPACE / "upstream/envs/openwam_arx-mn-native/bin/python"),
                    str(ROOT / "docs/handoff/native_openwam_arx_worker.py"), "--pipe-fd", str(inherited.fileno()), "--output", str(exchanges)],
                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.model_gpu)), pass_fds=(inherited.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                inherited.close(); logs = pool.submit(worker.communicate)
                connection.send({k: v for k, v in config["model"]["options"].items() if k not in ("runtime_python", "runtime_gpu")})
                if not connection.poll(900) or not connection.recv().get("ok"):
                    raise RuntimeError("native_model_loading_failed")
                def infer(observation):
                    data = {"vision": {name: {"color": pack(observation["vision"][name]["color"])} for name in cameras},
                        "state": {side + key: pack(observation["state"][side + key]) for side in ("left", "right") for key in ("_ee_pose", "_ee_joint_state")}, "instruction": observation["instruction"]}
                    connection.send(("infer", data))
                    if not connection.poll(900): raise TimeoutError("diagnostic_native_generation_bound")
                    reply = connection.recv()
                    if not reply.get("ok"): raise RuntimeError("native_model_generation_failed")
                    return [{k: unpack(v) for k, v in row.items()} for row in reply["actions"]]
            else:
                scope = {"np": np, "Any": Any}
                functions(source / "utils/process_data.py", ("_validate_config", "_get_state_keys", "_ensure_valid_state_array", "pack_robot_state", "unpack_robot_state"), scope)
                functions(source / "policy/Pi_05/model.py", ("encode_obs", "extract_image", "ensure_chw_uint8"), scope)
                sys.path.insert(0, str(source / "policy/Pi_05/openpi/packages/openpi-client/src"))
                from openpi_client.websocket_client_policy import WebsocketClientPolicy
                from openpi_client import msgpack_numpy
                client = WebsocketClientPolicy("127.0.0.1", args.port)
                def infer(observation):
                    payload = scope["encode_obs"](observation, "joint", {"arm_dim": [6, 6], "ee_dim": [1, 1]})
                    with (exchanges / f"request-{requests+1:04d}.msgpack").open("xb") as stream: stream.write(msgpack_numpy.packb(payload))
                    chunk = np.asarray(client.infer(payload)["actions"])
                    if chunk.shape != (50, 14) or not np.isfinite(chunk).all(): raise ValueError("finite_native_ARX_50_by_14_required")
                    with (exchanges / f"actions-{requests+1:04d}.msgpack").open("xb") as stream: stream.write(msgpack_numpy.packb({"actions": chunk}))
                    return scope["unpack_robot_state"](chunk, "joint", {"arm_dim": [6, 6], "ee_dim": [1, 1]}, source_type="obs")
            phase("native_policy_ready")
            budget = config["episodes"][0]["budgets"]; started = time.monotonic()
            while steps < budget["env_steps"] and not env.end_flag[0]:
                if budget["wall_time_s"] is not None and time.monotonic() - started >= budget["wall_time_s"]:
                    report["cooperative_wall_budget_reached"] = True; break
                if not cache: cache.extend(infer(raw)); requests += 1
                action = cache.popleft(); before, before_actions = counted[0], env.take_action_cnt[0]
                env.take_action(action); delta = counted[0] - before
                if env.take_action_cnt[0] != before_actions + 1: raise ValueError("native_action_counter_mismatch")
                total += delta; steps += 1; raw = env.get_obs(); save(next(saved, None), action, delta)
                if steps % 25 == 0: print(f"fresh native RoboDojo: {steps} actions", flush=True)
            same = lambda r: r["reference_available"] and all(v["exact_equal"] for v in r["fields"].values())
            success = bool(env.end_flag[0] and env.success[0])
            report.update(completed=True, fresh_native_policy_inference=requests > 0, native_actions=steps, native_requests=requests,
                native_physics_steps=total, native_sim_time_s=total * env.dt, native_metrics={"success": success, "score": 100. if success else float(env.reward_manager.get_score()[0])},
                initial_observation_exact=same(report["rows"][0]), first_divergence=next((r["env_step"] for r in report["rows"] if not same(r)), None),
                strict_equal=steps == result["execution_statistics"]["env_steps"] and next(saved, None) is None and all(same(r) for r in report["rows"]),
                native_reset_calls=reset_calls, action_cap=budget["env_steps"], cooperative_wall_budget_s=budget["wall_time_s"], additional_video_get_obs_render_or_physics_calls=0)
            if connection is not None:
                connection.send(("close", None))
                if not connection.poll(30) or not connection.recv().get("ok"): raise RuntimeError("native_model_close_ack_missing")
                worker.wait(timeout=30)
                if worker.returncode != 0: raise RuntimeError("native_model_close_exit_nonzero")
                report["native_model_close_acknowledged"] = True
        except Exception as error:
            report["completed"] = False; report["exception_chain"] = public_exception(error)
        finally:
            if frames is not None and frames.rows: frames.close()
            if client is not None and getattr(client, "_ws", None) is not None: client._ws.close()
            if connection is not None: connection.close()
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try: worker.wait(timeout=10)
                except subprocess.TimeoutExpired: worker.kill(); worker.wait(timeout=10)
            if logs is not None:
                out, err = logs.result(timeout=15); report["native_model_public_failure_categories"] = categories((out+err).decode(errors="replace"))
            try:
                if env is not None: env.close()
                if app is not None: app.close()
                report["close_completed"] = app is not None
            except Exception as error: report["close_exception_chain"] = public_exception(error)
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat(); phase("native_cleanup_finished")
            write(args.output / "native-result.json", report)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0 if report["completed"] and report["close_completed"] else 1)
