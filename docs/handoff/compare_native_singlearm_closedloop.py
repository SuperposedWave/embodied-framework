"""Fresh fixed-source LIBERO/VLABench policy reference, no Adapter execution.

Use official policy clients/pure methods and direct native environment calls.
Record consumed RGB only. A fresh-policy mismatch is retained separately from
successful task completion and from saved-action mapping evidence.
"""
import argparse
import ast
import collections
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

from probe_mn_components import ROOT, write, public_exception
sys.path.insert(0, str(ROOT))


class NativePolicy:
    def __init__(self, config, output):
        import numpy as np
        self.config, self.output, self.np = config, output, np
        self.requests, self.cache = 0, collections.deque()
        self.profile = config["model"]["options"]["io_profile"]
        self.source = Path(config["model"]["options"]["repo"])
        port = config["model"]["options"]["port"]
        self.output.mkdir()
        sys.path.insert(0, str(self.source))
        if config["model"]["adapter"] == "openwam":
            from benchmarks.utils.transport import WSPolicyClient
            # Match the production no-resend rule, using the source's explicit
            # predict_once method. No changes to preprocessing or decoding.
            class SingleSendClient(WSPolicyClient):
                def predict(current, payload):
                    return current.predict_once(payload)
            self.client = SingleSendClient(f"ws://127.0.0.1:{port}")
            original = self.client.predict_once
            def observed(payload):
                if set(payload) != {"images", "state", "prompt"}:
                    raise ValueError("source_public_OpenWAM_payload_required")
                write(self.output / f"request-{self.requests + 1:04d}.json", payload)
                reply = original(payload)
                write(self.output / f"action-{self.requests + 1:04d}.json", {"action": np.asarray(reply["action"]).tolist()})
                self.requests += 1
                return reply
            self.client.predict_once = observed
            if self.profile == "eef10_osc":
                from benchmarks.libero.openwam2libero_interface import OpenWAMLiberoPolicy
                self.policy = OpenWAMLiberoPolicy(_client=self.client)
            else:
                from benchmarks.vlabench.openwam2vlabench_interface import OpenWAMVLABenchPolicy
                self.policy = OpenWAMVLABenchPolicy.__new__(OpenWAMVLABenchPolicy)
                # Inject transport without creating a spare connection; the
                # native constructor initializes constants and pings once.
                import benchmarks.vlabench.openwam2vlabench_interface as module
                original_factory = module.WSPolicyClient
                module.WSPolicyClient = lambda *a, **kw: self.client
                try:
                    self.policy.__init__(host="127.0.0.1", port=port)
                finally:
                    module.WSPolicyClient = original_factory
            self.policy.reset()
        else:
            sys.path.insert(0, str(self.source / "packages/openpi-client/src"))
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            from openpi_client import msgpack_numpy
            self.client = WebsocketClientPolicy("127.0.0.1", port)
            original = self.client.infer
            def observed(payload):
                expected = {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
                if self.profile == "state7_absolute":
                    expected.add("observation/second_image")
                if set(payload) != expected:
                    raise ValueError("source_public_openpi_payload_required")
                with (self.output / f"request-{self.requests + 1:04d}.msgpack").open("xb") as stream:
                    stream.write(msgpack_numpy.packb(payload))
                reply = original(payload)
                with (self.output / f"action-{self.requests + 1:04d}.msgpack").open("xb") as stream:
                    stream.write(msgpack_numpy.packb({"actions": reply["actions"]}))
                self.requests += 1
                return reply
            self.client.infer = observed
            if self.profile == "state7_absolute":
                from check_mn_extension_native_mappings import native_class
                from VLABench.utils.utils import quaternion_to_euler
                self.policy = native_class(self.source / "examples/vlabench/eval.py", "Pi0", ("__init__", "reset", "predict"),
                    {"np": np, "collections": collections, "quaternion_to_euler": quaternion_to_euler})
                self.policy.__init__(self.client, replan_steps=5)
                self.policy.reset()
            else:
                from check_mn_extension_native_mappings import functions
                from openpi_client import image_tools
                scope = {"np": np, "math": math}
                functions(self.source / "examples/libero/main.py", ("_quat2axisangle",), scope)
                self.quaternion = scope["_quat2axisangle"]
                self.images = image_tools

    def act_libero(self, raw, instruction):
        np = self.np
        if self.profile == "eef10_osc":
            # Benchmark publicly casts achieved state to float32 in this named
            # protocol. Preserve that boundary before native client conversion.
            obs = dict(raw)
            for name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                obs[name] = np.asarray(raw[name], np.float32)
            return self.policy.act(obs, instruction)
        if not self.cache:
            images = {key: self.images.convert_to_uint8(self.images.resize_with_pad(np.ascontiguousarray(raw[name][::-1, ::-1]), 224, 224))
                for name, key in (("agentview_image", "observation/image"), ("robot0_eye_in_hand_image", "observation/wrist_image"))}
            payload = {**images, "observation/state": np.concatenate((raw["robot0_eef_pos"], self.quaternion(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])), "prompt": instruction}
            actions = np.asarray(self.client.infer(payload)["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 5 or not np.isfinite(actions).all():
                raise ValueError("five_finite_native_OSC_actions_required")
            self.cache.extend(actions[:5])
        return self.cache.popleft()

    def act_vlabench(self, raw):
        return self.policy.predict(raw)

    def close(self):
        if self.config["model"]["adapter"] == "openwam":
            self.policy.close()
        else:
            self.client._ws.close()


def action_fields(action, native, family):
    import numpy as np
    from replay_mn_native_actions import field
    if action is None:
        return {"action": {"exact_equal": False}}
    if family == "libero":
        old = np.asarray(action["commands"][0]["parameters"]["osc_action7"], dtype=action["native_dtype"])
        return {"action": {**field(old, native), "dtype_equal": old.dtype == native.dtype}}
    commands = {item["kind"]: item for item in action["commands"]}
    pose, grip = commands["pose_target"], commands["gripper_command"]
    pos, euler, fingers = native
    return {"position": field(pose["pose"]["position_m"], pos),
        "euler": field(action["source_euler_xyz_rad"][pose["target_id"]], euler),
        "gripper_width": field(grip["value"], np.sum(fingers))}


def run(args):
    import numpy as np
    from replay_mn_native_actions import trace, snapshot, digest, field
    from comparison_video import Frames
    from embodied_eval.native.model_assets import verify_model
    from embodied_eval.native.benchmark_assets import verify_benchmark
    config = json.loads(args.config.read_text())
    manifest = json.loads((args.framework / "manifest.json").read_text())
    if manifest["synthetic"] or manifest["config"] != config or config["benchmark"]["adapter"] not in ("libero", "vlabench"):
        raise ValueError("matching_real_independent_0.2_framework_manifest_required")
    assets = {"model": verify_model(config["model"]["adapter"], config["model"]["options"]),
        "benchmark": verify_benchmark(config["benchmark"]["adapter"], config["benchmark"]["options"])}
    if assets != manifest["provenance"]["verified_assets"]:
        raise ValueError("matching_component_asset_fingerprints_required")
    episode = args.framework / "episodes" / config["episodes"][0]["episode_id"]
    result = json.loads((episode / "result.json").read_text())
    old = iter(trace(episode / "trace.internal.jsonl"))
    reference = next(old)
    report = {"kind": "fresh_same_condition_fixed_source_singlearm_closedloop", "completed": False, "strict_equal": False,
        "synthetic": False, "fresh_native_policy_inference": True, "recorded_actions_control_native_environment": False,
        "Adapter_or_Runner_execution_called": False, "formal_acceptance": False,
        "setup_rng_seed_both_sides": args.setup_seed, "verified_assets": assets,
        "framework_manifest_sha256": digest(args.framework / "manifest.json"), "rows": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    env = policy = frames = temporary = None
    try:
        random.seed(args.setup_seed); np.random.seed(args.setup_seed)
        family = config["benchmark"]["adapter"]
        options = config["benchmark"]["options"]
        repo = Path(options["repo"])
        sys.path.insert(0, str(repo))
        if family == "libero":
            import torch
            import yaml
            package = repo / "libero/libero"
            temporary = tempfile.TemporaryDirectory(prefix="native-comparison-libero-")
            paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"), "init_states": str(package / "init_files"),
                "datasets": str(package.parent.parent / "datasets"), "assets": str(package / "assets")}
            with (Path(temporary.name) / "config.yaml").open("x") as stream:
                yaml.safe_dump(paths, stream)
            os.environ["LIBERO_CONFIG_PATH"] = temporary.name
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            seed = config["episodes"][0]["seed"]
            np.random.seed(seed)
            task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
            initial = torch.load(package / "init_files" / task.problem_folder / task.init_states_file, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(package / "bddl_files" / task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
            env.seed(seed); env.reset()
            raw = env.set_init_state(initial[0])
            settle = 30 if options["protocol"] == "spatial-30settle" else 10
            for _ in range(settle):
                raw, _, _, _ = env.step([0.] * 6 + [-1.])
            instruction = str(task.language)
            cameras = ("agentview_image", "robot0_eye_in_hand_image")
            period = .05
            report.update(native_settle_steps=settle, native_initial_state_index=0)
        else:
            os.environ["VLABENCH_ROOT"] = str(repo / "VLABench")
            import VLABench.robots
            import VLABench.tasks
            from VLABench.envs import load_env
            from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler
            random.seed(args.setup_seed); np.random.seed(args.setup_seed)
            frozen = json.loads((repo / "VLABench/configs/evaluation/tracks" / (options["track"] + ".json")).read_text())[options["task_name"]][options["trial_index"]]
            env = load_env(options["task_name"], episode_config=copy.deepcopy(frozen), random_init=False, eval=False, run_mode="eval")
            env.reset()
            raw = env.get_observation(require_pcd=False)
            instruction = env.task.get_instruction()
            cameras = ("forward_rgb", "franka_wrist_rgb", "right_rgb")
            period = float(env.control_timestep())
            report["frozen_track_scene_double_reset_and_native_one_substep"] = True
        if instruction != manifest["tasks"][0]["public"]["instruction"]:
            raise ValueError("initial_public_instruction_not_same")
        frames = Frames(args.output / "frames", cameras)
        policy = NativePolicy(config, args.output / "native-exchanges")
        steps, clock, done, last_action = 0, 0., False, None
        def save(ref, native_action=None, elapsed=None):
            if family == "libero":
                state_dtype = np.float32 if options["protocol"] == "spatial-30settle" else np.float64
                values = {"eef_position": np.asarray(raw["robot0_eef_pos"], state_dtype), "eef_quaternion": np.asarray(raw["robot0_eef_quat"], state_dtype),
                    "gripper_positions": np.asarray(raw["robot0_gripper_qpos"], state_dtype)}
                images = {name: raw[name] for name in cameras}
            else:
                ee = np.asarray(raw["ee_state"], np.float64)
                values = {"eef_position": ee[:3], "eef_quaternion": ee[[4, 5, 6, 3]], "gripper_open_fraction": [1. - ee[-1]], "robot_base_position": env.get_robot_frame_position()}
                images = {name: raw["rgb"][index] for name, index in zip(cameras, (2, 3, 0))}
            frames.append(images, steps, clock)
            fields = snapshot(ref[0], values, images) if ref else {}
            if native_action is not None:
                fields.update(action_fields(ref[1] if ref else None, native_action, family))
            if ref is not None:
                fields["simulation_clock"] = {"exact_equal": math.isclose(clock, ref[0]["sim_time_s"], rel_tol=0, abs_tol=1e-8),
                    "max_abs_diff": abs(clock - ref[0]["sim_time_s"]), "source_clock_rounding_tolerance_s": 1e-8}
                if steps:
                    fields["native_success"] = field(bool(done), ref[2]["terminated"])
                if family == "vlabench":
                    fields["dynamic_instruction"] = {"exact_equal": instruction == ref[0]["public_instruction"]}
            report["rows"].append({"env_step": steps, "native_sim_time_s": clock, "native_elapsed_s": elapsed,
                "reference_available": ref is not None, "fields": fields})
        save(reference)
        started = __import__("time").monotonic()
        cap = config["episodes"][0]["budgets"]["env_steps"]
        wall = config["episodes"][0]["budgets"]["wall_time_s"]
        while steps < cap and not done:
            if wall is not None and __import__("time").monotonic() - started >= wall:
                report["wall_budget_reached"] = True
                break
            if family == "libero":
                action = policy.act_libero(raw, instruction)
                raw, _, done, _ = env.step(action)
                elapsed = period
            else:
                if last_action is None:
                    last_action = np.concatenate([raw["ee_state"][:3], quaternion_to_euler(raw["ee_state"][3:7])])
                supplied = dict(raw, robot_frame=env.get_robot_frame_position(), instruction=instruction, last_action=last_action)
                pos, euler, fingers = policy.act_vlabench(supplied)
                action = (pos, euler, fingers)
                last_action = np.concatenate((pos, euler))
                _, joints = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=euler_to_quaternion(*euler))
                before = float(env.physics.time())
                native = env.step(np.concatenate([joints, fingers]))
                elapsed = float(env.physics.time()) - before
                done = bool(native.last())
                raw = env.get_observation(require_pcd=False)
                if not done:
                    instruction = env.task.get_instruction()
            steps += 1; clock += elapsed
            reference = next(old, None)
            save(reference, action, elapsed)
            if steps % 25 == 0:
                print(f"independent native {family}: {steps} actions", flush=True)
        same = lambda row: row["reference_available"] and all(value["exact_equal"] and value.get("dtype_equal", True) for value in row["fields"].values())
        metrics = {"success": bool(done)}
        if family == "vlabench":
            metrics["progress_score"] = float(env.get_task_progress())
            try:
                metrics["intention_score"] = float(env.get_intention_score(threshold=.1))
            except KeyError:
                metrics["intention_score"] = None
            if metrics["intention_score"] is not None and not math.isfinite(metrics["intention_score"]):
                metrics["intention_score"] = None
        report.update(completed=True, native_actions=steps, native_requests=policy.requests, native_sim_time_s=clock,
            native_metrics=metrics, framework_metrics=result["metrics"], action_cap=cap, cooperative_wall_budget_s=wall,
            initial_observation_exact=same(report["rows"][0]), first_divergence=next((row["env_step"] for row in report["rows"] if not same(row)), None),
            strict_equal=steps == result["execution_statistics"]["env_steps"] and next(old, None) is None and all(same(row) for row in report["rows"]),
            additional_video_get_obs_render_or_simulation_calls=0)
    except Exception as error:
        report["exception_chain"] = public_exception(error)
    finally:
        if frames is not None and frames.rows:
            frames.close()
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
        if temporary is not None:
            temporary.cleanup()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / "native-result.json", report)
    print(json.dumps({key: report.get(key) for key in ("completed", "strict_equal", "native_actions", "native_metrics", "initial_observation_exact", "first_divergence")}), flush=True)
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
