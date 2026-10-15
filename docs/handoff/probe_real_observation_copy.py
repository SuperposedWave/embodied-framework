"""Same-execution real RGB/state projection check, not an independent replay.

Observe exactly the raw observation already consumed by a Benchmark Adapter,
then compare every public image/state and provenance. No extra rendering or
model inference, no changed controller/score/timing, no relaxed tolerances.
This proves this execution's mapping, not historical trajectory reproduction.
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


def compare_snapshot(raw, public, benchmark):
    import numpy as np
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if benchmark == "droid_sim":
        values = raw["policy"]
        images = {name: array(values[name])[0] for name in ("external_cam", "external_cam_2", "wrist_cam")}
        states = {"arm_joint_positions": array(values["arm_joint_pos"]).reshape(-1).astype(np.float64),
            "gripper_open_fraction": 1. - array(values["gripper_pos"]).reshape(-1).astype(np.float64)}
        instruction = {1: "put the cube in the bowl", 2: "pick up the can and put it in the mug",
            3: "put the banana in the bin"}[1]
    else:
        images = {name: array(raw["vision"][name]["color"]) for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
        states = {}
        for side in ("left", "right"):
            original = np.asarray(raw["state"][side + "_ee_pose"], np.float64)
            states.update({side + "_joint_positions": np.asarray(raw["state"][side + "_arm_joint_state"], np.float64),
                side + "_eef_position": original[:3], side + "_eef_quaternion": original[[4, 5, 6, 3]],
                side + "_gripper_target": np.asarray(raw["state"][side + "_ee_joint_state"], np.float64)})
        instruction = str(raw["instruction"])
    captures, image_checks, state_checks = {}, {}, {}
    for name, original in images.items():
        sample = public.sensors[name]
        projected = np.asarray(sample.payload.data, dtype=sample.payload.dtype)
        captures["raw_" + name], captures["public_" + name] = original, projected
        image_checks[name] = {"native_shape": list(original.shape), "native_dtype": str(original.dtype),
            "public_shape": list(projected.shape), "public_dtype": str(projected.dtype),
            "pixel_values_and_types_exact": original.dtype == projected.dtype and np.array_equal(original, projected),
            "native_pixel_bytes_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
            "public_pixel_bytes_sha256": hashlib.sha256(projected.tobytes()).hexdigest(),
            "frame_timestamp_and_sensor_provenance_exact": sample.frame_id == name
                and sample.captured_at_s == public.sim_time_s and sample.provenance == "sensor"}
    for name, original in states.items():
        sample = public.states[name]
        projected = np.asarray(sample.payload.data, dtype=sample.payload.dtype)
        captures["raw_" + name], captures["public_" + name] = original, projected
        expected_provenance = "controller_target" if name.endswith("_target") else "measurement"
        state_checks[name] = {"native_values": original.tolist(), "public_values": projected.tolist(),
            "native_dtype": str(original.dtype), "public_dtype": str(projected.dtype),
            "values_and_types_exact": original.dtype == projected.dtype and np.array_equal(original, projected),
            "provenance": sample.provenance, "provenance_and_timestamp_exact":
                sample.provenance == expected_provenance and sample.captured_at_s == public.sim_time_s}
    all_keys = set(images) == set(public.sensors) and set(states) == set(public.states)
    passed = all_keys and instruction == public.public_instruction and all(
        row["pixel_values_and_types_exact"] and row["frame_timestamp_and_sensor_provenance_exact"]
        for row in image_checks.values()) and all(row["values_and_types_exact"] and row["provenance_and_timestamp_exact"]
        for row in state_checks.values())
    return captures, {"passed": bool(passed), "all_native_public_keys_accounted_for": all_keys,
        "instruction_exact": instruction == public.public_instruction, "images": image_checks, "states": state_checks,
        "env_step": public.env_step, "sim_time_s": public.sim_time_s}


def child(args):
    import numpy as np
    from dataclasses import replace
    from embodied_eval import contracts as leaf
    from embodied_eval.mn import contracts as c
    from embodied_eval.codec import decode, to_data
    from embodied_eval.mn.validation import validate_observation, validate_step
    from replay_mn_native_actions import trace
    manifest = json.loads((args.trial / "framework/manifest.json").read_text())
    if manifest["synthetic"] or manifest["config"]["benchmark"]["adapter"] != args.benchmark:
        raise ValueError("saved_real_trial_for_selected_benchmark_required")
    options = dict(manifest["config"]["benchmark"]["options"], device="cuda:0")
    if args.benchmark == "droid_sim" and options["scene"] != 1:
        raise ValueError("reviewed_native_scene_one_instruction_required")
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    receipt = {"kind": "same_execution_real_benchmark_projection_not_independent_native_replay", "passed": False,
        "synthetic": False, "benchmark_adapter_runtime_used": True, "model_adapter_runtime_used": False,
        "new_policy_inference": False, "additional_camera_capture_or_rendering": False,
        "raw_logs_saved": False, "formal_acceptance": False, "benchmark": args.benchmark, "snapshots": []}
    simulator = None
    try:
        if args.benchmark == "droid_sim":
            from embodied_eval.mn.droid_simulator import DroidSimulator as Simulator, verify
        else:
            from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator as Simulator, verify
        receipt["verified_assets_before_native_load"] = verify(options)
        receipt["adapter_source_sha256"] = hashlib.sha256(Path(sys.modules[Simulator.__module__].__file__).read_bytes()).hexdigest()
        simulator = Simulator(options, args.output)
        original = simulator._snapshot
        def save(raw, observation, calls):
            captures, check = compare_snapshot(raw, observation, args.benchmark)
            destination = args.output / f"snapshot-{len(receipt['snapshots']):04d}.npz"
            with destination.open("xb") as stream:
                np.savez_compressed(stream, **captures)
            check.update(raw_observations_consumed=calls, file=destination.name,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            check["passed"] = check["passed"] and calls == 1
            receipt["snapshots"].append(check)
            return observation
        if args.benchmark == "droid_sim":
            def observed(raw, context):
                return save(raw, original(raw, context), 1)
        else:
            def observed(context):
                captures = []
                original_get_obs = simulator.env.get_obs
                def record(*values, **keywords):
                    raw = original_get_obs(*values, **keywords)
                    captures.append(raw)
                    return raw
                simulator.env.get_obs = record
                try:
                    observation = original(context)
                finally:
                    simulator.env.get_obs = original_get_obs
                if len(captures) != 1:
                    raise ValueError("diagnostic_must_not_add_native_camera_capture")
                return save(captures[0], observation, len(captures))
        simulator._snapshot = observed
        ep = manifest["config"]["episodes"][0]
        from embodied_eval.mn.config import EpisodeConfig
        episode = decode(EpisodeConfig, ep)
        task = simulator.enumerate_tasks((episode,))[0]
        context = c.Context(run_id="same-execution-projection-diagnostic", env_id="native0", episode_id=episode.episode_id, message_id="reset")
        observation = simulator.reset(task, context)
        validate_observation(observation, simulator.describe())
        feedback = []
        for _, action, _ in trace(args.trial / "framework/episodes/ep-0/trace.internal.jsonl"):
            if action is None:
                continue
            request = replace(context, message_id=f"saved-action:{len(feedback)+1}")
            action = decode(c.Action, action)
            reply = simulator.advance(action, request, leaf.Budgets(env_steps=4-len(feedback)))
            validate_step(reply, observation, request, action, simulator.describe())
            observation = reply.observation
            feedback.append(to_data(reply.execution_feedback))
            if reply.terminated or reply.truncated or len(feedback) == 4:
                break
        receipt.update(actual_saved_actions=len(feedback), execution_feedback=feedback,
            benchmark_diagnostics=simulator.diagnostics(),
            passed=len(feedback) == 4 and len(receipt["snapshots"]) == 5
                and all(row["passed"] for row in receipt["snapshots"]))
        simulator.close()
        simulator = None
        receipt["native_close_returned"] = True
    except Exception as error:
        receipt["passed"] = False
        receipt["exception_chain"] = public_exception(error)
    finally:
        if simulator is not None:
            try:
                simulator.close()
                receipt["native_close_returned"] = True
            except Exception as error:
                receipt["close_exception_chain"] = public_exception(error)
    write(args.output / "component-result.json", receipt)
    # Native close has returned; avoid unsafe plugin interpreter finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if receipt["passed"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("droid_sim", "robodojo"), required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name in ("output", "trial", "isaac_libraries"):
        value = getattr(args, name).resolve()
        value.relative_to(ROOT / "outputs")
        setattr(args, name, value)
    if args.child:
        return child(args)
    if not args.accept_runtime_eula or not 0 <= args.gpu <= 7:
        raise ValueError("explicit_project_EULA_approval_and_local_GPU_required")
    args.output.mkdir(parents=True, exist_ok=False)
    library = json.loads((args.isaac_libraries / "receipt.json").read_text())
    if not library["passed"] or any(hashlib.sha256((args.isaac_libraries / item["file"]).read_bytes()).hexdigest() != item["sha256"] for item in library["libraries"]):
        raise ValueError("verified_private_native_libraries_required")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    directory = args.isaac_libraries / "library-root/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = str(directory) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    command = [str(WORKSPACE / "upstream/envs" / ("droid_sim-mn-native" if args.benchmark == "droid_sim" else "robodojo-mn-native") / "bin/python"),
        __file__, "--child", "--benchmark", args.benchmark, "--trial", str(args.trial), "--output", str(args.output),
        "--isaac-libraries", str(args.isaac_libraries), "--gpu", str(args.gpu)]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = process.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_owned(process)
        out, err = process.communicate()
    result = args.output / "component-result.json"
    receipt = {"kind": "supervised_real_same_execution_projection_not_historical_replay_pass",
        "at_utc": datetime.now(timezone.utc).isoformat(), "benchmark": args.benchmark, "gpu": args.gpu,
        "exit_code": process.returncode, "supervisor_timed_out": timed_out,
        "passed": not timed_out and process.returncode == 0 and result.is_file() and json.loads(result.read_text())["passed"],
        "public_log_categories": categories((out + err).decode(errors="replace")),
        "raw_logs_saved": False, "global_proxy_or_driver_changed": False, "occupancy_process_changed": False,
        "formal_acceptance": False, "historical_strict_replay_failures_retained": True}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
