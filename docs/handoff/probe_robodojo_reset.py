"""Supervised native stack_bowls reset/holding diagnostic, not policy scoring.

Keeps native logs in memory, uses project-only EULA acceptance, and preserves
reset evidence even if native application close exits early or crashes.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
from probe_mn_components import write, public_exception, categories
from prepare_robodojo_environment import bounded


def child(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    record = {"kind": "native_robodojo_reset_hold_not_model_inference_or_task_acceptance", "passed": False,
        "simulator_started": False, "policy_inference_performed": False, "official_score": None}
    benchmark = None
    try:
        from prepare_mn_extension import configurations
        from embodied_eval.native.robodojo_simulator import RoboDojoProcessSimulator, verify
        from embodied_eval.native import contracts as c
        from embodied_eval.core import types as leaf
        from embodied_eval.codec import to_data
        from tests.legacy_support.frozen.mn.validation import validate_observation, validate_step
        cfg = configurations()["pi05_arx_robodojo"]
        options = dict(cfg.benchmark.options, device="cuda:0")
        record["verified_native_assets"] = verify(options)
        benchmark = RoboDojoProcessSimulator(options, args.output)
        task = benchmark.enumerate_tasks(cfg.episodes)[0]
        context = c.Context(run_id="robodojo-reset-hold-diagnostic", env_id="native0", episode_id=task.episode_id, message_id="reset0")
        observation = benchmark.reset(task, context)
        record["simulator_started"] = True
        validate_observation(observation, benchmark.describe())
        record.update(native_reset_passed=True, observation_synthetic=False,
            measured_or_target_states={name: {"values": sample.payload.data, "provenance": sample.provenance}
                for name, sample in observation.states.items()},
            camera_shapes={name: list(sample.payload.shape) for name, sample in observation.sensors.items()},
            benchmark_diagnostics=benchmark.diagnostics())
        feedback = []
        for index in range(args.hold_steps):
            request = replace(context, message_id=f"hold:{index}")
            commands = []
            for side in ("left", "right"):
                commands.extend(leaf.JointTarget(command_id=f"hold:{index}:{side}:{i}", target_id=f"robot0/{side}_j{i}",
                    mode="position", unit="rad", value=value) for i, value in enumerate(observation.states[side + "_joint_positions"].payload.data))
                commands.append(leaf.GripperCommand(command_id=f"hold:{index}:{side}:grip", target_id=f"robot0/{side}_grip",
                    mode="open_fraction", value=observation.states[side + "_gripper_target"].payload.data[0]))
            action = c.Action(commands=tuple(commands), native_dtype="float64")
            reply = benchmark.advance(action, request, leaf.Budgets(env_steps=args.hold_steps-index))
            validate_step(reply, observation, request, action, benchmark.describe())
            feedback.append(to_data(reply.execution_feedback))
            observation = reply.observation
            record.update(execution_feedback=feedback, benchmark_diagnostics=benchmark.diagnostics(),
                native_terminated=reply.terminated, native_truncated=reply.truncated)
            # The diagnostic must obey the same native episode boundary as
            # Runner, not request another action from an already ended scene.
            if reply.terminated or reply.truncated:
                break
        record.update(execution_feedback=feedback, benchmark_diagnostics=benchmark.diagnostics())
        write(args.output / "validated-native-reset.json", dict(record, passed=True))
        benchmark.close()
        benchmark = None
        record["passed"] = True
    except Exception as error:
        record["exception_chain"] = public_exception(error)
    finally:
        # Capture a failure BEFORE a native close can exit this process.
        if not record["passed"]:
            write(args.output / "failure-before-native-close.json", record)
        if benchmark is not None:
            try:
                benchmark.close()
            except Exception as error:
                record["close_exception_chain"] = public_exception(error)
    write(args.output / "component-result.json", record)
    return 0 if record["passed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaac-libraries", type=Path, required=True)
    parser.add_argument("--accept-runtime-eula", action="store_true")
    parser.add_argument("--hold-steps", type=int, default=0)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.child:
        return child(args)
    if not args.accept_runtime_eula or not 0 <= args.hold_steps <= 2:
        raise ValueError("project-only EULA approval and bounded zero-to-two hold actions required")
    args.output.mkdir(parents=True, exist_ok=False)
    preparation = args.isaac_libraries.resolve()
    preparation.relative_to(ROOT / "outputs")
    libraries = json.loads((preparation / "receipt.json").read_text())
    if not libraries["passed"]:
        raise ValueError("private shared-library preparation required")
    for row in libraries["libraries"]:
        if hashlib.sha256((preparation / row["file"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("private shared library changed")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMNI_KIT_ACCEPT_EULA="YES", PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    directory = preparation / "library-root/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = str(directory) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    command = [str(WORKSPACE / "upstream/envs/robodojo-mn-native/bin/python"), __file__, "--child", "--output", str(args.output),
        "--isaac-libraries", str(preparation), "--hold-steps", str(args.hold_steps)]
    code, stdout, stderr = bounded(command, env, 600)
    result = args.output / "component-result.json"
    record = {"kind": "supervised_robodojo_native_reset_not_acceptance", "exit_code": code,
        "passed": code == 0 and result.is_file() and json.loads(result.read_text())["passed"],
        "public_log_categories": categories((stdout + stderr).decode(errors="replace")),
        "raw_logs_saved": False, "shared_driver_modified": False, "global_proxy_modified": False,
        "runtime_eula_project_process_only": True, "gpu": 0, "hold_actions_requested": args.hold_steps,
        "validated_reset_checkpoint_exists": (args.output / "validated-native-reset.json").is_file(),
        "formal_acceptance": False}
    write(args.output / "receipt.json", record)
    print(json.dumps(record), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
