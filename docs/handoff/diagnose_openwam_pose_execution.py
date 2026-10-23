"""Four saved real actions: independent native vs Benchmark execution audit.

No model server or fresh inference. Observe returned plans, exact low-level
commands and each completed PhysX step. Original controller/scoring/source and
physics parameters remain unchanged. Own child logs remain memory-only.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from itertools import islice
import json
import os
from pathlib import Path
import resource
import sys

from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
sys.path.insert(0, str(ROOT))


def child(args):
    from importlib import metadata
    import numpy as np
    from embodied_eval.codec import decode, to_data
    from embodied_eval.core import types as leaf
    from embodied_eval.native import contracts as c
    from tests.legacy_support.frozen.mn.config import RunConfig, validate_config
    from embodied_eval.native.registry import default_registry
    from robotwin_execution_observer import ExecutionObserver, articulation
    import replay_mn_native_actions as replay
    source = args.framework.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    config = validate_config(decode(RunConfig, manifest["config"]))
    if (manifest["synthetic"] or config.benchmark.adapter != "robotwin"
            or config.model.options["io_profile"] != "eef20_dual_wxyz_encoded_absolute"
            or config.episodes[0].budgets.env_steps != 4):
        raise ValueError("saved_real_four_action_physical_dual_pose_trial_required")
    verified = default_registry().benchmark("robotwin").verify_assets(config.benchmark.options)
    if verified != manifest["provenance"]["verified_assets"]["benchmark"]:
        raise ValueError("matching_verified_benchmark_assets_required")
    episode = source / "episodes" / config.episodes[0].episode_id
    prefix = list(islice(replay.trace(episode / "trace.internal.jsonl"), 5))
    if len(prefix) != 5 or prefix[0][1] is not None or any(row[1] is None for row in prefix[1:]):
        raise ValueError("four_completed_actions_required")
    plan_record = json.loads((args.plan_source / "component-result.json").read_text()) if args.plan_source else None
    if plan_record is not None and (not plan_record["completed"] or plan_record["manifest_sha256"] != replay.digest(source / "manifest.json")
            or plan_record["mode"] != "native" or len(plan_record["actions"]) != 4 or args.no_observer):
        raise ValueError("matching_completed_native_plan_record_required")
    observer = ExecutionObserver(plan_record["actions"] if plan_record is not None else None)
    benchmark = None
    result = {"kind": "saved_real_four_action_openwam_pose_execution_audit", "mode": args.mode,
        "synthetic": False, "fresh_model_inference": False, "formal_acceptance": False, "completed": False,
        "source_assets_verified": verified, "manifest_sha256": replay.digest(source / "manifest.json"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "observer_enabled": not args.no_observer,
        "planner_seed_or_settings_changed": False, "physics_or_controller_or_scoring_changed": False,
        "observer_adds_render_random_draws_or_physics_steps": False,
        "observer_reads_CPU_CUDA_results_and_may_change_wall_clock_timing": not args.no_observer,
        "raw_native_logs_or_environment_saved": False,
        "planning_mode": "saved_native_plan_controlled_diagnostic_not_evaluation" if plan_record is not None else "unaltered_native_planner",
        "real_native_planner_called_for_four_actions": plan_record is None,
        "plan_source_sha256": replay.digest(args.plan_source / "component-result.json") if args.plan_source else None}
    source_files = [Path(__file__), ROOT / "docs/handoff/replay_mn_native_actions.py", ROOT / "docs/handoff/robotwin_execution_observer.py",
                    ROOT / "docs/handoff/compare_native_robotwin_closedloop.py"]
    result["diagnostic_source_sha256_at_start"] = {str(path.relative_to(ROOT)): replay.digest(path) for path in source_files}
    try:
        if args.mode == "native":
            expected = json.loads((episode / "result.json").read_text())
            result["comparison_with_historical_framework"] = replay.replay_robotwin(config, episode, expected, args.output,
                observer=None if args.no_observer else observer)
        else:
            from contextlib import nullcontext
            from embodied_eval.native.native_benchmarks import NativeSimulator
            benchmark = NativeSimulator("robotwin", config.benchmark.options, args.output / "diagnostic")
            benchmark.describe()
            task = benchmark.enumerate_tasks(config.episodes)[0]
            historical_task = json.loads((episode / "task.internal.json").read_text())
            if to_data(task) != historical_task:
                raise ValueError("prepared_public_and_private_task_identity_differs")
            context = c.Context(run_id="pose-execution-audit", env_id="env0", episode_id=task.episode_id, message_id="reset")
            current = benchmark.reset(task, context)
            rows = []
            def compare(observation, old, feedback=None):
                fields = replay.snapshot(old, {name: sample.payload.data for name, sample in observation.states.items()},
                    {name: sample.payload.data for name, sample in observation.sensors.items()})
                fields["sim_time_s"] = replay.field(old["sim_time_s"], observation.sim_time_s)
                if feedback:
                    fields["physics_steps"] = replay.field(feedback["consumed_physics_steps"], reply.execution_feedback.consumed_physics_steps)
                rows.append({"env_step": observation.env_step, "fields": fields})
            compare(current, prefix[0][0])
            with observer.watch(benchmark.runtime.env) if not args.no_observer else nullcontext():
                for index, (old, recorded, old_reply) in enumerate(prefix[1:], 1):
                    if not args.no_observer:
                        observer.begin_action(index)
                    action = decode(c.Action, recorded)
                    native_row, mode = replay.robotwin_action_row(recorded)
                    mapped, mapped_mode = benchmark.robotwin_native_action(action)
                    if mode != mapped_mode or native_row.dtype != mapped.dtype or native_row.tobytes() != mapped.tobytes():
                        raise ValueError("independent_native_row_and_Benchmark_mapping_differ")
                    reply = benchmark.advance(action, replace(context, message_id=f"action{index}"),
                        leaf.Budgets(env_steps=5-index, model_calls=5-index))
                    if reply.execution_feedback.status != "executed":
                        raise ValueError("recorded_native_pose_execution_not_complete")
                    compare(reply.observation, old, old_reply["execution_feedback"])
            result["comparison_with_historical_framework"] = {"rows": rows,
                "native_physx_steps": benchmark.runtime.executor.physics_steps,
                "first_divergence": next((row["env_step"] for row in rows if not all(v["exact_equal"] for v in row["fields"].values())), None)}
            result["final_articulation"] = articulation(benchmark.runtime.env.robot)
            result["benchmark_diagnostics"] = benchmark.diagnostics()
        result.update(completed=True, actions=observer.actions,
            package_versions={name: metadata.version(name) for name in ("torch", "numpy", "sapien", "warp-lang")})
    except Exception as error:
        result.update(exception_chain=public_exception(error), actions=observer.actions)
    finally:
        if benchmark is not None:
            try:
                benchmark.close()
            except Exception as error:
                result.update(completed=False, close_exception_chain=public_exception(error))
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["diagnostic_source_unchanged_during_run"] = all(replay.digest(path) == result["diagnostic_source_sha256_at_start"][str(path.relative_to(ROOT))] for path in source_files)
        if not result["diagnostic_source_unchanged_during_run"]:
            result["completed"] = False
        write(args.output / "component-result.json", result)
    return 0 if result["completed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("native", "adapter"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-observer", action="store_true", help="Paired control: same saved actions, no added planner/physics observers")
    parser.add_argument("--plan-source", type=Path, help="Controlled diagnostic only: replay returned native plans; not a fresh-planning evaluation")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.framework, args.output = args.framework.resolve(), args.output.resolve()
    if args.plan_source:
        args.plan_source = args.plan_source.resolve()
        args.plan_source.relative_to(ROOT / "outputs")
    if args.child:
        try:
            return child(args)
        except Exception as error:
            write(args.output / "early-child-failure.json", {"completed": False, "exception_chain": public_exception(error)})
            return 1
    from prepare_robodojo_environment import bounded
    args.framework.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    if not 0 <= args.gpu <= 7:
        raise ValueError("available_gpu_ordinal_required")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"))
    command = [str(WORKSPACE / "upstream/envs/robotwin-stage4/bin/python"), __file__, "--child", "--framework", str(args.framework),
        "--output", str(args.output), "--mode", args.mode, "--gpu", str(args.gpu)]
    if args.no_observer:
        command.append("--no-observer")
    if args.plan_source:
        command.extend(("--plan-source", str(args.plan_source)))
    code, out, err = bounded(command, environment, 1200)
    path = args.output / "component-result.json"
    receipt = {"kind": "supervised_saved_four_action_execution_diagnostic", "mode": args.mode,
        "exit_code": code, "completed": code == 0 and path.is_file() and json.loads(path.read_text())["completed"],
        "gpu": args.gpu, "observer_enabled": not args.no_observer,
        "controlled_saved_plan_replay_not_fresh_planning": args.plan_source is not None,
        "public_log_categories": categories((out + err).decode(errors="replace")), "raw_logs_saved": False,
        "formal_acceptance": False, "fresh_model_inference": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
