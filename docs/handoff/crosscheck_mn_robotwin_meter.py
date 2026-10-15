"""Observe PhysX and production execution feedback during the SAME 0.2 run.

This diagnostic has no Model Adapter or fresh inference. Recorded actions drive
the independent Benchmark Adapter. A counter pass does NOT repair a failed
cross-process trajectory replay, or replace its historical physics count.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    target=args.output.resolve(); target.mkdir(parents=True,exist_ok=False)
    report={"kind":"mn_simultaneous_production_physx_counter_crosscheck_not_trajectory_acceptance",
        "at_utc":datetime.now(timezone.utc).isoformat(),"synthetic":False,"fresh_model_inference":False,
        "historical_trajectory_reproduced":False,"formal_acceptance":False,"passed":False}
    benchmark=None
    try:
        from embodied_eval.codec import decode
        from embodied_eval import contracts as leaf
        from embodied_eval.mn import contracts as c
        from embodied_eval.mn.config import RunConfig,validate_config
        from embodied_eval.mn.registry import default_registry
        from embodied_eval.mn.native_benchmarks import NativeSimulator
        from replay_mn_native_actions import trace,digest
        from replay_robotwin_native_actions import observe_physx
        root=args.framework.resolve()
        manifest=json.loads((root/"manifest.json").read_text())
        config=validate_config(decode(RunConfig,manifest["config"]))
        if manifest["synthetic"] or config.benchmark.adapter!="robotwin":
            raise ValueError("real_0.2_robotwin_manifest_required")
        default_registry().benchmark("robotwin").verify_assets(config.benchmark.options)
        benchmark=NativeSimulator("robotwin",config.benchmark.options,target/"diagnostic")
        space=benchmark.describe()
        task=benchmark.enumerate_tasks(config.episodes)[0]
        context=c.Context(run_id="mn-meter-check",env_id="env0",episode_id=task.episode_id,message_id="start")
        benchmark.reset(task,context)
        folder=root/"episodes"/task.episode_id
        previous=json.loads((folder/"result.json").read_text())
        rows=[]
        with observe_physx(benchmark.runtime.env.scene.physx_system) as counter:
            for _,recorded,_ in trace(folder/"trace.internal.jsonl"):
                if recorded is None:
                    continue
                request=replace(context,message_id=f"request:{len(rows)+1}")
                action=decode(c.Action,recorded)
                before=counter["completed"]
                reply=benchmark.advance(action,request,leaf.Budgets(env_steps=task.budgets.env_steps-len(rows)))
                actual=counter["completed"]-before
                feedback=reply.execution_feedback
                rows.append({"action":len(rows)+1,"independent_completed_physx_steps":actual,
                    "production_completed_physics_steps":feedback.consumed_physics_steps,
                    "elapsed_sim_time_s":feedback.elapsed_sim_time_s,
                    "counts_equal":actual==feedback.consumed_physics_steps,
                    "time_equal":math.isclose(actual*space.physics_period_s,feedback.elapsed_sim_time_s,abs_tol=1e-12),
                    "status":feedback.status})
                if reply.terminated or reply.truncated or feedback.status!="executed":
                    break
            report.update(rows=rows,executed_actions=len(rows),independent_physx_steps=counter["completed"],
                production_physics_steps=benchmark.runtime.executor.physics_steps,
                historical_physics_steps=previous["execution_statistics"]["physics_steps"],
                counter_backend=type(benchmark.runtime.env.scene.physx_system).__name__,
                source_manifest_sha256=digest(root/"manifest.json"),source_trace_sha256=digest(folder/"trace.internal.jsonl"),
                passed=bool(rows) and all(row["counts_equal"] and row["time_equal"] and row["status"]=="executed" for row in rows)
                    and counter["completed"]==benchmark.runtime.executor.physics_steps)
        report["official_success_in_this_diagnostic"]=benchmark.score(replace(context,message_id="score"))["success"].value
    except Exception as error:
        report.update(error_type=type(error).__name__,error_code=getattr(error,"code",None))
        import traceback
        traceback.print_exc()
    finally:
        if benchmark is not None:
            try:
                benchmark.close()
            except Exception as error:
                report.update(passed=False,cleanup_error_type=type(error).__name__)
    (target/"receipt.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({key:value for key,value in report.items() if key!="rows"}),flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
