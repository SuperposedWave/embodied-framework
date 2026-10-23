"""Passive final RoboTwin call arguments, outside public Runner/Adapter APIs.

Attach after native task preparation, so expert initialization is not confused
with policy actions. Preserve arguments, return objects, dtype, and call count.
No camera, RNG, planner, physics or scoring calls are added.
"""
from contextlib import contextmanager
import json
from pathlib import Path


@contextmanager
def record_controller(directory):
    from embodied_eval.v03.benchmarks import CanonicalBenchmark
    from embodied_eval.adapters.native_robotwin_executor import MeasuredRobotwinExecutor
    directory = Path(directory)
    original = CanonicalBenchmark.enumerate_tasks
    restored, rows = [], []
    def enumerated(benchmark, *args, **kwargs):
        tasks = original(benchmark, *args, **kwargs)
        runtime = getattr(benchmark.engine, "runtime", None)
        executor = getattr(runtime, "executor", None)
        if not isinstance(executor, MeasuredRobotwinExecutor):
            return tasks
        env = executor.env
        call = env.take_action
        def observed(action, *values, **keywords):
            import numpy as np
            native = np.asarray(action)
            mode = keywords.get("action_type", values[0] if values else "qpos")
            row = {"index": len(rows) + 1, "action_type": mode, "dtype": str(native.dtype),
                   "shape": list(native.shape), "values": native.tolist(), "completed": False,
                   "counter_before": int(env.take_action_cnt)}
            rows.append(row)
            result = call(action, *values, **keywords)
            row.update(completed=True, counter_after=int(env.take_action_cnt))
            return result
        env.take_action = observed
        restored.append((env, call))
        return tasks
    CanonicalBenchmark.enumerate_tasks = enumerated
    try:
        yield
    finally:
        CanonicalBenchmark.enumerate_tasks = original
        for env, call in reversed(restored):
            env.take_action = call
        if rows:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "receipt.json").write_text(json.dumps({
                "kind": "actual_environment_take_action_arguments_not_reconstructed_values",
                "records": rows, "all_calls_completed": all(row["completed"] for row in rows),
                "expert_initialization_actions_included": False,
                "extra_get_obs_planning_simulation_or_scoring_calls": 0,
                "arguments_return_objects_and_native_dtype_preserved": True}, indent=2) + "\n")
