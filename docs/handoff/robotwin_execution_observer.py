"""Passive, process-local planning/control/physics evidence for diagnostics.

No random draws, extra planning, rendering or physical steps. Reading CUDA
results synchronizes them, so observer-on results are not silently substituted
for historical observer-off evidence. Every binding is restored on failure.
"""
from contextlib import contextmanager
import hashlib
import pickle
import random
import sys

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    data = np.asarray(value).copy()
    if data.dtype.kind not in "biuf" or not np.isfinite(data).all():
        raise ValueError("finite_numeric_observer_value_required")
    return {"dtype": str(data.dtype), "shape": list(data.shape), "values": data.tolist(),
            "sha256": hashlib.sha256(data.tobytes()).hexdigest()}


def rng_fingerprints():
    """Only pseudorandom states, never process environment or credentials."""
    digest = lambda state: hashlib.sha256(pickle.dumps(state, protocol=4)).hexdigest()
    result = {"python": digest(random.getstate()), "numpy": digest(np.random.get_state())}
    torch = sys.modules.get("torch")
    if torch is not None:
        result["torch_cpu"] = array(torch.get_rng_state())["sha256"]
        result["torch_cuda"] = [array(state)["sha256"] for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
    return result


def articulation(robot):
    return {side: {"qpos": array(getattr(robot, side + "_entity").get_qpos()),
                   "qvel": array(getattr(robot, side + "_entity").get_qvel())}
            for side in ("left", "right")}


class ExecutionObserver:
    def __init__(self, recorded_plans=None):
        self.actions = []
        self.current = None
        self.recorded_plans = recorded_plans

    def begin_action(self, index):
        if index != len(self.actions) + 1:
            raise ValueError("sequential_observer_action_required")
        self.current = {"action_index": index, "plans": [], "motion_gen": [], "controls": [], "physics": []}
        self.actions.append(self.current)

    @contextmanager
    def watch(self, env, motion_class=None):
        if motion_class is None:
            from curobo.wrap.reacher.motion_gen import MotionGen
            motion_class = MotionGen
        robot, restored = env.robot, []
        def install(owner, name, replacement):
            original = getattr(owner, name)
            # Restore inherited methods by deleting the temporary instance
            # override, rather than leaving a bound-method shadow behind.
            own = name in getattr(owner, "__dict__", {})
            setattr(owner, name, replacement)
            restored.append((owner, name, original, own))
            return original
        try:
            for side in ("left", "right"):
                original = getattr(robot, side + "_plan_path")
                def plan(*args, _original=original, _side=side, **kwargs):
                    if self.current is None:
                        return _original(*args, **kwargs)
                    entry = {"side": _side, "target_pose": array(args[0]),
                             "before": articulation(robot), "rng": rng_fingerprints()}
                    self.current["plans"].append(entry)
                    if self.recorded_plans is None:
                        result = _original(*args, **kwargs)
                    else:
                        # Controlled diagnostic intervention, NEVER a policy
                        # evaluation or a proposed production planner fallback.
                        saved = self.recorded_plans[self.current["action_index"]-1]["plans"]
                        selected = [value for value in saved if value["side"] == _side]
                        if len(selected) != 1 or selected[0]["target_pose"] != entry["target_pose"]:
                            raise ValueError("recorded_plan_target_or_dtype_mismatch")
                        entry["recorded_initial_articulation_exact"] = selected[0]["before"] == entry["before"]
                        result = {key: np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"]).copy()
                                  if isinstance(value, dict) and "dtype" in value else value
                                  for key, value in selected[0]["result"].items()}
                    entry["result"] = {key: array(value) if isinstance(value, np.ndarray) else value
                                       for key, value in result.items()}
                    return result
                install(robot, side + "_plan_path", plan)
            original_motion = motion_class.plan_single
            def motion(instance, start, goal, *args, **kwargs):
                if self.current is None:
                    return original_motion(instance, start, goal, *args, **kwargs)
                entry = {"start_position": array(start.position), "goal_position": array(goal.position),
                         "goal_quaternion_wxyz": array(goal.quaternion), "rng": rng_fingerprints()}
                self.current["motion_gen"].append(entry)
                result = original_motion(instance, start, goal, *args, **kwargs)
                entry["success"] = array(result.success)
                if result.interpolated_plan is not None:
                    entry["position"] = array(result.interpolated_plan.position)
                    entry["velocity"] = array(result.interpolated_plan.velocity)
                return result
            install(motion_class, "plan_single", motion)
            for name in ("set_arm_joints", "set_gripper"):
                original = getattr(robot, name)
                def control(*args, _name=name, _original=original, **kwargs):
                    if self.current is not None:
                        encode = lambda value: value if isinstance(value, str) else array(value)
                        self.current["controls"].append({"operation": _name,
                            "physics_step_before": len(self.current["physics"]),
                            "arguments": [encode(value) for value in args],
                            "kwargs": {key: encode(value) for key, value in kwargs.items()}})
                    return _original(*args, **kwargs)
                install(robot, name, control)
            system = env.scene.physx_system
            system_class, original_step = type(system), type(system).step
            def step(current, *args, **kwargs):
                entry = {"before": articulation(robot)} if current is system and self.current is not None else None
                result = original_step(current, *args, **kwargs)
                if entry is not None:
                    entry["after"] = articulation(robot)
                    self.current["physics"].append(entry)
                return result
            install(system_class, "step", step)
            yield self
        finally:
            for owner, name, original, own in reversed(restored):
                if own:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
