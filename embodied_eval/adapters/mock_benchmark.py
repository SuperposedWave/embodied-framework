"""Synthetic robot state machine. Kinematic conventions live in this adapter."""
from copy import deepcopy
from dataclasses import replace
import math
import time

from .. import contracts as c
from ..codec import decode
from ..errors import AdapterError
from ..geometry import apply_pose_delta, frame_transform, multiply_quaternions, rotate_vector, transform_pose
from ..validation import require, same_episode
from .mock_spec import make_case, IDENTITY


class MockBenchmark:
    def __init__(self, config):
        self.config = config
        space, _, _, task, obs = make_case()
        self.template_task, self.template_obs = task, obs
        dt = config.control_period_s
        self.space = replace(space, spec_id="mock-space@1", revision="mock-benchmark-1",
            protocol_id="synthetic-manipulation", control_period_s=dt,
            frames=space.frames | {"base": c.FrameSpec(parent_id="world", dynamic=True)},
            sensors={key: replace(value, available_rate_hz=1/dt) for key, value in space.sensors.items()},
            discrete_specs={key: replace(value, duration_s=dt) for key, value in space.discrete_specs.items()},
            action_specs={
                "robot-control@1": replace(space.action_specs["robot-control@1"],
                    command_kinds=tuple(kind for kind in space.action_specs["robot-control@1"].command_kinds if kind != "controller_command"),
                    controller_refs=()),
                "mock-controller@1": c.ActionSpec(command_kinds=("controller_command",),
                    target_ids=("robot0/left_eef", "robot0/right_eef"), controller_refs=("normalized-controller@1",)),
            },
            tools={key: replace(value, intermediate_observations_supported=config.intermediate_observations_supported)
                   for key, value in space.tools.items()})
        self.context = None
        self.closed = False
        self.steps = self.serial = 0
        self._observations, self._previous_commands = {}, {}
        self.reset_count = 0
        self.control_events = []

    def describe(self):
        return deepcopy(self.space)

    def enumerate_tasks(self, selection):
        from ..config import EpisodeConfig
        tasks = []
        self.seeds = {}
        for raw in selection["episodes"]:
            ep = decode(EpisodeConfig, raw)
            self.seeds[ep.episode_id] = ep.seed
            tasks.append(replace(self.template_task, task_id="mock-task", episode_id=ep.episode_id,
                budgets=ep.budgets, initialization_ref=f"private-init:seed={ep.seed}",
                evaluation_config_ref="mock-scorer@1"))
        return tasks

    def _fault(self, name):
        return self.config.fault == name and (self.config.fault_episode_id is None
            or self.context.episode_id == self.config.fault_episode_id)

    def reset(self, task, context):
        self.context = context
        self.reset_count += 1
        self.steps = self.serial = 0
        self._observations.clear()
        self._previous_commands.clear()
        self.control_events.clear()
        if self._fault("reset_error"):
            raise AdapterError("benchmark", "reset_failed", "synthetic reset error", state_known=False)
        self.robot = deepcopy(self.template_obs.robots["robot0"])
        # Fixed nonzero rotation exposes pre/post multiplication mistakes.
        q = (0., 0., math.sqrt(.5), math.sqrt(.5))
        effectors = {key: replace(value, pose=replace(value.pose, quaternion_xyzw=q))
                     for key, value in self.robot.end_effectors.items()}
        self.robot = replace(self.robot, end_effectors=effectors)
        return self._snapshot()

    def _snapshot(self):
        self.serial += 1
        now = self.steps * self.space.control_period_s
        oid = f"{self.context.episode_id}:obs:{self.serial}"
        base = self.robot.base_pose
        transforms = {
            "base": c.TimedTransform(transform=c.Transform(translation_m=base.position_m, quaternion_xyzw=base.quaternion_xyzw), captured_at_s=now),
            "private_goal": c.TimedTransform(transform=IDENTITY, captured_at_s=now),
        }
        for side in ("left", "right"):
            pose = self.robot.end_effectors[side + "_eef"].pose
            transforms[side + "_tcp"] = c.TimedTransform(transform=c.Transform(
                translation_m=pose.position_m, quaternion_xyzw=pose.quaternion_xyzw), captured_at_s=now)
        obs = replace(self.template_obs, context=replace(self.context, message_id=oid), observation_id=oid,
            spec_ref=self.space.spec_id, env_step=self.steps, sim_time_s=now,
            sensors={key: replace(value, captured_at_s=now) for key, value in self.template_obs.sensors.items()},
            robots={"robot0": deepcopy(self.robot)}, frame_transforms=transforms)
        self._observations[oid] = deepcopy(obs)
        return deepcopy(obs)

    def _in_frame(self, pose, target_frame, obs):
        if pose.frame_id == target_frame:
            return pose
        return transform_pose(pose, frame_transform(self.space, obs, source_frame=pose.frame_id, target_frame=target_frame),
                              source_frame=pose.frame_id, target_frame=target_frame)

    def _put_pose(self, entity, pose, obs):
        pose = self._in_frame(pose, "base", obs)
        eff = replace(self.robot.end_effectors[entity], pose=pose)
        self.robot = replace(self.robot, end_effectors=self.robot.end_effectors | {entity: eff})
        self._previous_commands[entity] = pose

    def _command(self, cmd, obs):
        entity = cmd.target_id.split("/")[1]
        if isinstance(cmd, c.JointTarget):
            state = replace(self.robot.joints[entity], **{cmd.mode: cmd.value})
            self.robot = replace(self.robot, joints=self.robot.joints | {entity: state})
        elif isinstance(cmd, c.PoseTarget):
            self._put_pose(entity, cmd.pose, obs)
        elif isinstance(cmd, c.PoseDelta):
            if cmd.reference_policy == "chunk_start":
                require(cmd.reference_observation_id in self._observations, "invalid_delta_reference", cmd.command_id, "unknown chunk anchor")
                anchor = self._observations[cmd.reference_observation_id].robots["robot0"].end_effectors[entity].pose
            elif cmd.reference_policy == "previous_command":
                anchor = self._previous_commands.get(entity, self.robot.end_effectors[entity].pose)
            else:
                anchor = obs.robots["robot0"].end_effectors[entity].pose
            anchor = self._in_frame(anchor, cmd.frame_id, obs)
            result = apply_pose_delta(anchor, cmd.delta, composition=cmd.composition)
            self._put_pose(entity, result, obs)
        elif isinstance(cmd, c.GripperCommand):
            effectors = dict(self.robot.end_effectors)
            for key, spec in self.space.robots["robot0"].end_effectors.items():
                if spec.gripper_id == entity:
                    effectors[key] = replace(effectors[key], gripper_state=c.GripperState(mode=cmd.mode, value=deepcopy(cmd.value)))
            self.robot = replace(self.robot, end_effectors=effectors)
        elif isinstance(cmd, c.ControllerCommand):
            # Versioned mock controller: normalized [-1,1] delta maps to .01 metres.
            require(cmd.controller_profile_ref == "normalized-controller@1", "unknown_controller", cmd.command_id, "unknown mock controller")
            pose = self.robot.end_effectors[entity].pose
            delta = c.Transform(translation_m=tuple(.01*x for x in cmd.parameters["delta"]), quaternion_xyzw=IDENTITY.quaternion_xyzw)
            self._put_pose(entity, apply_pose_delta(pose, delta, composition="pre_multiply"), obs)
        elif isinstance(cmd, c.TwistCommand):
            base = entity == "base"
            original = self.robot.base_pose if base else self.robot.end_effectors[entity].pose
            pose = self._in_frame(original, cmd.twist.frame_id, obs)
            dt = self.space.control_period_s
            angular = tuple(x*dt for x in cmd.twist.angular_rad_s)
            theta = math.sqrt(sum(x*x for x in angular))
            q = IDENTITY.quaternion_xyzw if theta == 0 else (*tuple(x/theta*math.sin(theta/2) for x in angular), math.cos(theta/2))
            updated = replace(pose, position_m=tuple(x+v*dt for x, v in zip(pose.position_m, cmd.twist.linear_m_s)),
                              quaternion_xyzw=multiply_quaternions(q, pose.quaternion_xyzw))
            if base:
                self.robot = replace(self.robot, base_pose=self._in_frame(updated, "world", obs))
            else:
                self._put_pose(entity, updated, obs)

    def _discrete(self, control):
        pose = self.robot.base_pose
        if control.action == "forward":
            delta = rotate_vector(pose.quaternion_xyzw, (.1, 0., 0.))
            pose = replace(pose, position_m=tuple(x+y for x, y in zip(pose.position_m, delta)))
        elif control.action in ("left", "right"):
            angle = math.pi/8 * (1 if control.action == "left" else -1)
            pose = replace(pose, quaternion_xyzw=multiply_quaternions(
                (0., 0., math.sin(angle/2), math.cos(angle/2)), pose.quaternion_xyzw))
        self.robot = replace(self.robot, base_pose=pose)

    def _success(self):
        return self.config.success_at_step is not None and self.steps >= self.config.success_at_step

    def _execute(self, tick_count, request, budget, commands=(), tool=None, discrete=None):
        same_episode(request, self.context)
        start = self.steps
        deadline = time.monotonic() + budget.wall_time_s if budget.wall_time_s is not None else None
        observations, command_ids = [], []
        status, code, truncated = "executed", "ok", False
        if tool is not None and self._fault("planning_failed"):
            status, code, tick_count = "planning_failed", "planning_failed", 0
        for _ in range(tick_count):
            used = self.steps - start
            dt = self.space.control_period_s
            if ((budget.env_steps is not None and used >= budget.env_steps)
                or (budget.sim_time_s is not None and (used+1)*dt > budget.sim_time_s + 1e-9)
                or (deadline is not None and time.monotonic() >= deadline)):
                status, code, truncated = "partially_executed", "budget_exhausted", True
                break
            # All commands in one tick use the same pre-tick observation.
            before = next(reversed(self._observations.values()))
            if tool is not None:
                entity = tool.target_id.split("/")[1]
                pose = self.robot.end_effectors[entity].pose
                self._put_pose(entity, replace(pose, position_m=tuple(tool.arguments["position_m"])), before)
            elif discrete is not None:
                self._discrete(discrete)
            else:
                for cmd in commands:
                    self._command(cmd, before)
                    command_ids.append(cmd.command_id)
            self.steps += 1
            observations.append(self._snapshot())
            self.control_events.append({"step": self.steps, "request_id": request.message_id,
                                        "command_ids": [cmd.command_id for cmd in commands]})
            if self._fault("partial_execution_error") and self.steps >= self.config.fault_after_step:
                status, code = "execution_failed", "execution_failed"
                break
            if self._success():
                break
        final = observations[-1] if observations else self._snapshot()
        intermediates = tuple(observations[:-1])
        if self._fault("missing_history"):
            intermediates = ()
        elif not self.config.intermediate_observations_supported:
            intermediates = ()
        feedback = c.ExecutionFeedback(request_id=request.message_id, call_id=tool.call_id if tool else None,
            command_ids=tuple(dict.fromkeys(command_ids)), status=status, code=code,
            consumed_env_steps=self.steps-start, elapsed_sim_time_s=(self.steps-start)*self.space.control_period_s)
        return c.StepResult(context=final.context, observation=final, intermediate_observations=intermediates,
            execution_feedback=feedback, terminated=self._success(), truncated=truncated,
            end_reason="mock_task_success" if self._success() else ("budget_exhausted" if truncated else None),
            evaluation_private={"success": self._success(), "secret_score": 987654321})

    def advance(self, control, request, remaining_budget):
        if isinstance(control, c.DiscreteAction):
            ticks = round(control.duration_s/self.space.control_period_s)
            return self._execute(ticks, request, remaining_budget, discrete=control)
        return self._execute(1, request, remaining_budget, commands=control.commands)

    def execute_tool(self, call, request, remaining_budget):
        return self._execute(call.arguments["max_ticks"], request, remaining_budget, tool=call)

    def score(self, context):
        same_episode(context, self.context)
        if self._fault("score_error"):
            raise AdapterError("scoring", "scorer_failed", "synthetic scorer failure")
        return {"success": c.Metric(value=self._success(), unit="boolean", source="mock-scorer@1", valid=True)}

    def close(self):
        self.closed = True
        if self.context is not None and self._fault("close_error"):
            raise AdapterError("infrastructure", "benchmark_close_failed", "synthetic close error")

    def diagnostics(self):
        return {"closed": self.closed, "reset_count": self.reset_count, "env_steps": self.steps,
                "robot": deepcopy(self.robot) if hasattr(self, "robot") else None, "controls": deepcopy(self.control_events)}
