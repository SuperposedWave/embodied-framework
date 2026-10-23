"""VLABench fixed-track reset, one native IK/control tick, and official metrics."""
from dataclasses import replace
import copy
import json
import math
import os
from pathlib import Path

import numpy as np

import tests.legacy_support.frozen.environment_records as c
from embodied_eval.codec import decode
from embodied_eval.errors import AdapterError
from embodied_eval.core.validation import same_episode
from tests.legacy_support.frozen.adapters.native_common import check_tick_budget, finite_array, native_module
import tests.legacy_support.frozen.adapters.vlabench_spec as spec


class VLABenchBenchmark:
    space_factory = staticmethod(spec.space)

    def __init__(self, config):
        self.config = config
        self.root = Path(config.vlabench_repo).resolve()
        self.package = self.root / "VLABench"
        self.env = self.space = self.context = self.prepared = None
        self.steps = self.serial = 0
        self.success = False
        self.last_instruction = None
        self.last_ik_status = None

    def _prepare(self):
        if self.env is not None:
            return
        os.environ["VLABENCH_ROOT"] = str(self.package)
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        native_module(self.root, "VLABench.robots")
        native_module(self.root, "VLABench.tasks")
        environments = native_module(self.root, "VLABench.envs")
        path = self.package / "configs/evaluation/tracks" / (self.config.track + ".json")
        frozen = json.loads(path.read_text())[self.config.task_name][self.config.trial_index]
        try:
            self.env = environments.load_env(self.config.task_name, episode_config=copy.deepcopy(frozen),
                                            random_init=False, eval=False, run_mode="eval")
            # The official evaluator calls reset after load_env's own reset.
            self.env.reset()
            period = float(self.env.control_timestep())
            if not math.isfinite(period) or period <= 0:
                raise ValueError("invalid native control timestep")
            self.space = self.space_factory(period)
            self.prepared = self._native_observation(sample_instruction=True)
        except Exception as error:
            if self.env is not None:
                self.env.close()
                self.env = None
            raise AdapterError("benchmark", "vlabench_prepare_failed", type(error).__name__) from error

    def describe(self):
        self._prepare()
        return self.space

    def enumerate_tasks(self, selection):
        from tests.legacy_support.frozen.adapters.runtime_options import EpisodeConfig
        self._prepare()
        return [c.TaskSpec(task_id=f"{self.config.task_name}/episode_{self.config.trial_index:03d}",
            protocol_id=self.space.protocol_id, protocol_version=self.space.protocol_version,
            episode_id=decode(EpisodeConfig, raw).episode_id,
            public=c.PublicTask(instruction=self.prepared["instruction"]),
            initialization_ref=f"{self.config.track}/{self.config.task_name}/{self.config.trial_index}",
            observation_policy_ref="vlabench-public-three-rgb-native-proprio@1",
            budgets=decode(EpisodeConfig, raw).budgets,
            evaluation_config_ref="VLABench-official-success-progress-intention@cf588fe")
                for raw in selection["episodes"]]

    def _native_observation(self, sample_instruction):
        raw = self.env.get_observation(require_pcd=False)
        if sample_instruction:
            self.last_instruction = self.env.task.get_instruction()
        if not isinstance(self.last_instruction, str) or not self.last_instruction:
            raise AdapterError("benchmark", "instruction_missing", "VLABench returned no public task instruction")
        return {"rgb": raw["rgb"], "ee_state": raw["ee_state"],
                "robot_frame": self.env.get_robot_frame_position(), "instruction": self.last_instruction}

    def _snapshot(self, raw):
        self.serial += 1
        now = self.steps * self.space.control_period_s
        identity = replace(self.context, message_id=f"{self.context.episode_id}:obs:{self.serial}")
        sensors = {}
        for key, index in zip(spec.SENSORS, spec.CAMERA_INDICES):
            image = np.asarray(raw["rgb"][index])
            if image.shape != (480, 480, 3) or image.dtype != np.uint8:
                raise AdapterError("benchmark", "image_shape_mismatch", f"{key} requires raw uint8 480x480 RGB")
            sensors[key] = c.SensorSample(captured_at_s=now, frame_id=key, validity="valid", provenance="sensor",
                payload=c.Tensor(dtype="uint8", shape=image.shape, layout="HWC", data=image.tolist()))
        return c.Observation(context=identity, observation_id=identity.message_id, spec_ref=self.space.spec_id,
            env_step=self.steps, sim_time_s=now, sensors=sensors, robots={"robot0": c.RobotState()},
            extensions={spec.EXTENSION: {
                "ee_state_world_wxyz": finite_array(raw["ee_state"], (8,), "ee_state").tolist(),
                "robot_base_world_m": finite_array(raw["robot_frame"], (3,), "robot_frame").tolist(),
                "instruction": raw["instruction"]}})

    def reset(self, task, context):
        self.context = context
        self.steps = self.serial = 0
        self.success = False
        self.last_ik_status = None
        self._prepare()
        raw = self.prepared
        self.prepared = None
        if raw is None:
            raise AdapterError("benchmark", "episode_already_reset", "fixed single-episode adapter must be rebuilt")
        return self._snapshot(raw)

    def advance(self, control, request, remaining_budget):
        same_episode(request, self.context)
        check_tick_budget(remaining_budget, self.space.control_period_s)
        if not isinstance(control, c.ActionStep) or len(control.commands) != 1:
            raise AdapterError("execution", "invalid_action", "VLABench needs one native controller command")
        command = control.commands[0]
        if not isinstance(command, c.ControllerCommand) or command.controller_profile_ref != spec.CONTROLLER:
            raise AdapterError("execution", "invalid_action", "VLABench controller profile mismatch")
        parameters = command.parameters
        position = finite_array(parameters["position_world_m"], (3,), "position_world_m")
        euler = finite_array(parameters["euler_xyz_rad"], (3,), "euler_xyz_rad")
        fingers = finite_array(parameters["finger_targets_m"], (2,), "finger_targets_m")
        try:
            utilities = native_module(self.root, "VLABench.utils.utils")
            quat = utilities.euler_to_quaternion(*euler)
            self.last_ik_status, joints = self.env.robot.get_qpos_from_ee_pos(
                physics=self.env.physics, pos=position, quat=quat)
        except Exception as error:
            raise AdapterError("execution", "planning_failed", type(error).__name__,
                               execution_started=False, state_known=True) from error
        try:
            before = float(self.env.physics.time())
            timestep = self.env.step(np.concatenate([joints, fingers]))
            self.steps += 1
            elapsed = float(self.env.physics.time()) - before
            if not math.isclose(elapsed, self.space.control_period_s, rel_tol=0, abs_tol=1e-8):
                raise RuntimeError("native step elapsed time differs from declared control period")
            self.success = bool(timestep.last())
            raw = self._native_observation(sample_instruction=not self.success)
            observation = self._snapshot(raw)
        except Exception as error:
            raise AdapterError("execution", "vlabench_step_failed", type(error).__name__,
                               execution_started=True, state_known=False) from error
        feedback = c.ExecutionFeedback(request_id=request.message_id, command_ids=(command.command_id,),
            status="executed", code="ok", consumed_env_steps=1, elapsed_sim_time_s=self.space.control_period_s)
        return c.StepResult(context=observation.context, observation=observation, execution_feedback=feedback,
            terminated=self.success, truncated=False,
            end_reason="vlabench_official_success" if self.success else None,
            evaluation_private={"official_last": self.success})

    def execute_tool(self, call, request, remaining_budget):
        raise AdapterError("execution", "tool_unsupported", "VLABench policy profile declares no tools")

    def score(self, context):
        same_episode(context, self.context)
        source = "VLABench official native evaluator@cf588fe"
        result = {"success": c.Metric(value=self.success, unit="boolean", source=source, valid=True)}
        progress = float(self.env.get_task_progress())
        result["progress_score"] = c.Metric(value=progress if math.isfinite(progress) else None,
            unit="score", source=source, valid=math.isfinite(progress),
            unavailable_reason=None if math.isfinite(progress) else "native_nonfinite_metric")
        try:
            intention = float(self.env.get_intention_score(threshold=0.1))
        except KeyError:
            intention = float("nan")
        result["intention_score"] = c.Metric(value=intention if math.isfinite(intention) else None,
            unit="score", source=source, valid=math.isfinite(intention),
            unavailable_reason=None if math.isfinite(intention) else "native_intention_score_unavailable")
        return result

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.prepared = None

    def diagnostics(self):
        return {"adapter": "vlabench", "env_steps": self.steps, "success": self.success,
                "track": self.config.track, "task": self.config.task_name,
                "episode_index": self.config.trial_index, "max_substeps": 1,
                "control_period_s": self.space.control_period_s if self.space else None,
                "initialization": "frozen track config; seed field is not used for scene sampling"}
