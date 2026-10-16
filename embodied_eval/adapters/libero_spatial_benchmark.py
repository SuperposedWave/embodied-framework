"""LIBERO spatial task 0, trial 0, using the official environment and scorer."""
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from .. import _legacy_contracts as c
from ..codec import decode
from ..errors import AdapterError
from .._legacy_validation import same_episode
from .openwam_libero_spec import CONTROLLER, EXTENSION, space


class LiberoSpatialBenchmark:
    space_factory = staticmethod(space)
    controller_ref = CONTROLLER
    settle_steps = 30
    native_state_dtype = np.float32
    action_dtype = np.float32
    score_revision = "8f1084e"

    def __init__(self, config):
        self.config = config
        self.space = self.space_factory()
        self.env = None
        self.context = None
        self.steps = 0
        self.serial = 0
        self.success = False
        self.actions = []
        self._config_dir = tempfile.TemporaryDirectory(prefix="embodied-eval-libero-")
        root = Path(config.libero_repo).resolve()
        package = root / "libero" / "libero"
        if not (package / "bddl_files").is_dir():
            raise AdapterError("benchmark", "source_missing", f"LIBERO source unavailable: {root}")
        setup = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                 "init_states": str(package / "init_files"), "datasets": str(root / "datasets"),
                 "assets": str(package / "assets")}
        (Path(self._config_dir.name) / "config.yaml").write_text(yaml.safe_dump(setup, sort_keys=True))
        os.environ["LIBERO_PATH"] = str(root)
        os.environ["LIBERO_CONFIG_PATH"] = self._config_dir.name
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    def describe(self):
        return self.space

    def enumerate_tasks(self, selection):
        from libero.libero import benchmark
        from .runtime_options import EpisodeConfig
        self.suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        self.native_task = self.suite.get_task(0)
        self.init_states = self._load_init_states()
        if len(self.init_states) == 0:
            raise AdapterError("benchmark", "init_state_missing", "LIBERO spatial task 0 has no initialization state")
        tasks = []
        self.seeds = {}
        for raw in selection["episodes"]:
            ep = decode(EpisodeConfig, raw)
            self.seeds[ep.episode_id] = ep.seed
            tasks.append(c.TaskSpec(task_id="libero_spatial/task_00/trial_000", protocol_id="libero_spatial",
                protocol_version=self.space.protocol_version, episode_id=ep.episode_id,
                public=c.PublicTask(instruction=self.native_task.language),
                initialization_ref=f"libero_spatial/task_00/init_state_000/seed_{ep.seed}",
                observation_policy_ref="libero-two-rgb-native-proprio@1", budgets=ep.budgets,
                evaluation_config_ref="libero-official-success@" + self.score_revision))
        return tasks

    def _load_init_states(self):
        return self.suite.get_task_init_states(0)

    def _snapshot(self, raw):
        self.serial += 1
        now = self.steps * self.space.control_period_s
        oid = f"{self.context.episode_id}:obs:{self.serial}"
        sensors = {}
        for key, frame in (("agentview_image", "agentview_camera"),
                           ("robot0_eye_in_hand_image", "eye_in_hand_camera")):
            image = np.asarray(raw[key], dtype=np.uint8)
            if image.shape != (256, 256, 3):
                raise AdapterError("benchmark", "image_shape_mismatch", f"{key}: {image.shape}")
            sensors[key] = c.SensorSample(captured_at_s=now, frame_id=frame, validity="valid",
                provenance="sensor", payload=c.Tensor(dtype="uint8", shape=(256, 256, 3), layout="HWC",
                                                data=image.tolist()))
        native = {
            "eef_position_m": np.asarray(raw["robot0_eef_pos"], dtype=self.native_state_dtype).reshape(3).tolist(),
            "eef_quaternion_xyzw": np.asarray(raw["robot0_eef_quat"], dtype=self.native_state_dtype).reshape(4).tolist(),
            "gripper_qpos_m": np.asarray(raw["robot0_gripper_qpos"], dtype=self.native_state_dtype).reshape(2).tolist(),
        }
        return c.Observation(context=c.Context(run_id=self.context.run_id, env_id=self.context.env_id,
                episode_id=self.context.episode_id, message_id=oid), observation_id=oid,
            spec_ref=self.space.spec_id, env_step=self.steps, sim_time_s=now, sensors=sensors,
            robots={"robot0": c.RobotState()}, extensions={EXTENSION: native})

    def reset(self, task, context):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.context = context
        self.steps = self.serial = 0
        self.success = False
        self.actions = []
        bddl = Path(get_libero_path("bddl_files")) / self.native_task.problem_folder / self.native_task.bddl_file
        try:
            for attempt in range(5):
                try:
                    self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                    break
                except Exception as error:
                    if error.__class__.__name__ != "RandomizationError" or attempt == 4:
                        raise
            self.env.seed(self.seeds[task.episode_id])
            raw = self.env.reset()
            raw = self.env.set_init_state(self.init_states[0])
            settle = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            for _ in range(self.settle_steps):
                raw, _, _, _ = self.env.step(settle)
            return self._snapshot(raw)
        except Exception as error:
            raise AdapterError("benchmark", "libero_reset_failed", str(error), state_known=False) from error

    def advance(self, control, request, remaining_budget):
        same_episode(request, self.context)
        if not isinstance(control, c.ActionStep) or len(control.commands) != 1:
            raise AdapterError("execution", "invalid_action", "LIBERO pilot requires one controller command")
        command = control.commands[0]
        if not isinstance(command, c.ControllerCommand) or command.controller_profile_ref != self.controller_ref:
            raise AdapterError("execution", "invalid_action", "LIBERO pilot requires its declared OSC controller")
        action = np.asarray(command.parameters["osc_action7"], dtype=self.action_dtype)
        try:
            raw, reward, done, _ = self.env.step(action)
            self.steps += 1
            self.success = bool(done)
            self.actions.append(action.tolist())
            obs = self._snapshot(raw)
        except Exception as error:
            raise AdapterError("execution", "libero_step_failed", str(error),
                               execution_started=True, state_known=False) from error
        feedback = c.ExecutionFeedback(request_id=request.message_id, command_ids=(command.command_id,),
            status="executed", code="ok", consumed_env_steps=1, elapsed_sim_time_s=self.space.control_period_s)
        return c.StepResult(context=obs.context, observation=obs, execution_feedback=feedback,
            terminated=self.success, truncated=False,
            end_reason="libero_official_success" if self.success else None,
            evaluation_private={"official_done": self.success, "last_reward": float(reward)})

    def execute_tool(self, call, request, remaining_budget):
        raise AdapterError("execution", "tool_unsupported", "LIBERO spatial pilot has no tools")

    def score(self, context):
        same_episode(context, self.context)
        return {"success": c.Metric(value=self.success, unit="boolean",
            source="LIBERO official environment done@" + self.score_revision, valid=True)}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
        self._config_dir.cleanup()

    def diagnostics(self):
        return {"adapter": "libero_spatial", "env_steps": self.steps, "success": self.success,
                "actions_osc7": self.actions, "seed": self.seeds.get(self.context.episode_id) if self.context else None,
                "init_state_index": 0, "settle_steps": self.settle_steps}
