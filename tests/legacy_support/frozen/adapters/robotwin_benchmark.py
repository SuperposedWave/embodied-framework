"""Single reviewed RoboTwin trial; native task/planning/control/scoring stay here.

No XPolicyLab, source edits, planner fallback, fabricated fixed macro duration or
extra observation within a LingBot chunk. Each family runs in a separate process.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np

import tests.legacy_support.frozen.environment_records as c
from embodied_eval.codec import decode
from embodied_eval.errors import AdapterError
from embodied_eval.core.validation import same_episode, schema_value
from tests.legacy_support.frozen.adapters.native_common import finite_array, native_module
from tests.legacy_support.frozen.adapters.native_robotwin_executor import MeasuredRobotwinExecutor
import tests.legacy_support.frozen.adapters.robotwin_spec as spec


class _RngDomain:
    """Serial benchmark-owned Python/NumPy/Torch state, isolated from model calls.

LingBot's original text generator does not seed Python random. This port seeds
it explicitly, records the resulting text, and uses a distinct protocol version.
"""
    def __init__(self, seed):
        self.torch = sys.modules.get("torch")
        outside = self._capture()
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            if self.torch:
                self.torch.manual_seed(seed)
            self.state = self._capture()
        finally:
            self._restore(outside)

    def _capture(self):
        values = [random.getstate(), np.random.get_state()]
        if self.torch:
            values.append(self.torch.get_rng_state().clone())
            values.append([v.clone() for v in self.torch.cuda.get_rng_state_all()]
                          if self.torch.cuda.is_initialized() else None)
        return values

    def _restore(self, values):
        random.setstate(values[0])
        np.random.set_state(values[1])
        if self.torch:
            self.torch.set_rng_state(values[2])
            if values[3] is not None:
                self.torch.cuda.set_rng_state_all(values[3])

    @contextmanager
    def activate(self):
        outside = self._capture()
        self._restore(self.state)
        try:
            yield
        finally:
            self.state = self._capture()
            self._restore(outside)


class RobotwinBenchmark:
    def __init__(self, config, variant, output, model_repo=None):
        self.config, self.variant = config, variant
        self.model_repo = model_repo
        self.root = Path(config.robotwin_repo).resolve()
        run_root = Path(output).resolve()
        # describe/enumerate may prepare a native scene before ArtifactWriter
        # exclusively creates run_root. Keep native setup beside, not inside,
        # that root so the normal Runner cannot reject its own prepared output.
        self.runtime = run_root.parent / (run_root.name + "-robotwin-runtime")
        self.space = spec.space(variant)
        self.env = self.executor = self.context = self.raw = self.domain = None
        self.serial = 0
        self.captured_at = 0.
        self.selected_seed = self.seed_index = None
        self.instruction = None
        self.seed_attempts = []
        self._task_created = False

    @contextmanager
    def _scope(self):
        old = Path.cwd()
        os.chdir(self.runtime)
        try:
            if self.domain:
                with self.domain.activate():
                    yield
            else:
                yield
        finally:
            os.chdir(old)

    def _load(self):
        import yaml
        if self.env is not None:
            return
        if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK") == "1":
            raise AdapterError("benchmark", "unreviewed_planner_fallback", "native planner fallback changes the reviewed protocol")
        self.runtime.mkdir(parents=True, exist_ok=False)
        # Writable outputs are NOT symlinked into the read-only checkout.
        for name in ("envs", "assets", "task_config", "description"):
            (self.runtime / name).symlink_to(self.root / name, target_is_directory=True)
        for path in (self.root, self.root / "description/utils"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        with self._scope():
            # Prewarm native CUDA before SAPIEN import; do not patch planners.
            import torch
            if not torch.cuda.is_available():
                raise AdapterError("infrastructure", "cuda_unavailable", "RoboTwin native planning requires CUDA")
            torch.empty(1, device="cuda")
            self._prepare_cuda_runtime()
            native_module(self.root, "envs")
            generator = native_module(self.root, "generate_episode_instructions", "description/utils")
            module_spec = importlib.util.spec_from_file_location("embodied_eval_robotwin_native_eval", self.root / "script/eval_policy.py")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            self.generator = generator.generate_episode_descriptions
            args = yaml.safe_load((self.root / "task_config" / (self.config.task_config + ".yml")).read_text())
            args.update(task_name=self.config.task_name, task_config=self.config.task_config,
                        policy_name="embodied_eval_native", ckpt_setting=self.variant, eval_mode=True,
                        eval_video_log=False, collect_data=False)
            embodiments = yaml.safe_load((self.root / "task_config/_embodiment_config.yml").read_text())
            robot_file = embodiments["aloha-agilex"]["file_path"]
            args.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
                        left_embodiment_config=module.get_embodiment_config(robot_file),
                        right_embodiment_config=module.get_embodiment_config(robot_file))
            camera = module.get_camera_config(args["camera"]["head_camera_type"])
            args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
            self.args = args
            self.env = module.class_decorator(self.config.task_name)

    def describe(self):
        # Native source sets 1/250, but SAPIEN returns its float32 representation.
        # Negotiate that actual value BEFORE profile/manifest construction; no
        # fabricated fixed macro duration or post-negotiation spec mutation.
        self._load()
        scene = getattr(self.env, "scene", None)
        if scene is None:
            import sapien
            scene = sapien.Scene()
            scene.set_timestep(.004)
        from tests.legacy_support.frozen.adapters.native_robotwin_executor import _period
        period = _period(scene)
        import math
        if not math.isclose(period, .004, rel_tol=0, abs_tol=5e-10):
            raise AdapterError("benchmark", "native_topp_period_mismatch", "source-fixed 1/250 timestep required")
        self.space = replace(self.space, physics_period_s=period)
        return self.space

    def enumerate_tasks(self, selection):
        from tests.legacy_support.frozen.adapters.runtime_options import EpisodeConfig
        if self._task_created:
            raise AdapterError("benchmark", "single_trial_only", "rebuild adapter for a new trial")
        selected = [decode(EpisodeConfig, raw) for raw in selection["episodes"]]
        if len(selected) != 1:
            raise AdapterError("benchmark", "single_trial_only", "one reviewed trial required")
        self._load()
        ep = selected[0]
        self.seed_index = ep.seed
        start = 100000 * (1 + ep.seed)
        for seed in range(start, start + self.config.max_seed_candidates):
            self.domain = _RngDomain(seed)
            info = None
            with self._scope():
                try:
                    self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **copy.deepcopy(self.args))
                    info = self.env.play_once()
                    self.env.close_env()
                    valid = bool(self.env.plan_success and self.env.check_success())
                except Exception as error:
                    self.seed_attempts.append({"seed": seed, "valid": False, "error_type": type(error).__name__})
                    try:
                        self.env.close_env()
                    except Exception:
                        # Cleanup failures are infrastructure failures, not invalid seeds.
                        raise AdapterError("benchmark", "native_cleanup_failed", "expert seed scene cleanup failed") from error
                    continue
            self.seed_attempts.append({"seed": seed, "valid": valid})
            if valid:
                self.selected_seed = seed
                break
        else:
            raise AdapterError("benchmark", "expert_valid_seed_not_found", "bounded native expert-valid seed search exhausted")
        self.domain = _RngDomain(self.selected_seed)
        with self._scope():
            self.env.setup_demo(now_ep_num=0, seed=self.selected_seed, is_test=True, **copy.deepcopy(self.args))
            self.instruction = self._choose_instruction(info["info"])
            self.env.set_instruction(instruction=self.instruction)
            self.executor = MeasuredRobotwinExecutor(self.env,
                benchmark_revision=spec.VARIANTS[self.variant][1], action_type=spec.VARIANTS[self.variant][3])
            if self.executor.physics_period_s != self.space.physics_period_s:
                raise AdapterError("benchmark", "native_scene_period_differs", "actual task period differs from negotiated SAPIEN value")
        self._task_created = True
        return [c.TaskSpec(task_id="adjust_bottle/episode_000", episode_id=ep.episode_id,
            protocol_id=self.space.protocol_id, protocol_version=self.space.protocol_version,
            public=c.PublicTask(instruction=self.instruction),
            initialization_ref=f"demo_clean/seed-index-{ep.seed}/expert-valid-{self.selected_seed}",
            observation_policy_ref="robotwin-public-raw-native-chunk-cadence@1", budgets=ep.budgets,
            evaluation_config_ref="robotwin-native-eval-success@" + self.space.revision[:7],
            evaluation_private={"selected_native_seed": self.selected_seed, "seed_attempts": self.seed_attempts})]

    def _prepare_cuda_runtime(self):
        """Legacy bootstrap unchanged; independent simulator can override it."""
        if self.variant == "openwam_robotwin":
            helper = native_module(self.model_repo, "benchmarks.robotwin.eval_policy_wrapper")
            helper._prewarm_cuda_for_curobo()
            helper._patch_warp_torch_namespace()
        else:
            from tests.legacy_support.frozen.adapters.native_robotwin_runtime import prewarm_native_cuda
            prewarm_native_cuda()

    def _choose_instruction(self, episode_info):
        if self.variant == "openwam_robotwin":
            choices = native_module(self.model_repo, "benchmarks.robotwin.eval_manifest")
            return choices.choose_instructions(self.generator, task=self.config.task_name,
                candidate_seed=self.selected_seed, episode_info=episode_info)[self.config.instruction_type]
        descriptions = self.generator(self.config.task_name, [episode_info], 100)
        return str(np.random.choice(descriptions[0][self.config.instruction_type]))

    def _snapshot(self):
        self.serial += 1
        clock = self.executor.physics_steps * self.executor.physics_period_s
        identity = replace(self.context, message_id=f"{self.context.episode_id}:obs:{self.serial}")
        sensors = {}
        for key in spec.SENSORS:
            image = np.asarray(self.raw["observation"][key]["rgb"])
            if image.shape != (240, 320, 3) or image.dtype != np.uint8:
                raise AdapterError("benchmark", "native_camera_mismatch", "source D435 raw uint8 240x320 RGB required")
            sensors[key] = c.SensorSample(captured_at_s=self.captured_at, frame_id=key, validity="valid", provenance="sensor",
                payload=c.Tensor(dtype="uint8", shape=image.shape, layout="HWC", data=image.tolist()))
        poses = self.raw["endpose"]
        extension = {"joint_drive_targets_and_grippers": finite_array(self.raw["joint_action"]["vector"], (14,), "drive targets").tolist(),
                     "captured_at_sim_time_s": self.captured_at}
        for side in ("left", "right"):
            extension[side + "_endpose_world_wxyz"] = finite_array(poses[side + "_endpose"], (7,), "native endpose").tolist()
            extension[side + "_gripper_open_fraction"] = float(finite_array(np.asarray(poses[side + "_gripper"]).reshape(-1), (1,), "gripper")[0])
        return c.Observation(context=identity, observation_id=identity.message_id, spec_ref=self.space.spec_id,
            env_step=self.executor.steps, sim_time_s=clock, sensors=sensors, robots={"robot0": c.RobotState()},
            extensions={spec.EXTENSION: extension})

    def reset(self, task, context):
        if not self._task_created or self.context is not None:
            raise AdapterError("benchmark", "invalid_trial_lifecycle", "enumerate the fixed trial once before reset")
        if task.public.instruction != self.instruction or task.initialization_ref != f"demo_clean/seed-index-{self.seed_index}/expert-valid-{self.selected_seed}":
            raise AdapterError("benchmark", "trial_identity_mismatch", "prepared native trial identity changed")
        self.context = context
        with self._scope():
            self.raw = copy.deepcopy(self.env.get_obs())
        self.captured_at = 0.
        return self._snapshot()

    def advance(self, control, request, remaining_budget):
        same_episode(request, self.context)
        action, controller = spec.refs(self.variant)
        if not isinstance(control, c.NativeAction) or control.action_spec_ref != action or len(control.commands) != 1:
            raise AdapterError("execution", "invalid_native_action", "one reviewed native controller row required")
        command = control.commands[0]
        if command.controller_profile_ref != controller or command.target_id != "robot0/base":
            raise AdapterError("execution", "invalid_native_controller", "dual-arm native controller mismatch")
        parameters = command.parameters
        schema_value(self.space.controllers[controller].parameter_schema, parameters, "native.parameters")
        row = finite_array(parameters["native_row"], (spec.VARIANTS[self.variant][4],), "native row")
        # Validate a lossless declared dtype conversion, including integer rows.
        cast = row.astype(parameters["native_dtype"])
        if not np.isfinite(cast).all() or not np.array_equal(row, cast):
            raise AdapterError("execution", "native_dtype_loss", "declared native dtype changes action values")
        with self._scope():
            receipt = self.executor.execute(cast, remaining_budget)
        final = receipt.official_success or receipt.native_action_limit_reached
        can_continue = (remaining_budget.env_steps is None or remaining_budget.env_steps > 1)
        can_continue &= remaining_budget.model_calls is None or remaining_budget.model_calls > 0
        if parameters["acquire_observation_after"] and not final and not receipt.error_type and can_continue:
            try:
                with self._scope():
                    self.raw = copy.deepcopy(self.env.get_obs())
                self.captured_at = receipt.cumulative_sim_time_s
            except Exception as error:
                raise AdapterError("execution", "native_observation_failed", type(error).__name__,
                                   execution_started=True, state_known=False) from error
        # Between chunk rows/final states, preserve the LAST acquisition timestamp.
        observation = self._snapshot()
        feedback = c.ExecutionFeedback(request_id=request.message_id, command_ids=(command.command_id,),
            status="execution_failed" if receipt.error_type else "executed",
            code="execution_failed" if receipt.error_type else "ok", consumed_env_steps=1,
            consumed_physics_steps=receipt.consumed_physics_steps, elapsed_sim_time_s=receipt.elapsed_sim_time_s)
        return c.StepResult(context=observation.context, observation=observation, execution_feedback=feedback,
            terminated=receipt.official_success, truncated=receipt.native_action_limit_reached and not receipt.official_success,
            end_reason="robotwin_official_success" if receipt.official_success else "robotwin_native_action_limit" if receipt.native_action_limit_reached else None,
            evaluation_private={"native_error_type": receipt.error_type})

    def execute_tool(self, call, request, remaining_budget):
        raise AdapterError("execution", "tool_unsupported", "native RoboTwin policy declares no tools")

    def score(self, context):
        same_episode(context, self.context)
        return {"success": c.Metric(value=bool(self.env.eval_success), unit="boolean",
                source="RoboTwin native eval_success@" + self.space.revision[:7], valid=True)}

    def close(self):
        if self.env is not None:
            with self._scope():
                self.env.close_env()
            self.env = None
        self.raw = None

    def diagnostics(self):
        return {"adapter": "robotwin", "variant": self.variant, "framework_registered": True,
                "real_scene_meter_verified": False, "native_seed_index": self.seed_index,
                "selected_native_seed": self.selected_seed, "seed_attempts": self.seed_attempts,
                "instruction": self.instruction, "protocol_version": self.space.protocol_version,
                "video_and_collection_disabled": True, "planner_fallback_enabled": False,
                "camera_cadence": "each OpenWAM row; each LingBot chunk boundary only",
                "executor": self.executor.diagnostics() if self.executor else None}
