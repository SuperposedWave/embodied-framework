"""Policy-free RoboDojo simulator boundary, with pinned native control/score.

Only environment methods are extracted from the source-pinned EvalEnv class.
Its model client, policy dispatch, recovery and output writers are never loaded.
The sole removed reset statement resets the policy, which the Model Adapter owns.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from .. import _legacy_contracts as leaf
from ..codec import decode
from ..errors import ContractError
from .._legacy_validation import require
from . import contracts as c
from .registry import BenchmarkRegistration
from .assets_common import _revision, _hf_objects
from .joint_native import RAW_GRIPPER_PROTOCOL, RAW_GRIPPER_SIGNATURE, RAW_GRIPPER_CONTROLLER

REVISION = "266130a3ec41ba9b20b0e5648d2f3542f219c06c"
METHODS = ("_post_setup_scene", "reset", "setup_scene", "get_action_type", "take_action", "take_action_batch",
           "process_control_info", "validate_action_dict", "step", "is_episode_end", "_align_layout_success", "mark_env_unstable",
           "have_empty", "get_control_empty")


@dataclass(frozen=True, kw_only=True)
class DojoOptions:
    repo: str
    protocol: str = "stack-bowls-arx-x5-native-v1"
    layout_index: int = 0
    device: str = "cuda:0"


def options(value):
    cfg = decode(DojoOptions, value)
    require(cfg.protocol == "stack-bowls-arx-x5-native-v1" and cfg.layout_index >= 0,
            "unknown_protocol", "benchmark", "explicit stack_bowls/arx_x5 native layout protocol required")
    require(cfg.device.startswith("cuda:") and cfg.device[5:].isdigit() and bool(cfg.repo),
            "invalid_device", "benchmark", "explicit CUDA ordinal and local source required")
    return cfg


def space(value):
    options(value)
    joints = {side + f"_j{i}": leaf.JointSpec(joint_type="revolute", position_unit="rad", velocity_unit="rad/s", effort_unit="Nm")
              for side in ("left", "right") for i in range(6)}
    grippers = {side + "_grip": leaf.GripperSpec(modes=("open_fraction",)) for side in ("left", "right")}
    robot = leaf.RobotSpec(embodiment_id="arx-x5-dual", base_frame_id="world", joints=joints, grippers=grippers,
        end_effectors={side + "_eef": leaf.EndEffectorSpec(frame_id="world", tcp_id=side + "_native_ee_link") for side in ("left", "right")})
    sensors = {name: leaf.SensorSpec(modality="rgb", role=name, mounted_on="robot0" if "wrist" in name else "environment",
        frame_id=name, color_order="RGB", tensor=leaf.TensorSpec(dtype="uint8", shape=(-1, -1, 3), layout="HWC"))
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
    states = {}
    for side in ("left", "right"):
        for suffix, semantic, unit, width, origin in (
            ("joint_positions", "joint_position", "rad", 6, "native get_joint reads measured articulation qpos, in configured arm_joint_indices order"),
            ("eef_position", "eef_position", "m", 3, "native get_real_endpose achieved ee_link, local scene origin (single env)"),
            ("eef_quaternion", "eef_orientation", "quaternion_xyzw", 4, "native achieved ee_link wxyz reordered to xyzw"),
            ("gripper_target", "gripper_drive_target", "open_fraction", 1, "native ObsManager.state ee_joint_state comes from prev_control, NOT measured qpos")):
            states[side + "_" + suffix] = c.StateSpec(semantic=semantic, unit=unit,
                frame_id="world" if semantic.startswith("eef_") else None,
                tensor=leaf.TensorSpec(dtype="float64", shape=(width,), layout="vector"), origin=side + " " + origin)
    jt = tuple("robot0/" + name for name in joints) + tuple("robot0/" + name for name in grippers)
    pt = tuple("robot0/" + side + suffix for side in ("left", "right") for suffix in ("_eef", "_grip"))
    ctrl = leaf.ControllerSpec(frame_id="world", parameter_schema={"type": "object", "properties": {
        "requested_open_fraction": {"type": "number"}}, "required": ["requested_open_fraction"], "additionalProperties": False},
        parameter_units={"requested_open_fraction": "native_unbounded_open_fraction_request"})
    return c.SpaceSpec(spec_id="robodojo-stack-bowls-arx-x5@" + REVISION[:7], revision=REVISION,
        protocol_id="robodojo_sim", protocol_version=options(value).protocol, timing_mode="measured_native",
        control_period_s=None, physics_period_s=.004, frames={name: leaf.FrameSpec(parent_id=None) for name in ("world", *sensors)},
        robots={"robot0": robot}, sensors=sensors, state_fields=states, controllers={RAW_GRIPPER_CONTROLLER: ctrl},
        native_protocols={RAW_GRIPPER_PROTOCOL: RAW_GRIPPER_SIGNATURE},
        action_specs={"dual-joint-absolute@1": leaf.ActionSpec(command_kinds=("joint_target", "gripper_command"), target_ids=jt),
            "dual-joint-raw-gripper@1": leaf.ActionSpec(command_kinds=("joint_target", "controller_command"), target_ids=jt,
                controller_refs=(RAW_GRIPPER_CONTROLLER,)),
            "dual-pose-absolute@1": leaf.ActionSpec(command_kinds=("pose_target", "gripper_command"), target_ids=pt)},
        required_action_targets={"dual-joint-absolute@1": jt, "dual-joint-raw-gripper@1": jt, "dual-pose-absolute@1": pt},
        action_joint_modes={"dual-joint-absolute@1": ("position",), "dual-joint-raw-gripper@1": ("position",)})


def extract_environment_class(source, task_class, namespace):
    """Source-locked reuse of ENV methods, never execute the module/imports/factory."""
    tree = ast.parse(source)
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "EvalEnv"]
    require(len(candidates) == 1, "native_source_changed", "EvalEnv", "exactly one reviewed environment class required")
    chosen, hashes, removed = [], {}, 0
    for method in candidates[0].body:
        if not isinstance(method, ast.FunctionDef) or method.name not in METHODS:
            continue
        hashes[method.name] = hashlib.sha256(ast.get_source_segment(source, method).encode()).hexdigest()
        if method.name == "reset":
            retained = []
            for statement in method.body:
                if ast.unparse(statement) == "self.model_client.call(func_name='reset')":
                    removed += 1
                else:
                    retained.append(statement)
            method.body = retained
        require(not any(isinstance(n, ast.Attribute) and n.attr in ("model_client", "deploy_cfg", "policy_name")
            for n in ast.walk(method)), "policy_dependency_not_removed", method.name, "environment method still accesses policy state")
        chosen.append(method)
    require(set(hashes) == set(METHODS) and removed == 1, "native_source_changed", "methods", "reviewed method set and sole model reset removal required")
    cls = ast.ClassDef(name="IndependentNativeEnv", bases=[ast.Name(id="TaskClass", ctx=ast.Load())], keywords=[], body=chosen, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {**namespace, "TaskClass": task_class, "__name__": __name__}
    exec(compile(module, "pinned-robodojo-environment-methods", "exec"), scope)
    return scope["IndependentNativeEnv"], hashes


def readiness(value):
    cfg = options(value)
    root = Path(cfg.repo)
    missing = [str(root / relative) for relative in ("Assets/Robots/x5", "Assets/Eval_Layout/RoboDojo/arx_x5/0") if not (root / relative).is_dir()]
    return {"local_paths_available": root.is_dir() and not missing, "missing_paths": missing, "hashes_verified": False, "dependencies_verified": False}


def verify(value):
    cfg = options(value)
    _revision(cfg.repo, REVISION)
    return {"source_revision": REVISION, "sha256": _hf_objects(Path(cfg.repo).resolve(),
        Path(__file__).parent / "asset_manifests/robodojo.json")}


def metrics(value):
    options(value)
    return {name: leaf.Metric(value=None, unit=unit, source="robodojo:official-native", valid=False, unavailable_reason="not_scored")
        for name, unit in (("success", "boolean"), ("score", "score"))}


def launch_application(sync, launcher, device):
    from argparse import Namespace
    launch = Namespace(kit_args="", headless=True, enable_cameras=True, device=device)
    sync.add_zero_delay_kit_args(launch)
    # Native fast shutdown may exit the embedding interpreter before evidence
    # is written. This affects teardown only, not controller or scene settings.
    return launcher(launch, fast_shutdown=False).app


def prepare_robot_paths(repo, output):
    """Native template replacement in an owned runtime tree, not the source.

    The fixed update_embodiment_config_path.py replaces these two literal
    placeholders with the repository root. Preserve every other byte/setting.
    Robot constructors require curobo.yml beside their other assets, so link
    immutable assets into this small tree rather than editing the reference.
    """
    root = Path(repo).resolve()
    template = root / "Assets/Robots/x5/curobo_tmp.yml"
    source = template.read_text()
    require("${ASSETS_PATH}/Assets/Robots/x5" in source, "native_source_changed", "curobo", "native repository-root template required")
    generated = source.replace("${ASSETS_PATH}", str(root)).replace("$ASSETS_PATH", str(root))
    target = Path(output) / "native-robot-paths/x5"
    target.mkdir(parents=True, exist_ok=False)
    for path in template.parent.iterdir():
        if path.name != "curobo.yml":
            (target / path.name).symlink_to(path)
    with (target / "curobo.yml").open("x") as stream:
        stream.write(generated)
    return target.parent, {"native_template_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generated_config_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "native_path_template_only": True, "reference_assets_modified": False}


class RoboDojoSimulator:
    def __init__(self, value, output=None):
        self.cfg, self.space = options(value), space(value)
        self.output = Path(output) if output else None
        self.app = self.env = None
        self.method_hashes = {}
        self.reward_registration = None
        self.robot_path_evidence = {}
        self.temporary_runtime = None
        self.failed_ik_targets = set()
        self.steps = self.physics_steps = 0
        self.sim_time = 0.

    def describe(self):
        return self.space

    def _load(self, seed):
        from ..adapters.native_common import native_module
        from isaaclab.app import AppLauncher
        import numpy as np
        repo = self.cfg.repo
        sync = native_module(repo, "env.camera_manager.capture.render_sync")
        self.app = launch_application(sync, AppLauncher, self.cfg.device)
        from omegaconf import OmegaConf
        loaders = native_module(repo, "utils.load_file")
        pipeline = native_module(repo, "utils.pipeline_utils")
        if self.output is None:
            import tempfile
            self.temporary_runtime = tempfile.TemporaryDirectory(prefix="embodiedeval-robodojo-")
            runtime = Path(self.temporary_runtime.name)
        else:
            runtime = self.output
        robot_paths, self.robot_path_evidence = prepare_robot_paths(repo, runtime)
        pipeline.ROBOTS_PATH = str(robot_paths)
        registry = native_module(repo, "task.RoboDojo.task_registry")
        _, task_class = registry.load_task_class("stack_bowls")
        ObsManager = native_module(repo, "env.observation_manager.obs_manager").ObsManager
        SeedManager = native_module(repo, "env.seed_manager.seed_manager").SeedManager
        UnStableError = native_module(repo, "utils.cluttered_generator").UnStableError
        root = Path(repo)
        read = lambda relative: loaders.load_yaml(str(root / relative))
        eval_cfg = read("env_cfg/arx_x5.yml")
        eval_cfg.update(task_name="stack_bowls", num_envs=1, seed=seed, eval_num=1, physx_monitor_enabled=False)
        cfg = OmegaConf.create({"sim": read("env_cfg/sim/sim_config.yml"), "scene": read("env_cfg/scene/default.yml"),
            "camera": read("env_cfg/camera/camera_config.yml"), "robot": read("env_cfg/robot/dual_x5.yml"),
            "task_env": read("task/RoboDojo/config/stack_bowls.yml"), "eval_cfg": eval_cfg})
        cfg = pipeline.process_randomization(cfg)
        cfg, _ = pipeline.process_config(cfg, task_name="stack_bowls")
        cfg.sim.scene.num_envs = 1
        # Native main.py does NOT copy the application/render device into the
        # simulation config. This fixed sim_config.yml omits device and BaseEnv
        # consequently selects CPU physics. CuRobo still uses native CUDA.
        # Forcing GPU physics changes the reference protocol and also feeds
        # CUDA tensors into the unchanged NumPy scorer. Preserve native choice.
        require(cfg.sim.get("device", "cpu") == "cpu", "native_source_changed", "sim.device", "reviewed native CPU physics required")
        cfg.sim.seed = [0]
        cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
        cls, self.method_hashes = extract_environment_class((root / "src/eval_client/eval_env.py").read_text(), task_class,
            {"np": np, "deepcopy": deepcopy, "UnStableError": UnStableError})
        e = cls.__new__(cls)
        # Native task constructor provides robot, scene, reward and control;
        # no EvalEnv.__init__, deploy.yml, policy module or WS client is used.
        task_class.__init__(e, cfg, self.app)
        self.env = e
        require(e.dt == .004 and cfg.sim.decimation == 1, "timing_protocol_changed", "sim",
                "pinned stack_bowls native physics period and decimation required")
        original_ik = e.robot_manager.solve_ik
        def observed_ik(*args, **kwargs):
            result = original_ik(*args, **kwargs)
            if result["status"] != "Success":
                robot = kwargs.get("robot")
                require(robot is not None, "native_source_changed", "IK", "reviewed keyword robot identity required")
                self.failed_ik_targets.add("robot0/" + robot.arm_name.split("_")[0] + "_eef")
            return result
        e.robot_manager.solve_ik = observed_ik
        e.eval_cfg = cfg.eval_cfg
        e.physx_monitor_enabled = False
        e.obs_manager = ObsManager(deepcopy(eval_cfg["observation"]), 1, e.dt, "stack_bowls", {}, e.env_seed_list)
        e.success, e.end_flag, e.take_action_cnt = [True], [False], [0]
        e.episode_nums = 1
        e.unstable_nums = 0
        e.unstable_envs = set()
        e.robot_action_dim_info = pipeline.get_robot_action_dim_info(eval_cfg)
        e.seed_manager = SeedManager(eval_cfg)
        e.seed_manager.init_eval()
        e.scene_manager.layout_manager.replay = True
        e._abort_video_writers = lambda: None
        def observations(env_idx_list=None, last_frame=False):
            e.obs_manager.render_for_capture()
            return [deepcopy(e.obs_manager.get_obs(env_idx_list=[0])[0])]
        e.get_obs_batch = observations
        e.get_obs = lambda: observations()[0]
        self.actual_physics_steps = 0
        original_step = e.sim_step
        def counted_step(*args, **kwargs):
            native_sim = e.sim.unwrapped
            before = native_sim._sim_step_counter
            result = original_step(*args, **kwargs)
            consumed = native_sim._sim_step_counter - before
            require(consumed == native_sim.cfg.decimation, "execution_count_mismatch", "physics",
                    "native sim counter must match actual decimated physics steps")
            self.actual_physics_steps += consumed
            return result
        e.sim_step = counted_step

    def enumerate_tasks(self, episodes):
        require(len(episodes) == 1 and episodes[0].seed == 0, "pilot_scope", "episodes", "native layout collection seed0, one selected layout")
        ep = episodes[0]
        return [c.TaskSpec(task_id="robodojo_stack_bowls", episode_id=ep.episode_id,
            public=leaf.PublicTask(instruction="Stack the three bowls together."), budgets=ep.budgets,
            initialization_ref=f"official-layout-seed0:index{self.cfg.layout_index}", evaluation_config_ref=self.cfg.protocol)]

    def _snapshot(self, context):
        import numpy as np
        from ..adapters.native_common import finite_array
        raw = self.env.get_obs()
        sensors = {}
        for name in self.space.sensors:
            array = np.asarray(raw["vision"][name]["color"])
            require(array.ndim == 3 and array.shape[-1] == 3 and array.dtype == np.uint8,
                    "invalid_native_image", name, "native RGB camera required")
            sensors[name] = leaf.SensorSample(captured_at_s=self.sim_time, frame_id=name, validity="valid", provenance="sensor",
                payload=leaf.Tensor(dtype="uint8", shape=array.shape, layout="HWC", data=array.tolist()))
        states = {}
        def add(name, array):
            states[name] = c.StateSample(captured_at_s=self.sim_time, provenance="controller_target" if name.endswith("_target") else "measurement",
                payload=leaf.Tensor(dtype="float64", shape=array.shape, layout="vector", data=array.tolist()))
        for side in ("left", "right"):
            joints = finite_array(raw["state"][side + "_arm_joint_state"], (6,), "native measured joints", np.float64)
            pose = finite_array(raw["state"][side + "_ee_pose"], (7,), "native measured ee link", np.float64)
            grip = finite_array(raw["state"][side + "_ee_joint_state"], (1,), "native commanded gripper", np.float64)
            add(side + "_joint_positions", joints)
            add(side + "_eef_position", pose[:3])
            add(side + "_eef_quaternion", pose[[4, 5, 6, 3]])
            add(side + "_gripper_target", grip)
        after = replace(context, message_id=context.message_id + ":observation")
        return c.Observation(context=after, observation_id=after.message_id, spec_ref=self.space.spec_id,
            env_step=self.steps, sim_time_s=self.sim_time, sensors=sensors, states=states, public_instruction=str(raw["instruction"]))

    def reset(self, task, context):
        if self.env is None:
            self._load(0)
        self.env.reset(seed=[self.cfg.layout_index])
        require(self.env.episode_nums == 1 and not self.env.unstable_envs, "unstable_native_layout", "reset", "selected layout must be stable; no silent replacement")
        # Native run_eval registers checks and process-score transitions AFTER
        # reset clears them. Without this, an empty check list is a false win.
        self.env.run_reward()
        self.env.get_score()
        reward = self.env.reward_manager
        self.reward_registration = {"native_reward_check_groups": len(reward.check_list[0]),
            "native_process_score_stages": len(reward.score_list[0]), "native_score_mode": reward.score_meta[0]["mode"]}
        require(self.reward_registration == {"native_reward_check_groups": 1,
            "native_process_score_stages": 2, "native_score_mode": "transition"},
            "native_reward_registration_missing", "reset", "fixed stack_bowls task checks and score transitions must be registered before actions")
        self.steps = self.physics_steps = 0
        self.sim_time = 0.
        self.actual_physics_steps = 0
        return self._snapshot(context)

    def native_action(self, action):
        import numpy as np
        cmds = {cmd.target_id: cmd for cmd in action.commands}
        require(len(cmds) == len(action.commands), "duplicate_target", "action", "one action per target")
        pose_mode = any(isinstance(cmd, leaf.PoseTarget) for cmd in action.commands)
        mode = "dual-pose-absolute@1" if pose_mode else "dual-joint-raw-gripper@1" if action.native_protocol else "dual-joint-absolute@1"
        require(set(cmds) == set(self.space.required_action_targets[mode]), "missing_execution_target", "action", "complete dual arms and grippers required")
        require(action.native_protocol in (None, RAW_GRIPPER_PROTOCOL), "native_protocol_mismatch", "action", "matching raw gripper protocol required")
        result = {}
        def native_array(values):
            original = np.asarray(values)
            cast = original.astype(action.native_dtype or "float64")
            require(np.isfinite(cast).all() and np.array_equal(original, cast),
                    "native_dtype_loss", "action", "lossless declared native dtype required")
            return cast
        for side in ("left", "right"):
            if pose_mode:
                cmd = cmds["robot0/" + side + "_eef"]
                require(isinstance(cmd, leaf.PoseTarget),
                        "unsupported_action", "pose", "absolute native pose required")
                pose = cmd.pose
                require(pose.frame_id == "world", "frame_mismatch", "pose", "native scene coordinates required")
                q = pose.quaternion_xyzw
                result[side + "_ee_pose"] = native_array((*pose.position_m, q[3], *q[:3]))
            else:
                joints = [cmds[f"robot0/{side}_j{i}"] for i in range(6)]
                require(all(isinstance(cmd, leaf.JointTarget) and cmd.mode == "position" and cmd.unit == "rad" for cmd in joints),
                        "unsupported_action", "joints", "absolute radian joints required")
                result[side + "_arm_joint_state"] = native_array([cmd.value for cmd in joints])
            grip = cmds["robot0/" + side + "_grip"]
            if isinstance(grip, leaf.ControllerCommand):
                require(action.native_protocol == RAW_GRIPPER_PROTOCOL and grip.controller_profile_ref == RAW_GRIPPER_CONTROLLER,
                        "native_protocol_mismatch", "grip", "declared raw opening request required")
                value = grip.parameters["requested_open_fraction"]
            else:
                require(isinstance(grip, leaf.GripperCommand) and grip.mode == "open_fraction" and 0 <= grip.value <= 1,
                        "unsupported_action", "grip", "public opening fraction required")
                value = grip.value
            result[side + "_ee_joint_state"] = native_array([value])
        require(all(np.isfinite(array).all() for array in result.values()), "invalid_native_payload", "action", "finite native arrays required")
        return result

    def advance(self, action, context, remaining):
        require(remaining.sim_time_s is None, "unsupported_native_budget", "advance", "no hard simulation-time cap for native interpolation")
        require(remaining.env_steps is None or remaining.env_steps > 0, "budget_exhausted", "advance", "one native action required")
        row = self.native_action(action)
        self.failed_ik_targets.clear()
        before = self.actual_physics_steps
        count = self.env.take_action_cnt[0]
        self.env.take_action(row)
        consumed = self.actual_physics_steps - before
        require(self.env.take_action_cnt[0] == count + 1, "execution_count_mismatch", "advance", "native action was not executed")
        self.steps += 1
        self.physics_steps += consumed
        self.sim_time = self.physics_steps * .004
        success = bool(self.env.end_flag[0] and self.env.success[0])
        # Native code can omit an arm after IK failure while moving grippers.
        # Preserve that native execution, but never claim all targets executed.
        feedback = leaf.ExecutionFeedback(request_id=context.message_id,
            status="execution_failed" if self.failed_ik_targets else "executed",
            code="planning_failed" if self.failed_ik_targets else "ok", consumed_env_steps=1,
            consumed_physics_steps=consumed, elapsed_sim_time_s=consumed * .004,
            command_ids=tuple(cmd.command_id for cmd in action.commands if cmd.target_id not in self.failed_ik_targets))
        return c.StepResult(observation=self._snapshot(context), execution_feedback=feedback, terminated=success,
            truncated=bool(self.env.end_flag[0] and not success))

    def score(self, context):
        e = self.env
        # Task.get_score registers/resets score transitions; native run_eval
        # calls it before actions, never again while collecting the result.
        success = bool(e.end_flag[0] and e.success[0])
        score = 100. if success else float(e.reward_manager.get_score()[0])
        return {name: leaf.Metric(value=value, unit=unit, source="robodojo:official-native", valid=True)
            for name, value, unit in (("success", success, "boolean"), ("score", score, "score"))}

    def close(self):
        try:
            if self.env is not None:
                try:
                    self.env.obs_manager.reset()
                finally:
                    self.env.close()
        finally:
            self.env = None
            if self.app is not None:
                self.app.close()
                self.app = None
            if self.temporary_runtime is not None:
                self.temporary_runtime.cleanup()
                self.temporary_runtime = None

    def diagnostics(self):
        return {"synthetic": False, "native_actions": self.steps, "physics_steps": self.physics_steps,
            "sim_time_s": self.sim_time, "source_method_sha256": self.method_hashes, "policy_client_created": False,
            "gripper_controller": "native clamp before interpolation", "full_suite_verified": False,
            "layout_index": self.cfg.layout_index, "robot_path_evidence": self.robot_path_evidence,
            "native_reward_registration": self.reward_registration,
            "physics_device": self.env.device if self.env is not None else "not_loaded",
            "application_render_device": self.cfg.device}


def _native_worker(connection, value, output):
    """Benchmark-only native process; exit only AFTER native close and reply."""
    import os
    import resource
    import sys
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    native = None
    failed = False
    method = "initialize"
    try:
        native = RoboDojoSimulator(value, output)
        connection.send({"ok": True, "method": method, "result": None})
        while True:
            method, args = connection.recv()
            if method not in ("reset", "advance", "score", "diagnostics", "close") or (failed and method != "close"):
                raise ValueError("private benchmark lifecycle changed")
            try:
                result = getattr(native, method)(*args)
            except Exception as error:
                failed = True
                # Safe traceback locations only, not arbitrary native text.
                import traceback
                locations = [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                    for frame in traceback.extract_tb(error.__traceback__)[-10:]]
                connection.send({"ok": False, "method": method, "error_type": type(error).__name__, "locations": locations,
                    "error_code": error.issue.code if isinstance(error, ContractError) else None,
                    "native_counts": {"action_count": int(native.env.take_action_cnt[0]),
                        "physics_steps": native.actual_physics_steps, "end_flag": bool(native.env.end_flag[0]),
                        "success_flag": bool(native.env.success[0])} if native.env is not None and method == "advance" else None})
                if method == "close":
                    os._exit(1)
            else:
                connection.send({"ok": True, "method": method, "result": result})
                if method == "close":
                    connection.close()
                    os._exit(0)  # Native env/app are closed; avoid plugin Python finalizers.
    except BaseException:
        if native is not None:
            try:
                native.close()
            except BaseException:
                pass
        os._exit(1)


class RoboDojoProcessSimulator(RoboDojoSimulator):
    """Native application isolation; no Runner concurrency/retry/recovery."""
    def __init__(self, value, output=None):
        super().__init__(value, output)
        import multiprocessing
        self.closed = self.failed = False
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.worker = context.Process(target=_native_worker, args=(child, value, output), daemon=False)
        self.worker.start()
        child.close()
        try:
            self._receive("initialize")
        except BaseException:
            self.connection.close()
            self.worker.join()
            self.closed = True
            raise

    def _receive(self, method):
        from ..errors import AdapterError
        advancing = method == "advance"
        try:
            response = self.connection.recv()
        except (EOFError, OSError) as error:
            self.failed = True
            raise AdapterError("infrastructure", "robodojo_native_worker_exited", "native simulator reply unavailable",
                execution_started=advancing, state_known=not advancing) from error
        if (not isinstance(response, dict) or type(response.get("ok")) is not bool or response.get("method") != method
                or (response["ok"] and "result" not in response)):
            self.failed = True
            raise AdapterError("infrastructure", "robodojo_invalid_worker_reply", "invalid private simulator reply",
                execution_started=advancing, state_known=not advancing)
        if not response["ok"]:
            self.failed = True
            error = AdapterError("execution" if advancing else "infrastructure", "robodojo_native_worker_error",
                "native simulator operation failed; no retry or restart", execution_started=advancing, state_known=not advancing)
            error.native_failure = {key: response[key] for key in ("error_type", "locations", "error_code", "native_counts") if key in response}
            raise error
        return response["result"]

    def _call(self, method, *args):
        from ..errors import AdapterError
        require(not self.closed, "adapter_closed", "robodojo", "native simulator already closed")
        require(not self.failed or method == "close", "native_simulator_failed", "robodojo", "only cleanup after native failure")
        try:
            self.connection.send((method, args))
        except (BrokenPipeError, EOFError, OSError) as error:
            self.failed = True
            raise AdapterError("infrastructure", "robodojo_native_worker_exited", "native simulator unavailable",
                execution_started=False, state_known=method != "advance") from error
        return self._receive(method)

    def reset(self, task, context):
        return self._call("reset", task, context)

    def advance(self, action, context, remaining):
        return self._call("advance", action, context, remaining)

    def score(self, context):
        return self._call("score", context)

    def diagnostics(self):
        return dict(self._call("diagnostics"), native_application_isolated=True, native_worker_pid=self.worker.pid, retries_or_restarts=0)

    def close(self):
        from ..errors import AdapterError
        if self.closed:
            return
        acknowledged = False
        try:
            if self.worker.is_alive():
                self._call("close")
                acknowledged = True
            self.worker.join()
            if self.worker.exitcode != 0 or not acknowledged:
                raise AdapterError("infrastructure", "robodojo_native_close_failed", "native close reply and clean exit required")
        finally:
            self.worker.join()
            self.connection.close()
            self.closed = True


def register(registry):
    registry.register_benchmark("robodojo", BenchmarkRegistration(space, RoboDojoProcessSimulator, options, verify, readiness,
        lambda value: {"source_revision": REVISION, "protocol": options(value).protocol, "native_policy_dependency_removed": True}, metrics))
