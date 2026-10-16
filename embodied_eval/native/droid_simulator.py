"""DROID sim_evals execution protocol, independent of the selected policy.

This exact upstream has timeout termination but NO task-success scorer. Never
invent a success rule, equate timeout with failure, or rename it R021/R022.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from ..errors import AdapterError, ContractError
from ..core import types as leaf
from ..codec import decode
from ..core.validation import require
from ..adapters.native_common import native_module, finite_array
from . import contracts as c
from .assets_common import _revision, _hf_objects
from .registry import BenchmarkRegistration

REVISION = "3a6b0e8177f696947f49c082728f10ef5133ede1"
INSTRUCTIONS = {1: "put the cube in the bowl", 2: "pick up the can and put it in the mug",
                3: "put the banana in the bin"}


@dataclass(frozen=True, kw_only=True)
class DroidOptions:
    repo: str
    assets_dir: str
    protocol: str = "droid-three-scenes-native-v1"
    scene: int = 1
    device: str = "cuda:0"


def options(value):
    cfg = decode(DroidOptions, value)
    require(cfg.protocol == "droid-three-scenes-native-v1" and cfg.scene in INSTRUCTIONS,
            "unknown_protocol", "benchmark", "explicit native DROID scene 1, 2 or 3 required")
    require(cfg.device in ("cpu",) or (cfg.device.startswith("cuda:") and cfg.device[5:].isdigit()),
            "invalid_device", "benchmark.device", "explicit CPU or CUDA ordinal required")
    require(bool(cfg.repo and cfg.assets_dir), "missing_source", "benchmark", "explicit source and USD asset paths required")
    return cfg


def space(value):
    cfg = options(value)
    # Unconnected camera roots do not invent extrinsics or an identity transform.
    frames = {name: leaf.FrameSpec(parent_id=None) for name in ("world", "external_cam", "external_cam_2", "wrist_cam")}
    robot = leaf.RobotSpec(embodiment_id="droid-franka-robotiq", base_frame_id="world",
        joints={f"j{i}": leaf.JointSpec(joint_type="revolute", position_unit="rad", velocity_unit="rad/s",
            effort_unit="Nm") for i in range(7)}, grippers={"grip": leaf.GripperSpec(modes=("open_fraction",))})
    targets = tuple(f"robot0/j{i}" for i in range(7)) + ("robot0/grip",)
    return c.SpaceSpec(spec_id=f"droid-sim-scene{cfg.scene}@{REVISION[:7]}", revision=REVISION,
        protocol_id="droid_sim_evals", protocol_version=cfg.protocol, timing_mode="fixed_period",
        control_period_s=1 / 15, physics_period_s=1 / 120, frames=frames, robots={"robot0": robot},
        sensors={name: leaf.SensorSpec(modality="rgb", role=name, mounted_on="robot0" if name == "wrist_cam" else "environment",
            frame_id=name, color_order="RGB", available_rate_hz=15.,
            tensor=leaf.TensorSpec(dtype="uint8", shape=(720, 1280, 3), layout="HWC"))
            for name in ("external_cam", "external_cam_2", "wrist_cam")},
        state_fields={"arm_joint_positions": c.StateSpec(semantic="joint_position", unit="rad", frame_id=None,
            tensor=leaf.TensorSpec(dtype="float64", shape=(7,), layout="vector"),
            origin="native arm_joint_pos; measured qpos, verified panda_joint1..7 order"),
            "gripper_open_fraction": c.StateSpec(semantic="gripper_open_fraction", unit="open_fraction", frame_id=None,
            tensor=leaf.TensorSpec(dtype="float64", shape=(1,), layout="vector"),
            origin="1 - measured finger_joint qpos/(pi/4), native policy observation; corruption disabled")},
        action_specs={"joint7-binary-gripper@1": leaf.ActionSpec(command_kinds=("joint_target", "gripper_command"),
            target_ids=targets)}, required_action_targets={"joint7-binary-gripper@1": targets},
        action_joint_modes={"joint7-binary-gripper@1": ("position",)})


def readiness(value):
    cfg = options(value)
    missing = [key for key in ("repo", "assets_dir") if not Path(getattr(cfg, key)).is_dir()]
    return {"local_paths_available": not missing, "missing_paths": missing, "hashes_verified": False,
            "dependencies_verified": False, "official_success_scorer_available": False}


def verify(value):
    cfg = options(value)
    _revision(cfg.repo, REVISION)
    return {"source_revision": REVISION, "sha256": _hf_objects(Path(cfg.assets_dir).resolve(),
        Path(__file__).parent / "asset_manifests/droid_sim.json"), "official_success_scorer_available": False}


def metrics(value):
    options(value)
    return {"success": leaf.Metric(value=None, unit="boolean", source="droid_sim_evals:no-official-scorer",
        valid=False, unavailable_reason="official_task_success_scorer_not_defined")}


class DroidSimulator:
    def __init__(self, value, output=None):
        self.cfg, self.space = options(value), space(value)
        self.app = self.env = None
        self.steps = self.physics_steps = 0
        self.sim_time = 0.
        try:
            from isaaclab.app import AppLauncher
            # Application creation must precede all simulator/task imports.
            # Kit's default fast shutdown may terminate the embedding Python
            # process before Runner writes its summary. This changes teardown
            # only: native environment, cameras, controller and stepping stay
            # untouched. The adapter must return from close() to its caller.
            self.app = AppLauncher(headless=True, enable_cameras=True, device=self.cfg.device,
                                   fast_shutdown=False).app
            native_module(self.cfg.repo, "sim_evals.environments", "src")
            import gymnasium as gym
            from isaaclab_tasks.utils import parse_env_cfg
            from copy import deepcopy
            module = sys.modules["sim_evals.environments.droid_environment"]
            module.DATA_PATH = Path(self.cfg.assets_dir).resolve()
            env_cfg = parse_env_cfg("DROID", device=self.cfg.device, num_envs=1, use_fabric=True)
            env_cfg.scene.robot = deepcopy(env_cfg.scene.robot)
            env_cfg.scene.robot.spawn.usd_path = str(module.DATA_PATH / "franka_robotiq_2f_85_flattened.usd")
            env_cfg.set_scene(self.cfg.scene)
            self.env = gym.make("DROID", cfg=env_cfg)
        except Exception:
            self.close()
            raise

    def describe(self):
        return self.space

    def enumerate_tasks(self, episodes):
        require(len(episodes) == 1, "pilot_scope", "episodes", "one native DROID scene/trial per process")
        ep = episodes[0]
        return [c.TaskSpec(task_id=f"droid_scene{self.cfg.scene}", episode_id=ep.episode_id,
            public=leaf.PublicTask(instruction=INSTRUCTIONS[self.cfg.scene]), budgets=ep.budgets,
            initialization_ref=f"native-two-render-resets:scene{self.cfg.scene}:seed{ep.seed}",
            evaluation_config_ref=self.cfg.protocol, evaluation_private={"seed": ep.seed})]

    @staticmethod
    def _numpy(value):
        import numpy as np
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _snapshot(self, raw, context):
        import numpy as np
        policy = raw["policy"]
        sensors = {}
        for name in self.space.sensors:
            value = self._numpy(policy[name])
            require(value.shape == (1, 720, 1280, 3) and value.dtype == np.uint8,
                    "invalid_native_image", name, "one native uint8 RGB camera required")
            array = value[0]
            sensors[name] = leaf.SensorSample(frame_id=name, captured_at_s=self.sim_time, validity="valid", provenance="sensor",
                payload=leaf.Tensor(dtype="uint8", shape=array.shape, layout="HWC", data=array.tolist()))
        joints = finite_array(self._numpy(policy["arm_joint_pos"]).reshape(-1), (7,), "measured arm joints", np.float64)
        closed = finite_array(self._numpy(policy["gripper_pos"]).reshape(-1), (1,), "measured gripper", np.float64)
        require(0 <= closed[0] <= 1, "state_out_of_bounds", "gripper", "native closed fraction outside bounds")
        states = {name: c.StateSample(captured_at_s=self.sim_time,
            payload=leaf.Tensor(dtype="float64", shape=array.shape, layout="vector", data=array.tolist()))
            for name, array in (("arm_joint_positions", joints), ("gripper_open_fraction", 1. - closed))}
        after = replace(context, message_id=context.message_id + ":observation")
        return c.Observation(context=after, observation_id=after.message_id, spec_ref=self.space.spec_id,
            env_step=self.steps, sim_time_s=self.sim_time, sensors=sensors, states=states,
            public_instruction=INSTRUCTIONS[self.cfg.scene])

    def reset(self, task, context):
        native = self.env.unwrapped
        robot = native.scene["robot"]
        names = [name for name in robot.data.joint_names if name in [f"panda_joint{i}" for i in range(1, 8)]]
        require(names == [f"panda_joint{i}" for i in range(1, 8)], "joint_order_mismatch", "robot",
                "native observation helper uses articulation order; do not guess joint ordering")
        self.env.reset(seed=task.evaluation_private["seed"])
        raw, _ = self.env.reset()  # preserve native second render reset
        self.steps = self.physics_steps = 0
        self.sim_time = 0.
        return self._snapshot(raw, context)

    def native_action(self, action):
        import numpy as np
        require(action.native_protocol is None, "native_protocol_mismatch", "action", "canonical physical actions required")
        commands = {cmd.target_id: cmd for cmd in action.commands}
        expected = set(self.space.required_action_targets["joint7-binary-gripper@1"])
        require(len(commands) == len(action.commands) and set(commands) == expected,
                "missing_execution_target", "action", "all seven joints and the gripper are atomic")
        values = []
        for i in range(7):
            cmd = commands[f"robot0/j{i}"]
            require(isinstance(cmd, leaf.JointTarget) and cmd.mode == "position" and cmd.unit == "rad",
                    "unsupported_action", cmd.target_id, "absolute joint radians required")
            values.append(cmd.value)
        grip = commands["robot0/grip"]
        require(isinstance(grip, leaf.GripperCommand) and grip.mode == "open_fraction" and 0 <= grip.value <= 1,
                "unsupported_action", "grip", "public opening fraction required")
        # Source BinaryJointPositionZeroToOneAction treats numeric >.5 as CLOSE.
        # Pass 1-opening, retaining its native strict threshold (including .5).
        values.append(1. - grip.value)
        original = finite_array(values, (8,), "native execution action")
        cast = original.astype(action.native_dtype or "float64")
        require(np.isfinite(cast).all() and np.array_equal(original, cast),
                "native_dtype_loss", "action", "lossless declared native dtype required")
        return cast

    def advance(self, action, context, remaining):
        require(remaining.env_steps is None or remaining.env_steps >= 1, "budget_exhausted", "advance", "one action required")
        require(remaining.sim_time_s is None or remaining.sim_time_s + 1e-9 >= 1 / 15,
                "budget_exhausted", "advance", "one fixed native cycle required")
        row = self.native_action(action)
        import torch
        native = self.env.unwrapped
        before = native.sim.current_time_step_index
        raw, _, terminated, truncated, _ = self.env.step(torch.as_tensor(row, device=native.device)[None])
        consumed = native.sim.current_time_step_index - before
        require(consumed == 8, "execution_count_mismatch", "physics", "native fixed cycle must actually consume eight physics steps")
        self.steps += 1
        self.physics_steps += consumed
        self.sim_time = self.steps / 15
        feedback = leaf.ExecutionFeedback(request_id=context.message_id,
            command_ids=tuple(cmd.command_id for cmd in action.commands), status="executed",
            consumed_env_steps=1, elapsed_sim_time_s=1 / 15, consumed_physics_steps=consumed)
        return c.StepResult(observation=self._snapshot(raw, context), execution_feedback=feedback,
            terminated=bool(terminated.item()), truncated=bool(truncated.item()))

    def score(self, context):
        return metrics({"repo": self.cfg.repo, "assets_dir": self.cfg.assets_dir})

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            if self.app is not None:
                self.app.close()
                self.app = None

    def diagnostics(self):
        return {"synthetic": False, "native_actions": self.steps, "physics_steps": self.physics_steps,
                "sim_time_s": self.sim_time, "official_success_scorer_available": False,
                "full_suite_verified": False, "original_R021_R022_mapping": "unresolved"}


def _native_worker(connection, value):
    """One owned native application; no model, retries or episode concurrency.

    Spawn (never fork initialized CUDA). Messages are trusted local Python
    objects on a private pipe, not a new public/remote contract. After native
    close returns, exit without unsafe NVIDIA plugin Python finalizers.
    """
    import os
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if os.geteuid() == 0:
        sys.argv.append("--allow-root")
    sys.argv.append("--/log/file=")
    native = None
    method = "initialize"
    try:
        native = DroidSimulator(value)
        connection.send({"ok": True, "result": None})
        while True:
            method, args = connection.recv()
            if method not in ("reset", "advance", "score", "diagnostics", "close"):
                raise ValueError("unknown private native worker method")
            try:
                result = getattr(native, method)(*args)
                connection.send({"ok": True, "result": result})
                if method == "close":
                    connection.close()
                    os._exit(0)  # native environment AND application already closed
            except Exception as error:
                response = {"ok": False, "error_type": type(error).__name__, "method": method}
                if isinstance(error, ContractError):
                    response["contract_issue"] = error.issue
                connection.send(response)
                # No retry/recovery after execution uncertainty. Only close is
                # still allowed, so the owning caller can release resources.
                if method == "close":
                    os._exit(1)
                while True:
                    cleanup, _ = connection.recv()
                    if cleanup == "close":
                        native.close()
                        connection.send({"ok": True, "result": None})
                        connection.close()
                        os._exit(0)
    except BaseException as error:
        try:
            connection.send({"ok": False, "error_type": type(error).__name__, "method": method})
        except (BrokenPipeError, EOFError, OSError):
            pass
        if native is not None:
            try:
                native.close()
            except BaseException:
                pass
        os._exit(1)


class DroidProcessSimulator(DroidSimulator):
    """Benchmark-local application isolation, keeping the same public API.

    The parent never imports Isaac. All controller/IK/physics remain in the
    native worker. Synchronous pipe calls have no new hard timeout, retries,
    restart, concurrent trials or recovery semantics.
    """
    def __init__(self, value, output=None):
        import multiprocessing
        self.cfg, self.space = options(value), space(value)
        self.closed = False
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.worker = context.Process(target=_native_worker, args=(child, value), daemon=False)
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
        advancing = method == "advance"
        try:
            response = self.connection.recv()
        except (EOFError, OSError) as error:
            raise AdapterError("infrastructure", "droid_native_worker_exited",
                "owned native simulator ended without a reply", execution_started=advancing,
                state_known=not advancing) from error
        if (not isinstance(response, dict) or not isinstance(response.get("ok"), bool)
                or (response["ok"] and "result" not in response)):
            raise AdapterError("infrastructure", "droid_invalid_worker_reply",
                "invalid private native simulator reply", execution_started=advancing, state_known=not advancing)
        if not response["ok"]:
            raise AdapterError("execution" if advancing else "infrastructure", "droid_native_worker_error",
                "native simulator operation failed; no retry or restart", execution_started=advancing,
                state_known=not advancing)
        return response["result"]

    def _call(self, method, *args):
        require(not self.closed, "adapter_closed", "droid", "native simulator already closed")
        try:
            self.connection.send((method, args))
        except (BrokenPipeError, EOFError, OSError) as error:
            raise AdapterError("infrastructure", "droid_native_worker_exited",
                "owned native simulator is unavailable", execution_started=False,
                state_known=method != "advance") from error
        return self._receive(method)

    def reset(self, task, context):
        return self._call("reset", task, context)

    def advance(self, action, context, remaining):
        return self._call("advance", action, context, remaining)

    def score(self, context):
        return self._call("score", context)

    def diagnostics(self):
        return dict(self._call("diagnostics"), native_application_isolated=True,
                    native_worker_pid=self.worker.pid, retries_or_restarts=0)

    def close(self):
        if self.closed:
            return
        try:
            if self.worker.is_alive():
                self._call("close")
            self.worker.join()
            if self.worker.exitcode != 0:
                raise AdapterError("infrastructure", "droid_native_close_failed",
                    "owned native simulator did not close cleanly")
        finally:
            self.worker.join()
            self.connection.close()
            self.closed = True


def register(registry):
    registry.register_benchmark("droid_sim", BenchmarkRegistration(space, DroidProcessSimulator, options, verify, readiness,
        lambda value: {"source_revision": REVISION, "protocol": options(value).protocol,
            "scene": options(value).scene, "original_R021_R022_mapping": "unresolved"}, metrics))
