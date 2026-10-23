"""Measured native macro execution, NOT a registered RoboTwin Benchmark pair.

Counts completed scene.step calls without modifying native source, planning,
controls, rendering, camera acquisition or the official success predicate.
Task setup/expert-valid seed selection and public observation mapping belong to
the higher-level Benchmark adapter. Separate real PhysX checks establish only
their recorded execution scope; this component does not self-attest that proof.
"""
from dataclasses import dataclass
import math

from embodied_eval.errors import AdapterError
from tests.legacy_support.frozen.adapters.native_common import finite_array
from tests.legacy_support.frozen.adapters.native_robotwin_models import OPENWAM_ROBOTWIN_REVISION, LINGBOT_ROBOTWIN_REVISION


def _count(value, label):
    if type(value) is not int or value < 0:
        raise AdapterError("execution", "invalid_native_counter", label, state_known=False)
    return value


def _period(scene):
    value = scene.get_timestep()
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise AdapterError("execution", "invalid_native_period", "positive physical timestep required", state_known=False)
    return float(value)


class _SceneMeter:
    """Temporary Python proxy; delegated bound methods remain native methods."""
    def __init__(self, scene, period):
        self.scene, self.period = scene, period
        self.completed_steps = 0
        self.ambiguous = False

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self, *args, **kwargs):
        try:
            if not math.isclose(_period(self.scene), self.period, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError("native physical period changed")
            result = self.scene.step(*args, **kwargs)
            self.completed_steps += 1
            if not math.isclose(_period(self.scene), self.period, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError("native physical period changed")
            return result
        except Exception:
            # A thrown native physics call may already have advanced state.
            # Successful calls before it are known, but the total is not.
            self.ambiguous = True
            raise


@dataclass(frozen=True, kw_only=True)
class NativeMacroReceipt:
    consumed_env_steps: int
    consumed_physics_steps: int
    elapsed_sim_time_s: float
    cumulative_env_steps: int
    cumulative_physics_steps: int
    cumulative_sim_time_s: float
    official_success: bool
    native_action_limit_reached: bool
    error_type: str | None = None


class MeasuredRobotwinExecutor:
    """Executes one native row in an already initialized, single-owner episode.

The caller must verify source/checkpoint/task identity before full registration.
Passing a revision here selects allowed source semantics, not proof of identity.
No extra get_obs/check_success call, no simulation-time budget or interruption.
"""
    def __init__(self, env, *, benchmark_revision, action_type):
        if benchmark_revision not in (OPENWAM_ROBOTWIN_REVISION, LINGBOT_ROBOTWIN_REVISION):
            raise ValueError("unreviewed RoboTwin source revision")
        if action_type not in ("ee", "qpos"):
            raise ValueError("native action type must be ee or qpos")
        self.env, self.benchmark_revision, self.action_type = env, benchmark_revision, action_type
        self.physics_period_s = _period(env.scene)
        # SAPIEN3 exposes this source-fixed timestep through a float32 API:
        # 0.004 is reported as 0.004000000189989805. Accept that representation
        # at bootstrap, but retain the exact measured value for all receipts
        # and continue checking runtime changes against that captured value.
        if not math.isclose(self.physics_period_s, 1 / 250, rel_tol=0, abs_tol=5e-10):
            raise AdapterError("benchmark", "native_topp_period_mismatch",
                               "reviewed native trajectory timing is 1/250 seconds")
        self.steps = self.physics_steps = 0
        self.last_receipt = None
        self.valid = True
        if _count(env.take_action_cnt, "take_action_cnt") != 0:
            raise AdapterError("benchmark", "native_episode_not_fresh", "native action clock must start at zero")
        self.limit = _count(env.step_lim, "step_lim")
        if self.limit < 1:
            raise AdapterError("benchmark", "invalid_native_limit", "positive native action limit required")

    def execute(self, action, remaining):
        if not self.valid:
            raise AdapterError("execution", "native_state_unknown", "executor cannot resume ambiguous native state",
                               state_known=False)
        if remaining.sim_time_s is not None:
            raise AdapterError("budget", "unsupported_native_budget", "native macro has no hard simulation-time limit")
        if ((remaining.env_steps is not None and remaining.env_steps < 1)
                or (remaining.wall_time_s is not None and remaining.wall_time_s <= 0)):
            raise AdapterError("budget", "budget_exhausted", "no native action budget remains")
        # Preserve supplied numeric dtype/order; do not scale, clip or convert.
        row = finite_array(action, (16 if self.action_type == "ee" else 14,), "native action")
        before = _count(self.env.take_action_cnt, "take_action_cnt")
        if before != self.steps or _count(self.env.step_lim, "step_lim") != self.limit:
            self.valid = False
            raise AdapterError("execution", "native_counter_changed", "native action counter/limit changed outside the executor",
                               state_known=False)
        if self.env.eval_success or before >= self.limit:
            raise AdapterError("budget", "native_episode_finished", "success or native action limit already reached")
        scene = self.env.scene
        if not math.isclose(_period(scene), self.physics_period_s, rel_tol=0, abs_tol=1e-12):
            self.valid = False
            raise AdapterError("execution", "native_period_changed", "physical timestep changed", state_known=False)
        meter = _SceneMeter(scene, self.physics_period_s)
        native_error = None
        self.env.scene = meter
        try:
            self.env.take_action(row.copy(), action_type=self.action_type)
        except Exception as error:
            native_error = type(error).__name__  # Never expose arbitrary native diagnostic text.
        finally:
            scene_changed = self.env.scene is not meter
            if not scene_changed:
                self.env.scene = scene
        try:
            after = _count(self.env.take_action_cnt, "take_action_cnt")
            known = (not scene_changed and not meter.ambiguous and after - before == 1
                     and _count(self.env.step_lim, "step_lim") == self.limit
                     and math.isclose(_period(scene), self.physics_period_s, rel_tol=0, abs_tol=1e-12))
        except Exception:
            known = False
        if not known:
            self.valid = False
            raise AdapterError("execution", "native_execution_state_unknown", native_error or "NativeMeasurementError",
                               execution_started=True, state_known=False)
        self.steps = after
        self.physics_steps += meter.completed_steps
        self.last_receipt = NativeMacroReceipt(consumed_env_steps=1,
            consumed_physics_steps=meter.completed_steps, elapsed_sim_time_s=meter.completed_steps * self.physics_period_s,
            cumulative_env_steps=self.steps, cumulative_physics_steps=self.physics_steps,
            cumulative_sim_time_s=self.physics_steps * self.physics_period_s,
            official_success=bool(self.env.eval_success), native_action_limit_reached=after >= self.limit,
            error_type=native_error)
        return self.last_receipt

    def diagnostics(self):
        return {"framework_registered": False, "real_scene_meter_verified": False,
                "source_identity_verified": False, "benchmark_revision": self.benchmark_revision,
                "action_type": self.action_type, "native_state_known": self.valid,
                "env_steps_last_confirmed": self.steps, "physics_steps_last_confirmed": self.physics_steps,
                "physics_period_s": self.physics_period_s}


class LingbotNativeActionCache:
    """Consume every returned native joint14 row in order, reset between episodes.

Observation acquisition is deliberately left to the native integration boundary:
the native LingBot client does not call get_obs between rows of a chunk.
"""
    def __init__(self, bridge):
        self.bridge = bridge
        self.rows = []
        self.instruction = None
        self.active = False

    def begin_episode(self):
        self.rows.clear()
        self.instruction = None
        self.active = False
        self.bridge.reset()
        self.active = True

    @property
    def needs_observation(self):
        return not self.rows

    def next_action(self, observation, instruction):
        if not self.active:
            raise AdapterError("model_service", "episode_not_started", "native cache is not active")
        if self.instruction is not None and instruction != self.instruction:
            self.rows.clear()
            # No silent cache reset/replanning on a changed text; native episode
            # instruction stays fixed after generation in the reviewed client.
            raise AdapterError("model_input", "native_instruction_changed", "native episode instruction changed")
        if not self.rows:
            chunk = self.bridge.predict(observation, instruction)
            if getattr(chunk, "ndim", None) != 2 or chunk.shape[0] < 1:
                raise AdapterError("model_output", "invalid_native_chunk", "joint14 chunk must be nonempty")
            chunk = finite_array(chunk, (chunk.shape[0], 14), "native joint14 chunk")
            self.rows = [row.copy() for row in chunk]
            self.instruction = instruction
        return self.rows.pop(0)

    def end_episode(self):
        self.rows.clear()
        self.instruction = None
        self.active = False
