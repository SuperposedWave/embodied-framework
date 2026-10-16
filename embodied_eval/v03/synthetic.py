"""Explicit SYNTHETIC fixtures for canonical architecture and CLI regression."""
from dataclasses import dataclass, replace
from typing import Literal
from .. import _legacy_contracts as leaf
from ..codec import decode
from ..errors import AdapterError
from .._legacy_validation import require
from ..mn.registry import ModelRegistration, BenchmarkRegistration
from . import contracts as c


@dataclass(frozen=True, kw_only=True)
class ModelOptions:
    value: float = .1
    cache_rows: int = 1


@dataclass(frozen=True, kw_only=True)
class BenchmarkOptions:
    native_camera_name: str = "native_camera"
    period_s: float = .05
    timing_mode: Literal["fixed_period", "measured_native"] = "fixed_period"
    fault: Literal["none", "score_error"] = "none"


def model_options(options):
    cfg = decode(ModelOptions, options)
    require(cfg.cache_rows > 0, "invalid_horizon", "cache_rows", "positive cache length")
    return cfg


def benchmark_options(options):
    cfg = decode(BenchmarkOptions, options)
    require(cfg.period_s > 0 and bool(cfg.native_camera_name), "invalid_timing", "benchmark", "positive period and native camera identity required")
    return cfg


def capability(options, ramp=False):
    model_options(options)
    return c.CapabilitySpec(model_id="synthetic-ramp" if ramp else "synthetic-policy", io_profile_id="synthetic-head-joint",
        source_revision="synthetic-v03", checkpoint_ref="synthetic:none", normalization_ref="synthetic:none",
        inputs=(c.InputSpec(key="rgb", public_path="views.main" if ramp else "sensors.head", kind="sensor", semantic="rgb",
            tensor=leaf.TensorSpec(dtype="uint8", shape=(-1, -1, 3), layout="HWC")),
            c.InputSpec(key="instruction", public_path="task_public.instruction", kind="task")),
        output=c.OutputSpec(command_kinds=("joint_target",), targets={"robot0/j0": "joint"}, joint_modes=("position",)),
        supported_timing_modes=("fixed_period", "measured_native"))


def space(options):
    cfg = benchmark_options(options)
    return c.SpaceSpec(spec_id="synthetic-canonical-v03", revision="fixture-v03", protocol_id="synthetic",
        protocol_version="canonical-1", timing_mode=cfg.timing_mode,
        control_period_s=cfg.period_s if cfg.timing_mode == "fixed_period" else None,
        physics_period_s=.004 if cfg.timing_mode == "measured_native" else None,
        frames={"world": leaf.FrameSpec(parent_id=None)},
        robots={"robot0": c.RobotSpec(embodiment_id="synthetic-arm", base_frame_id="world",
            joints={"j0": leaf.JointSpec(joint_type="revolute", position_unit="rad", velocity_unit="rad/s", effort_unit="Nm")})},
        sensors={"head": c.SensorSpec(modality="rgb", role="head", mounted_on="scene", frame_id="world", color_order="RGB",
            tensor=leaf.TensorSpec(dtype="uint8", shape=(2, 2, 3), layout="HWC"), native_source={"field": cfg.native_camera_name, "synthetic": True})},
        state_fields={}, view_aliases={"main": "head"},
        action_specs={"joint-position@1": leaf.ActionSpec(command_kinds=("joint_target",), target_ids=("robot0/j0",))},
        required_action_targets={"joint-position@1": ("robot0/j0",)}, action_joint_modes={"joint-position@1": ("position",)})


class SyntheticModel:
    def __init__(self, options, output=None):
        self.config = model_options(options)
        self.calls = self.forwards = self.observes = 0
        self.latest = None
        self.events = []

    def begin_episode(self, context, task, profile, space):
        self.calls = self.forwards = self.observes = 0
        self.latest = None
        self.events = [{"event": "begin", "episode_id": context.episode_id, "cache_at_begin": 0}]

    def observe(self, model_input):
        self.latest = model_input
        self.observes += 1

    def predict(self, context):
        forward = int(self.calls % self.config.cache_rows == 0)
        self.forwards += forward
        self.calls += 1
        command = leaf.JointTarget(command_id=context.message_id + ":j0", target_id="robot0/j0",
                                  mode="position", value=self.config.value, unit="rad")
        return c.Prediction(context=context, based_on_observation_id=self.latest.observation_id,
            action=c.Action(commands=(command,)), model_forward_count=forward)

    def end_episode(self, context):
        self.latest = None

    def close(self):
        self.latest = None

    def diagnostics(self):
        return {"synthetic": True, "calls": self.calls, "forwards": self.forwards,
                "observe_count": self.observes, "events": self.events}


class SyntheticRampModel(SyntheticModel):
    def predict(self, context):
        result = super().predict(context)
        command = replace(result.action.commands[0], value=result.action.commands[0].value + .01 * self.calls)
        return replace(result, action=replace(result.action, commands=(command,)))


class SyntheticBenchmark:
    def __init__(self, options, output=None):
        self.config = benchmark_options(options)
        self.space = space(options)
        self.steps = self.physics = 0
        self.sim_time = 0.

    def describe(self):
        return self.space

    def enumerate_tasks(self, episodes):
        return [c.TaskSpec(task_id="synthetic", episode_id=ep.episode_id,
            public=leaf.PublicTask(instruction="move the joint"), budgets=ep.budgets,
            initialization_ref="synthetic-seed:" + str(ep.seed), evaluation_config_ref="synthetic-never-success") for ep in episodes]

    def snapshot(self):
        context = replace(self.context, message_id="obs:" + str(self.steps))
        return c.Observation(context=context, observation_id=context.message_id, spec_ref=self.space.spec_id,
            env_step=self.steps, sim_time_s=self.sim_time,
            sensors={"head": leaf.SensorSample(captured_at_s=self.sim_time, frame_id="world", validity="valid", provenance="sensor",
                payload=leaf.Tensor(dtype="uint8", shape=(2, 2, 3), layout="HWC", data=[[[self.steps % 256, 0, 0]] * 2] * 2))}, states={})

    def reset(self, task, context):
        self.context = context
        self.steps = self.physics = 0
        self.sim_time = 0.
        return self.snapshot()

    def advance(self, action, context, remaining):
        self.steps += 1
        physics = self.steps + 2 if self.space.timing_mode == "measured_native" else None
        elapsed = physics * .004 if physics is not None else self.config.period_s
        self.physics += physics or 0
        self.sim_time += elapsed
        return c.StepResult(observation=self.snapshot(), terminated=False, truncated=False,
            execution_feedback=leaf.ExecutionFeedback(request_id=context.message_id, status="executed", consumed_env_steps=1,
                elapsed_sim_time_s=elapsed, consumed_physics_steps=physics, command_ids=tuple(cmd.command_id for cmd in action.commands)))

    def score(self, context):
        if self.config.fault == "score_error":
            raise AdapterError("scoring", "synthetic_score_error", "synthetic scorer failure")
        return {"success": leaf.Metric(value=False, unit="boolean", source="synthetic-never-success", valid=True)}

    def diagnostics(self):
        return {"synthetic": True, "steps": self.steps, "physics_steps": self.physics}

    def close(self):
        pass


class SyntheticImageBenchmark(SyntheticBenchmark):
    def __init__(self, options, output=None):
        super().__init__({"period_s": .1, "native_camera_name": "another_native_view", **options}, output)


def register(registry):
    for name, cls in (("synthetic_policy_a", SyntheticModel), ("synthetic_policy_b", SyntheticRampModel)):
        registry.register_model(name, ModelRegistration(
            lambda o, ramp=cls is SyntheticRampModel: capability(o, ramp), cls, model_options,
            lambda o: {"synthetic": True}, lambda o: {"available": True, "synthetic": True}, lambda o: {"synthetic": True}, True))
    for name, cls in (("synthetic_simulator_a", SyntheticBenchmark), ("synthetic_simulator_b", SyntheticImageBenchmark)):
        defaults = {} if cls is SyntheticBenchmark else {"period_s": .1, "native_camera_name": "another_native_view"}
        registry.register_benchmark(name, BenchmarkRegistration(lambda o, d=defaults: space({**d, **o}), cls, benchmark_options,
            lambda o: {"synthetic": True}, lambda o: {"available": True, "synthetic": True}, lambda o: {"synthetic": True},
            lambda o: {"success": leaf.Metric(value=None, unit="boolean", source="synthetic-never-success", valid=False, unavailable_reason="not_scored")}, True))
