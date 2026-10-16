"""Small independent policy/simulator plugins, explicitly synthetic evidence."""
from dataclasses import dataclass, replace
from .. import _legacy_contracts as leaf
from ..codec import decode
from .._legacy_validation import require
from . import contracts as c
from .registry import ModelRegistration, BenchmarkRegistration


@dataclass(frozen=True, kw_only=True)
class ModelOptions:
    value: float = 0.1
    cache_rows: int = 1


@dataclass(frozen=True, kw_only=True)
class BenchmarkOptions:
    camera_name: str = "camera"
    robot_id: str = "robot0"
    timing_mode: str = "fixed_period"
    period_s: float = .05


def model_options(options):
    result = decode(ModelOptions, options)
    require(result.cache_rows > 0, "invalid_horizon", "cache_rows", "positive cache length")
    return result


def benchmark_options(options):
    result = decode(BenchmarkOptions, options)
    require(result.timing_mode in ("fixed_period", "measured_native") and result.period_s > 0,
            "invalid_timing", "benchmark", "known mode and positive period required")
    require(bool(result.camera_name and result.robot_id) and "/" not in result.robot_id,
            "invalid_id", "benchmark", "nonempty camera and simple robot ID required")
    return result


def capability(options):
    model_options(options)
    return c.CapabilitySpec(model_id="synthetic-joint-policy", source_revision="fixture-1",
        checkpoint_ref="synthetic:none", normalization_ref="synthetic:none",
        inputs=(c.InputSpec(key="image", kind="sensor", semantic="rgb", tensor=leaf.TensorSpec(dtype="uint8", shape=(-1,-1,3), layout="HWC")),
                c.InputSpec(key="instruction", kind="task")),
        output=c.OutputSpec(command_kinds=("joint_target",), target_slots={"joint": "joint"}, joint_modes=("position",)),
        supported_timing_modes=("fixed_period", "measured_native"))


def space(options):
    cfg = benchmark_options(options)
    rid = cfg.robot_id
    return c.SpaceSpec(spec_id=f"synthetic-simulator-{rid}", revision="fixture-1", protocol_id="synthetic", protocol_version="1",
        timing_mode=cfg.timing_mode, control_period_s=cfg.period_s if cfg.timing_mode == "fixed_period" else None,
        physics_period_s=.004 if cfg.timing_mode == "measured_native" else None,
        frames={"world": leaf.FrameSpec(parent_id=None)},
        robots={rid: leaf.RobotSpec(embodiment_id="synthetic-arm", base_frame_id="world",
                                  joints={"j0": leaf.JointSpec(joint_type="revolute", position_unit="rad", velocity_unit="rad/s", effort_unit="Nm")})},
        sensors={cfg.camera_name: leaf.SensorSpec(modality="rgb", role="head_camera", mounted_on="scene", frame_id="world",
            tensor=leaf.TensorSpec(dtype="uint8", shape=(2,2,3), layout="HWC"), color_order="RGB")}, state_fields={},
        action_specs={"joint-position@1": leaf.ActionSpec(command_kinds=("joint_target",), target_ids=(rid+"/j0",))},
        action_joint_modes={"joint-position@1":("position",)})


class SyntheticModel:
    def __init__(self, options, output=None):
        self.config = model_options(options)
        self.latest = self.profile = None
        self.calls = self.forwards = 0

    def begin_episode(self, context, task, profile, space):
        self.profile = profile
        self.latest = None
        self.calls = self.forwards = 0

    def observe(self, model_input):
        self.latest = model_input

    def predict(self, context):
        forward = int(self.calls % self.config.cache_rows == 0)
        self.forwards += forward
        self.calls += 1
        cmd = leaf.JointTarget(command_id=context.message_id+":joint", target_id=self.profile.binding.targets["joint"],
                               mode="position", value=self.config.value, unit="rad")
        return c.Prediction(context=context, based_on_observation_id=self.latest.observation_id,
                            action=c.Action(commands=(cmd,)), model_forward_count=forward)

    def end_episode(self, context):
        self.latest = None

    def close(self):
        self.latest = None

    def diagnostics(self):
        return {"synthetic": True, "calls": self.calls, "forwards": self.forwards}


class SyntheticBenchmark:
    def __init__(self, options, output=None):
        self.config = benchmark_options(options)
        self.space = space(options)
        self.steps, self.physics, self.sim_time = 0, 0, 0.

    def describe(self):
        return self.space

    def enumerate_tasks(self, episodes):
        return [c.TaskSpec(task_id="synthetic", episode_id=ep.episode_id, public=leaf.PublicTask(instruction="move the joint"),
                           budgets=ep.budgets, initialization_ref=f"synthetic-seed:{ep.seed}", evaluation_config_ref="synthetic-never-success") for ep in episodes]

    def _observation(self):
        context = replace(self.context, message_id=f"obs:{self.steps}")
        return c.Observation(context=context, observation_id=context.message_id, spec_ref=self.space.spec_id,
            env_step=self.steps, sim_time_s=self.sim_time, sensors={self.config.camera_name:
                leaf.SensorSample(captured_at_s=self.sim_time, frame_id="world", validity="valid", provenance="sensor",
                    payload=leaf.Tensor(dtype="uint8", shape=(2,2,3), layout="HWC", data=[[[0,0,0]]*2]*2))}, states={})

    def reset(self, task, context):
        self.context = context
        self.steps, self.physics, self.sim_time = 0, 0, 0.
        return self._observation()

    def advance(self, action, context, remaining):
        self.steps += 1
        physics = self.steps + 2 if self.space.timing_mode == "measured_native" else None
        elapsed = physics*.004 if physics is not None else self.config.period_s
        self.physics += physics or 0
        self.sim_time += elapsed
        return c.StepResult(observation=self._observation(), terminated=False, truncated=False,
            execution_feedback=leaf.ExecutionFeedback(request_id=context.message_id, status="executed",
                command_ids=tuple(cmd.command_id for cmd in action.commands), consumed_env_steps=1,
                elapsed_sim_time_s=elapsed, consumed_physics_steps=physics))

    def score(self, context):
        return {"success": leaf.Metric(value=False, unit="boolean", source="synthetic-never-success", valid=True)}

    def close(self):
        pass

    def diagnostics(self):
        return {"synthetic": True, "steps": self.steps, "physics_steps": self.physics}


class SyntheticRampModel(SyntheticModel):
    """A different policy: advances its joint target with each decision."""
    def predict(self, context):
        prediction = super().predict(context)
        command = prediction.action.commands[0]
        command = replace(command, value=command.value + self.calls * .01)
        return replace(prediction, action=replace(prediction.action, commands=(command,)))


class SyntheticImageBenchmark(SyntheticBenchmark):
    """A different simulator: 100ms control and a changing camera stream."""
    def __init__(self, options, output=None):
        super().__init__({"period_s": .1, **options}, output)

    def _observation(self):
        observation = super()._observation()
        name = self.config.camera_name
        sample = observation.sensors[name]
        image = [[[self.steps % 256, 0, 0] for _ in range(2)] for _ in range(2)]
        return replace(observation, sensors={name: replace(sample, payload=replace(sample.payload, data=image))})


def register(registry):
    for name, cls in (("synthetic_policy_a", SyntheticModel), ("synthetic_policy_b", SyntheticRampModel)):
        describe = (capability if cls is SyntheticModel else
                    lambda o: replace(capability(o), model_id="synthetic-ramp-policy"))
        registry.register_model(name, ModelRegistration(describe, cls, model_options,
            lambda o: {"synthetic": True}, lambda o: {"available": True, "synthetic": True},
            lambda o: {"synthetic": True}, True))
    for name, cls in (("synthetic_simulator_a", SyntheticBenchmark), ("synthetic_simulator_b", SyntheticImageBenchmark)):
        describe = space if cls is SyntheticBenchmark else lambda o: space({"period_s": .1, **o})
        registry.register_benchmark(name, BenchmarkRegistration(describe, cls, benchmark_options,
            lambda o: {"synthetic": True}, lambda o: {"available": True, "synthetic": True},
            lambda o: {"synthetic": True}, lambda o: {"success": leaf.Metric(value=None, unit="boolean", source="synthetic-never-success", valid=False, unavailable_reason="not_scored")}, True))
