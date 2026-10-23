"""Synthetic contract fixtures: no real model, simulator or benchmark claims."""
import tests.legacy_support.frozen._legacy_contracts as c

IDENTITY = c.Transform(translation_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.))


def make_case():
    context = c.Context(run_id="synthetic-run", env_id="env-0", episode_id="ep-0", message_id="obs-0")
    frames = {
        "world": c.FrameSpec(parent_id=None),
        "base": c.FrameSpec(parent_id="world", static_transform=IDENTITY),
        "left_tcp": c.FrameSpec(parent_id="base", dynamic=True),
        "right_tcp": c.FrameSpec(parent_id="base", dynamic=True),
        "private_goal": c.FrameSpec(parent_id="world", dynamic=True),
    }
    joint = c.JointSpec(joint_type="revolute", position_unit="rad", velocity_unit="rad/s",
                        effort_unit="Nm", position_limits=(-3., 3.))
    robot = c.RobotSpec(embodiment_id="synthetic-dual-arm-v1", base_frame_id="base",
        joints={"left_j0": joint, "right_j0": joint},
        end_effectors={
            "left_eef": c.EndEffectorSpec(frame_id="left_tcp", tcp_id="left-tip", gripper_id="left_gripper"),
            "right_eef": c.EndEffectorSpec(frame_id="right_tcp", tcp_id="right-tip", gripper_id="right_gripper"),
        },
        grippers={key: c.GripperSpec(modes=("open_fraction", "width_m"), max_width_m=.08)
                  for key in ("left_gripper", "right_gripper")})
    tensor_spec = c.TensorSpec(dtype="uint8", shape=(2, 2, 3), layout="HWC")
    sensors = {key: c.SensorSpec(modality="rgb", role=role, mounted_on="robot0", frame_id=frame,
                                tensor=tensor_spec, color_order="RGB", available_rate_hz=20.)
               for key, role, frame in (("head_rgb", "overview", "base"),
                   ("left_wrist_rgb", "left_wrist", "left_tcp"), ("right_wrist_rgb", "right_wrist", "right_tcp"))}
    controller_schema = {
        "type": "object", "properties": {
            "delta": {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1},
                      "minItems": 3, "maxItems": 3},
        }, "required": ["delta"], "additionalProperties": False,
    }
    targets = tuple("robot0/" + key for key in ("left_j0", "right_j0", "left_eef", "right_eef", "left_gripper", "right_gripper", "base"))
    space = c.SpaceSpec(spec_id="synthetic-space@1", revision="fixture-1",
        protocol_id="synthetic-manipulation", protocol_version="1", control_period_s=.05,
        frames=frames, robots={"robot0": robot}, sensors=sensors,
        action_specs={"robot-control@1": c.ActionSpec(
            command_kinds=("joint_target", "pose_target", "pose_delta", "twist_command", "gripper_command", "controller_command"),
            target_ids=targets, controller_refs=("normalized-controller@1",))},
        controllers={"normalized-controller@1": c.ControllerSpec(frame_id="base", parameter_schema=controller_schema,
            parameter_units={"delta": "controller_native"})},
        discrete_specs={"navigation@1": c.DiscreteSpec(vocabulary=("forward", "left", "right", "halt"),
            target_ids=("robot0/base",), duration_s=.05)},
        tools={"move_eef@1": c.ToolSpec(target_ids=("robot0/left_eef", "robot0/right_eef"), intermediate_observations_supported=True,
            argument_schema={"type": "object", "properties": {
                "position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "frame_id": {"const": "base"}, "max_ticks": {"type": "integer", "minimum": 1},
            }, "required": ["position_m", "frame_id", "max_ticks"], "additionalProperties": False})},
        extensions={"bench.private": {"hidden_threshold": .12345}, "example.robot_metadata": {"version": 1}})
    cap = c.CapabilitySpec(model_id="synthetic-model", source_revision="fixture-1",
        checkpoint_ref="synthetic-checkpoint@1", normalization_ref="identity-normalization@1",
        roles=("action_policy", "agent", "world_model"),
        required_inputs=(
            c.InputRequirement(key="image", kind="sensor", modality="rgb", dtype="uint8", shape=(2, 2, 3), layout="HWC", frame_id="base", max_age_s=.1),
            c.InputRequirement(key="joint", kind="joint", unit="rad"),
            c.InputRequirement(key="instruction", kind="task"),
        ), supported_embodiments=("synthetic-dual-arm-v1",), observation_update="decision_boundary",
        chunk_mode="runner_owned", produced_action_specs=("robot-control@1",),
        produced_discrete_specs=("navigation@1",), produced_tools=("move_eef@1",),
        can_stop=True, produced_predictions=("future_state",), supported_extensions=("example.robot_metadata",))
    policy = c.ModelViewPolicy(sensor_ids=tuple(sensors), frame_ids=("world", "base", "left_tcp", "right_tcp"),
        robot_fields=("robot0/joints/left_j0/position", "robot0/end_effectors/left_eef/pose",
                      "robot0/end_effectors/right_eef/pose"), extension_names=("example.robot_metadata",))
    profile = c.AdaptationProfile(profile_id="synthetic-pair@1", revision="fixture-1",
        benchmark_spec_ref=space.spec_id, model_id=cap.model_id, model_source_revision=cap.source_revision,
        checkpoint_ref=cap.checkpoint_ref, normalization_ref=cap.normalization_ref, role="action_policy",
        robot_ids=("robot0",), input_bindings={
            "image": c.InputBinding(kind="sensor", source_id="head_rgb"),
            "joint": c.InputBinding(kind="joint", source_id="robot0/left_j0", field_name="position"),
            "instruction": c.InputBinding(kind="task", source_id="instruction")},
        action_spec_refs=cap.produced_action_specs, discrete_spec_refs=cap.produced_discrete_specs,
        tool_refs=(), chunk_mode=cap.chunk_mode, execution_horizon_steps=2, view_policy=policy)
    tensor = c.Tensor(dtype="uint8", shape=(2, 2, 3), layout="HWC",
        data=[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [0, 0, 0]]])
    sample = lambda sid: c.SensorSample(captured_at_s=0., frame_id=sensors[sid].frame_id,
                                       validity="valid", provenance="sensor", payload=tensor)
    eff = lambda x: c.EndEffectorState(pose=c.Pose(position_m=(x, 0., .3), quaternion_xyzw=(0., 0., 0., 1.), frame_id="base"),
                                      gripper_state=c.GripperState(mode="open_fraction", value=.5))
    obs = c.Observation(context=context, observation_id="obs-0", spec_ref=space.spec_id, env_step=0, sim_time_s=0.,
        sensors={sid: sample(sid) for sid in sensors},
        robots={"robot0": c.RobotState(joints={
            "left_j0": c.JointState(position=0., velocity=.25, effort=.1),
            "right_j0": c.JointState(position=.5)},
            end_effectors={"left_eef": eff(-.2), "right_eef": eff(.2)},
            base_pose=c.Pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.), frame_id="world"))},
        frame_transforms={fid: c.TimedTransform(transform=IDENTITY, captured_at_s=0.)
                          for fid in ("left_tcp", "right_tcp", "private_goal")},
        extensions={"bench.private": {"secret_score": 987654321}, "example.robot_metadata": {"version": 1}})
    task = c.TaskSpec(task_id="synthetic-task", protocol_id=space.protocol_id, protocol_version=space.protocol_version,
        episode_id=context.episode_id, public=c.PublicTask(instruction="Move the two arms to the public targets."),
        initialization_ref="private-init@1", observation_policy_ref="synthetic-visibility@1",
        budgets=c.Budgets(env_steps=10, model_calls=10, tool_calls=3, sim_time_s=.5, wall_time_s=30.),
        evaluation_config_ref="private-evaluation@1", evaluation_private={"hidden_success": True, "secret_goal": "do-not-send"})
    return space, cap, profile, task, obs
