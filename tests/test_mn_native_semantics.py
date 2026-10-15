"""SYNTHETIC executor probes; no simulator, GPU, model or official score."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest

from embodied_eval import contracts as leaf
from embodied_eval.errors import ContractError
from embodied_eval.mn import contracts as c
from embodied_eval.mn.binding import check_compatibility, resolve
from embodied_eval.mn.native_catalog import space
from embodied_eval.mn.validation import validate_step


class CanonicalNativeSemanticsTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional native client NumPy unavailable")
        self.np = np
        self.space = space("robotwin", {"repo": "/not-loaded", "protocol": "seeded-instruction", "task_name": "adjust_bottle"})
        self.context = c.Context(run_id="r", env_id="e", episode_id="episode", message_id="request")

    def bench(self):
        from embodied_eval.mn.native_benchmarks import NativeSimulator
        runtime = SimpleNamespace(raw={}, captured_at=0., _scope=nullcontext)
        pose_plan = lambda p: {"status": "Success", "position": self.np.zeros((1, 6))}
        joint_plan = lambda *a, **kw: (None, self.np.zeros((1, 6)), None, None, None)
        runtime.env = SimpleNamespace(get_obs=lambda: {"fresh": True}, robot=SimpleNamespace(
            left_plan_path=pose_plan, right_plan_path=pose_plan,
            left_mplib_planner=SimpleNamespace(TOPP=joint_plan), right_mplib_planner=SimpleNamespace(TOPP=joint_plan)))
        def execute(row, remaining):
            runtime.row = row.copy()
            if runtime.executor.action_type == "ee":
                runtime.env.robot.left_plan_path(row[:7])
                runtime.env.robot.right_plan_path(row[8:15])
            else:
                runtime.env.robot.left_mplib_planner.TOPP(row[:6], .004)
                runtime.env.robot.right_mplib_planner.TOPP(row[7:13], .004)
            return SimpleNamespace(official_success=False, native_action_limit_reached=False, error_type=None,
                cumulative_sim_time_s=.036, consumed_physics_steps=9, elapsed_sim_time_s=.036)
        runtime.executor = SimpleNamespace(execute=execute, action_type=None)
        obs = c.Observation(context=replace(self.context, message_id="after"), observation_id="after", spec_ref=self.space.spec_id,
                            env_step=1, sim_time_s=.036, sensors={}, states={})
        runtime._snapshot = lambda: obs
        simulator = NativeSimulator.__new__(NativeSimulator)
        simulator.runtime, simulator.family, simulator.space = runtime, "robotwin", self.space
        simulator.cfg = SimpleNamespace(protocol="seeded-instruction")
        runtime.diagnostics = lambda: {"variant": "legacy-label", "camera_cadence": "legacy-combination-label"}
        simulator._project = lambda value: value
        return simulator, runtime

    def dual_capability(self):
        targets = {side + "_eef": "end_effector" for side in ("left", "right")}
        targets.update({side + "_grip": "gripper" for side in ("left", "right")})
        return c.CapabilitySpec(model_id="synthetic-dual-poses", source_revision="fixture", checkpoint_ref="synthetic:none",
            normalization_ref="synthetic:none", inputs=(), output=c.OutputSpec(command_kinds=("pose_target", "gripper_command"),
                target_slots=targets, gripper_modes={side + "_grip": ("open_fraction",) for side in ("left", "right")}),
            supported_timing_modes=("measured_native",))

    def test_variable_canonical_dual_pose_executor(self):
        cap = self.dual_capability()
        # Explicit crossed arm wiring is data; neither Model nor Bench changes.
        mapping = {slot: "robot0/" + slot.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right")
                   for slot in cap.output.target_slots}
        binding = c.BindingConfig(inputs={}, targets=mapping, action_spec_ref="dual-cartesian-absolute@1")
        resolve(self.space, cap, binding)
        commands = []
        for side, position in (("left", (.1, .2, .3)), ("right", (.4, .5, .6))):
            commands.extend((leaf.PoseTarget(command_id=side + "-pose", target_id=mapping[side + "_eef"],
                pose=leaf.Pose(position_m=position, quaternion_xyzw=(0., 0., 0., 1.), frame_id="world")),
                leaf.GripperCommand(command_id=side + "-grip", target_id=mapping[side + "_grip"], mode="open_fraction", value=.5)))
        action = c.Action(commands=tuple(commands))
        benchmark, runtime = self.bench()
        reply = benchmark.advance(action, self.context, leaf.Budgets(env_steps=2))
        self.assertEqual(runtime.executor.action_type, "ee")
        self.assertEqual(runtime.row.tolist(), [.4, .5, .6, 1., 0., 0., 0., .5, .1, .2, .3, 1., 0., 0., 0., .5])
        self.assertEqual(reply.execution_feedback.consumed_physics_steps, 9)
        self.assertEqual(reply.execution_feedback.command_ids, tuple(cmd.command_id for cmd in commands))
        diagnostics = benchmark.diagnostics()
        self.assertNotIn("variant", diagnostics["native"])
        self.assertIn("independent of model family", diagnostics["native"]["camera_cadence"])
        self.assertEqual(diagnostics["native_environment_loader_legacy_labels"]["loader_variant_label"], "legacy-label")

    def test_variable_canonical_joint_executor(self):
        values = self.np.arange(14, dtype=self.np.float32) / 16
        commands = []
        for side, offset in (("left", 0), ("right", 7)):
            commands.extend(leaf.JointTarget(command_id=side + str(i), target_id=f"robot0/{side}_j{i}", mode="position",
                unit="rad", value=float(values[offset + i])) for i in range(6))
            commands.append(leaf.GripperCommand(command_id=side + "-grip", target_id=f"robot0/{side}_grip",
                mode="open_fraction", value=float(values[offset + 6])))
        benchmark, runtime = self.bench()
        reply = benchmark.advance(c.Action(commands=tuple(commands), native_dtype="float32"), self.context, leaf.Budgets(env_steps=2))
        self.assertEqual(runtime.executor.action_type, "qpos")
        self.assertEqual(runtime.row.dtype, values.dtype)
        self.assertTrue(self.np.array_equal(runtime.row, values))
        self.assertEqual(reply.execution_feedback.elapsed_sim_time_s, .036)
        # Raw normalized requests MUST NOT be clipped before native interpolation.
        from embodied_eval.mn.native_catalog import GRIPPER_PROTOCOL, GRIPPER_CONTROLLER, capability
        values[6],values[13]=1.02,-.03
        raw=[]
        for command in commands:
            if isinstance(command,leaf.GripperCommand):
                index=6 if "left" in command.target_id else 13
                raw.append(leaf.ControllerCommand(command_id=command.command_id,target_id=command.target_id,
                    controller_profile_ref=GRIPPER_CONTROLLER,parameters={"requested_open_fraction":float(values[index])}))
            else:
                raw.append(command)
        cap=capability("lingbot_vla",{"repo":"/not-loaded","checkpoint_dir":"/not-loaded","io_profile":"joint14"})
        binding=c.BindingConfig(inputs={req.key:"instruction" if req.kind=="task" else {"head":"head_camera","left_wrist":"left_camera", "right_wrist":"right_camera", "drive_targets":"joint_drive_targets"}[req.key] for req in cap.inputs},
            targets={key:"robot0/"+key for key in cap.output.target_slots},action_spec_ref="joint-absolute-native-gripper@1")
        resolve(self.space,cap,binding)
        result=benchmark.advance(c.Action(commands=tuple(raw),native_dtype="float32",native_protocol=GRIPPER_PROTOCOL),self.context,leaf.Budgets(env_steps=2))
        self.assertTrue(self.np.array_equal(runtime.row,values))
        self.assertGreater(runtime.row[6],1.)
        self.assertLess(runtime.row[13],0.)
        self.assertEqual(result.execution_feedback.consumed_physics_steps,9)

    def test_single_arm_cannot_satisfy_atomic_dual_execution(self):
        cap = self.dual_capability()
        cap = replace(cap, output=replace(cap.output, target_slots={key: value for key, value in cap.output.target_slots.items() if key.startswith("left")},
            gripper_modes={"left_grip": ("open_fraction",)}))
        binding = c.BindingConfig(inputs={}, targets={key: "robot0/" + key for key in cap.output.target_slots}, action_spec_ref="dual-cartesian-absolute@1")
        self.assertIn("missing_execution_target", {issue.code for issue in check_compatibility(self.space, cap, binding).issues})

    def test_declared_joint_mode_mismatch_rejected(self):
        from embodied_eval.mn.synthetic import capability, space as synthetic_space
        cap = capability({})
        cap = replace(cap, output=replace(cap.output, joint_modes=("velocity",)))
        binding = c.BindingConfig(inputs={"image": "camera", "instruction": "instruction"}, targets={"joint": "robot0/j0"}, action_spec_ref="joint-position@1")
        self.assertIn("unsupported_joint_mode", {issue.code for issue in check_compatibility(synthetic_space({}), cap, binding).issues})

    def test_gripper_width_not_treated_as_fraction(self):
        cap = self.dual_capability()
        cap = replace(cap, output=replace(cap.output, gripper_modes={side + "_grip": ("width_m",) for side in ("left", "right")}))
        binding = c.BindingConfig(inputs={}, targets={key: "robot0/" + key for key in cap.output.target_slots}, action_spec_ref="dual-cartesian-absolute@1")
        self.assertIn("unsupported_gripper_mode", {issue.code for issue in check_compatibility(self.space, cap, binding).issues})

    def test_executed_feedback_cannot_omit_one_command(self):
        from embodied_eval.mn.synthetic import SyntheticBenchmark
        env = SyntheticBenchmark({})
        obs = env.reset(None, self.context)
        action = c.Action(commands=(leaf.JointTarget(command_id="command", target_id="robot0/j0", mode="position", unit="rad", value=0.),))
        reply = env.advance(action, self.context, leaf.Budgets(env_steps=1))
        reply = replace(reply, execution_feedback=replace(reply.execution_feedback, command_ids=()))
        with self.assertRaises(ContractError):
            validate_step(reply, obs, self.context, action, env.space)
