"""Synthetic native failure observation; no GPU, planner or task score."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import numpy as np
from embodied_eval import _legacy_contracts as leaf
from embodied_eval.mn import contracts as c
from embodied_eval.mn.native_benchmarks import NativeSimulator, observe_native_motion_failures


class ExecutionFailureTests(unittest.TestCase):
    def test_pose_failure_observed_and_native_methods_restored(self):
        failed_result = {"status": "Failure"}
        robot = SimpleNamespace(left_plan_path=lambda p: {"status": "Success", "position": np.zeros((1, 6))},
            right_plan_path=lambda p: failed_result)
        original = robot.right_plan_path
        with observe_native_motion_failures(robot, "ee") as failures:
            self.assertIs(robot.right_plan_path([0.]), failed_result)
            robot.left_plan_path([0.])
            self.assertEqual(failures, {"right"})
        self.assertIs(robot.right_plan_path, original)

    def test_empty_native_joint_trajectory_not_reported_as_executed(self):
        result = (None, np.zeros((0, 6)), None, None, None)
        original = SimpleNamespace(TOPP=lambda *a, **kw: result, other="preserved")
        robot = SimpleNamespace(left_mplib_planner=original, right_mplib_planner=original)
        with observe_native_motion_failures(robot, "qpos") as failures:
            self.assertEqual(robot.left_mplib_planner.other, "preserved")
            self.assertIs(robot.left_mplib_planner.TOPP([], .004), result)
            self.assertEqual(failures, {"left"})
        self.assertIs(robot.left_mplib_planner, original)

    def test_native_joint_exception_still_reaches_native_catcher(self):
        error = ValueError("synthetic planner")
        def fail(*args, **kwargs):
            raise error
        robot = SimpleNamespace(left_mplib_planner=SimpleNamespace(TOPP=fail), right_mplib_planner=SimpleNamespace(TOPP=fail))
        with observe_native_motion_failures(robot, "qpos") as failures:
            try:
                robot.left_mplib_planner.TOPP([])
            except ValueError as observed:
                self.assertIs(observed, error)
            self.assertEqual(failures, {"left"})

    def test_methods_restored_when_execution_raises(self):
        original = lambda p: {"status": "Success", "position": np.zeros((1, 6))}
        robot = SimpleNamespace(left_plan_path=original, right_plan_path=original)
        with self.assertRaises(RuntimeError):
            with observe_native_motion_failures(robot, "ee"):
                raise RuntimeError("synthetic execution error")
        self.assertIs(robot.left_plan_path, original)
        self.assertIs(robot.right_plan_path, original)

    def test_partial_pose_failure_preserves_counts_but_not_false_arm_confirmation(self):
        sim = NativeSimulator.__new__(NativeSimulator)
        robot = SimpleNamespace(left_plan_path=lambda p: {"status": "Success", "position": np.zeros((1, 6))},
            right_plan_path=lambda p: {"status": "Failure"})
        def execute(row, remaining):
            robot.left_plan_path(row[:7])
            robot.right_plan_path(row[8:15])
            return SimpleNamespace(official_success=False, native_action_limit_reached=False, error_type=None,
                consumed_physics_steps=50, elapsed_sim_time_s=.2, cumulative_sim_time_s=.2)
        sim.runtime = SimpleNamespace(_scope=nullcontext, env=SimpleNamespace(robot=robot),
            executor=SimpleNamespace(execute=execute), _snapshot=lambda: None)
        sim._project = lambda obs: None
        cmds = tuple(cmd for side in ("left", "right") for cmd in (
            leaf.PoseTarget(command_id=side + "p", target_id="robot0/" + side + "_eef",
                pose=leaf.Pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.), frame_id="world")),
            leaf.GripperCommand(command_id=side + "g", target_id="robot0/" + side + "_grip", mode="open_fraction", value=.5)))
        ctx = c.Context(run_id="r", env_id="e", episode_id="ep", message_id="request")
        result = sim._advance_robotwin(c.Action(commands=cmds), ctx, leaf.Budgets(env_steps=2))
        self.assertEqual(result.execution_feedback.code, "planning_failed")
        from embodied_eval.codec import decode
        self.assertEqual(decode(leaf.ExecutionFeedback, result.execution_feedback), result.execution_feedback)
        self.assertEqual(result.execution_feedback.status, "execution_failed")
        self.assertEqual(result.execution_feedback.command_ids, ("leftp", "leftg", "rightg"))
        self.assertEqual(result.execution_feedback.consumed_physics_steps, 50)
        self.assertAlmostEqual(result.execution_feedback.elapsed_sim_time_s, .2)


if __name__ == "__main__":
    unittest.main()
