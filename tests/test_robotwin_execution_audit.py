"""Synthetic saved-row/observer safety tests; not a real robot evaluation."""
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from replay_mn_native_actions import robotwin_action_row, snapshot
from robotwin_execution_observer import ExecutionObserver, array, rng_fingerprints
from compare_robotwin_execution_audits import compare
import build_repro_followup_report as report_builder


def pose_action():
    commands = []
    for side in ("left", "right"):
        commands.extend([{"target_id": "robot0/" + side + "_eef", "kind": "pose_target",
            "pose": {"frame_id": "world", "position_m": [.25, -.5, .75], "quaternion_xyzw": [0., 0., 0., 1.]}},
            {"target_id": "robot0/" + side + "_grip", "kind": "controller_command",
             "controller_profile_ref": "gripper-open-fraction-raw-request@1", "parameters": {"requested_open_fraction": 1.00390625}}])
    return {"commands": commands, "native_dtype": "float32", "native_protocol": "gripper-open-fraction-interpolate-then-clamp@1"}


class RecordedRowTests(unittest.TestCase):
    def test_physical_pose_explicit_xyzw_to_native_wxyz_both_arms(self):
        row, mode = robotwin_action_row(pose_action())
        self.assertEqual(mode, "ee")
        self.assertEqual(row.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(row, [.25, -.5, .75, 1., 0., 0., 0., 1.00390625] * 2)

    def test_legacy_extension_keeps_bytes_and_dtype(self):
        values = np.arange(16, dtype=np.float32)
        action = {"native_protocol": "dual-eef16-legacy-quaternion@1", "commands": [{"target_id": "robot0/base",
            "parameters": {"native_row": values.tolist(), "native_dtype": "float32"}}]}
        row, mode = robotwin_action_row(action)
        self.assertEqual((row.tobytes(), mode), (values.tobytes(), "ee"))

    def test_joint_route_keeps_values_and_raw_grippers(self):
        commands = []
        for side in ("left", "right"):
            commands.extend({"target_id": "robot0/" + side + f"_j{i}", "kind": "joint_target", "value": i*.125} for i in range(6))
            commands.append({"target_id": "robot0/" + side + "_grip", "kind": "controller_command",
                "parameters": {"requested_open_fraction": -.00390625}})
        row, mode = robotwin_action_row({"commands": commands, "native_protocol": "gripper-open-fraction-interpolate-then-clamp@1", "native_dtype": "float64"})
        self.assertEqual(mode, "qpos")
        np.testing.assert_array_equal(row, [0, .125, .25, .375, .5, .625, -.00390625]*2)

    def test_missing_arm_rejected(self):
        action = pose_action(); action["commands"].pop()
        with self.assertRaises(ValueError):
            robotwin_action_row(action)

    def test_duplicate_target_rejected(self):
        action = pose_action(); action["commands"].append(copy.deepcopy(action["commands"][0]))
        with self.assertRaises(ValueError):
            robotwin_action_row(action)

    def test_wrong_frame_rejected(self):
        action = pose_action(); action["commands"][0]["pose"]["frame_id"] = "unknown_base"
        with self.assertRaises(ValueError):
            robotwin_action_row(action)

    def test_wrong_native_gripper_protocol_rejected(self):
        action = pose_action(); action["native_protocol"] = "unknown"
        with self.assertRaises(ValueError):
            robotwin_action_row(action)

    def test_lossy_dtype_rejected_in_physical_pose(self):
        action = pose_action(); action["commands"][0]["pose"]["position_m"][0] = .123456789
        with self.assertRaises(ValueError):
            robotwin_action_row(action)

    def test_nonfinite_pose_rejected(self):
        action = pose_action(); action["commands"][0]["pose"]["position_m"][0] = float("nan")
        with self.assertRaises(ValueError):
            robotwin_action_row(action)

    def test_unknown_command_not_interpreted_as_joint(self):
        with self.assertRaises(ValueError):
            robotwin_action_row({"commands": [{"kind": "twist", "target_id": "robot0/base"}], "native_protocol": None, "native_dtype": "float32"})


class FakeEntity:
    def __init__(self):
        self.qpos = np.zeros(2, np.float32)
    def get_qpos(self):
        return self.qpos
    def get_qvel(self):
        return np.zeros(2, np.float32)


class FakeMotion:
    def plan_single(self, start, goal):
        return SimpleNamespace(success=np.array([True]), interpolated_plan=SimpleNamespace(position=start.position.copy(), velocity=np.zeros_like(start.position)))


class FakeRobot:
    def __init__(self):
        self.left_entity, self.right_entity = FakeEntity(), FakeEntity()
        self.commands = []
    def left_plan_path(self, pose):
        result = FakeMotion().plan_single(SimpleNamespace(position=np.ones((2, 2), np.float32)),
            SimpleNamespace(position=np.zeros((1, 3), np.float32), quaternion=np.array([[1., 0., 0., 0.]], np.float32)))
        return {"status": "Success", "position": result.interpolated_plan.position, "velocity": result.interpolated_plan.velocity}
    right_plan_path = left_plan_path
    def set_arm_joints(self, position, velocity, side):
        self.commands.append((np.asarray(position).copy(), side))
    def set_gripper(self, value, side):
        self.commands.append((float(value), side))


class FakeSystem:
    def __init__(self, robot):
        self.robot, self.steps = robot, 0
    def step(self):
        self.steps += 1
        self.robot.left_entity.qpos += np.float32(.125)


class ObserverSafetyTests(unittest.TestCase):
    def env(self):
        robot = FakeRobot()
        return SimpleNamespace(robot=robot, scene=SimpleNamespace(physx_system=FakeSystem(robot)))

    def test_observer_preserves_plans_controls_step_counts_and_rng(self):
        env, audit = self.env(), ExecutionObserver()
        before = rng_fingerprints()
        with audit.watch(env, FakeMotion):
            audit.begin_action(1)
            result = env.robot.left_plan_path(np.zeros(7, np.float32))
            env.robot.set_arm_joints(result["position"][0], result["velocity"][0], "left")
            env.robot.set_gripper(1.00390625, "right")
            env.scene.physx_system.step()
        self.assertEqual(before, rng_fingerprints())
        self.assertEqual(env.scene.physx_system.steps, 1)
        self.assertEqual(len(env.robot.commands), 2)
        row = audit.actions[0]
        self.assertEqual([len(row[key]) for key in ("plans", "motion_gen", "controls", "physics")], [1, 1, 2, 1])
        np.testing.assert_array_equal(row["physics"][0]["before"]["left"]["qpos"]["values"], [0, 0])
        np.testing.assert_array_equal(row["physics"][0]["after"]["left"]["qpos"]["values"], [.125, .125])

    def test_all_bindings_restored_after_failure(self):
        env, audit = self.env(), ExecutionObserver()
        step, motion = FakeSystem.step, FakeMotion.plan_single
        with self.assertRaises(RuntimeError):
            with audit.watch(env, FakeMotion):
                audit.begin_action(1)
                raise RuntimeError("synthetic")
        self.assertIs(FakeSystem.step, step)
        self.assertIs(FakeMotion.plan_single, motion)
        for name in ("left_plan_path", "right_plan_path", "set_arm_joints", "set_gripper"):
            self.assertNotIn(name, env.robot.__dict__)

    def test_other_physics_system_is_not_counted(self):
        env, other, audit = self.env(), self.env(), ExecutionObserver()
        with audit.watch(env, FakeMotion):
            audit.begin_action(1)
            other.scene.physx_system.step()
        self.assertEqual(audit.actions[0]["physics"], [])
        self.assertEqual(other.scene.physx_system.steps, 1)

    def test_record_is_copied_not_a_mutable_native_view(self):
        values = np.ones(3, np.float32)
        encoded = array(values); values[:] = 0
        self.assertEqual(encoded["values"], [1., 1., 1.])

    def test_nonfinite_observation_rejected(self):
        with self.assertRaises(ValueError):
            array([float("inf")])

    def test_action_sequence_explicit(self):
        with self.assertRaises(ValueError):
            ExecutionObserver().begin_action(2)

    def saved(self):
        env, audit = self.env(), ExecutionObserver()
        with audit.watch(env, FakeMotion):
            audit.begin_action(1)
            env.robot.left_plan_path(np.zeros(7, np.float32))
            env.robot.right_plan_path(np.zeros(7, np.float32))
            env.robot.set_arm_joints([1., 1.], [0., 0.], "left")
            env.scene.physx_system.step()
        return audit.actions

    def test_saved_plan_replay_is_explicit_and_does_not_call_motion_generator(self):
        saved = self.saved()
        env, audit = self.env(), ExecutionObserver(saved)
        with audit.watch(env, FakeMotion):
            audit.begin_action(1)
            result = env.robot.left_plan_path(np.zeros(7, np.float32))
        self.assertEqual(audit.actions[0]["motion_gen"], [])
        self.assertEqual(array(result["position"]), saved[0]["plans"][0]["result"]["position"])
        self.assertTrue(audit.actions[0]["plans"][0]["recorded_initial_articulation_exact"])

    def test_saved_plan_wrong_target_rejected_and_bindings_restored(self):
        saved = self.saved()
        env, audit = self.env(), ExecutionObserver(saved)
        with self.assertRaises(ValueError):
            with audit.watch(env, FakeMotion):
                audit.begin_action(1)
                env.robot.left_plan_path(np.ones(7, np.float32))
        self.assertNotIn("left_plan_path", env.robot.__dict__)


class EvidenceComparisonTests(unittest.TestCase):
    def test_git_chinese_paths_are_read_without_quoted_escapes(self):
        with patch.object(report_builder, "git", return_value="docs/GPU实例交接文档.md\0README.md\0") as call:
            self.assertEqual(report_builder.tracked_paths(), ["docs/GPU实例交接文档.md", "README.md"])
        call.assert_called_once_with("ls-files", "-z")

    def test_git_space_and_newline_paths_are_not_split(self):
        with patch.object(report_builder, "git", return_value="docs/with space.md\0docs/with\nnewline.md\0"):
            self.assertEqual(report_builder.tracked_paths(), ["docs/with space.md", "docs/with\nnewline.md"])

    def receipt(self):
        return {"completed": True, "observer_enabled": True, "manifest_sha256": "synthetic-manifest",
            "source_assets_verified": {"synthetic": True}, "package_versions": {"numpy": "synthetic"},
            "mode": "native", "planning_mode": "unaltered_native_planner", "plan_source_sha256": None,
            "actions": ObserverSafetyTests().saved()}

    def test_identical_record_does_not_report_planning_difference(self):
        a = self.receipt(); result = compare(a, copy.deepcopy(a))
        self.assertEqual(result["same_recorded_MotionGen_input_different_output_witnesses"], [])
        self.assertTrue(result["all_control_operations_exact"])
        self.assertTrue(result["all_recorded_articulation_physics_states_exact"])

    def test_same_input_changed_output_is_evidence_not_internal_root_proof(self):
        a = self.receipt(); b = copy.deepcopy(a)
        b["actions"][0]["motion_gen"][1]["position"] = array(np.zeros((2, 2), np.float32))
        result = compare(a, b)
        self.assertEqual(result["same_recorded_MotionGen_input_different_output_witnesses"], [{"action_index": 1, "side": "right", "global_rng_equal": True}])
        self.assertTrue(result["native_planner_output_variation_observed_without_Adapter"])
        self.assertFalse(result["internal_planner_rng_all_state_or_kernel_root_uniquely_identified"])

    def test_changed_manifest_rejected(self):
        a = self.receipt(); b = copy.deepcopy(a); b["manifest_sha256"] = "other"
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_unfinished_observer_evidence_rejected(self):
        a = self.receipt(); b = copy.deepcopy(a); b["completed"] = False
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_mixed_real_and_saved_planning_not_comparable(self):
        a = self.receipt(); b = copy.deepcopy(a); b["planning_mode"] = "saved_native_plan_controlled_diagnostic_not_evaluation"
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_fresh_qpos_and_stale_camera_can_have_different_correct_times(self):
        observation = {"states": {"left_joint_positions": {"captured_at_s": 2., "payload": {"data": [0., 1.]}}},
            "sensors": {"head_camera": {"captured_at_s": 1., "payload": {"data": [[[0, 0, 0]]]}}}}
        result = snapshot(observation, {"left_joint_positions": [0., 1.]}, {"head_camera": [[[0, 0, 0]]]},
            acquired_at=1., state_acquired_at={"left_joint_positions": 2.})
        self.assertTrue(all(value["exact_equal"] for value in result.values()))

    def test_wrong_qpos_time_not_hidden_by_matching_camera_time(self):
        observation = {"states": {"left_joint_positions": {"captured_at_s": 1., "payload": {"data": [0., 1.]}}}, "sensors": {}}
        result = snapshot(observation, {"left_joint_positions": [0., 1.]}, {}, acquired_at=1.,
            state_acquired_at={"left_joint_positions": 2.})
        self.assertFalse(result["left_joint_positions:acquisition_time"]["exact_equal"])


if __name__ == "__main__":
    unittest.main()
