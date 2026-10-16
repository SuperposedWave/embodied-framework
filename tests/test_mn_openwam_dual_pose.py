"""Synthetic boundary/architecture tests, NOT real weights or robot execution.

The fake native codec is intentionally independent geometry. Executing the
fixed upstream training reader/codec on saved real samples is a separate gate.
"""
from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import numpy as np

from embodied_eval import _legacy_contracts as leaf
from embodied_eval.errors import AdapterError, ContractError
from embodied_eval.geometry import quaternion_to_rot6d, rot6d_to_quaternion, normalize_quaternion
from embodied_eval.config import load_config
from embodied_eval.mn import contracts as c
from embodied_eval.mn.binding import check_compatibility, resolve
from embodied_eval.mn.native_catalog import capability, space, GRIPPER_PROTOCOL
from embodied_eval.mn.native_models import NativeActionModel
from embodied_eval.mn.native_benchmarks import NativeSimulator
from embodied_eval.mn.openwam_dual_pose import PROFILE, encode_state, decode_action
from embodied_eval.v03.preflight import check
from embodied_eval.mn.validation import validate_prediction


class SyntheticCodec:
    @staticmethod
    def robotwin_endpose_to_eef20d(left, right, lg, rg):
        rows = []
        for pose, grip in ((left, lg), (right, rg)):
            pose = np.asarray(pose, np.float32)
            rotation = quaternion_to_rot6d(normalize_quaternion(pose[3:]))
            rows.extend((*pose[:3], *rotation, grip))
        return np.asarray(rows, np.float32)

    @staticmethod
    def eef20d_to_ee16d(raw):
        rows = []
        for offset in (0, 10):
            q = rot6d_to_quaternion(raw[offset+3:offset+9])
            rows.extend((*raw[offset:offset+3], *q, raw[offset+9]))
        return np.asarray(rows, np.float32)


class OpenWAMEncodedDualPoseTests(unittest.TestCase):
    def setUp(self):
        self.options = {"io_profile": PROFILE, "repo": "/not-loaded", "checkpoint_dir": "/not-loaded"}
        self.cap = capability("openwam", self.options)
        self.space = space("robotwin", {"repo": "/not-loaded", "protocol": "sealed-instruction", "task_name": "adjust_bottle"})
        self.context = c.Context(run_id="synthetic", env_id="env", episode_id="ep", message_id="request")
        inputs = {"instruction": "instruction", "head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}
        for side in ("left", "right"):
            inputs.update({side+"_position": side+"_eef_position", side+"_quaternion": side+"_eef_quaternion",
                           side+"_grip": side+"_gripper_open_fraction"})
        self.targets = {key: "robot0/"+key for key in self.cap.output.target_slots}
        self.binding = c.BindingConfig(inputs=inputs, targets=self.targets, action_spec_ref="dual-cartesian-native-gripper@1")
        self.profile = resolve(self.space, self.cap, self.binding)
        self.states = {"left_position": [.125, .25, .5], "right_position": [-.25, .125, .75],
                       "left_quaternion": [0., 0., 0., 1.], "right_quaternion": [0., 0., 0., 1.],
                       "left_grip": [1.], "right_grip": [.25]}

    def raw(self, q=(0., 0., 0., 1.)):
        self.states["left_quaternion"] = q
        return encode_state(self.states.__getitem__, SyntheticCodec)

    def action(self, raw):
        return decode_action(raw, self.context, self.targets, SyntheticCodec)

    def assert_rotation(self, q, expected):
        a, b = np.asarray(q, np.float64), np.asarray(expected, np.float64)
        a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
        # q and -q are the SAME physical orientation.
        self.assertLess(min(np.linalg.norm(a-b), np.linalg.norm(a+b)), 3e-7)

    def view(self):
        samples = {key: c.StateSample(captured_at_s=0., provenance="controller_target" if key.endswith("grip") else "measurement",
                   payload=leaf.Tensor(dtype="float64", shape=(len(value),), layout="vector", data=list(value)))
                   for key, value in self.states.items()}
        sensors = {key: leaf.SensorSample(captured_at_s=0., frame_id=key, validity="valid", provenance="sensor",
                   payload=leaf.Tensor(dtype="uint8", shape=(2, 3, 3), layout="HWC", data=np.full((2,3,3), i, np.uint8).tolist()))
                   for i, key in enumerate(("head", "left_wrist", "right_wrist"))}
        return c.ModelInput(context=self.context, observation_id="obs", sim_time_s=0.,
                            task_public=leaf.PublicTask(instruction="synthetic instruction"), sensors=sensors, states=samples)

    def model(self):
        model = NativeActionModel("openwam", self.options)
        model.profile, model.space, model.conversion = self.profile, self.space, SyntheticCodec
        return model

    def test_named_profile_has_same_weights_not_same_ambiguous_output(self):
        old = capability("openwam", {**self.options, "io_profile": "eef20_dual_legacy"})
        self.assertEqual((old.source_revision, old.checkpoint_ref, old.normalization_ref),
                         (self.cap.source_revision, self.cap.checkpoint_ref, self.cap.normalization_ref))
        self.assertNotEqual(old.model_id, self.cap.model_id)
        self.assertEqual(old.output.command_kinds, ("controller_command",))
        self.assertEqual(self.cap.output.command_kinds, ("pose_target", "controller_command"))
        self.assertEqual(self.cap.output.native_protocol, GRIPPER_PROTOCOL)

    def test_example_preflight_is_structural_not_assets_or_execution(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/"examples/mn_openwam_dual_wxyz_encoded_absolute.json")
        with patch.object(NativeActionModel, "__init__", side_effect=AssertionError("model factory called")), \
             patch.object(NativeSimulator, "__init__", side_effect=AssertionError("environment factory called")):
            self.assertTrue(check(cfg)["structurally_compatible"])
        import importlib.util
        import sys
        root=Path(__file__).resolve().parents[1]
        path=root/"docs/handoff/run_openwam_dual_pose_comparison.py"
        spec=importlib.util.spec_from_file_location("synthetic_comparison_config_probe",path)
        module=importlib.util.module_from_spec(spec)
        with patch.object(sys,"path",[str(root/"docs/handoff"),*sys.path]):
            spec.loader.exec_module(module)
        canonical=module.canonical_config(cfg)
        self.assertEqual(set(canonical["episodes"][0]["budgets"]),{"env_steps","sim_time_s","model_calls","tool_calls","wall_time_s"})
        self.assertIsNone(canonical["episodes"][0]["budgets"]["sim_time_s"])

    def test_identity_requires_special_encoding_not_naive_identity_rot6d(self):
        raw = self.raw()
        np.testing.assert_array_equal(raw[3:9], [1., 0., 0., 0., -1., 0.])
        action = self.action(raw)
        self.assert_rotation(action.commands[0].pose.quaternion_xyzw, [0., 0., 0., 1.])
        naive = SyntheticCodec.eef20d_to_ee16d(raw)[3:7]
        self.assertGreater(abs(float(naive[0])), .99)  # naive tagging would rotate X by 180 degrees

    def test_three_axes_ninety_degrees(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                q = np.zeros(4); q[axis] = math.sqrt(.5); q[3] = math.sqrt(.5)
                self.assert_rotation(self.action(self.raw(q)).commands[0].pose.quaternion_xyzw, q)

    def test_three_axes_one_hundred_eighty_degrees(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                q = np.zeros(4); q[axis] = 1.
                self.assert_rotation(self.action(self.raw(q)).commands[0].pose.quaternion_xyzw, q)

    def test_general_rotation_and_both_arms(self):
        left = normalize_quaternion((.2, -.3, .4, .7))
        right = normalize_quaternion((-.1, .6, -.2, .3))
        self.states["right_quaternion"] = right
        action = self.action(self.raw(left))
        self.assert_rotation(action.commands[0].pose.quaternion_xyzw, left)
        self.assert_rotation(action.commands[2].pose.quaternion_xyzw, right)
        self.assertEqual({x.target_id for x in action.commands}, set(self.targets.values()))

    def test_quaternion_sign_equivalence(self):
        q = np.asarray(normalize_quaternion((.3, .2, -.6, .4)))
        np.testing.assert_array_equal(self.raw(q), self.raw(-q))
        original = self.action(self.raw(q))
        class NegatedCodec(SyntheticCodec):
            @staticmethod
            def eef20d_to_ee16d(raw):
                native = SyntheticCodec.eef20d_to_ee16d(raw)
                native[3:7] *= -1; native[11:15] *= -1
                return native
        negated = decode_action(self.raw(q), self.context, self.targets, NegatedCodec)
        for offset in (0, 2):
            self.assert_rotation(negated.commands[offset].pose.quaternion_xyzw, original.commands[offset].pose.quaternion_xyzw)

    def test_roundtrip_preserves_native_float32_numbers_and_raw_grippers(self):
        raw = self.raw(normalize_quaternion((.2, .3, -.4, .8)))
        raw[9], raw[19] = np.float32(1.00390625), np.float32(-.00390625)
        action = self.action(raw)
        actual, mode = NativeSimulator.robotwin_native_action(action)
        np.testing.assert_array_equal(actual, SyntheticCodec.eef20d_to_ee16d(raw))
        self.assertEqual(actual.dtype, np.dtype("float32"))
        self.assertEqual(mode, "ee")
        self.assertEqual(action.commands[1].parameters["requested_open_fraction"], 1.00390625)
        self.assertEqual(action.commands[3].parameters["requested_open_fraction"], -.00390625)

    def test_explicit_crossed_arm_targets_not_positional_guess(self):
        targets = {key: target.replace("left", "TEMP").replace("right", "left").replace("TEMP", "right") for key, target in self.targets.items()}
        action = decode_action(self.raw(), self.context, targets, SyntheticCodec)
        row, _ = NativeSimulator.robotwin_native_action(action)
        np.testing.assert_array_equal(row[:3], self.states["right_position"])
        np.testing.assert_array_equal(row[8:11], self.states["left_position"])

    def test_input_payload_images_prompt_and_state_match_legacy(self):
        view = self.view()
        old = NativeActionModel("openwam", {**self.options, "io_profile": "eef20_dual_legacy"})
        new = self.model()
        utils = SimpleNamespace(encode_numpy_b64=lambda frame: (frame.dtype.str, frame.shape, frame.tobytes()),
            resize_for_lshape_slot=lambda *args: self.fail("native dual camera unexpectedly resized"), build_payload=lambda **kwargs: kwargs)
        old.utils, old.conversion, new.utils = utils, SyntheticCodec, utils
        prompt = SimpleNamespace(format_prompt_for_inference=lambda text: "source prompt: "+text)
        with patch("embodied_eval.mn.native_models.native_module", return_value=prompt):
            self.assertEqual(new.native_input(view), old.native_input(view))

    def test_prediction_validation_accepts_physical_poses_and_raw_grippers(self):
        raw = self.raw(); raw[9] = 1.00390625
        action = self.action(raw)
        obs = c.Observation(context=replace(self.context, message_id="obs"), observation_id="obs", spec_ref=self.space.spec_id,
                            env_step=0, sim_time_s=0., sensors={}, states={})
        pred = c.Prediction(context=self.context, based_on_observation_id="obs", action=action)
        validate_prediction(pred, self.context, obs, self.space, self.cap, self.profile)
        self.assertTrue(action.acquire_observation_after)

    def test_single_row_cadence_cache_requests_and_instruction_reset(self):
        row = self.raw()
        class Peer:
            resets = 0
            def reset(self):
                self.resets += 1
                return {"type": "reset_ack"}
            def predict_once(self, payload):
                return {"action": row.copy()}
        model = self.model(); model.peer = Peer(); model.native_input = lambda view: {}
        model.begin_episode(self.context, leaf.PublicTask(instruction="synthetic instruction"), self.profile, self.space)
        model.observe(self.view())
        for index in range(3):
            pred = model.predict(replace(self.context, message_id=f"request-{index}"))
            self.assertTrue(pred.action.acquire_observation_after)
            self.assertIsNone(pred.model_forward_count)
            self.assertEqual(len(model.cache), 0)
        self.assertEqual(model.requests, 3)
        self.assertEqual(model.peer.resets, 1)
        model.observe(replace(self.view(), task_public=leaf.PublicTask(instruction="changed")))
        model.predict(replace(self.context, message_id="changed"))
        self.assertEqual(model.peer.resets, 2)
        model.cache.append(row.copy()); model.end_episode(self.context)
        self.assertFalse(model.cache)
        self.assertIsNone(model.latest)

    def test_wrong_action_width_including_internal_eighty_rejected(self):
        for width in (14, 16, 19, 21, 80):
            with self.subTest(width=width), self.assertRaises(AdapterError):
                self.action(np.zeros(width, np.float32))

    def test_nonfinite_action_rejected(self):
        for value in (float("nan"), float("inf")):
            raw = self.raw(); raw[0] = value
            with self.assertRaises(AdapterError):
                self.action(raw)

    def test_zero_and_parallel_rotation_columns_rejected(self):
        for rotation in ([0.]*6, [1.,0.,0.,2.,0.,0.]):
            for offset in (3, 13):
                raw = self.raw(); raw[offset:offset+6] = rotation
                with self.subTest(offset=offset, rotation=rotation), self.assertRaises(ContractError):
                    self.action(raw)

    def test_zero_nonunit_and_nonfinite_input_quaternions_rejected(self):
        for q in ([0.]*4, [0.,0.,0.,2.], [0.,0.,float("nan"),1.]):
            with self.subTest(q=q), self.assertRaises((ContractError, AdapterError)):
                self.raw(q)

    def test_wrong_input_state_dimensions_rejected(self):
        for key in ("left_position", "right_quaternion", "right_grip"):
            state = dict(self.states); state[key] = [0.]
            if key == "right_grip": state[key] = [0., 0.]
            with self.subTest(key=key), self.assertRaises(AdapterError):
                encode_state(state.__getitem__, SyntheticCodec)

    def test_out_of_bounds_public_gripper_input_rejected(self):
        self.states["right_grip"] = [1.01]
        with self.assertRaises(ContractError):
            encode_state(self.states.__getitem__, SyntheticCodec)

    def test_missing_arm_and_duplicate_target_rejected(self):
        for targets in ({k:v for k,v in self.targets.items() if k != "right_eef"}, {**self.targets, "right_eef": self.targets["left_eef"]}):
            with self.assertRaises(ContractError):
                decode_action(self.raw(), self.context, targets, SyntheticCodec)
            self.assertFalse(check_compatibility(self.space, self.cap, replace(self.binding, targets=targets)).compatible)

    def test_missing_input_and_wrong_state_meaning_rejected(self):
        self.assertFalse(check_compatibility(self.space, self.cap, replace(self.binding,
            inputs={k:v for k,v in self.binding.inputs.items() if k != "right_quaternion"})).compatible)
        states = dict(self.space.state_fields)
        states["left_gripper_open_fraction"] = replace(states["left_gripper_open_fraction"], semantic="gripper_open_fraction")
        self.assertFalse(check_compatibility(replace(self.space, state_fields=states), self.cap, self.binding).compatible)

    def test_unmatched_native_gripper_protocol_rejected(self):
        self.assertFalse(check_compatibility(replace(self.space, native_protocols={}), self.cap, self.binding).compatible)

    def test_lossy_native_output_cast_rejected(self):
        raw = self.raw().astype(np.float64); raw[0] = .1234567890123
        with self.assertRaises(ContractError):
            self.action(raw)

    def test_decoder_cannot_invent_nonunit_physical_quaternion(self):
        class BadCodec(SyntheticCodec):
            @staticmethod
            def eef20d_to_ee16d(raw):
                row = SyntheticCodec.eef20d_to_ee16d(raw); row[3:7] = 0.
                return row
        with self.assertRaises(ContractError):
            decode_action(self.raw(), self.context, self.targets, BadCodec)

    def test_peer_lossy_float64_is_rejected_before_cache_cast(self):
        row=self.raw().astype(np.float64); row[0]=.1234567890123
        model=self.model()
        model.peer=SimpleNamespace(predict_once=lambda payload:{"action":row})
        model.latest=self.view(); model.last_instruction=model.latest.task_public.instruction
        model.native_input=lambda view:{}
        with self.assertRaises(ContractError):
            model.predict(self.context)
        self.assertFalse(model.cache)

    def test_benchmark_public_observation_reorders_physical_wxyz(self):
        q=normalize_quaternion((.2,-.3,.4,.7))
        ext={"captured_at_sim_time_s":0.,"joint_drive_targets_and_grippers":[0.]*14}
        for side in ("left","right"):
            ext[side+"_endpose_world_wxyz"]=[.1,.2,.3,q[3],q[0],q[1],q[2]]
            ext[side+"_gripper_open_fraction"]=.5
        old_context=leaf.Context(run_id="synthetic",env_id="env",episode_id="ep",message_id="obs")
        raw=leaf.Observation(context=old_context,observation_id="obs",spec_ref="synthetic",env_step=0,
            sim_time_s=0.,sensors={},robots={},extensions={"robotwin.native_proprio.v1":ext})
        benchmark=NativeSimulator.__new__(NativeSimulator)
        benchmark.family,benchmark.space="robotwin",self.space
        benchmark.runtime=SimpleNamespace(_scope=nullcontext,env=SimpleNamespace(robot=SimpleNamespace(
            get_left_arm_real_jointState=lambda:np.zeros(7),get_right_arm_real_jointState=lambda:np.zeros(7))))
        public=benchmark._project(raw)
        for side in ("left","right"):
            self.assertEqual(public.states[side+"_eef_quaternion"].payload.data,list(q))
            self.assertEqual(public.states[side+"_eef_quaternion"].provenance,"measurement")
            self.assertEqual(public.states[side+"_gripper_open_fraction"].provenance,"controller_target")

    def test_pose_and_raw_gripper_use_existing_native_execution_and_counts(self):
        from test_mn_native_semantics import CanonicalNativeSemanticsTests
        fixture=CanonicalNativeSemanticsTests()
        fixture.np=np; fixture.space=self.space; fixture.context=self.context
        benchmark,runtime=fixture.bench()
        raw=self.raw(); raw[9],raw[19]=1.00390625,-.00390625
        action=self.action(raw)
        result=benchmark.advance(action,self.context,leaf.Budgets(env_steps=2))
        np.testing.assert_array_equal(runtime.row,SyntheticCodec.eef20d_to_ee16d(raw))
        self.assertEqual(result.execution_feedback.status,"executed")
        self.assertEqual(result.execution_feedback.consumed_env_steps,1)
        self.assertEqual(result.execution_feedback.consumed_physics_steps,9)
        self.assertEqual(result.execution_feedback.elapsed_sim_time_s,.036)
        self.assertEqual(set(result.execution_feedback.command_ids),{cmd.command_id for cmd in action.commands})


if __name__ == "__main__":
    unittest.main()
