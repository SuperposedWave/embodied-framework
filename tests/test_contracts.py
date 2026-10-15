from dataclasses import replace
import hashlib
import json
import math
import unittest

from embodied_eval import contracts as c
from embodied_eval.codec import decode, dumps, loads, to_data
from embodied_eval.compatibility import check_compatibility, validate_model_input
from embodied_eval.errors import ContractError
from embodied_eval.geometry import (
    compose, inverse, transform_pose, apply_pose_delta, quaternion_to_rot6d,
    rot6d_to_quaternion, convert_quantity, rotate_vector, frame_transform,
)
from embodied_eval.validation import (
    validate_space, validate_tensor, validate_observation, validate_task,
    validate_prediction, validate_step_result, validate_episode_result, schema_value,
)
from embodied_eval.visibility import build_model_input, public_space
from examples.contract_sample import make_case, prediction, sequence, IDENTITY
from examples.validate_contracts import build_examples


class ContractCase(unittest.TestCase):
    def setUp(self):
        self.space, self.cap, self.profile, self.task, self.obs = make_case()

    def fails(self, code, fn, *args, **kwargs):
        with self.assertRaises(ContractError) as error:
            fn(*args, **kwargs)
        self.assertEqual(error.exception.issue.code, code)

    def pred(self, decision, profile=None):
        return prediction(self.obs, profile or self.profile, decision)

    def validate(self, pred, profile=None):
        return validate_prediction(pred, self.obs, self.space, self.cap, profile or self.profile)

    def report_codes(self, profile=None, cap=None, space=None):
        return {issue.code for issue in check_compatibility(
            space or self.space, cap or self.cap, profile or self.profile).issues}


class SerializationTests(ContractCase):
    def test_full_wire_examples_round_trip(self):
        examples = build_examples()
        for case in examples["prediction_cases"]:
            with self.subTest(case=case["name"]):
                wire = json.dumps(case["prediction"])
                self.assertEqual(to_data(loads(c.Prediction, wire)), case["prediction"])
        self.assertEqual(len(examples["prediction_cases"]), 8)

    def test_unknown_field_is_not_ignored(self):
        raw = to_data(self.obs)
        raw["hidden_score"] = 1
        self.fails("unknown_field", decode, c.Observation, raw)

    def test_boolean_is_not_integer_budget(self):
        self.fails("invalid_type", decode, c.Budgets, {"env_steps": True})

    def test_future_version_rejected(self):
        raw = to_data(self.obs.context) | {"contract_version": "0.2"}
        self.fails("invalid_type", decode, c.Context, raw)

    def test_duplicate_json_keys_rejected(self):
        self.fails("duplicate_key", loads, c.Budgets, '{"env_steps":1,"env_steps":2}')

    def test_nonfinite_numbers_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.fails("invalid_type", dumps, replace(self.obs, sim_time_s=value))
        self.fails("non_finite", loads, c.Budgets, '{"sim_time_s":NaN}')

    def test_bool_is_not_a_boolean_metric_float(self):
        # A bool remains bool across the float|bool|null union.
        metric = c.Metric(value=False, unit="boolean", source="fixture", valid=True)
        self.assertIs(loads(c.Metric, dumps(metric)).value, False)

    def test_tensor_shape_range_and_reference_checks(self):
        good = c.Tensor(dtype="uint8", shape=(1, 3), layout="NC", data=[[0, 1, 255]])
        validate_tensor(good)
        self.fails("tensor_dtype_mismatch", validate_tensor, replace(good, data=[[0, 1, 256]]))
        self.fails("tensor_shape_mismatch", validate_tensor, replace(good, data=[[1, 2]]))
        ref = c.DataRef(storage="file", key="synthetic.raw", byte_length=3, sha256=hashlib.sha256(b"\0\1\xff").hexdigest())
        validate_tensor(replace(good, data=None, data_ref=ref))
        self.fails("invalid_buffer_size", validate_tensor, replace(good, data=None, data_ref=replace(ref, byte_length=2)))
        self.fails("ambiguous_payload", validate_tensor, replace(good, data_ref=ref))


class CompatibilityTests(ContractCase):
    def test_bad_local_schema_reference_is_reported(self):
        self.fails("invalid_schema_ref", schema_value, {"$ref": "#/$defs/missing"}, {}, "tool")

    def test_float32_tensor_overflow_is_not_finite_float32(self):
        self.fails("tensor_dtype_mismatch", validate_tensor,
                   c.Tensor(dtype="float32", shape=(1,), layout="F", data=[1e40]))

    def test_fixture_has_declared_compatibility(self):
        self.assertTrue(check_compatibility(self.space, self.cap, self.profile).compatible)

    def test_weights_normalization_and_source_are_pinned(self):
        for name in ("checkpoint_ref", "normalization_ref", "model_source_revision", "benchmark_spec_ref"):
            with self.subTest(field=name):
                self.assertIn("profile_mismatch", self.report_codes(replace(self.profile, **{name: "other@1"})))

    def test_checkpoint_embodiment_mismatch(self):
        self.assertIn("unsupported_embodiment", self.report_codes(cap=replace(self.cap, supported_embodiments=("other-robot",))))

    def test_input_unit_mismatch(self):
        reqs = tuple(replace(req, unit="deg") if req.key == "joint" else req for req in self.cap.required_inputs)
        self.assertIn("unit_mismatch", self.report_codes(cap=replace(self.cap, required_inputs=reqs)))

    def test_unresolved_mapping_is_blocking(self):
        self.assertIn("unresolved_profile", self.report_codes(replace(self.profile, unresolved=("rotation semantics unknown",))))

    def test_hidden_camera_and_misspelled_fields_are_blocking(self):
        v = replace(self.profile.view_policy, sensor_ids=("left_wrist_rgb",))
        self.assertIn("hidden_required_input", self.report_codes(replace(self.profile, view_policy=v)))
        v = replace(self.profile.view_policy, robot_fields=self.profile.view_policy.robot_fields + ("robot0/joints/no-such/position",))
        self.assertIn("invalid_binding", self.report_codes(replace(self.profile, view_policy=v)))

    def test_duplicate_chunk_owner_and_horizon(self):
        self.assertIn("profile_mismatch", self.report_codes(replace(self.profile, chunk_mode="adapter_owned")))
        p = replace(self.profile, chunk_mode="adapter_owned", execution_horizon_steps=2)
        self.assertIn("double_chunk_owner", self.report_codes(p, cap=replace(self.cap, chunk_mode="adapter_owned")))

    def test_every_step_history_requires_tool_stream(self):
        p = replace(self.profile, role="agent", tool_refs=("move_eef@1",))
        cap = replace(self.cap, observation_update="every_env_step")
        tools = self.space.tools | {"move_eef@1": replace(self.space.tools["move_eef@1"], intermediate_observations_supported=False)}
        self.assertIn("observation_history_mismatch", self.report_codes(p, cap=cap, space=replace(self.space, tools=tools)))
        self.assertTrue(check_compatibility(self.space, cap, p).compatible)
        self.assertTrue(check_compatibility(self.space, self.cap, p).compatible)

    def test_required_extension_cannot_silently_disappear(self):
        space = replace(self.space, required_extensions=("bench.private",))
        codes = self.report_codes(space=space)
        self.assertIn("unsupported_extension", codes)
        self.assertIn("hidden_required_input", codes)

    def test_frame_conversion_is_explicit(self):
        reqs = tuple(replace(req, frame_id="world") if req.key == "image" else req for req in self.cap.required_inputs)
        cap = replace(self.cap, required_inputs=reqs)
        self.assertIn("frame_mismatch", self.report_codes(cap=cap))
        profile = replace(self.profile, frame_conversions=(("base", "world"),))
        self.assertTrue(check_compatibility(self.space, cap, profile).compatible)

    def test_invalid_frame_graph_is_a_contract_error(self):
        frames = dict(self.space.frames)
        frames["world"] = c.FrameSpec(parent_id="base", static_transform=IDENTITY)
        self.fails("frame_cycle", validate_space, replace(self.space, frames=frames))
        frames["world"] = c.FrameSpec(parent_id="missing", static_transform=IDENTITY)
        self.fails("unknown_frame", validate_space, replace(self.space, frames=frames))

    def test_schema_reference_cannot_trigger_network_fetch(self):
        tools = {"remote": c.ToolSpec(target_ids=("robot0/left_eef",), argument_schema={"$ref": "https://example.com/schema"})}
        self.fails("external_schema_ref", validate_space, replace(self.space, tools=tools))

    def test_schema_validates_nested_oneof_and_local_refs(self):
        schema = {"$defs": {"positive": {"type": "number", "exclusiveMinimum": 0}},
                  "type": "object", "properties": {"amount": {"$ref": "#/$defs/positive"},
                  "choice": {"oneOf": [{"const": "left"}, {"const": "right"}]}},
                  "required": ["amount", "choice"], "additionalProperties": False}
        schema_value(schema, {"amount": .1, "choice": "left"}, "tool")
        self.fails("schema_mismatch", schema_value, schema, {"amount": 0, "choice": "left"}, "tool")

    def test_gripper_joint_semantics_required(self):
        r = self.space.robots["robot0"]
        grippers = dict(r.grippers) | {"fingers": c.GripperSpec(modes=("joint_targets",))}
        space = replace(self.space, robots={"robot0": replace(r, grippers=grippers)})
        self.fails("missing_semantics", validate_space, space)
        grippers["fingers"] = c.GripperSpec(modes=("joint_targets",), joint_ids=("left_j0",))
        validate_space(replace(self.space, robots={"robot0": replace(r, grippers=grippers)}))


class VisibilityTests(ContractCase):
    def view(self, obs=None):
        return build_model_input(self.task, obs or self.obs, self.profile)

    def test_private_scores_initialization_frames_and_robot_fields_are_removed(self):
        model_input = self.view()
        validate_model_input(model_input, self.space, self.cap, self.profile)
        wire = dumps(model_input)
        for secret in ("secret_goal", "hidden_success", "987654321", "bench.private", "private_goal", "private-init", "private-evaluation"):
            self.assertNotIn(secret, wire)
        joint = model_input.observation_view.robots["robot0"].joints["left_j0"]
        self.assertEqual(joint.position, 0.)
        self.assertIsNone(joint.velocity)
        self.assertIsNone(joint.effort)
        self.assertNotIn("right_j0", model_input.observation_view.robots["robot0"].joints)
        self.assertIsNone(model_input.observation_view.robots["robot0"].end_effectors["left_eef"].gripper_state)
        static_wire = dumps(public_space(self.space, self.profile))
        self.assertNotIn("private_goal", static_wire)
        self.assertNotIn("bench.private", static_wire)
        validate_space(public_space(self.space, self.profile))

    def test_public_view_is_not_a_shared_mutable_object(self):
        model_input = self.view()
        model_input.observation_view.sensors["head_rgb"].payload.data[0][0][0] = 123
        model_input.observation_view.extensions["example.robot_metadata"]["version"] = 999
        self.assertEqual(self.obs.sensors["head_rgb"].payload.data[0][0][0], 255)
        self.assertEqual(self.obs.extensions["example.robot_metadata"]["version"], 1)
        static = public_space(self.space, self.profile)
        static.robots["robot0"].joints.clear()
        self.assertEqual(len(self.space.robots["robot0"].joints), 2)

    def test_injecting_a_full_observation_is_rejected(self):
        unsafe = replace(self.view(), observation_view=self.obs)
        self.fails("visibility_violation", validate_model_input, unsafe, self.space, self.cap, self.profile)

    def test_zero_missing_and_invalid_are_different(self):
        validate_model_input(self.view(), self.space, self.cap, self.profile)  # zero is measured and valid
        obs = replace(self.obs, sensors={})
        self.fails("missing_required_input", validate_model_input, self.view(obs), self.space, self.cap, self.profile)
        sample = replace(self.obs.sensors["head_rgb"], validity="invalid", payload=None, invalid_reason="secret_goal detected")
        obs = replace(self.obs, sensors=self.obs.sensors | {"head_rgb": sample})
        view = self.view(obs)
        self.assertEqual(view.observation_view.sensors["head_rgb"].invalid_reason, "unavailable")
        self.assertNotIn("secret_goal", dumps(view))
        self.fails("missing_required_input", validate_model_input, view, self.space, self.cap, self.profile)
        robot = self.obs.robots["robot0"]
        obs = replace(self.obs, robots={"robot0": replace(robot, joints={"left_j0": c.JointState(velocity=.1)})})
        self.fails("missing_required_input", validate_model_input, self.view(obs), self.space, self.cap, self.profile)

    def test_stale_and_future_camera_samples(self):
        obs = replace(self.obs, sim_time_s=.2, env_step=4)
        self.fails("stale_input", validate_model_input, self.view(obs), self.space, self.cap, self.profile)
        sample = replace(self.obs.sensors["head_rgb"], captured_at_s=.1)
        obs = replace(self.obs, sensors=self.obs.sensors | {"head_rgb": sample})
        self.fails("invalid_time", validate_observation, obs, self.space)

    def test_end_hook_cannot_accept_score_fields_or_arbitrary_text(self):
        self.fails("unknown_field", decode, c.ModelEndContext, {
            "context": to_data(self.obs.context), "reason": "model_stop", "success": True})
        self.fails("invalid_type", decode, c.ModelEndContext, {
            "context": to_data(self.obs.context), "reason": "hidden_success=True"})

    def test_feedback_cannot_expose_extra_scoring_or_bad_type(self):
        view = replace(self.view(), execution_feedback_view={"success": True})
        self.fails("visibility_violation", validate_model_input, view, self.space, self.cap, self.profile)
        view = replace(self.view(), execution_feedback_view={"consumed_env_steps": "3"})
        self.fails("invalid_type", validate_model_input, view, self.space, self.cap, self.profile)

    def test_reserved_world_model_candidate_input_is_rejected(self):
        view = replace(self.view(), candidate_actions=sequence())
        self.fails("reserved_input", validate_model_input, view, self.space, self.cap, self.profile)


class DecisionAndResultTests(ContractCase):
    def test_chunk_8_execute_2_is_preserved(self):
        pred = self.pred(sequence(8, 2))
        self.validate(pred)
        self.assertEqual(len(pred.decision.steps), 8)
        self.assertEqual(pred.decision.execution_horizon_steps, 2)

    def test_chunk_index_tick_horizon_and_duplicate_targets(self):
        seq = sequence(2, 2)
        bad = replace(seq, steps=(replace(seq.steps[0], index=1), seq.steps[1]))
        self.fails("invalid_sequence", self.validate, self.pred(bad))
        bad = replace(seq, steps=(replace(seq.steps[0], duration_s=.1), seq.steps[1]))
        self.fails("timing_mismatch", self.validate, self.pred(bad))
        self.fails("invalid_horizon", self.validate, self.pred(replace(seq, execution_horizon_steps=3)))
        cmd = seq.steps[0].commands[0]
        bad = replace(seq, steps=(replace(seq.steps[0], commands=(cmd, replace(cmd, command_id="second"))), seq.steps[1]))
        self.fails("duplicate_id", self.validate, self.pred(bad))

    def test_adapter_owned_chunk_exposes_one_action(self):
        profile = replace(self.profile, chunk_mode="adapter_owned", execution_horizon_steps=1)
        cap = replace(self.cap, chunk_mode="adapter_owned")
        check_compatibility(self.space, cap, profile).require_compatible()
        validate_prediction(self.pred(sequence(), profile), self.obs, self.space, cap, profile)
        self.fails("double_chunk_owner", validate_prediction, self.pred(sequence(2, 1), profile),
                   self.obs, self.space, cap, profile)

    def test_unit_and_joint_limits_do_not_silently_scale_or_clip(self):
        cmd = sequence().steps[0].commands[0]
        for command, code in ((replace(cmd, unit="deg"), "unit_mismatch"), (replace(cmd, value=4.), "action_out_of_bounds")):
            self.fails(code, self.validate, self.pred(sequence(commands=lambda i: (command,))))

    def test_controller_command_is_separate_from_si_units(self):
        cmd = c.ControllerCommand(command_id="ctrl", target_id="robot0/left_eef",
            controller_profile_ref="normalized-controller@1", parameters={"delta": [.1, 0., 0.]})
        self.validate(self.pred(sequence(commands=lambda i: (cmd,))))
        bad = replace(cmd, parameters={"delta": [2., 0., 0.]})
        self.fails("schema_mismatch", self.validate, self.pred(sequence(commands=lambda i: (bad,))))
        bad = replace(cmd, controller_profile_ref="unknown@1")
        self.fails("unknown_controller", self.validate, self.pred(sequence(commands=lambda i: (bad,))))
        self.fails("invalid_type", decode, c.ControllerCommand, to_data(cmd) | {"parameter_domain": "m"})

    def test_gripper_fraction_semantics_and_bounds(self):
        cmd = c.GripperCommand(command_id="grip", target_id="robot0/left_gripper", mode="open_fraction", value=0.)
        self.validate(self.pred(sequence(commands=lambda i: (cmd,))))
        self.fails("invalid_gripper", self.validate, self.pred(sequence(commands=lambda i: (replace(cmd, value=1.1),))))

    def test_pose_delta_reference_is_explicit(self):
        cmd = c.PoseDelta(command_id="delta", target_id="robot0/left_eef", delta=IDENTITY, frame_id="base",
            reference_policy="chunk_start", composition="pre_multiply", reference_observation_id=self.obs.observation_id)
        self.validate(self.pred(sequence(commands=lambda i: (cmd,))))
        self.fails("invalid_delta_reference", self.validate,
                   self.pred(sequence(commands=lambda i: (replace(cmd, reference_observation_id="older"),))))

    def test_cross_episode_and_stale_observation_output_rejected(self):
        pred = self.pred(sequence())
        self.fails("episode_mismatch", self.validate, replace(pred, context=replace(pred.context, episode_id="previous-episode")))
        self.fails("stale_prediction", self.validate, replace(pred, based_on_observation_id="old-observation"))
        self.fails("request_mismatch", self.validate, replace(pred, request_id="other-request"))
        self.fails("request_mismatch", validate_prediction, pred, self.obs, self.space, self.cap, self.profile,
                   request_context=replace(pred.context, message_id="newer-pending-request"))

    def test_tool_argument_schema_rejects_extra_missing_and_wrong_type(self):
        p = replace(self.profile, role="agent", tool_refs=("move_eef@1",))
        call = c.ToolCall(call_id="call", tool_ref="move_eef@1", target_id="robot0/left_eef",
                         arguments={"position_m": [.1, 0., .3], "frame_id": "base", "max_ticks": 3})
        self.validate(self.pred(call, p), p)
        for arguments in (call.arguments | {"secret": 1}, {"position_m": [1., 2., 3.]}, call.arguments | {"max_ticks": True}):
            with self.subTest(args=arguments):
                self.fails("schema_mismatch", self.validate, self.pred(replace(call, arguments=arguments), p), p)

    def test_future_prediction_is_not_robot_action(self):
        p = replace(self.profile, role="world_model", action_spec_refs=(), discrete_spec_refs=(), tool_refs=())
        future = c.FuturePrediction(modality="future_state", relative_time_s=.1,
            payload=c.Tensor(dtype="float32", shape=(2,), layout="F", data=[0., .1]))
        self.validate(prediction(self.obs, p, None, predictions=(future,)), p)
        self.fails("invalid_world_prediction", self.validate, prediction(self.obs, p, sequence(), predictions=(future,)), p)

    def test_tool_consumption_matches_actual_three_ticks(self):
        examples = build_examples()
        result = decode(c.StepResult, examples["tool_step_result"])
        case = next(case for case in examples["prediction_cases"] if case["name"] == "tool")
        pred = decode(c.Prediction, case["prediction"])
        validate_step_result(result, self.obs, self.space, expected_prediction=pred)
        self.fails("execution_count_mismatch", validate_step_result,
            replace(result, execution_feedback=replace(result.execution_feedback, consumed_env_steps=1)),
            self.obs, self.space)
        self.fails("call_mismatch", validate_step_result,
            replace(result, execution_feedback=replace(result.execution_feedback, call_id="other")),
            self.obs, self.space, expected_prediction=pred)

    def test_rejected_execution_does_not_fabricate_steps(self):
        result = decode(c.StepResult, build_examples()["tool_step_result"])
        bad = replace(result, execution_feedback=replace(result.execution_feedback, status="rejected", code="invalid_argument"))
        self.fails("invalid_execution_count", validate_step_result, bad, self.obs, self.space)

    def test_per_tick_model_requires_complete_tool_observation_history(self):
        result = decode(c.StepResult, build_examples()["tool_step_result"])
        cap = replace(self.cap, observation_update="every_env_step")
        validate_step_result(result, self.obs, self.space, capability=cap)
        self.fails("missing_step_observations", validate_step_result,
                   replace(result, intermediate_observations=()), self.obs, self.space, capability=cap)
        self.fails("invalid_observation_history", validate_step_result,
                   replace(result, intermediate_observations=tuple(reversed(result.intermediate_observations))),
                   self.obs, self.space, capability=cap)

    def test_native_termination_and_budget_flags_are_both_preserved(self):
        result = decode(c.StepResult, build_examples()["tool_step_result"])
        result = replace(result, terminated=True, truncated=True, end_reason="native-terminal-and-budget")
        validate_step_result(result, self.obs, self.space)

    def test_active_stop_does_not_imply_success_or_unavailable_metrics(self):
        result = decode(c.EpisodeResult, build_examples()["stopped_episode_result"])
        validate_episode_result(result)
        self.assertEqual(result.termination_reason, "model_stop")
        self.assertIs(result.metrics["success"].value, False)
        unavailable = c.Metric(value=None, unit="boolean", source="fixture", valid=False, unavailable_reason="scorer_failed")
        validate_episode_result(replace(result, run_status="error", metrics={"success": unavailable}))
        self.fails("invalid_metric", validate_episode_result,
            replace(result, metrics={"success": replace(unavailable, value=False)}))


class GeometryTests(unittest.TestCase):
    def test_static_dynamic_frame_graph_uses_rotation_and_translation(self):
        space, _, _, _, obs = make_case()
        q = (0., 0., math.sqrt(.5), math.sqrt(.5))
        frames = dict(space.frames)
        frames["base"] = replace(frames["base"], static_transform=c.Transform(translation_m=(1., 2., 0.), quaternion_xyzw=q))
        space = replace(space, frames=frames)
        transforms = dict(obs.frame_transforms)
        transforms["left_tcp"] = c.TimedTransform(transform=c.Transform(translation_m=(1., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.)), captured_at_s=0.)
        obs = replace(obs, frame_transforms=transforms)
        t = frame_transform(space, obs, source_frame="left_tcp", target_frame="world")
        self.close_vec(t.translation_m, (1., 3., 0.))
        reverse = frame_transform(space, obs, source_frame="world", target_frame="left_tcp")
        self.close_vec(compose(t, reverse).translation_m, (0., 0., 0.))
        with self.assertRaises(ContractError):
            frame_transform(space, replace(obs, sim_time_s=1.), source_frame="left_tcp", target_frame="world", max_age_s=.1)

    def close_vec(self, a, b):
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=7)

    def test_nontrivial_frame_transform_and_inverse(self):
        q = (0., 0., math.sqrt(.5), math.sqrt(.5))
        t = c.Transform(translation_m=(1., 2., 0.), quaternion_xyzw=q)
        p = c.Pose(position_m=(1., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.), frame_id="base")
        world = transform_pose(p, t, source_frame="base", target_frame="world")
        self.close_vec(world.position_m, (1., 3., 0.))
        restored = transform_pose(world, inverse(t), source_frame="world", target_frame="base")
        self.close_vec(restored.position_m, p.position_m)
        self.close_vec(compose(t, inverse(t)).translation_m, (0., 0., 0.))

    def test_delta_pre_and_post_multiplication_are_different(self):
        p = c.Pose(position_m=(0., 0., 0.), quaternion_xyzw=(0., 0., math.sqrt(.5), math.sqrt(.5)), frame_id="base")
        delta = c.Transform(translation_m=(1., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.))
        self.close_vec(apply_pose_delta(p, delta, composition="pre_multiply").position_m, (1., 0., 0.))
        self.close_vec(apply_pose_delta(p, delta, composition="post_multiply").position_m, (0., 1., 0.))

    def test_rot6d_has_documented_column_convention(self):
        q = (0., 0., math.sqrt(.5), math.sqrt(.5))
        self.close_vec(quaternion_to_rot6d(q), (0., 1., 0., -1., 0., 0.))
        for q in ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.),
                  (0., 0., 0., -1.), q, tuple(-x for x in q)):
            with self.subTest(q=q):
                restored = rot6d_to_quaternion(quaternion_to_rot6d(q))
                self.close_vec(rotate_vector(restored, (1., 2., 3.)), rotate_vector(q, (1., 2., 3.)))

    def test_explicit_unit_conversion_and_dimension_rejection(self):
        self.assertAlmostEqual(convert_quantity(180., "deg", "rad"), math.pi)
        self.assertAlmostEqual(convert_quantity(100., "cm", "m"), 1.)
        for source, target in (("rad", "m"), ("controller_native", "m"), ("pixels", "m")):
            with self.subTest(source=source):
                with self.assertRaises(ContractError):
                    convert_quantity(1., source, target)

    def test_degenerate_rotations_are_rejected(self):
        with self.assertRaises(ContractError):
            rot6d_to_quaternion((1., 0., 0., 2., 0., 0.))
        with self.assertRaises(ContractError):
            quaternion_to_rot6d((0., 0., 0., 2.))


if __name__ == "__main__":
    unittest.main()
