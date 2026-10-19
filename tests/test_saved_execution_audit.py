"""Small synthetic log fixtures only, never new real model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from docs.handoff.execution_audit import analyze_events, image_digest, summary
from docs.handoff.audit_saved_execution import array_difference
from docs.handoff.audit_saved_native_clocks import resolve_binding


def fixture(n=3, *, reuse=False, same_pixels=False, measured=False, physics=False, v03=False):
    def obs(i):
        now = i * .05
        return {"observation_id": "obs-" + str(i), "env_step": i, "sim_time_s": now,
            "sensors": {"cam": {"captured_at_s": 0. if reuse else now,
                "validity": "valid", "payload": {"dtype": "uint8", "layout": "HWC", "shape": [1, 1, 3],
                "data": [[[1 if reuse or same_pixels else i, 2, 3]]]}}}, "states": {}, "frame_transforms": {}}
    manifest = {"space": {"timing_mode": "measured_native" if measured else "fixed_period",
        "control_period_s": None if measured else .05, "physics_period_s": .025 if physics else None},
        "capability": {"inputs": [{"key": "head", "kind": "sensor"}]},
        "profile": {"input_paths": {"head": "sensors.cam"}} if v03 else {"binding": {"inputs": {"head": "cam"}}}}
    events = [{"event": "reset", "data": obs(0)}]
    inputs = []
    for i in range(n):
        view = obs(i)
        if not v03:
            view["sensors"] = {"head": view["sensors"]["cam"]}
        inputs.append(view)
        events.append({"event": "prediction", "data": {"context": {"message_id": "req-" + str(i)},
            "based_on_observation_id": "obs-" + str(i), "model_forward_count": None,
            "action": {"commands": [], "acquire_observation_after": not reuse}}})
        events.append({"event": "step_result", "data": {"observation": obs(i + 1),
            "execution_feedback": {"request_id": "req-" + str(i), "status": "executed", "consumed_env_steps": 1,
                "elapsed_sim_time_s": .05, "consumed_physics_steps": 2 if physics else None},
            "terminated": i == n - 1, "truncated": False}})
    result = {"execution_statistics": {"env_steps": n, "model_calls": n, "sim_time_s": n * .05,
        "physics_steps": 2 * n if measured and physics else None, "model_forward_count": None}}
    diagnostics = {"model": {"native_requests": 1, "policy_calls": n, "cache_rows_remaining": 0}}
    return events, inputs, manifest, result, diagnostics


class SavedExecutionAuditTests(unittest.TestCase):
    def test_fixed_clock_and_all_counters_reconcile(self):
        x = analyze_events(*fixture())
        self.assertTrue(x["passed"], x["quality_issues"])
        self.assertEqual(x["executed_actions"], 3)
        self.assertAlmostEqual(x["execution_elapsed_s"]["sum"], .15)

    def test_reset_plus_steps_is_not_number_of_inputs(self):
        x = analyze_events(*fixture())
        self.assertEqual((x["recorded_observation_snapshots"], x["public_model_input_records"]), (4, 3))

    def test_repeated_cache_frame_is_not_multiple_captures(self):
        x = analyze_events(*fixture(reuse=True))
        s = x["model_input_sensors"]["head"]
        self.assertTrue(x["passed"])
        self.assertEqual((s["occurrences"], s["distinct_capture_timestamps"], s["repeated_capture_occurrences"]), (3, 1, 2))

    def test_new_timestamp_same_pixels_is_valid_not_silently_deduplicated(self):
        x = analyze_events(*fixture(same_pixels=True))
        s = x["model_input_sensors"]["head"]
        self.assertTrue(x["passed"])
        self.assertEqual(s["distinct_capture_timestamps"], 3)
        self.assertEqual(s["adjacent_new_timestamps_with_same_pixels"], 2)

    def test_same_capture_time_different_pixels_is_reported(self):
        args = fixture(reuse=True)
        args[0][2]["data"]["observation"]["sensors"]["cam"]["payload"]["data"][0][0][0] = 9
        x = analyze_events(*args)
        self.assertFalse(x["passed"])
        self.assertIn("trace:same_capture_time_different_pixels:cam", x["quality_issues"])

    def test_future_image_timestamp_is_reported(self):
        args = fixture()
        args[1][0]["sensors"]["head"]["captured_at_s"] = 1.
        x = analyze_events(*args)
        self.assertFalse(x["passed"])
        self.assertIn("input:invalid_or_future_capture:head", x["quality_issues"])

    def test_sensor_renaming_from_historical_profile_is_verified(self):
        x = analyze_events(*fixture())
        self.assertTrue(x["passed"])
        self.assertEqual(x["public_input_capture_and_pixel_matches"], 3)

    def test_current_public_names_are_not_treated_as_model_slot_names(self):
        x = analyze_events(*fixture(v03=True))
        self.assertTrue(x["passed"])
        self.assertEqual(set(x["model_input_sensors"]), {"cam"})

    def test_input_pixel_mismatch_is_not_hidden_by_same_capture_time(self):
        args = fixture()
        args[1][0]["sensors"]["head"]["payload"]["data"][0][0][0] = 9
        x = analyze_events(*args)
        self.assertIn("public_input_capture_or_pixels_differ_from_source_snapshot", x["quality_issues"])

    def test_missing_camera_is_not_silently_dropped(self):
        args = fixture()
        args[1][1]["sensors"].clear()
        x = analyze_events(*args)
        self.assertIn("public_sensor_inventory_mismatch", x["quality_issues"])

    def test_predict_calls_and_service_requests_are_distinct(self):
        x = analyze_events(*fixture())
        self.assertEqual((x["predict_calls"], x["recorded_native_requests"]), (3, 1))
        self.assertIsNone(x["verified_model_forward_count"])
        self.assertIsNone(x["observed_request_positions"])

    def test_cleared_cache_is_not_proof_all_rows_executed(self):
        x = analyze_events(*fixture())
        self.assertEqual(x["reported_cache_rows_after_episode"], 0)
        self.assertIn("cache_remaining_diagnostic_is_after_episode_cleanup_not_chunk_consumption_proof", x["warnings"])

    def test_measured_physics_count_reconciles_with_real_elapsed(self):
        x = analyze_events(*fixture(measured=True, physics=True))
        self.assertTrue(x["passed"])
        self.assertEqual(x["physics_steps"], 6)

    def test_fixed_mode_feedback_physics_not_confused_with_null_result_summary(self):
        x = analyze_events(*fixture(physics=True))
        self.assertTrue(x["passed"])
        self.assertEqual(x["physics_steps"], 6)
        self.assertIsNone(x["result_physics_steps"])
        self.assertIn("physics_count_available_in_feedback_but_fixed_mode_result_keeps_null", x["warnings"])

    def test_physics_elapsed_contradiction_is_reported(self):
        args = fixture(measured=True, physics=True)
        args[0][2]["data"]["execution_feedback"]["consumed_physics_steps"] = 3
        x = analyze_events(*args)
        self.assertIn("physics_count_vs_elapsed_mismatch", x["quality_issues"])

    def test_observation_clock_contradiction_is_reported(self):
        args = fixture()
        args[0][2]["data"]["observation"]["sim_time_s"] = .07
        x = analyze_events(*args)
        self.assertIn("observation_clock_vs_feedback_mismatch", x["quality_issues"])

    def test_result_aggregate_mismatch_is_reported(self):
        args = fixture()
        args[3]["execution_statistics"]["env_steps"] = 4
        self.assertIn("result_counter_mismatch:env_steps", analyze_events(*args)["quality_issues"])

    def test_public_input_extra_and_missing_records_are_detected(self):
        args = fixture()
        args[1].append(copy.deepcopy(args[1][-1]))
        self.assertEqual(analyze_events(*args)["extra_public_input_records"], 1)
        args = fixture()
        args[1].pop()
        self.assertIn("missing_public_input_record", analyze_events(*args)["quality_issues"])

    def test_wrong_prediction_observation_and_execution_request_are_detected(self):
        args = fixture()
        args[0][1]["data"]["based_on_observation_id"] = "other"
        args[0][2]["data"]["execution_feedback"]["request_id"] = "other"
        issues = analyze_events(*args)["quality_issues"]
        self.assertIn("prediction_uses_wrong_observation", issues)
        self.assertIn("execution_request_id_mismatch", issues)

    def test_verified_forward_count_only_when_every_prediction_records_it(self):
        args = fixture()
        for row in args[0]:
            if row["event"] == "prediction":
                row["data"]["model_forward_count"] = 1
        args[3]["execution_statistics"]["model_forward_count"] = 3
        self.assertEqual(analyze_events(*args)["verified_model_forward_count"], 3)

    def test_raw_rgb_differences_do_not_wrap_unsigned_arithmetic(self):
        x = array_difference([[[0, 255, 2]]], [[[255, 0, 2]]], image=True)
        self.assertEqual((x["changed_values"], x["changed_pixels"], x["max_abs_difference"]), (2, 1, 255.))

    def test_flip_check_is_diagnostic_not_an_image_rewrite(self):
        a, b = [[[1, 2, 3]], [[4, 5, 6]]], [[[4, 5, 6]], [[1, 2, 3]]]
        before = copy.deepcopy((a, b))
        x = array_difference(a, b, image=True)
        self.assertTrue(x["exact_after_flip"]["vertical"])
        self.assertEqual((a, b), before)

    def test_missing_image_payload_does_not_create_a_fake_pixel_hash(self):
        self.assertIsNone(image_digest({"data": None}))

    def test_image_invalid_values_and_shape_are_rejected_without_uint8_narrowing(self):
        for values in ([[[256, 1, 2]]], [[[.5, 1, 2]]], [[[-1, 1, 2]]], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                image_digest({"dtype": "uint8", "layout": "HWC", "shape": [1, 1, 3], "data": values})

    def test_empty_interval_population_is_missing_not_zero_frequency(self):
        self.assertEqual(summary([]), {"count": 0, "min": None, "max": None, "mean": None, "sum": 0.})


class NativeReferenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"config": {"episodes": [{"episode_id": "ep"}]}}))
        base = self.root / "episodes/ep"
        base.mkdir(parents=True)
        self.view = base / "model_input.public.jsonl"
        self.view.write_text('{}\n')
        native = self.root / "native-exchanges"
        native.mkdir()
        self.request = native / "request-0001.msgpack"
        self.request.write_bytes(b"synthetic fixture bytes, not real model data")
        self.sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.companion = {"framework_manifest_sha256": self.sha, "passed": False,
            "native_request_sha256": hashlib.sha256(self.request.read_bytes()).hexdigest(),
            "framework_public_input_first_line_sha256": hashlib.sha256(self.view.read_bytes()).hexdigest()}

    def save_companion(self):
        (self.root / "first-native-request-check.json").write_text(json.dumps(self.companion))

    def test_explicit_native_manifest_binding_matches(self):
        x = resolve_binding(self.manifest, {"framework_manifest_sha256": self.sha}, self.root)
        self.assertEqual(x["basis"], "native_result_manifest_sha256")

    def test_wrong_explicit_binding_is_not_hidden_by_valid_companion(self):
        self.save_companion()
        with self.assertRaisesRegex(ValueError, "native_framework_manifest_hash_mismatch"):
            resolve_binding(self.manifest, {"framework_manifest_sha256": "0" * 64}, self.root)

    def test_matching_source_identity_does_not_require_matching_inference_input(self):
        self.save_companion()
        x = resolve_binding(self.manifest, {}, self.root)
        self.assertFalse(x["companion_input_equality_check_passed"])
        self.assertEqual(len(x["files"]), 2)

    def test_companion_wrong_manifest_is_rejected(self):
        self.companion["framework_manifest_sha256"] = "0" * 64
        self.save_companion()
        with self.assertRaisesRegex(ValueError, "companion_framework_manifest_hash_mismatch"):
            resolve_binding(self.manifest, {}, self.root)

    def test_companion_wrong_native_request_is_rejected(self):
        self.save_companion()
        self.request.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "actual_native_request_hash_mismatch"):
            resolve_binding(self.manifest, {}, self.root)

    def test_companion_wrong_public_input_is_rejected(self):
        self.save_companion()
        self.view.write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, "framework_first_input_hash_mismatch"):
            resolve_binding(self.manifest, {}, self.root)


if __name__ == "__main__":
    unittest.main()
