"""Read-only saved execution checks. No model, simulator, or GPU imports.

Grains: one trace transition, one public model input, one sensor capture time.
Service request counters, Adapter predict calls and verified forwards are NOT
interchangeable. All durations below describe execution, never weight training.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

EPS_S = 1e-8  # Numerical clock comparison tolerance, not physical-step tolerance.


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def summary(values):
    values = list(values)
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": math.fsum(values) / len(values) if values else None,
            "sum": math.fsum(values) if values else 0.}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def image_digest(payload):
    import numpy as np
    if payload is None or payload.get("data") is None:
        return None  # A reference is not evidence of bytes we did not read.
    if payload.get("dtype") != "uint8" or payload.get("layout") != "HWC":
        raise ValueError("audit_requires_declared_uint8_HWC_image")
    array = np.asarray(payload["data"])
    if (array.shape != tuple(payload["shape"]) or array.ndim != 3 or array.shape[-1] != 3
            or array.dtype.kind not in "iu" or (array.size and (array.min() < 0 or array.max() > 255))):
        raise ValueError("invalid_saved_RGB_values_or_shape")
    return hashlib.sha256(array.astype("uint8").tobytes()).hexdigest()


def compact_observation(raw):
    sensors = {}
    for key, sample in raw.get("sensors", {}).items():
        payload = sample.get("payload")
        sensors[key] = {"captured_at_s": sample.get("captured_at_s"),
            "pixel_sha256": image_digest(payload), "validity": sample.get("validity"),
            "dtype": payload.get("dtype") if payload else None,
            "shape": payload.get("shape") if payload else None}
    return {"observation_id": raw.get("observation_id"), "sim_time_s": raw.get("sim_time_s"),
            "env_step": raw.get("env_step"), "sensors": sensors,
            "states": raw.get("states", {}), "frame_transforms": raw.get("frame_transforms", {}),
            "instruction": raw.get("public_instruction", raw.get("task_public", {}).get("instruction"))}


class Captures:
    def __init__(self):
        self.rows = {}

    def add(self, obs, issues, label):
        now = obs["sim_time_s"]
        for key, sample in obs["sensors"].items():
            row = self.rows.setdefault(key, {"occurrences": 0, "captures": {}, "ages": []})
            row["occurrences"] += 1
            at, pixel = sample["captured_at_s"], sample["pixel_sha256"]
            if not finite(at) or not finite(now) or at < -EPS_S or at > now + EPS_S:
                issues.append(label + ":invalid_or_future_capture:" + key)
                continue
            row["ages"].append(now - at)
            if row["captures"] and at < next(reversed(row["captures"])) - EPS_S:
                issues.append(label + ":capture_time_went_backwards:" + key)
            if at in row["captures"] and row["captures"][at] != pixel:
                issues.append(label + ":same_capture_time_different_pixels:" + key)
            row["captures"][at] = pixel

    def result(self):
        result = {}
        for key, row in self.rows.items():
            items = sorted(row["captures"].items())
            result[key] = {"occurrences": row["occurrences"], "distinct_capture_timestamps": len(items),
                "repeated_capture_occurrences": row["occurrences"] - len(items),
                "capture_times_s": [t for t, _ in items],
                "distinct_capture_intervals_s": summary(b[0] - a[0] for a, b in zip(items, items[1:])),
                "age_at_delivery_s": summary(row["ages"]),
                "adjacent_new_timestamps_with_same_pixels": sum(a[1] is not None and a[1] == b[1] for a, b in zip(items, items[1:])),
                "missing_inline_image_captures": sum(pixel is None for _, pixel in items),
                "timestamps_are_declared_capture_events_not_independent_camera_probe": True}
        return result


def sensor_mapping(manifest):
    profile = manifest["profile"]
    if "input_paths" in profile:
        # Current public ModelInput keeps canonical public keys, not model slots.
        return {path.split(".", 1)[1]: path.split(".", 1)[1]
                for path in profile["input_paths"].values() if path.startswith("sensors.")}
    return {req["key"]: profile["binding"]["inputs"][req["key"]]
            for req in manifest["capability"]["inputs"] if req["kind"] == "sensor"}


def analyze_events(events, inputs, manifest, result, diagnostics):
    """Pure streaming calculation; return compact inspectable per-action rows."""
    issues, warnings, transitions, predictions = [], [], [], []
    trace_captures, input_captures = Captures(), Captures()
    previous = initial = pending = None
    snapshots = 0
    seen_ids = set()
    input_count = input_exact = 0
    inputs = iter(inputs)
    mapping = sensor_mapping(manifest)
    known_forwards = []
    unknown_forwards = 0
    input_clock_rows = []
    for event in events:
        kind, data = event["event"], event["data"]
        if kind in ("reset", "step_result"):
            raw = data if kind == "reset" else data["observation"]
            obs = compact_observation(raw)
            if obs["observation_id"] in seen_ids:
                issues.append("duplicate_observation_id")
            seen_ids.add(obs["observation_id"])
            snapshots += 1
            trace_captures.add(obs, issues, "trace")
            if kind == "reset":
                if previous is not None or obs["sim_time_s"] != 0 or obs["env_step"] != 0:
                    issues.append("invalid_or_repeated_reset")
                initial = obs
            else:
                if previous is None or pending is None:
                    raise ValueError("step_without_previous_snapshot_and_prediction")
                feedback = data["execution_feedback"]
                elapsed = feedback.get("elapsed_sim_time_s")
                physical = feedback.get("consumed_physics_steps")
                consumed = feedback.get("consumed_env_steps")
                if not finite(elapsed) or elapsed < 0:
                    raise ValueError("invalid_elapsed_time")
                if not isinstance(consumed, int) or isinstance(consumed, bool) or consumed < 0:
                    raise ValueError("invalid_consumed_action_count")
                if physical is not None and (not isinstance(physical, int) or isinstance(physical, bool) or physical < 0):
                    raise ValueError("invalid_physics_count")
                if not finite(obs["sim_time_s"]) or abs(obs["sim_time_s"] - previous["sim_time_s"] - elapsed) > EPS_S:
                    issues.append("observation_clock_vs_feedback_mismatch")
                if obs["env_step"] != previous["env_step"] + consumed:
                    issues.append("observation_action_counter_mismatch")
                if feedback.get("request_id") != pending.get("context", {}).get("message_id"):
                    issues.append("execution_request_id_mismatch")
                space = manifest["space"]
                physics_dt = space.get("physics_period_s")
                if physical is not None and physics_dt is not None and abs(physical * physics_dt - elapsed) > EPS_S:
                    issues.append("physics_count_vs_elapsed_mismatch")
                control_dt = space.get("control_period_s")
                if space.get("timing_mode") == "fixed_period" and control_dt is not None and abs(control_dt * consumed - elapsed) > EPS_S:
                    issues.append("fixed_control_period_mismatch")
                flag = pending["action"].get("acquire_observation_after")
                terminal = bool(data.get("terminated") or data.get("truncated"))
                transitions.append({"action_index": len(transitions) + 1,
                    "before_s": previous["sim_time_s"], "after_s": obs["sim_time_s"],
                    "elapsed_s": elapsed, "physics_steps": physical, "consumed_actions": consumed,
                    "status": feedback.get("status"), "acquire_requested": flag,
                    "terminal": terminal,
                    "capture_time_advanced": {k: s["captured_at_s"] != previous["sensors"].get(k, {}).get("captured_at_s") for k, s in obs["sensors"].items()}})
                pending = None
            previous = obs
        elif kind == "prediction":
            if previous is None or pending is not None:
                raise ValueError("prediction_without_snapshot_or_double_prediction")
            pending = data
            if data.get("based_on_observation_id") != previous["observation_id"]:
                issues.append("prediction_uses_wrong_observation")
            view = next(inputs, None)
            if view is None:
                issues.append("missing_public_input_record")
            else:
                packed = compact_observation(view)
                input_count += 1
                input_captures.add(packed, issues, "input")
                same = packed["observation_id"] == previous["observation_id"] and packed["sim_time_s"] == previous["sim_time_s"]
                if set(packed["sensors"]) != set(mapping):
                    issues.append("public_sensor_inventory_mismatch")
                    same = False
                for key, source in mapping.items():
                    if packed["sensors"].get(key) != previous["sensors"].get(source):
                        same = False
                input_exact += int(same)
                if not same:
                    issues.append("public_input_capture_or_pixels_differ_from_source_snapshot")
                input_clock_rows.append({"predict_index": len(predictions) + 1, "sim_time_s": packed["sim_time_s"],
                    "sensors": packed["sensors"]})
            count = data.get("model_forward_count")
            if count is None:
                unknown_forwards += 1
            elif isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                known_forwards.append(count)
            else:
                issues.append("invalid_model_forward_count")
            predictions.append(data["action"])
    extra_inputs = sum(1 for _ in inputs)
    if extra_inputs:
        issues.append("extra_public_input_records")
    if initial is None:
        issues.append("missing_reset_observation")
    if pending is not None:
        issues.append("prediction_not_followed_by_execution")
    stats = result["execution_statistics"]
    elapsed = summary(r["elapsed_s"] for r in transitions)
    physics = None if any(r["physics_steps"] is None for r in transitions) else sum(r["physics_steps"] for r in transitions)
    actions = sum(r["consumed_actions"] for r in transitions)
    for name, actual in (("env_steps", actions), ("model_calls", len(predictions))):
        if stats.get(name) != actual:
            issues.append("result_counter_mismatch:" + name)
    if stats.get("physics_steps") is not None or manifest["space"].get("timing_mode") == "measured_native":
        if stats.get("physics_steps") != physics:
            issues.append("result_counter_mismatch:physics_steps")
    elif physics is not None:
        warnings.append("physics_count_available_in_feedback_but_fixed_mode_result_keeps_null")
    if not finite(stats.get("sim_time_s")) or abs(stats["sim_time_s"] - elapsed["sum"]) > EPS_S:
        issues.append("result_sim_time_mismatch")
    forward_total = None if unknown_forwards else sum(known_forwards)
    if stats.get("model_forward_count") != forward_total:
        issues.append("result_forward_count_mismatch")
    model_diag = diagnostics.get("model", {})
    request_fields = [(key, model_diag[key]) for key in ("native_requests", "policy_requests", "native_generation_requests") if key in model_diag]
    if len(request_fields) > 1 and len({value for _, value in request_fields}) > 1:
        issues.append("conflicting_native_request_counters")
    request_count = request_fields[0][1] if request_fields else None
    for key in ("policy_calls", "predict_calls"):
        if key in model_diag and model_diag[key] != len(predictions):
            issues.append("diagnostics_predict_count_mismatch")
    if request_count is not None and (not isinstance(request_count, int) or isinstance(request_count, bool) or not 0 <= request_count <= len(predictions)):
        issues.append("invalid_native_request_count")
    warnings.extend(["request_positions_not_logged_cannot_prove_per_request_execution_lengths",
                     "capture_timestamps_are_adapter_records_not_independent_camera_trigger_telemetry"])
    if unknown_forwards:
        warnings.append("true_model_forward_count_unavailable_not_equal_to_requests_or_predict_calls")
    if "cache_rows_remaining" in model_diag:
        warnings.append("cache_remaining_diagnostic_is_after_episode_cleanup_not_chunk_consumption_proof")
    return {"passed": not issues, "quality_issues": dict(Counter(issues)), "warnings": warnings,
        "recorded_observation_snapshots": snapshots, "public_model_input_records": input_count,
        "public_input_capture_and_pixel_matches": input_exact,
        "extra_public_input_records": extra_inputs,
        "predict_calls": len(predictions), "executed_actions": actions, "physics_steps": physics,
        "result_physics_steps": stats.get("physics_steps"),
        "verified_model_forward_count": forward_total,
        "unknown_forward_records": unknown_forwards,
        "recorded_native_requests": request_count,
        "native_request_counter_field": request_fields[0][0] if request_fields else None,
        "reported_cache_rows_after_episode": model_diag.get("cache_rows_remaining"),
        "execution_elapsed_s": elapsed, "trace_sensors": trace_captures.result(),
        "model_input_sensors": input_captures.result(), "transitions": transitions,
        "input_clock_rows": input_clock_rows, "initial_observation": initial,
        "actions": predictions, "training_action_period_verified": False,
        "observed_request_positions": None}


def analyze_directory(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episode = manifest["config"]["episodes"][0]["episode_id"]
    base = directory / "episodes" / episode
    paths = [manifest_path, base / "result.json", base / "diagnostics.internal.json",
             base / "trace.internal.jsonl", base / "model_input.public.jsonl"]
    protected = {str(p): digest_file(p) for p in paths[:3]}
    def records(path):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                hasher.update(line)
                yield json.loads(line)
        protected[str(path)] = hasher.hexdigest()
    result = analyze_events(records(paths[3]), records(paths[4]), manifest,
        json.loads(paths[1].read_text()), json.loads(paths[2].read_text()))
    if any(digest_file(p) != sha for p, sha in protected.items()):
        raise ValueError("historical_file_changed_during_read_only_audit")
    result.update(directory=str(directory), contract_version=manifest["config"].get("contract_version"),
        synthetic=manifest.get("synthetic"), model_family=manifest["config"]["model"]["adapter"],
        io_profile=manifest["config"]["model"]["options"].get("io_profile"),
        benchmark=manifest["config"]["benchmark"]["adapter"],
        protocol=manifest["config"]["benchmark"]["options"].get("protocol"),
        declared_timing={k: manifest["space"].get(k) for k in ("timing_mode", "control_period_s", "physics_period_s")},
        protected_file_sha256=protected, historical_files_unchanged=True)
    return result
