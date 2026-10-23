"""Read saved real execution replies and reconcile their actual counters.

This checks evidence consistency, not a new simulation, independent physics
measurement, policy quality or stage acceptance. A correctly recorded native
failure may pass this audit while the trial itself remains failed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def analyze(events):
    from embodied_eval.core import types as leaf
    from embodied_eval.codec import decode
    resets = predictions = replies = env_steps = physics_steps = 0
    sim_time = 0.0
    pending = previous_observation = last = scope = None
    failed, errors, scored = [], [], None
    observations = set()
    for event in events:
        kind, data = event["event"], event["data"]
        if kind == "reset":
            if resets or predictions or replies or data["env_step"] != 0 or data["sim_time_s"] != 0:
                raise ValueError("one_initial_reset_required")
            resets += 1
            scope = {key: data["context"][key] for key in ("run_id", "env_id", "episode_id", "contract_version")}
            previous_observation = data["observation_id"]
            observations.add(previous_observation)
        elif kind == "prediction":
            if resets != 1 or pending is not None or failed or errors or scored is not None:
                raise ValueError("prediction_without_observation_or_unreplied_action")
            predictions += 1
            request = data["context"]["message_id"]
            commands = [item["command_id"] for item in data["action"]["commands"]]
            if (request != f"request:{predictions}" or data["based_on_observation_id"] != previous_observation
                    or any(data["context"][key] != value for key, value in scope.items())
                    or not commands or len(commands) != len(set(commands))):
                raise ValueError("prediction_sequence_or_complete_unique_commands_invalid")
            pending = (request, set(commands))
        elif kind == "step_result":
            if pending is None:
                raise ValueError("reply_without_prediction")
            feedback = decode(leaf.ExecutionFeedback, data["execution_feedback"])
            request, commands = pending
            acknowledged = set(feedback.command_ids)
            if (feedback.request_id != request or len(acknowledged) != len(feedback.command_ids)
                    or not acknowledged <= commands or feedback.consumed_env_steps not in (0, 1)
                    or feedback.consumed_physics_steps is None or feedback.consumed_physics_steps < 0
                    or not math.isfinite(feedback.elapsed_sim_time_s) or feedback.elapsed_sim_time_s < 0):
                raise ValueError("reply_identity_commands_or_actual_counts_invalid")
            if feedback.status == "executed":
                if acknowledged != commands or feedback.code != "ok" or feedback.consumed_env_steps != 1:
                    raise ValueError("successful_reply_silently_lost_commands")
            else:
                failed.append({"prediction_number": predictions, "status": feedback.status, "code": feedback.code,
                    "confirmed_command_ids": sorted(acknowledged), "unconfirmed_command_ids": sorted(commands - acknowledged),
                    "actual_env_steps": feedback.consumed_env_steps, "actual_physics_steps": feedback.consumed_physics_steps,
                    "elapsed_sim_time_s": feedback.elapsed_sim_time_s})
            obs = data["observation"]
            env_steps += feedback.consumed_env_steps
            physics_steps += feedback.consumed_physics_steps
            sim_time += feedback.elapsed_sim_time_s
            if (any(obs["context"][key] != value for key, value in scope.items()) or obs["observation_id"] in observations
                    or obs["env_step"] != env_steps or not math.isclose(obs["sim_time_s"], sim_time, abs_tol=1e-7)):
                raise ValueError("post_action_observation_identity_or_actual_time_count_mismatch")
            observations.add(obs["observation_id"])
            previous_observation = obs["observation_id"]
            replies += 1
            last = {"terminated": data["terminated"], "truncated": data["truncated"], "status": feedback.status}
            pending = None
        elif kind == "error":
            if pending is not None or not data.get("state_known"):
                raise ValueError("unknown_or_unreplied_execution_cannot_be_certified")
            errors.append({key: data.get(key) for key in ("code", "phase", "state_known")})
        elif kind == "score":
            if pending is not None or scored is not None:
                raise ValueError("one_score_after_completed_replies_required")
            for metric in data.values():
                decode(leaf.Metric, metric)
            scored = data
        else:
            raise ValueError("unreviewed_trace_event")
    if resets != 1 or not replies or pending is not None:
        raise ValueError("complete_saved_real_execution_required")
    return {"reset_count": resets, "prediction_count": predictions, "reply_count": replies,
        "actual_env_steps": env_steps, "actual_physics_steps": physics_steps, "actual_sim_time_s": sim_time,
        "last_reply": last, "failed_replies": failed, "recorded_errors": errors, "recorded_metrics": scored,
        "all_successful_actions_acknowledged_every_command": True}


def verify(directory):
    def read(relative):
        return json.loads((directory / relative).read_text())
    receipt = read("receipt.json")
    manifest = read("framework/manifest.json")
    summary = read("framework/summary.json")
    result = read("framework/episodes/ep-0/result.json")
    diagnostics = read("framework/episodes/ep-0/diagnostics.internal.json")
    if manifest["synthetic"] or summary["synthetic"] or receipt.get("synthetic", True):
        raise ValueError("real_trial_receipts_required")
    trace_path = directory / "framework/episodes/ep-0/trace.internal.jsonl"
    trace_hash = hashlib.sha256()
    def events():
        with trace_path.open("rb") as stream:
            for line in stream:
                trace_hash.update(line)
                yield json.loads(line)
    accounting = analyze(events())
    stats = result["execution_statistics"]
    for key, actual in (("env_steps", accounting["actual_env_steps"]), ("physics_steps", accounting["actual_physics_steps"]),
            ("model_calls", accounting["prediction_count"])):
        if stats[key] != actual:
            raise ValueError("saved_reply_and_runner_counter_mismatch")
    if not math.isclose(stats["sim_time_s"], accounting["actual_sim_time_s"], abs_tol=1e-8):
        raise ValueError("saved_reply_and_runner_time_mismatch")
    row = summary["episodes"][0]
    if (len(summary["episodes"]) != 1 or not row["execution_statistics_complete"]
            or Path(row["result_ref"]).resolve() != (directory / "framework/episodes/ep-0/result.json").resolve()
            or row["termination_reason"] != result["termination_reason"]
            or receipt["execution_statistics"] != stats or result["metrics"] != receipt["metrics"]
            or accounting["recorded_metrics"] != result["metrics"]
            or summary["errors"] != receipt["framework_errors"] or summary["close_errors"]):
        raise ValueError("known_complete_receipt_summary_result_required")
    benchmark = diagnostics["benchmark"]
    if benchmark.get("family") == "robotwin":
        executor = benchmark["native"]["executor"]
        if not executor["native_state_known"]:
            raise ValueError("native_executor_state_unknown")
        native_steps, native_physics = executor["env_steps_last_confirmed"], executor["physics_steps_last_confirmed"]
        native_time = native_physics * executor["physics_period_s"]
    else:
        native_steps, native_physics, native_time = benchmark["native_actions"], benchmark["physics_steps"], benchmark["sim_time_s"]
    if native_steps != stats["env_steps"] or native_physics != stats["physics_steps"]:
        raise ValueError("native_benchmark_diagnostic_counter_mismatch")
    if not math.isclose(native_time, stats["sim_time_s"], abs_tol=1e-7):
        raise ValueError("native_benchmark_diagnostic_time_mismatch")
    if accounting["failed_replies"] and (receipt["passed"] or not receipt["framework_errors"]):
        raise ValueError("failed_action_must_not_become_trial_pass")
    if result["termination_reason"] == "environment_terminated" and not accounting["last_reply"]["terminated"]:
        raise ValueError("native_terminal_flag_required")
    files = ["receipt.json", "framework/manifest.json", "framework/summary.json",
        "framework/episodes/ep-0/result.json", "framework/episodes/ep-0/diagnostics.internal.json"]
    return {"kind": "saved_real_reply_accounting_consistency_not_independent_physics_or_task_acceptance",
        "at_utc": datetime.now(timezone.utc).isoformat(), "passed": True, "synthetic": False,
        "trial": str(directory.relative_to(ROOT)), "framework_trial_passed": receipt["passed"],
        "framework_client_exit_code": receipt["client_exit_code"], "framework_errors": receipt["framework_errors"],
        "actual_saved_reply_accounting": accounting, "metrics": result["metrics"],
        "termination_reason": result["termination_reason"], "execution_statistics_complete": True,
        "trace_sha256": trace_hash.hexdigest(),
        "evidence_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files},
        "loaded_source_sha256_from_manifest": manifest["source_sha256"],
        "current_accounting_audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_inference_or_simulation": False, "tolerance_only_for_accumulated_time_roundoff": 1e-7,
        "failed_native_actions_reclassified_as_success": False, "formal_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.trial, args.output = args.trial.resolve(), args.output.resolve()
    args.trial.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    value = verify(args.trial)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value[key] for key in ("passed", "framework_trial_passed", "framework_errors", "termination_reason")}))


if __name__ == "__main__":
    main()
