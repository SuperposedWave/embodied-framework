"""Read-only real DreamZero public input, complete chunk and cache audit.

Native fixed-source client methods replay saved real observations and outputs.
No new inference, simulator, RNG change, source edits or relaxed tolerance.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from probe_mn_components import ROOT, write


def compare(directory):
    import numpy as np
    import torch
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.codec import decode
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.additional_catalog import verify_policy
    from embodied_eval.mn.additional_models import JointPolicyModel
    from embodied_eval.mn.droid_simulator import DroidSimulator
    from check_mn_extension_native_mappings import native_class, same
    from compare_arx_recorded_exchanges import predictions
    manifest = json.loads((directory / "framework/manifest.json").read_text())
    cfg = manifest["config"]
    if manifest["synthetic"] or cfg["model"]["adapter"] != "dreamzero" or cfg["benchmark"]["adapter"] != "droid_sim":
        raise ValueError("completed_real_dreamzero_droid_trial_required")
    trial_receipt = json.loads((directory / "receipt.json").read_text())
    if not trial_receipt["passed"]:
        raise ValueError("completed_runtime_receipt_required")
    assets = verify_policy("dreamzero", cfg["model"]["options"])
    model = JointPolicyModel("dreamzero", cfg["model"]["options"])
    model.space = decode(c.SpaceSpec, manifest["space"])
    model.profile = decode(c.BoundProfile, manifest["profile"])
    model.images = native_module(model.cfg.codec_repo, "openpi_client.image_tools", "packages/openpi-client/src")
    codec = native_module(model.cfg.codec_repo, "openpi_client.msgpack_numpy", "packages/openpi-client/src")
    executor = DroidSimulator.__new__(DroidSimulator)
    executor.space = model.space
    source = Path(model.cfg.repo) / "eval_utils/run_sim_eval.py"
    exchanges_dir = directory / "native-exchanges"
    exchanges = json.loads((exchanges_dir / "receipt.json").read_text())
    if not exchanges["all_observed_exchanges_completed"]:
        raise ValueError("all_native_exchanges_must_complete")
    episode = directory / "framework/episodes/ep-0"
    result = json.loads((episode / "result.json").read_text())
    actual = iter(predictions(episode / "trace.internal.jsonl"))
    executed = generated = input_count = 0
    checks = []
    with (episode / "model_input.public.jsonl").open() as inputs:
        for exchange in exchanges["records"]:
            line = next(inputs)
            input_count += 1
            view = decode(c.ModelInput, json.loads(line))
            request_bytes = (exchanges_dir / exchange["request_file"]).read_bytes()
            response_bytes = (exchanges_dir / exchange["response_file"]).read_bytes()
            if hashlib.sha256(request_bytes).hexdigest() != exchange["request_sha256"] or hashlib.sha256(response_bytes).hexdigest() != exchange["response_sha256"]:
                raise ValueError("recorded_exchange_checksum_changed")
            request, response = codec.unpackb(request_bytes), codec.unpackb(response_bytes)
            rows = response["actions"]
            if rows.ndim != 2 or rows.shape[1] != 8 or rows.dtype != np.float32 or len(rows) < model.horizon or not np.isfinite(rows).all():
                raise ValueError("complete_finite_native_float32_joint8_chunk_required")
            oracle = native_class(source, "DreamZeroJointPosClient", ("infer", "_extract_observation"),
                {"np": np, "image_tools": model.images})
            # Exercising every saved generated row is a pure conversion audit;
            # actual executed cache horizon stays separately counted below.
            oracle.actions_from_chunk_completed, oracle.open_loop_horizon = 0, len(rows)
            oracle.session_id = model.session_id = request["session_id"]
            sent = []
            def recorded_request(value):
                value["endpoint"] = "infer"
                sent.append(value)
                return {"actions": rows}
            oracle.client = SimpleNamespace(infer=recorded_request)
            raw = {"policy": {**{name: torch.from_numpy(np.asarray(view.sensors[slot].payload.data, np.uint8))[None]
                for slot, name in (("exterior_right", "external_cam"), ("exterior_left", "external_cam_2"), ("wrist", "wrist_cam"))},
                "arm_joint_pos": torch.tensor(view.states["joints"].payload.data, dtype=torch.float64),
                "gripper_pos": 1. - torch.tensor(view.states["grip"].payload.data, dtype=torch.float64)}}
            with redirect_stdout(io.StringIO()):
                expected = [oracle.infer(raw, view.task_public.instruction)["action"] for _ in rows]
            generated += len(rows)
            whole_exact = all(same(target, executor.native_action(model.decode_action(row, view.context)))
                for row, target in zip(rows, expected))
            prefix = min(model.horizon, result["execution_statistics"]["env_steps"] - executed)
            if prefix < 1:
                raise ValueError("extra_native_request_without_executed_action")
            cache_checks = []
            for offset in range(prefix):
                if offset:
                    next(inputs)
                    input_count += 1
                prediction = decode(c.Prediction, next(actual))
                if prediction.context.message_id != f"request:{executed + 1}":
                    raise ValueError("native_cache_prediction_order_changed")
                cache_checks.append(same(expected[offset], executor.native_action(prediction.action)))
                executed += 1
            checks.append({"request_index": exchange["index"], "native_input_pixels_state_prompt_values_dtypes_exact": same(sent[0], request),
                "model_input_mapping_exact": same(model.native_input(view), request),
                "complete_generated_rows": len(rows), "generated_dtype": str(rows.dtype),
                "all_generated_action_conversions_exact": whole_exact, "executed_cache_rows": prefix,
                "all_executed_cache_rows_exact": all(cache_checks),
                "real_public_input_line_sha256": hashlib.sha256(line.encode()).hexdigest()})
        if next(inputs, None) is not None or next(actual, None) is not None:
            raise ValueError("unaccounted_public_input_or_prediction")
    return {"kind": "recorded_real_dreamzero_native_input_chunk_and_cache_audit_not_reinference",
        "at_utc": datetime.now(timezone.utc).isoformat(), "trial": str(directory.relative_to(ROOT)),
        "passed": executed == result["execution_statistics"]["env_steps"] and all(row["native_input_pixels_state_prompt_values_dtypes_exact"]
            and row["model_input_mapping_exact"] and row["all_generated_action_conversions_exact"] and row["all_executed_cache_rows_exact"] for row in checks),
        "native_requests": len(checks), "generated_rows_compared": generated, "executed_cache_rows_compared": executed,
        "public_input_lines_accounted_for": input_count, "rows": checks,
        "native_client_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "asset_verification": assets, "synthetic": False, "new_policy_inference": False,
        "new_simulator_episode": False, "fresh_native_reinference_compared": False,
        "native_source_unchanged": True, "task_success_used_as_mapping_proof": False, "formal_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.trial, args.output = args.trial.resolve(), args.output.resolve()
    for path in (args.trial, args.output):
        path.relative_to(ROOT / "outputs")
    proof = compare(args.trial)
    write(args.output, proof)
    print(json.dumps({key: proof[key] for key in ("passed", "native_requests", "generated_rows_compared", "executed_cache_rows_compared")}))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
