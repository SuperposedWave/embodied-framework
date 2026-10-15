"""Read-only real ARX exchange/input/cache comparison against pinned pure sources.

No inference, simulator, extra observations, sampling, retries or tolerances.
The whole generated chunk is checked separately from its executed prefix.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def source_observation(view):
    import numpy as np
    return {"vision": {name: {"color": np.asarray(view.sensors[slot].payload.data, np.uint8)}
        for slot, name in (("head", "cam_head"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist"))},
        "state": {side + original: np.asarray(view.states[side + slot].payload.data, np.float64)
            for side in ("left", "right") for slot, original in (("_joints", "_arm_joint_state"), ("_grip", "_ee_joint_state"))},
        "instruction": view.task_public.instruction}


def predictions(path):
    # Skip large RGB step/reset lines without decoding them. This is the
    # framework's compact ArtifactWriter format, not a generic trace parser.
    with path.open() as stream:
        for line in stream:
            if line.startswith('{"event":"prediction",'):
                yield json.loads(line)["data"]


def robotwin_conversion():
    """Execute only the unchanged production method's pre-execution prefix.

    Stop BEFORE any native method/command/RNG is called. The source hash and
    stop boundary make this a saved-action conversion check, not a replay.
    """
    import numpy as np
    from types import SimpleNamespace
    from embodied_eval import contracts as leaf
    from embodied_eval.validation import require
    from embodied_eval.adapters.native_common import finite_array
    from embodied_eval.mn.native_catalog import DUAL_PROTOCOL
    source = ROOT / "embodied_eval/mn/native_benchmarks.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "NativeSimulator")
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_advance_robotwin"))
    stop = next(index for index, node in enumerate(method.body) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and ast.unparse(node.targets[0]) == "runtime.executor.action_type")
    method.body = method.body[:stop] + [ast.Return(value=ast.Tuple(elts=[ast.Name(id="cast", ctx=ast.Load()),
        ast.Name(id="action_type", ctx=ast.Load())], ctx=ast.Load()))]
    scope = dict(np=np, leaf=leaf, require=require, finite_array=finite_array, DUAL_PROTOCOL=DUAL_PROTOCOL)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), scope)
    instance = SimpleNamespace(runtime=None)
    def convert(action):
        row, mode = scope["_advance_robotwin"](instance, action, None, None)
        if mode != "qpos":
            raise ValueError("saved_ARX_joint_actions_required")
        return row
    def native_row(value):
        # These four arrays come from the fixed native unpack_robot_state.
        # Concatenation is the unchanged RoboTwin take_action qpos row order.
        return np.concatenate([value[name] for name in ("left_arm_joint_state", "left_ee_joint_state",
            "right_arm_joint_state", "right_ee_joint_state")])
    return convert, native_row, hashlib.sha256(source.read_bytes()).hexdigest()


def compare(directory):
    import numpy as np
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.codec import decode
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.assets_common import _revision
    from embodied_eval.mn.native_catalog import PROFILES
    from embodied_eval.mn.native_models import NativeActionModel
    from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator
    from check_mn_extension_native_mappings import functions, same
    manifest = json.loads((directory / "framework/manifest.json").read_text())
    cfg = manifest["config"]
    if manifest["synthetic"] or cfg["model"]["adapter"] != "openpi" or cfg["model"]["options"]["io_profile"] != "arx_joint14":
        raise ValueError("recorded_real_arx_joint_policy_required")
    exchanges_dir = directory / "native-exchanges"
    exchanges = json.loads((exchanges_dir / "receipt.json").read_text())
    if not exchanges["all_observed_exchanges_completed"]:
        raise ValueError("incomplete_exchange_evidence")
    model = NativeActionModel("openpi", cfg["model"]["options"])
    _revision(model.cfg.repo, PROFILES["arx_joint14"][2])
    model.space = decode(c.SpaceSpec, manifest["space"])
    model.profile = decode(c.BoundProfile, manifest["profile"])
    benchmark = cfg["benchmark"]["adapter"]
    conversion_source = None
    if benchmark == "robodojo":
        convert = RoboDojoSimulator(cfg["benchmark"]["options"]).native_action
        expected_conversion = lambda value: value
    elif benchmark == "robotwin":
        convert, expected_conversion, conversion_source = robotwin_conversion()
    else:
        raise ValueError("reviewed_ARX_simulator_conversion_required")
    codec = native_module(model.cfg.repo, "openpi_client.msgpack_numpy", "policy/Pi_05/openpi/packages/openpi-client/src")
    scope = {"np": np, "Any": Any}
    repo = Path(model.cfg.repo)
    sources = [repo / "utils/process_data.py", repo / "policy/Pi_05/model.py"]
    hashes = functions(sources[0], ("_validate_config", "_get_state_keys", "_ensure_valid_state_array", "pack_robot_state", "unpack_robot_state"), scope)
    hashes.update(functions(sources[1], ("encode_obs", "extract_image", "ensure_chw_uint8"), scope))
    dims = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    episode = directory / "framework/episodes/ep-0"
    result = json.loads((episode / "result.json").read_text())
    actual_predictions = iter(predictions(episode / "trace.internal.jsonl"))
    chunks, input_count, executed_count, generated_count = [], 0, 0, 0
    with (episode / "model_input.public.jsonl").open() as inputs:
        for index, exchange in enumerate(exchanges["records"]):
            line = next(inputs)
            input_count += 1
            view = decode(c.ModelInput, json.loads(line))
            request_path = exchanges_dir / exchange["request_file"]
            response_path = exchanges_dir / exchange["response_file"]
            request_bytes, response_bytes = request_path.read_bytes(), response_path.read_bytes()
            if hashlib.sha256(request_bytes).hexdigest() != exchange["request_sha256"] or hashlib.sha256(response_bytes).hexdigest() != exchange["response_sha256"]:
                raise ValueError("captured_exchange_bytes_changed")
            request, response = codec.unpackb(request_bytes), codec.unpackb(response_bytes)
            expected_input = scope["encode_obs"](source_observation(view), "joint", dims)
            actions = np.asarray(response["actions"])
            if actions.shape != (50, 14) or not np.isfinite(actions).all():
                raise ValueError("pinned_native_50_row_chunk_required")
            expected_rows = scope["unpack_robot_state"](actions, "joint", dims, source_type="obs")
            whole_chunk_exact = all(same(expected_conversion(expected), convert(model.decode_action(row, view.context, view)))
                for expected, row in zip(expected_rows, actions))
            generated_count += len(actions)
            checks = []
            remaining = result["execution_statistics"]["env_steps"] - executed_count
            prefix_length = min(len(actions), remaining)
            if prefix_length < 1 or index + 1 < len(exchanges["records"]) and prefix_length != 50:
                raise ValueError("unexpected_replan_or_cache_boundary")
            for offset in range(prefix_length):
                if offset:
                    next(inputs)
                    input_count += 1
                prediction = decode(c.Prediction, next(actual_predictions))
                if prediction.context.message_id != f"request:{executed_count + 1}":
                    raise ValueError("recorded_prediction_order_changed")
                checks.append(same(expected_conversion(expected_rows[offset]), convert(prediction.action)))
                executed_count += 1
            chunks.append({"index": index + 1, "first_prediction_number": index * 50 + 1,
                "input_public_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                "input_native_source_exact": same(expected_input, request),
                "input_model_conversion_exact": same(expected_input, model.native_input(view)),
                "generated_chunk_shape": list(actions.shape), "generated_chunk_dtype": str(actions.dtype),
                "all_50_native_conversion_values_and_dtypes_exact": whole_chunk_exact,
                "executed_prefix_length": prefix_length, "all_executed_cache_rows_exact": all(checks),
                "source_state_provenance": {key: value.provenance for key, value in view.states.items()}})
        if next(inputs, None) is not None or next(actual_predictions, None) is not None:
            raise ValueError("unaccounted_model_input_or_prediction")
    return {"kind": "real_recorded_native_arx_mapping_and_executed_cache_order_not_reinference",
        "passed": executed_count == result["execution_statistics"]["env_steps"] and all(
            row["input_native_source_exact"] and row["input_model_conversion_exact"] and
            row["all_50_native_conversion_values_and_dtypes_exact"] and row["all_executed_cache_rows_exact"] for row in chunks),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "trial": str(directory.relative_to(ROOT)),
        "benchmark": benchmark, "benchmark_conversion_source_sha256": conversion_source,
        "native_requests": len(chunks), "generated_rows_compared": generated_count,
        "executed_cache_rows_compared": executed_count, "public_input_lines_accounted_for": input_count,
        "rows": chunks, "source_function_sha256": hashes,
        "source_file_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "new_policy_inference": False, "new_simulator_episode": False, "fresh_native_reinference_compared": False,
        "task_success_used_as_mapping_proof": False, "formal_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.trial, args.output = args.trial.resolve(), args.output.resolve()
    args.trial.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    result = compare(args.trial)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "native_requests", "generated_rows_compared", "executed_cache_rows_compared")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
