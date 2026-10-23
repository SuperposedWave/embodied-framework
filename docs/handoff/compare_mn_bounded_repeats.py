"""Read two saved ARX/RoboTwin trials; no inference, simulation or trace edits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_trial(root):
    from embodied_eval.codec import decode
    from embodied_eval.native import contracts as c
    folder = root / "pi05_arx_robotwin/framework/episodes/ep-0"
    with (folder / "model_input.public.jsonl").open() as stream:
        view = decode(c.ModelInput, json.loads(stream.readline()))
    with (folder / "trace.internal.jsonl").open() as stream:
        prediction = next(json.loads(line)["data"] for line in stream if json.loads(line)["event"] == "prediction")
    result = json.loads((folder / "result.json").read_text())
    return view, decode(c.Action, prediction["action"]), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from embodied_eval.native.native_models import NativeActionModel
    from embodied_eval.core import types as leaf
    a, pa, ra = read_trial(args.first)
    b, pb, rb = read_trial(args.repeat)
    model = NativeActionModel("openpi", {"repo": "/not-imported", "checkpoint_dir": "/not-loaded", "io_profile": "arx_joint14"})
    native_a, native_b = model.native_input(a), model.native_input(b)
    def same(left, right):
        if isinstance(left, dict):
            return isinstance(right, dict) and set(left) == set(right) and all(same(value, right[key]) for key, value in left.items())
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
        return left == right
    keys = {key: bool(same(native_a[key], native_b[key])) for key in native_a}
    def values(action):
        joints, grips = {}, {}
        for command in action.commands:
            if isinstance(command, leaf.JointTarget):
                joints[command.target_id] = command.value
            elif isinstance(command, leaf.ControllerCommand):
                grips[command.target_id] = command.parameters["requested_open_fraction"]
            else:
                raise ValueError("unexpected ARX command")
        return joints, grips
    ja, ga = values(pa)
    jb, gb = values(pb)
    if set(ja) != set(jb) or set(ga) != set(gb):
        raise ValueError("target identity changed")
    receipt = {"kind": "read_only_bounded_fresh_repeat_comparison_not_native_reference_acceptance",
        "at_utc": datetime.now(timezone.utc).isoformat(), "first": str(args.first), "repeat": str(args.repeat),
        "native_first_input_keys_exact": keys, "native_first_input_values_and_dtypes_exact": all(keys.values()),
        "first_joint_targets_exact": ja == jb, "maximum_first_joint_target_difference_rad": max(abs(ja[k] - jb[k]) for k in ja),
        "first_raw_gripper_requests_exact": ga == gb, "first_action_native_dtype_exact": pa.native_dtype == pb.native_dtype,
        "first_statistics": ra["execution_statistics"], "repeat_statistics": rb["execution_statistics"],
        "model_requests_performed_by_this_script": 0, "weights_loaded_by_this_script": False,
        "stochastic_or_backend_cause_proven": False, "independent_native_policy_comparison_performed": False,
        "formal_acceptance": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
