"""Reconstruct a saved current input's real model codec, without connecting.

Compare with the independently recorded native request, including dtype. This
is reconstruction, not a claim that a previously uncaptured packet was logged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def compare(a, b):
    import numpy as np
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return {"exact": False, "keys_a": sorted(a), "keys_b": sorted(b)}
        items = {key: compare(a[key], b[key]) for key in a}
        return {"exact": all(row["exact"] for row in items.values()), "fields": items}
    if isinstance(a, str) or isinstance(b, str):
        return {"exact": type(a) is type(b) and a == b}
    left, right = np.asarray(a), np.asarray(b)
    row = {"exact": left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right),
           "left_shape": list(left.shape), "right_shape": list(right.shape), "left_dtype": str(left.dtype), "right_dtype": str(right.dtype)}
    if left.shape == right.shape and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        row.update(max_abs_diff=float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.,
                   different_scalars=int(np.count_nonzero(left != right)))
    if left.size <= 80 and right.size <= 80:
        row.update(left_values=left.tolist(), right_values=right.tolist())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, required=True)
    p.add_argument("--native-exchanges", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from embodied_eval.codec import decode
    from embodied_eval.contracts import ModelInput
    from embodied_eval.native.model_catalog import capability
    from embodied_eval.native.native_models import NativeActionModel
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.v03.models import profile_paths, private_input
    manifest = json.loads((args.framework / "manifest.json").read_text())
    cfg = manifest["config"]
    if manifest["synthetic"] or cfg["contract_version"] != "0.3":
        raise ValueError("real_current_framework_required")
    family, options = cfg["model"]["adapter"], cfg["model"]["options"]
    episode = args.framework / "episodes" / cfg["episodes"][0]["episode_id"]
    with (episode / "model_input.public.jsonl").open() as stream:
        first = stream.readline()
    view = decode(ModelInput, json.loads(first))
    cap = capability(family, options)
    native_view = private_input(view, cap, profile_paths(options["io_profile"], cap))
    model = NativeActionModel(family, options)
    if family == "openwam":
        model.utils = native_module(options["repo"], "benchmarks.utils")
        model.conversion = native_module(options["repo"], "benchmarks.utils.action_conversion")
    elif options["io_profile"] == "state8_osc":
        model.images = native_module(options["repo"], "openpi_client.image_tools", "packages/openpi-client/src")
    actual = model.native_input(native_view)
    if family == "openwam":
        expected = json.loads((args.native_exchanges / "request-0001.json").read_text())
    else:
        codec = native_module(options["repo"], "deploy.msgpack_numpy" if family == "lingbot_vla" else "openpi_client.msgpack_numpy",
            "" if family == "lingbot_vla" else "packages/openpi-client/src")
        expected = codec.unpackb((args.native_exchanges / "request-0001.msgpack").read_bytes())
    result = compare(actual, expected)
    report = {"kind": "current_saved_public_input_native_codec_reconstruction_not_new_inference", "at_utc": datetime.now(timezone.utc).isoformat(),
              "passed": result["exact"], "framework": str(args.framework), "native_exchanges": str(args.native_exchanges),
              "saved_public_input_line_sha256": hashlib.sha256(first.encode()).hexdigest(),
              "additional_inference_reset_or_environment_calls": 0, "actual_packet_capture_claim": False, "comparison": result}
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
