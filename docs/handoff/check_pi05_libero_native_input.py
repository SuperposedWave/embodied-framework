"""Read-only first-input mapping and native wire-codec comparison, no inference."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("immutable_output_required")
    from embodied_eval import _legacy_contracts as c
    from embodied_eval.codec import decode
    from embodied_eval.config import load_config
    from embodied_eval.stage4_pairs import verify
    from embodied_eval.adapters.openpi_libero_model import OpenpiLiberoModel
    config = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
    verify(config)
    sys.path.insert(0, str(Path(config.model.openpi_repo) / "packages/openpi-client/src"))
    from openpi_client import image_tools, msgpack_numpy
    with args.public_inputs.open() as stream:
        wire = json.loads(stream.readline())
    model_input = decode(c.ModelInput, wire)
    model = OpenpiLiberoModel(config.model, args.output.parent)
    model.images = image_tools
    mapped = model.native_input(model_input)  # No begin/connect/predict or service call.
    source = Path(config.model.openpi_repo) / "examples/libero/main.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_quat2axisangle")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    raw = wire["observation_view"]
    ext = raw["extensions"]["libero.native_proprio.v1"]
    native = {"prompt": wire["task_public"]["instruction"],
              "observation/state": np.concatenate((ext["eef_position_m"],
                   namespace["_quat2axisangle"](np.asarray(ext["eef_quaternion_xyzw"])), ext["gripper_qpos_m"]))}
    for key, target in (("agentview_image", "observation/image"), ("robot0_eye_in_hand_image", "observation/wrist_image")):
        image = np.asarray(raw["sensors"][key]["payload"]["data"], dtype=np.uint8)
        native[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(image[::-1, ::-1]), 224, 224))
    fields = {}
    for name in native:
        a, b = mapped[name], native[name]
        if isinstance(b, str):
            fields[name] = {"exact_equal": a == b}
        else:
            fields[name] = {"exact_equal": bool(np.array_equal(a, b)), "dtype_equal": a.dtype == b.dtype,
                "adapter_dtype": str(a.dtype), "native_dtype": str(b.dtype), "shape": list(b.shape),
                "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b))),
                "native_sha256": hashlib.sha256(b.tobytes()).hexdigest()}
    # NativePolicyPeer uses packb; official WebsocketClientPolicy uses Packer.
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(mapped))
    official = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(native))
    codec_exact = all(decoded[k] == official[k] if isinstance(official[k], str)
                      else decoded[k].dtype == official[k].dtype and np.array_equal(decoded[k], official[k])
                      for k in official)
    passed = codec_exact and all(v["exact_equal"] and v.get("dtype_equal", True) for v in fields.values())
    receipt = {"kind": "pi05_libero_first_native_input_mapping_and_codec_not_inference",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "model_called": False, "native_environment_called": False, "formal_acceptance": False,
        "native_wire_decoded_values_and_dtypes_exact": codec_exact, "fields": fields}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
