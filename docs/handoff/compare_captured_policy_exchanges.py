"""Compare actually recorded model-wire payloads; no reconstruction/inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from compare_current_wire_input import compare
    from embodied_eval.adapters.native_common import native_module
    manifest = json.loads((args.case / "framework/manifest.json").read_text())
    options = manifest["config"]["model"]["options"]
    family = manifest["config"]["model"]["adapter"]
    captured = args.case / "model-exchanges"
    native = args.case / "native/native-exchanges"
    receipt = json.loads((captured / "receipt.json").read_text())
    records = receipt["records"]
    if not records or not all(row["completed"] for row in records):
        raise ValueError("completed_actual_framework_wire_capture_required")
    if family == "openwam":
        decode = lambda path: json.loads(path.read_text())
        suffix = ".json"
    else:
        codec = native_module(options["repo"], "deploy.msgpack_numpy" if family == "lingbot_vla" else "openpi_client.msgpack_numpy",
            "" if family == "lingbot_vla" else "packages/openpi-client/src")
        decode = lambda path: codec.unpackb(path.read_bytes())
        suffix = ".msgpack"
    rows = []
    for row in records:
        index = row["index"]
        request = compare(decode(captured / row["request_file"]), decode(native / f"request-{index:04d}{suffix}"))
        action = compare(decode(captured / row["actions_file"]), decode(native / f"action-{index:04d}{suffix}"))
        rows.append({"index": index, "request": request, "returned_actions": action})
    native_count = len(list(native.glob("request-*" + suffix)))
    report = {"kind": "actually_captured_native_wire_comparison_not_reconstructed_packets",
              "at_utc": datetime.now(timezone.utc).isoformat(), "case": str(args.case),
              "framework_requests": len(records), "native_requests": native_count,
              "request_count_equal": native_count == len(records), "rows": rows,
              "all_requests_equal": all(row["request"]["exact"] for row in rows),
              "all_returned_actions_equal": all(row["returned_actions"]["exact"] for row in rows),
              "extra_inference_or_simulation_calls": 0, "actual_packets_captured": True}
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("framework_requests", "native_requests", "all_requests_equal", "all_returned_actions_equal")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
