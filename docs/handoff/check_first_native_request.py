"""Read-only first native request versus framework public-input packing.

This checks the production model's CPU packing against a captured *actual*
native request. It does not claim the framework's historical wire was captured,
rerun inference, start an environment, or replace closedloop equality checks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from probe_mn_components import ROOT, write, public_exception
sys.path.insert(0, str(ROOT))


def compare(framework, native, output):
    import numpy as np
    from embodied_eval.codec import decode
    from embodied_eval.native import contracts as c
    from embodied_eval.native.native_models import NativeActionModel
    from embodied_eval.native.additional_models import JointPolicyModel
    from embodied_eval.adapters.native_common import native_module
    from check_mn_extension_native_mappings import same
    manifest_path = framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest["config"]
    if manifest["synthetic"]:
        raise ValueError("real_first_public_input_required")
    episode = framework / "episodes" / cfg["episodes"][0]["episode_id"]
    with (episode / "model_input.public.jsonl").open("rb") as stream:
        first_line = stream.readline()
    view = decode(c.ModelInput, json.loads(first_line))
    family, options = cfg["model"]["adapter"], cfg["model"]["options"]
    arx_wam = family == "openwam" and options["io_profile"] == "eef20_arx_absolute"
    if arx_wam:
        from embodied_eval.native.openwam_arx import OpenWAMArxModel, SOURCE_ROOT
        model = OpenWAMArxModel(options)
        model.poses = native_module(options["repo"], "openwam.dataloader.utils.poses", SOURCE_ROOT)
        model.calibration = native_module(options["repo"], "openwam.dataloader.robodojo_contract", SOURCE_ROOT).arx_x5_calibration()
        model.format_prompt = native_module(options["repo"], "openwam.dataloader.transforms.multiview", SOURCE_ROOT).format_prompt_for_inference
    else:
        model = JointPolicyModel(family, options) if family == "cosmos3" else NativeActionModel(family, options)
    report = {"kind": "CPU_packing_against_actual_native_first_request_not_framework_wire_capture",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False, "synthetic": False,
        "new_inference_or_simulator_started": False, "production_Adapter_CPU_packing_used_in_this_check": True,
        "production_Adapter_execution_used_in_native_closedloop": False, "actual_framework_wire_captured": False,
        "framework_public_input_first_line_sha256": hashlib.sha256(first_line).hexdigest(),
        "framework_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "formal_acceptance": False}
    if arx_wam:
        request_path = native / "native-exchanges/request-0001.npz"
        with np.load(request_path, allow_pickle=False) as archive:
            names = ("head_camera", "left_wrist_camera", "right_wrist_camera")
            if set(archive.files) != {"state", "prompt", *names}:
                raise ValueError("complete_native_payload_array_archive_required")
            actual = {"state": archive["state"].copy(), "prompt": str(archive["prompt"].item()),
                "images": {name: archive[name].copy() for name in names}}
        report["native_request_is_lossless_payload_array_archive_not_network_wire"] = True
    elif family == "openwam":
        model.utils = native_module(options["repo"], "benchmarks.utils")
        model.conversion = native_module(options["repo"], "benchmarks.utils.action_conversion")
        request_path = native / "native-exchanges/request-0001.json"
        actual = json.loads(request_path.read_text())
    else:
        request_path = native / "native-exchanges/request-0001.msgpack"
        if not request_path.exists():
            request_path = native / "native-exchanges/exchange-0001-request.msgpack"
        if family == "lingbot_vla":
            codec = native_module(options["repo"], "deploy.msgpack_numpy")
        else:
            prefix = "policy/Pi_05/openpi/packages/openpi-client/src" if options["io_profile"] == "arx_joint14" else "packages/openpi-client/src"
            repo = options.get("codec_repo", options["repo"])
            codec = native_module(repo, "openpi_client.msgpack_numpy", prefix)
            if options["io_profile"] == "state8_osc":
                model.images = native_module(repo, "openpi_client.image_tools", prefix)
            elif family == "cosmos3":
                model.images = native_module(options["client_repo"], "robolab.core.utils.image_utils")
        actual = codec.unpackb(request_path.read_bytes())
        if family == "cosmos3":
            # Compare all request fields, including the actual fresh UUID.
            # Historical UUID was not recorded and is not claimed equal.
            model.session_id = actual["session_id"]
            report["same_historical_framework_session_UUID_verified"] = False
    expected = model.native_input(view)
    if arx_wam:
        # Source encoding already returns float32 state values and PIL RGB.
        # Compare their exact archive representation, not Python object IDs.
        expected = {"state": np.asarray(expected["state"], np.float32), "prompt": expected["prompt"],
            "images": {name: np.asarray(value) for name, value in expected["images"].items()}}
    report.update(native_request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
        fields={key: {"exact_equal": key in actual and same(value, actual[key])} for key, value in expected.items()},
        request_field_inventory_equal=set(expected) == set(actual))
    report["passed"] = report["request_field_inventory_equal"] and all(row["exact_equal"] for row in report["fields"].values())
    write(output, report)
    print(json.dumps({"passed": report["passed"], "fields": report["fields"]}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new_check_receipt_required")
    for path in (args.framework, args.native, args.output):
        path.resolve().relative_to(ROOT / "outputs")
    try:
        return compare(args.framework, args.native, args.output)
    except Exception as error:
        write(args.output, {"passed": False, "exception_chain": public_exception(error)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
