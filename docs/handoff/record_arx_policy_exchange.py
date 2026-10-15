"""Diagnostic-only native ARX requests/chunks, not a new Model protocol.

Observe one normal native peer call without retrying, sampling or altering its
arguments/return object. Save only the source-defined public input and actions;
never transport headers, metadata, error text or authentication fields.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def write_binary(path, value):
    with path.open("xb") as stream:
        stream.write(value)
    return hashlib.sha256(value).hexdigest()


@contextmanager
def record_exchanges(directory):
    import numpy as np
    from embodied_eval.adapters.native_policy_transport import NativePolicyPeer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = NativePolicyPeer.infer
    records = []
    def observed(peer, value):
        record = {"index": len(records) + 1, "completed": False}
        records.append(record)
        if not isinstance(value, dict) or set(value) != {"state", "images", "prompt"}:
            raise ValueError("source_defined_public_arx_request_required")
        if not isinstance(value["prompt"], str) or not isinstance(value["images"], dict) or set(value["images"]) != {
                "cam_high", "cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("source_defined_arx_image_slots_required")
        state = np.asarray(value["state"])
        if state.shape != (14,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise ValueError("source_defined_arx_state_required")
        if any(not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3
               for image in value["images"].values()):
            raise ValueError("source_defined_public_arx_rgb_required")
        stem = f"exchange-{record['index']:04d}"
        request = peer.codec.packb(value)
        record["request_file"] = stem + "-request.msgpack"
        record["request_sha256"] = write_binary(directory / record["request_file"], request)
        result = original(peer, value)
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError("source_defined_arx_action_chunk_required")
        # Source response timing/extra fields are intentionally not saved.
        response = peer.codec.packb({"actions": result["actions"]})
        record["response_file"] = stem + "-actions.msgpack"
        record["response_sha256"] = write_binary(directory / record["response_file"], response)
        record["completed"] = True
        return result
    NativePolicyPeer.infer = observed
    try:
        yield
    finally:
        NativePolicyPeer.infer = original
        with (directory / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump({"kind": "diagnostic_observed_source_native_public_arx_exchange_not_new_protocol",
                "records": records, "all_observed_exchanges_completed": bool(records) and all(row["completed"] for row in records),
                "headers_authentication_metadata_or_raw_errors_saved": False, "response_only_actions_saved": True,
                "additional_inference_retry_or_sampling": False, "original_return_objects_preserved": True}, stream, indent=2)
            stream.write("\n")
