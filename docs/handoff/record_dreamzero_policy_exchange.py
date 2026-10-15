"""Diagnostic-only public DreamZero request/action recording, no new protocol.

Never records headers, server metadata, authentication or arbitrary error text.
Original peer arguments/results and native inference count/order are unchanged.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from record_arx_policy_exchange import write_binary


def check_request(value):
    import numpy as np
    images = {"observation/exterior_image_0_left", "observation/exterior_image_1_left", "observation/wrist_image_left"}
    keys = images | {"observation/joint_position", "observation/gripper_position",
        "observation/cartesian_position", "session_id", "prompt", "endpoint"}
    if not isinstance(value, dict) or set(value) != keys or value["endpoint"] != "infer":
        raise ValueError("exact_native_public_dreamzero_request_required")
    if not isinstance(value["prompt"], str) or not isinstance(value["session_id"], str):
        raise ValueError("public_instruction_and_session_required")
    for name in images:
        array = value[name]
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.shape != (180, 320, 3):
            raise ValueError("native_public_three_RGB_views_required")
    for name, shape in (("observation/joint_position", (7,)), ("observation/gripper_position", (1,)),
            ("observation/cartesian_position", (6,))):
        array = value[name]
        if not isinstance(array, np.ndarray) or array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError("native_public_state_shape_and_dtype_required")
    if np.any(value["observation/cartesian_position"] != 0):
        raise ValueError("source_defined_unused_cartesian_compatibility_field_required")


@contextmanager
def record_exchanges(directory):
    import numpy as np
    from embodied_eval.adapters.native_policy_transport import NativePolicyPeer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original = NativePolicyPeer.infer
    records = []
    def observed(peer, value):
        check_request(value)
        record = {"index": len(records) + 1, "completed": False}
        records.append(record)
        stem = f"exchange-{record['index']:04d}"
        record["request_file"] = stem + "-request.msgpack"
        record["request_sha256"] = write_binary(directory / record["request_file"], peer.codec.packb(value))
        result = original(peer, value)
        chunk = result.get("actions") if isinstance(result, dict) else result
        if not isinstance(chunk, np.ndarray) or chunk.ndim != 2 or chunk.shape[1] != 8 or chunk.dtype != np.float32 or not np.isfinite(chunk).all():
            raise ValueError("native_float32_joint8_response_required")
        record["response_file"] = stem + "-actions.msgpack"
        record["response_sha256"] = write_binary(directory / record["response_file"], peer.codec.packb({"actions": chunk}))
        record["completed"] = True
        return result
    NativePolicyPeer.infer = observed
    try:
        yield
    finally:
        NativePolicyPeer.infer = original
        with (directory / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump({"kind": "diagnostic_native_public_dreamzero_exchange_not_protocol",
                "records": records, "all_observed_exchanges_completed": bool(records) and all(row["completed"] for row in records),
                "headers_authentication_metadata_or_raw_errors_saved": False,
                "additional_inference_retry_or_sampling": False, "original_return_objects_preserved": True}, stream, indent=2)
            stream.write("\n")
