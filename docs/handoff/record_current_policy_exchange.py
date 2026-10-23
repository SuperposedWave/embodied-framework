"""Passive current model-wire evidence; no retries, extra calls or headers.

Install a peer observer only AFTER the model's normal connection/import order.
Reset/health traffic is not recorded. Source codecs preserve ndarray dtype;
only public inference payloads and returned actions enter these private files.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


@contextmanager
def record_exchanges(directory):
    import numpy as np
    before = np.random.get_state()
    from embodied_eval.native.native_models import NativeActionModel
    after = np.random.get_state()
    if before[0] != after[0] or not np.array_equal(before[1], after[1]) or before[2:] != after[2:]:
        raise ValueError("diagnostic_setup_must_not_consume_numpy_rng")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    connect = NativeActionModel._connect
    installed, records = [], []
    def save(path, payload):
        with path.open("xb") as stream:
            stream.write(payload)
        return hashlib.sha256(payload).hexdigest()
    def connected(model, *args, **kwargs):
        result = connect(model, *args, **kwargs)
        peer = model.peer
        method = "predict_once" if model.family == "openwam" else "infer"
        owner = type(peer)
        if any(cls is owner and name == method for cls, name, original in installed):
            return result
        original = getattr(owner, method)
        def observed(instance, value, *values, **keywords):
            if value.get("reset") is True:
                return original(instance, value, *values, **keywords)
            expected = {
                "openwam": {"images", "state", "prompt"},
                "openpi": {"observation/image", "observation/wrist_image", "observation/state", "prompt"},
                "lingbot_vla": {"observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist", "observation.state", "task"},
            }[model.family]
            if model.cfg.io_profile == "state7_absolute":
                expected = expected | {"observation/second_image"}
            if set(value) != expected:
                raise ValueError("only_source_defined_public_inference_keys_may_be_recorded")
            row = {"index": len(records) + 1, "family": model.family, "completed": False}
            records.append(row)
            suffix = ".json" if model.family == "openwam" else ".msgpack"
            encode = (lambda v: json.dumps(v, ensure_ascii=False).encode()) if suffix == ".json" else instance.codec.packb
            stem = f"request-{row['index']:04d}"
            row["request_file"] = stem + suffix
            row["request_sha256"] = save(directory / row["request_file"], encode(value))
            reply = original(instance, value, *values, **keywords)
            key = "actions" if model.family == "openpi" else "action"
            row["actions_file"] = f"action-{row['index']:04d}" + suffix
            row["actions_sha256"] = save(directory / row["actions_file"], encode({key: reply[key]}))
            row["completed"] = True
            return reply
        installed.append((owner, method, original))
        setattr(owner, method, observed)
        return result
    NativeActionModel._connect = connected
    try:
        yield
    finally:
        NativeActionModel._connect = connect
        for owner, name, original in reversed(installed):
            setattr(owner, name, original)
        (directory / "receipt.json").write_text(json.dumps({
            "kind": "passively_observed_native_wire_not_new_public_protocol", "records": records,
            "headers_authentication_metadata_or_raw_errors_saved": False,
            "extra_inference_reset_get_obs_or_render_calls": 0,
            "request_and_return_objects_preserved": True, "setup_numpy_rng_unchanged": True,
            "all_exchanges_completed": bool(records) and all(row["completed"] for row in records)}, indent=2) + "\n")
