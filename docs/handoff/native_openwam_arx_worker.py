"""Diagnostic native-source OpenWAM engine, never instantiate a ModelAdapter.

Private inherited pipe only. Execute pinned source encode/conditions/decode and
get_action methods with a real native engine. Save only public payload/actions.
"""
import argparse
from collections.abc import Mapping
from multiprocessing.connection import Connection
from pathlib import Path
import resource
import sys
from typing import Any

from probe_mn_components import ROOT, write, public_exception
sys.path.insert(0, str(ROOT))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe-fd", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = Connection(args.pipe_fd)
    model = server = None
    try:
        options = connection.recv()
        from embodied_eval.mn.assets import verify_model
        from check_mn_extension_native_mappings import functions, native_class
        from native_array_wire import pack, unpack
        import numpy as np
        from omegaconf import OmegaConf
        assets = verify_model("openwam", options)
        repo = Path(options["repo"])
        root = repo / "policy/OpenWAM/OpenWAM"
        sys.path.insert(0, str(root))
        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config
        from openwam.dataloader.utils import poses
        from openwam.dataloader.robodojo_contract import arx_x5_calibration
        from openwam.dataloader.transforms.multiview import format_prompt_for_inference
        source = repo / "policy/OpenWAM/model.py"
        scope = {"np": np, "Any": Any, "Mapping": Mapping, "_GRIPPER_ATOL": 1e-8, "_EEF20_DIM": 20,
            "_CAMERA_TO_PAYLOAD": {"cam_head": "head_camera", "cam_left_wrist": "left_wrist_camera", "cam_right_wrist": "right_wrist_camera"}}
        hashes = functions(source, ("_validated_pose", "_validated_gripper", "_to_pil", "_decode_instruction"), scope)
        model = native_class(source, "Model", ("_state_to_eef20", "_encode_obs", "_conditions", "_eef20_chunk_to_native",
            "update_obs", "update_obs_batch", "get_action", "get_action_batch", "reset"), scope)
        model._poses, model._calibration, model._format_prompt = poses, arx_x5_calibration(), format_prompt_for_inference
        model.default_instruction, model.allow_dummy_policy = "follow the instruction", False
        cfg = _load_deploy_yaml(str(root / "configs/deploy.yaml"))
        for key, value in (("optimization.dit_cache.enabled", False), ("optimization.compile.enabled", False),
                ("optimization.decode_video", False), ("inference.inference_mode", "sync")):
            OmegaConf.update(cfg, key, value, merge=False)
        server = build_server_from_config(cfg, options["checkpoint_dir"], device="cuda")
        server._init_policy()
        model._engine, model._wam_policy, model._preprocessor = server.engine, server._policy, server._obs_preprocessor
        model.replan_steps = OmegaConf.select(server.cfg, "inference.inference_horizon", default=None)
        model.reset()
        write(args.output / "native-model-loaded.json", {"model_loaded": True, "verified_assets": assets,
            "source_file_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(), "source_function_sha256": hashes,
            "production_ModelAdapter_instantiated": False, "allow_dummy_policy": False})
        connection.send({"ok": True})
        requests = 0
        while True:
            method, payload = connection.recv()
            if method == "close":
                connection.send({"ok": True, "requests": requests})
                return 0
            if method != "infer" or not isinstance(payload, dict) or set(payload) != {"vision", "state", "instruction"}:
                raise ValueError("source_public_observation_or_close_required")
            decoded = {"vision": {name: {"color": unpack(camera["color"])} for name, camera in payload["vision"].items()},
                "state": {name: unpack(value) for name, value in payload["state"].items()}, "instruction": payload["instruction"]}
            model.update_obs(decoded)
            encoded = model._batch[0]
            # Persist original public arrays in a lossless NumPy archive, not
            # pickle, authentication, logging metadata, or arbitrary objects.
            np.savez_compressed(args.output / f"request-{requests + 1:04d}.npz",
                state=np.asarray(encoded["state"], np.float32), prompt=np.asarray(encoded["prompt"]),
                **{name: np.asarray(image) for name, image in encoded["images"].items()})
            actions = model.get_action()
            np.savez_compressed(args.output / f"actions-{requests + 1:04d}.npz",
                **{key: np.stack([row[key] for row in actions]) for key in actions[0]})
            requests += 1
            connection.send({"ok": True, "actions": [{key: pack(value) for key, value in row.items()} for row in actions]})
    except Exception as error:
        write(args.output / "native-model-failure.json", {"exception_chain": public_exception(error)})
        try:
            connection.send({"ok": False, "error_type": type(error).__name__})
        except (BrokenPipeError, EOFError, OSError):
            pass
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
