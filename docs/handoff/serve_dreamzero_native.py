"""Owned, fixed-source DreamZero server; no simulator or framework Runner.

Deployment selects loopback, verified local assets/tokenizer and output paths.
Native mesh, loader, causal policy, worker loop and websocket methods execute
unchanged. Parent supervisor owns this process group and keeps logs in memory.
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import resource
import sys
from probe_mn_components import ROOT, WORKSPACE, write, public_exception


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if not 1 <= args.port <= 65535:
        raise ValueError("loopback_port_required")
    import torch
    import torch.distributed as dist
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.native.assets_common import _revision
    from embodied_eval.native.additional_catalog import POLICIES, OPENPI_REVISION
    from prepare_mn_extension import configurations
    cfg = configurations()["dreamzero_droid_sim"].model.options
    repo = Path(cfg["repo"])
    _revision(repo, POLICIES["dreamzero"]["source_revision"])
    _revision(cfg["codec_repo"], OPENPI_REVISION)
    client_modules = {name: native_module(cfg["codec_repo"], "openpi_client." + name,
        "packages/openpi-client/src") for name in ("base_policy", "msgpack_numpy", "image_tools")}
    sys.path.insert(0, str(repo))
    os.environ["ENABLE_DIT_CACHE"] = "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800
    import socket_test_optimized_AR as native
    mesh = native.init_mesh()
    rank = dist.get_rank()
    proof = {"kind": "owned_native_dreamzero_policy_server_not_acceptance", "rank": rank,
        "source_revision": POLICIES["dreamzero"]["source_revision"], "synthetic": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "model_loaded": False,
        "native_sampler_mesh_worker_loop_and_loader_unchanged": True,
        "client_module_sha256": {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in client_modules.items()}, "loopback_only": True,
        "model_forward_count_claimed": False, "raw_logs_saved": False}
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=2400))
        policy = native.GrootSimPolicy(embodiment_tag=native.EmbodimentTag("oxe_droid"),
            model_path=cfg["checkpoint_dir"], device="cuda", device_mesh=mesh,
            tokenizer_path_override=str(args.tokenizer.resolve()))
        if policy.trained_model.action_head.ip_size != 2:
            raise ValueError("native_two_rank_mesh_required")
        proof.update(model_loaded=True, native_inference_parallel_ranks=2,
            eval_bf16=bool(policy.eval_bf16), torch_version=torch.__version__,
            gpu_name=torch.cuda.get_device_name(rank), loaded_at_utc=datetime.now(timezone.utc).isoformat())
        write(args.output / f"rank-{rank}-model-loaded.json", dict(proof))
        if rank == 0:
            wrapper = native.ARDroidRoboarenaPolicy(policy, group, output_dir=str(args.output / "native-video"))
            server_config = native.PolicyServerConfig(image_resolution=(180, 320), needs_wrist_camera=True,
                n_external_cameras=2, needs_stereo_camera=False, needs_session_id=True,
                action_space="joint_position")
            server = native.RoboarenaServer(policy=wrapper, server_config=server_config,
                host="127.0.0.1", port=args.port)
            write(args.output / "server-ready.json", dict(proof, port=args.port,
                ready_receipt_precedes_socket_bind=True))
            server.serve_forever()
        else:
            worker = native.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                output_dir=None, signal_group=group)
            asyncio.run(worker._worker_loop())
    except Exception as error:
        proof["exception_chain"] = public_exception(error)
        proof["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / f"rank-{rank}-server-failure.json", proof)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
