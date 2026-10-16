"""Two-GPU, source-native DreamZero request with explicit evidence boundaries.

Uses a saved REAL DROID reset observation, not a new simulator episode. Model
loading, rank communication and sampling remain native. Logs stay in memory.
This supervisor bounds its owned diagnostic processes, not Runner executions.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
from probe_mn_components import write, public_exception, categories


def first_reset(episode):
    with (episode / "trace.internal.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "reset":
                return event["data"]
    raise ValueError("recorded real reset observation required")


def verify_preparation(path):
    from download_stage4_assets import verify
    path = path.resolve()
    path.relative_to(ROOT / "outputs")
    receipt = json.loads((path / "receipt.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    revision = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
    if not receipt["passed"] or receipt["revision"] != revision or manifest["revision"] != revision:
        raise ValueError("complete fixed native backbone preparation required")
    directory = Path(receipt["directory"]).resolve()
    expected = WORKSPACE.parent.parent / "Model/Wan2.1-I2V-14B-480P-DreamZero-runtime"
    if directory != expected or receipt["selected_files"] != 15:
        raise ValueError("explicit native asset directory and selected file set required")
    for item in manifest["objects"]:
        verify(directory / item["relative"], item)
    cache = Path(receipt["private_offline_cache"]).resolve()
    cache.relative_to(path)
    if (cache / "models--Wan-AI--Wan2.1-I2V-14B-480P/refs/main").read_text() != revision:
        raise ValueError("private offline main alias mismatch")
    snapshot = cache / "models--Wan-AI--Wan2.1-I2V-14B-480P/snapshots" / revision
    for item in manifest["objects"]:
        if (snapshot / item["relative"]).resolve() != (directory / item["relative"]).resolve():
            raise ValueError("offline native cache points outside the verified asset set")
    return directory, cache


def child(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch
    import torch.distributed as dist
    from embodied_eval.mn.assets_common import _revision
    from embodied_eval.mn.additional_catalog import POLICIES
    from prepare_mn_extension import configurations
    from embodied_eval.adapters.native_common import native_module
    cfg = configurations()["dreamzero_droid_sim"]
    # Resolve the configured client BEFORE native imports the same package.
    # Keep stale-import rejection; do not evict modules to hide a conflict.
    # Production server/client isolation does not have this combined-probe
    # import ordering issue. No native loader or sampling order changes.
    codec_modules = {name: native_module(cfg.model.options["codec_repo"],
        "openpi_client." + name, "packages/openpi-client/src")
        for name in ("base_policy", "msgpack_numpy", "image_tools")}
    repo = WORKSPACE / "upstream/dreamzero-mn"
    _revision(repo, POLICIES["dreamzero"]["source_revision"])
    sys.path.insert(0, str(repo))
    # Same switches and compilation limit as the pinned native entrypoint.
    os.environ["ENABLE_DIT_CACHE"] = "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800
    import socket_test_optimized_AR as native
    mesh = native.init_mesh()
    rank = dist.get_rank()
    record = {"kind": "real_dreamzero_request_on_recorded_real_observation_not_new_episode",
        "rank": rank, "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
        "model_loaded": False, "real_model_request": False, "synthetic_input": False,
        "source_revision": POLICIES["dreamzero"]["source_revision"], "native_sampler_modified": False,
        "native_loader_order_modified": False, "simulator_started": False, "official_score": None,
        "configured_client_source_resolved_before_native_server_import": True,
        "allocator_diagnostic": args.allocator, "global_allocator_configuration_modified": False,
        "native_precision_tiling_image_resolution_and_compile_modes_unchanged": True,
        "client_module_sha256": {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in codec_modules.items()}, "stale_import_guard_disabled": False}
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=args.timeout_s))
        policy = native.GrootSimPolicy(embodiment_tag=native.EmbodimentTag("oxe_droid"),
            model_path=cfg.model.options["checkpoint_dir"], device="cuda", device_mesh=mesh,
            tokenizer_path_override=str(args.tokenizer))
        if policy.trained_model.action_head.ip_size != 2:
            raise ValueError("native two-rank inference parallelization unavailable")
        record.update(model_loaded=True, native_inference_parallel_ranks=2,
            eval_bf16=bool(policy.eval_bf16), torch_version=torch.__version__,
            gpu_name=torch.cuda.get_device_name(rank))
        def memory_snapshot():
            free, total = torch.cuda.mem_get_info()
            statistics = torch.cuda.memory_stats()
            return {"device_total_bytes": total, "device_free_bytes": free,
                "torch_allocated_bytes": torch.cuda.memory_allocated(),
                "torch_reserved_bytes": torch.cuda.memory_reserved(),
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "inactive_split_bytes": statistics.get("inactive_split_bytes.all.current"),
                "allocator_backend": torch.cuda.memory.get_allocator_backend()}
        record["memory_after_native_loading"] = memory_snapshot()
        record["model_loading_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.output / f"rank-{rank}-model-loaded.json", dict(record))
        wrapper = native.ARDroidRoboarenaPolicy(policy, group, output_dir=None)
        if rank:
            # Native server worker has this exact broadcast/forward/barrier order.
            server = native.WebsocketPolicyServer(policy, host="127.0.0.1", port=1, signal_group=group)
            request = torch.zeros(1, dtype=torch.int32, device="cpu")
            dist.broadcast(request, src=0, group=group)
            if request.item() != 0:
                raise ValueError("native request signal mismatch")
            batch = server._receive_batch_from_rank0()
            dist.barrier()
            with torch.no_grad():
                policy.lazy_joint_forward_causal(batch)
            dist.barrier()
            record.update(real_model_request=True, passed=True)
        else:
            from embodied_eval import _legacy_contracts as leaf
            from embodied_eval.codec import decode
            from embodied_eval.mn import contracts as c
            from embodied_eval.mn.additional_models import JointPolicyModel
            from embodied_eval.mn.binding import resolve
            from embodied_eval.mn.droid_simulator import DroidSimulator, space as droid_space
            obs = decode(c.Observation, first_reset(args.input_episode))
            model = JointPolicyModel("dreamzero", cfg.model.options)
            space = droid_space(cfg.benchmark.options)
            profile = resolve(space, model.cap, cfg.binding)
            model.profile, model.space, model.session_id = profile, space, "fixed-dreamzero-runtime-diagnostic"
            model.images = codec_modules["image_tools"]
            view = c.ModelInput(context=obs.context, observation_id=obs.observation_id, sim_time_s=obs.sim_time_s,
                task_public=leaf.PublicTask(instruction=obs.public_instruction or "Pick up the blue mug and put it on the plate."),
                sensors={r.key: obs.sensors[profile.binding.inputs[r.key]] for r in model.cap.inputs if r.kind == "sensor"},
                states={r.key: obs.states[profile.binding.inputs[r.key]] for r in model.cap.inputs if r.kind == "state"},
                frame_transforms=obs.frame_transforms)
            from embodied_eval.mn.transforms import prepare_input
            prepared = prepare_input(view, model.cap, profile, space)
            wire = model.native_input(prepared)
            # Native wrapper consumes exactly this wire payload once; no second
            # inference is represented as a deterministic comparison.
            rows = wrapper.infer(wire)
            import numpy as np
            if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) < model.horizon or not np.isfinite(rows).all():
                raise ValueError("invalid native action chunk")
            executor = DroidSimulator.__new__(DroidSimulator)
            executor.space = space
            decoded = np.stack([executor.native_action(model.decode_action(row, obs.context)) for row in rows[:model.horizon]])
            # Replay this generated chunk through unchanged native client methods,
            # not a second implementation of gripper thresholds or concatenation.
            from check_mn_extension_native_mappings import native_class, same
            source = repo / "eval_utils/run_sim_eval.py"
            oracle = native_class(source, "DreamZeroJointPosClient", ("infer", "_extract_observation"),
                {"np": np, "image_tools": model.images})
            oracle.actions_from_chunk_completed, oracle.open_loop_horizon = 0, model.horizon
            oracle.session_id = model.session_id
            sent = []
            def request(payload):
                payload["endpoint"] = "infer"
                sent.append(payload)
                return {"actions": rows}
            oracle.client = SimpleNamespace(infer=request)
            raw = {"policy": {name: torch.from_numpy(np.asarray(obs.sensors[name].payload.data, np.uint8))[None]
                for name in ("external_cam", "external_cam_2", "wrist_cam")}}
            raw["policy"].update(arm_joint_pos=torch.tensor(obs.states["arm_joint_positions"].payload.data, dtype=torch.float64),
                gripper_pos=1. - torch.tensor(obs.states["gripper_open_fraction"].payload.data, dtype=torch.float64))
            with redirect_stdout(io.StringIO()):
                expected = np.stack([oracle.infer(raw, prepared.task_public.instruction)["action"] for _ in range(model.horizon)])
            input_exact = same(sent[0], wire)
            exact = decoded.dtype == expected.dtype and np.array_equal(decoded, expected)
            record.update(real_model_request=True, native_chunk_shape=list(rows.shape), native_dtype=str(rows.dtype),
                native_chunk_values=rows.tolist(), executed_mapping_rows=model.horizon,
                decode_values_and_dtypes_exact=bool(exact), wire_state_dtype=str(wire["observation/joint_position"].dtype),
                native_client_input_pixels_values_dtypes_exact=input_exact,
                native_client_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                recorded_input_episode=str(args.input_episode.relative_to(ROOT)),
                passed=bool(exact and input_exact), independent_native_reinference_compared=False)
            if not exact or not input_exact:
                raise ValueError("native action decoding mismatch")
        torch.cuda.synchronize()
    except Exception as error:
        record["exception_chain"] = public_exception(error)
        if isinstance(error, torch.OutOfMemoryError):
            # Extract only numeric memory facts, never save arbitrary messages.
            match = re.search(r"Tried to allocate ([0-9.]+) (KiB|MiB|GiB)", str(error))
            record["cuda_out_of_memory_confirmed"] = True
            record["oom_requested_allocation_bytes"] = (int(float(match.group(1)) * {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[match.group(2)])
                if match else None)
            try:
                record["memory_at_failure"] = memory_snapshot()
            except Exception as snapshot_error:
                record["memory_snapshot_exception_type"] = type(snapshot_error).__name__
    record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / f"rank-{rank}-result.json", record)
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--input-episode", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--allocator", choices=("native-default", "max-split-128", "expandable-native"), default="native-default",
        help="Process-only fragmentation diagnostic; precision, graph capture and sampler unchanged")
    parser.add_argument("--cuda-toolchain", type=Path,
        help="Verified project-local CUDA12.9 compiler, no driver or global PATH changes")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.input_episode = args.input_episode.resolve()
    args.input_episode.relative_to(ROOT / "outputs")
    if args.child:
        return child(args)
    parts = args.gpus.split(",")
    if len(parts) != 2 or any(not part.isdigit() or not 0 <= int(part) <= 7 for part in parts) or parts[0] == parts[1] or args.timeout_s <= 0:
        raise ValueError("two_distinct_local_GPUs_and_positive_diagnostic_supervisor_bound_required")
    selected_gpus = [int(part) for part in parts]
    if len(set(selected_gpus)) != 2:
        raise ValueError("two_distinct_local_GPUs_required")
    args.output.mkdir(parents=True, exist_ok=False)
    directory, cache = verify_preparation(args.preparation)
    from prepare_mn_extension import configurations
    from embodied_eval.mn.additional_catalog import verify_policy
    verified_model = verify_policy("dreamzero", configurations()["dreamzero_droid_sim"].model.options)
    write(args.output / "verified-model-assets.json", verified_model)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, HF_HUB_CACHE=str(cache),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    if args.allocator == "max-split-128":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,max_split_size_mb:128"
    elif args.allocator == "expandable-native":
        # PyTorch 2.8 documents this experimental allocation mode. This last
        # fragmentation probe does not change native precision, tiling, graph
        # capture, tensors, rank count or sampling. It is not a runtime pass.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native,expandable_segments:True"
    toolchain_receipt = None
    if args.cuda_toolchain is not None:
        from prepare_cuda129_toolchain import private_toolchain
        toolkit = private_toolchain(args.cuda_toolchain)
        env.update(CUDA_HOME=str(toolkit), PATH=str(toolkit / "bin") + os.pathsep + env.get("PATH", ""))
        toolchain_receipt = str(args.cuda_toolchain.resolve().relative_to(ROOT) / "receipt.json")
    command = [str(WORKSPACE / "upstream/envs/dreamzero-mn-native/bin/python"), "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=2", __file__, "--child", "--output", str(args.output),
        "--preparation", str(args.preparation.resolve()), "--input-episode", str(args.input_episode),
        "--tokenizer", str(directory / "google/umt5-xxl"), "--timeout-s", str(args.timeout_s), "--allocator", args.allocator]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
    ranks = [json.loads(p.read_text()) for p in sorted(args.output.glob("rank-*-result.json"))]
    receipt = {"kind": "supervised_two_gpu_native_dreamzero_request_not_closed_loop_acceptance",
        "passed": not timed_out and process.returncode == 0 and len(ranks) == 2 and all(r["passed"] for r in ranks),
        "exit_code": process.returncode, "supervisor_timed_out": timed_out,
        "public_log_categories": categories((stdout + stderr).decode(errors="replace")),
        "raw_logs_saved": False, "gpus": selected_gpus, "ranks_completed": len(ranks),
        "allocator_diagnostic": args.allocator, "global_allocator_configuration_modified": False,
        "rank_failure_categories": {key: any(item.get("public_failure_categories", {}).get(key, False)
            for rank in ranks for item in rank.get("exception_chain", [])) for key in categories("")},
        "private_cuda_toolchain_receipt": toolchain_receipt,
        "global_proxy_modified": False, "shared_driver_modified": False, "formal_acceptance": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
