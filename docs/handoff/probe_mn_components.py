"""Supervised component diagnostics; never equate synthetic inputs with a trial.

Upstream logs and arbitrary exception messages stay in memory. Receipts contain
only package/source facts, exception classes, safe traceback locations and
allow-listed failure categories. Only this probe's process group is terminated.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

COMPONENTS = {
    "cosmos3": ("cosmos3-mn-native", "cosmos-framework-mn", "cf5d68c00d97ccd2480a2320ed652b92dec63102"),
    "dreamzero": ("dreamzero-mn-native", "dreamzero-mn", "ab790c198fbce33503358efbbd4187ce9a89adf3"),
    "openwam_arx": ("openwam_arx-mn-native", "xpolicy-reference-mn", "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2"),
    "droid_sim": ("droid_sim-mn-native", "sim-evals-mn", "3a6b0e8177f696947f49c082728f10ef5133ede1"),
}
FAILURE_PATTERNS = {
    "cuda_driver_too_old": ("driver version is insufficient", "driver on your system is too old"),
    "cuda_illegal_address": ("illegal memory access", "cudaerrorillegaladdress"),
    "cuda_out_of_memory": ("cuda out of memory",),
    "cuda_compiler_missing": ("cuda_home", "nvcc not found", "no such file or directory: 'nvcc'"),
    "missing_shared_library": ("cannot open shared object file",),
    "missing_transformer_engine_library": ("could not find libtransformer_engine",),
    "unsupported_gpu_architecture": ("no kernel image is available", "unsupported gpu architecture"),
    "offline_asset_missing": ("localentrynotfounderror", "offline mode", "outgoing traffic has been disabled"),
    "runtime_eula_required": ("eula", "end user license agreement"),
    "torchvision_operator_missing": ("torchvision::nms",),
    "distributed_failure": ("nccl error", "ncclerror", "process group failed", "processgroupnccl error", "rendezvous failed"),
    "flash_attention_missing": ("flash_attn", "flash-attn"),
    "vulkan_device_unavailable": ("failed to create any gpu devices", "failed to create vulkan"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def categories(text):
    lowered = text.lower()
    return {name: any(pattern in lowered for pattern in patterns)
            for name, patterns in FAILURE_PATTERNS.items()}


def public_exception(error):
    chain = []
    seen = set()
    while error is not None and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        frames = []
        for frame in traceback.extract_tb(error.__traceback__):
            path = Path(frame.filename)
            try:
                name = str(path.relative_to(WORKSPACE))
            except ValueError:
                name = path.name
            frames.append({"file": name, "line": frame.lineno, "function": frame.name})
        item = {"error_type": type(error).__name__, "locations": frames[-16:],
                "public_failure_categories": categories(str(error))}
        code = getattr(error, "code", None) or getattr(getattr(error, "issue", None), "code", None)
        if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]+", code):
            item["public_error_code"] = code
        if hasattr(error, "native_failure"):
            item["private_native_failure"] = error.native_failure
        if isinstance(error, ModuleNotFoundError) and re.fullmatch(r"[A-Za-z0-9_.]+", error.name or ""):
            item["missing_module"] = error.name
        if isinstance(error, AttributeError) and re.fullmatch(r"[A-Za-z0-9_]+", error.name or ""):
            item["missing_attribute"] = error.name
        library = re.fullmatch(r"([A-Za-z0-9_]+) shared object not found\.", str(error))
        if library:
            item["missing_cuda_library"] = library.group(1)
        chain.append(item)
        error = error.__cause__ or error.__context__
    return chain


def packages():
    result = {}
    for name in ("torch", "torchvision", "numpy", "websockets", "transformers", "transformer-engine",
                 "transformer-engine-cu12", "megatron-core", "isaacsim", "isaaclab", "flash-attn"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def torch_probe():
    import torch
    result = {"torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()}
    if not result["cuda_available"]:
        raise RuntimeError("CUDA unavailable to component")
    value = torch.arange(32 * 32, device="cuda", dtype=torch.float32).reshape(32, 32)
    actual = (value @ value.T).sum().item()
    torch.cuda.synchronize()
    result.update(cuda_matrix_operation_passed=bool(actual > 0),
                  gpu_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    return result


def native_import(component):
    env_name, repo_name, _ = COMPONENTS[component]
    repo = WORKSPACE / "upstream" / repo_name
    if component == "cosmos3":
        sys.path.insert(0, str(repo))
        from cosmos_framework.inference.common.init import init_script
        init_script()  # native startup order, including source-defined flags
        from cosmos_framework.inference import inference
        return {"module": inference.__name__}
    if component == "dreamzero":
        sys.path.insert(0, str(repo))
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        return {"module": GrootSimPolicy.__module__}
    if component == "openwam_arx":
        sys.path.insert(0, str(repo / "policy/OpenWAM/OpenWAM"))
        from openwam.deploy import server
        return {"module": server.__name__}
    from isaaclab.app import AppLauncher
    return {"module": AppLauncher.__module__, "application_started": False}


def openwam_request(record, input_episode=None):
    import numpy as np
    from embodied_eval import _legacy_contracts as leaf
    from embodied_eval.codec import to_data
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.binding import resolve
    from embodied_eval.mn.registry import default_registry
    from embodied_eval.mn.validation import validate_prediction
    from prepare_mn_extension import configurations

    cfg = configurations()["openwam_arx_robodojo" if input_episode is not None else "openwam_arx_robotwin"]
    registry = default_registry()
    model_reg = registry.model(cfg.model.adapter)
    record["asset_verification"] = model_reg.verify_assets(cfg.model.options)
    cap = model_reg.describe(cfg.model.options)
    space = registry.benchmark(cfg.benchmark.adapter).describe(cfg.benchmark.options)
    profile = resolve(space, cap, cfg.binding)
    context = c.Context(run_id="openwam-arx-component-probe", env_id="synthetic0",
                           episode_id="synthetic0", message_id="obs0")
    sensors, states, digests = {}, {}, {}
    for req in (cap.inputs if input_episode is None else ()):
        if req.kind == "task":
            continue
        source = cfg.binding.inputs[req.key]
        if req.kind == "sensor":
            spec = space.sensors[source]
            image = np.full(spec.tensor.shape, 30 + 40 * len(sensors), dtype=np.uint8)
            tensor = leaf.Tensor(dtype="uint8", shape=image.shape, layout="HWC", data=image.tolist())
            sensors[req.key] = leaf.SensorSample(frame_id=spec.frame_id, captured_at_s=0.,
                validity="valid", provenance="sensor", payload=tensor)
            digests[req.key] = {"shape": list(image.shape), "dtype": "uint8", "sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        else:
            spec = space.state_fields[source]
            value = np.zeros(spec.tensor.shape, dtype=np.float64)
            if spec.semantic == "eef_orientation":
                value[-1] = 1.
            elif spec.semantic == "eef_position":
                value[:] = (.2, 0., .8)
            elif "gripper" in spec.semantic:
                value[:] = 1.
            tensor = leaf.Tensor(dtype="float64", shape=value.shape, layout="vector", data=value.tolist())
            states[req.key] = c.StateSample(captured_at_s=0., payload=tensor,
                provenance="controller_target" if spec.semantic == "gripper_drive_target" else "measurement")
            digests[req.key] = {"shape": list(value.shape), "dtype": "float64", "values": value.tolist()}
    view = c.ModelInput(context=context, observation_id="obs0", sim_time_s=0.,
        task_public=leaf.PublicTask(instruction="stack the bowls"), sensors=sensors, states=states)
    if input_episode is not None:
        from embodied_eval.codec import decode
        episode = input_episode.resolve()
        episode.relative_to(ROOT / "outputs")
        manifest = json.loads((episode.parent.parent / "manifest.json").read_text())
        if manifest["synthetic"] or manifest["config"]["model"]["options"]["io_profile"] != "eef20_arx_absolute":
            raise ValueError("recorded real input from the same named model profile required")
        with (episode / "model_input.public.jsonl").open("rb") as stream:
            first_line = stream.readline()
        view = decode(c.ModelInput, json.loads(first_line))
        context = view.context
        digests = {key: {"shape": list(sample.payload.shape), "dtype": sample.payload.dtype,
            "sha256": hashlib.sha256(np.asarray(sample.payload.data, np.uint8).tobytes()).hexdigest()}
            for key, sample in view.sensors.items()}
        digests.update({key: {"shape": list(sample.payload.shape), "dtype": sample.payload.dtype,
            "values": sample.payload.data, "provenance": sample.provenance} for key, sample in view.states.items()})
        record.update(observation_synthetic=False, recorded_real_public_input=True,
            input_first_line_sha256=hashlib.sha256(first_line).hexdigest(), input_episode=str(episode.relative_to(ROOT)),
            new_environment_episode_performed=False)
    model = model_reg.factory(cfg.model.options, None)
    try:
        model.begin_episode(context, view.task_public, profile, space)
        record["model_loaded"] = True
        record["recorded_input_summary" if input_episode is not None else "synthetic_input_summary"] = digests
        model.observe(view)
        generated = []
        original_generate = model.server.engine.generate_batch
        def capture_generate(*values, **keywords):
            result = original_generate(*values, **keywords)
            array = result["actions"]
            if hasattr(array, "detach"):
                array = array.detach().cpu().numpy()
            generated.append(np.asarray(array).copy())
            return result
        model.server.engine.generate_batch = capture_generate
        from dataclasses import replace
        request = replace(context, message_id="request0")
        prediction = model.predict(request)
        validate_prediction(prediction, request, view, space, cap, profile)
        record.update(real_model_request=True, public_prediction=to_data(prediction),
                      model_diagnostics=model.diagnostics(), closed_loop=False, official_score=None)
        # Independent source oracle: extract only fixed pure conversions. Never
        # instantiate/import the upstream pair model or access an environment.
        from check_mn_extension_native_mappings import functions, native_class, same
        from typing import Any, Mapping
        source = Path(cfg.model.options["repo"]) / "policy/OpenWAM/model.py"
        scope = {"np": np, "Any": Any, "Mapping": Mapping, "_GRIPPER_ATOL": 1e-8, "_EEF20_DIM": 20,
            "_CAMERA_TO_PAYLOAD": {"cam_head": "head_camera", "cam_left_wrist": "left_wrist_camera",
                                   "cam_right_wrist": "right_wrist_camera"}}
        hashes = functions(source, ("_validated_pose", "_validated_gripper", "_to_pil", "_decode_instruction"), scope)
        oracle = native_class(source, "Model", ("_state_to_eef20", "_encode_obs", "_eef20_chunk_to_native"), scope)
        oracle._poses, oracle._calibration = model.poses, model.calibration
        oracle._format_prompt, oracle.default_instruction = model.format_prompt, "unused"
        raw = {"vision": {name: {"color": np.asarray(view.sensors[slot].payload.data, dtype=np.uint8)}
            for slot, name in (("head", "cam_head"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist"))},
            "state": {}, "instruction": view.task_public.instruction}
        for side in ("left", "right"):
            position = np.asarray(view.states[side + "_position"].payload.data, dtype=np.float64)
            q = np.asarray(view.states[side + "_quaternion"].payload.data, dtype=np.float64)
            raw["state"][side + "_ee_pose"] = np.concatenate((position, q[[3, 0, 1, 2]]))
            raw["state"][side + "_ee_joint_state"] = np.asarray(view.states[side + "_grip"].payload.data, dtype=np.float64)
        expected_input, actual_input = oracle._encode_obs(raw), model.native_input(view)
        input_exact = (expected_input["state"] == actual_input["state"] and expected_input["prompt"] == actual_input["prompt"]
            and set(expected_input["images"]) == set(actual_input["images"]))
        input_exact = input_exact and all(same(np.asarray(expected_input["images"][key]), np.asarray(actual_input["images"][key]))
            for key in expected_input["images"])
        projected = model.server._policy._project_binary_dims(generated[0])
        count = len(projected[0]) if model.horizon is None else min(model.horizon, len(projected[0]))
        expected_rows = oracle._eef20_chunk_to_native(projected[0, :count])
        from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator
        executor = RoboDojoSimulator({"repo": "/absent"})  # pure conversion, no reset/application
        predictions = [prediction]
        for index in range(1, count):
            request = replace(context, message_id=f"request{index}")
            item = model.predict(request)
            validate_prediction(item, request, view, space, cap, profile)
            predictions.append(item)
        actual_rows = [executor.native_action(item.action) for item in predictions]
        action_exact = len(expected_rows) == len(actual_rows) and all(same(e, a) for e, a in zip(expected_rows, actual_rows))
        record.update(native_generated_chunk={"shape": list(generated[0].shape), "dtype": str(generated[0].dtype),
            "values": generated[0].tolist()}, consumed_native_chunk_rows=count,
            native_source_comparison={"input_pixels_state_prompt_exact": input_exact,
                "all_consumed_action_values_dtypes_exact": action_exact,
                "source_pure_function_sha256": hashes, "independent_native_reinference": False,
                "simulator_execution_compared": False}, model_diagnostics=model.diagnostics())
        if not input_exact or not action_exact:
            raise RuntimeError("source_pinned_native_conversion_comparison_failed")
    finally:
        model.close()


def droid_reset(record, args):
    from embodied_eval.mn.registry import default_registry
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.validation import validate_observation, validate_step
    from embodied_eval import _legacy_contracts as leaf
    from embodied_eval.codec import to_data
    from dataclasses import replace
    from prepare_mn_extension import configurations
    cfg = configurations()["dreamzero_droid_sim"]
    options = dict(cfg.benchmark.options, device="cuda:0")
    entry = default_registry().benchmark("droid_sim")
    record["asset_verification"] = entry.verify_assets(options)
    # Root execution is allowed only for this isolated application. Suppress
    # automatic Kit file logs; arbitrary upstream logs are not persisted.
    sys.argv += ["--allow-root", "--/log/file="]
    # New isolated benchmark owns the native application in its worker. Do
    # not import Isaac into this parent just to instrument the old lifecycle.
    benchmark = entry.factory(options, None)
    record["simulator_started"] = True
    try:
        context = c.Context(run_id="droid-native-reset-probe", env_id="native0", episode_id="ep-0", message_id="reset0")
        task = benchmark.enumerate_tasks(cfg.episodes)[0]
        observation = benchmark.reset(task, context)
        validate_observation(observation, benchmark.describe())
        record.update(native_reset_passed=True, observation_synthetic=False,
            measured_states={key: sample.payload.data for key, sample in observation.states.items()},
            camera_shapes={key: list(sample.payload.shape) for key, sample in observation.sensors.items()},
            benchmark_diagnostics=benchmark.diagnostics(), official_score=None)
        if args.droid_hold_steps:
            joint_values = observation.states["arm_joint_positions"].payload.data
            grip_value = observation.states["gripper_open_fraction"].payload.data[0]
            feedback = []
            for index in range(args.droid_hold_steps):
                request = replace(context, message_id=f"hold:{index}")
                commands = tuple(leaf.JointTarget(command_id=f"hold:{index}:j{i}", target_id=f"robot0/j{i}",
                    mode="position", unit="rad", value=value) for i, value in enumerate(joint_values))
                commands += (leaf.GripperCommand(command_id=f"hold:{index}:grip", target_id="robot0/grip",
                    mode="open_fraction", value=grip_value),)
                action = c.Action(commands=commands, native_dtype="float64")
                reply = benchmark.advance(action, request, leaf.Budgets(env_steps=args.droid_hold_steps-index))
                validate_step(reply, observation, request, action, benchmark.describe())
                feedback.append(to_data(reply.execution_feedback))
                observation = reply.observation
            record.update(native_hold_execution_passed=True, execution_feedback=feedback,
                          benchmark_diagnostics=benchmark.diagnostics(), policy_inference_performed=False)
        # Native fast shutdown can terminate Python from app.close(). Preserve
        # a validated-reset checkpoint before entering that native lifecycle.
        write(args.output / "validated-native-reset.json", dict(record, passed=True, finished_at_utc=now()))
    finally:
        benchmark.close()


def cosmos_request(record, args):
    import numpy as np
    from dataclasses import replace
    from embodied_eval import _legacy_contracts as leaf
    from embodied_eval.codec import to_data
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.binding import resolve
    from embodied_eval.mn.registry import default_registry
    from embodied_eval.mn.validation import validate_prediction
    from prepare_mn_extension import configurations
    cfg = configurations()["cosmos3_droid_sim"]
    registry = default_registry()
    registration = registry.model("cosmos3")
    record["asset_verification"] = registration.verify_assets(cfg.model.options)
    cap = registration.describe(cfg.model.options)
    space = registry.benchmark("droid_sim").describe(cfg.benchmark.options)
    profile = resolve(space, cap, cfg.binding)
    sys.path.insert(0, cfg.model.options["repo"])
    from cosmos_framework.scripts import action_policy_server_robolab as native
    # Only file logging is suppressed; setup/config/sampler are unchanged.
    native.init_output_dir = lambda directory: Path(directory).mkdir(parents=True, exist_ok=True)
    overrides = []
    if args.cosmos_native_vae:
        preparation = args.cosmos_native_vae.resolve()
        preparation.relative_to(ROOT / "outputs")
        receipt = json.loads((preparation / "receipt.json").read_text())
        manifest = json.loads((preparation / "manifest.json").read_text())
        if not receipt["passed"] or manifest["revision"] != "921dbaf3f1674a56f47e83fb80a34bac8a8f203e":
            raise ValueError("fixed native Cosmos VAE not ready")
        from download_stage4_assets import verify
        item = manifest["objects"][0]
        path = Path(receipt["directory"]) / item["relative"]
        digest = verify(path, item)
        # The upstream documented dot-list override changes only the asset
        # location, not architecture, sampler, state or controller semantics.
        overrides = [f"model.config.tokenizer.vae_path={path}", "model.config.tokenizer.bucket_name="]
        record["native_vae_resolution"] = {"file": str(path), "sha256": digest,
            "source_pinned_revision": manifest["revision"], "source_supported_path_override": True}
    service = native.RobolabPolicyService(native.RobolabServerArgs(
        checkpoint_path=cfg.model.options["checkpoint_dir"], host="127.0.0.1",
        experiment_overrides=overrides, output_dir=args.output / "native-output"))
    record.update(model_loaded=True, native_file_logging_disabled=True,
                  inference_transport="in-process-native-policy-service-not-websocket")
    if args.mode == "cosmos-server":
        record["inference_transport"] = "source-native-openpi-websocket-server"
        write(args.output / "server-ready.json", dict(record, server_ready=True, port=args.port, at_utc=now()))
        # Exactly the serving class selected by the fixed native serve() entry.
        server_class = native._load_openpi_websocket_policy_server()
        try:
            server_class(policy=service, host="127.0.0.1", port=args.port, metadata={}).serve_forever()
        finally:
            service.shutdown_worker()
        return
    context = c.Context(run_id="cosmos3-native-component-probe", env_id="synthetic0",
                        episode_id="synthetic0", message_id="obs0")
    sensors, states = {}, {}
    for req in cap.inputs:
        if req.kind == "sensor":
            spec = space.sensors[cfg.binding.inputs[req.key]]
            value = np.full(spec.tensor.shape, 30 + 40 * len(sensors), dtype=np.uint8)
            sensors[req.key] = leaf.SensorSample(frame_id=spec.frame_id, captured_at_s=0.,
                validity="valid", provenance="sensor", payload=leaf.Tensor(dtype="uint8",
                    shape=value.shape, layout="HWC", data=value.tolist()))
        elif req.kind == "state":
            value = np.zeros(req.tensor.shape, dtype=np.float64)
            if req.key == "grip":
                value[:] = 1.
            states[req.key] = c.StateSample(captured_at_s=0., payload=leaf.Tensor(
                dtype="float64", shape=value.shape, layout="vector", data=value.tolist()))
    view = c.ModelInput(context=context, observation_id="obs0", sim_time_s=0.,
        task_public=leaf.PublicTask(instruction="put the cube in the bowl"), sensors=sensors, states=states)
    if args.cosmos_input_episode:
        from embodied_eval.codec import decode
        episode = args.cosmos_input_episode.resolve()
        episode.relative_to(ROOT / "outputs")
        manifest = json.loads((episode.parent.parent / "manifest.json").read_text())
        if manifest["synthetic"] or manifest["config"]["model"]["adapter"] != "cosmos3":
            raise ValueError("recorded real Cosmos component input required")
        with (episode / "model_input.public.jsonl").open("rb") as stream:
            first_line = stream.readline()
        view = decode(c.ModelInput, json.loads(first_line))
        context = view.context
        record.update(observation_synthetic=False, recorded_real_public_input=True, new_environment_episode_performed=False,
            input_episode=str(episode.relative_to(ROOT)), input_first_line_sha256=hashlib.sha256(first_line).hexdigest())
    generated = []
    class NativePeer:
        def infer(self, value):
            result = service.infer(value)
            generated.append(np.asarray(result["action"]).copy())
            return result
        def close(self):
            pass
    model = registration.factory(cfg.model.options, None)
    model.peer = NativePeer()
    try:
        model.begin_episode(context, view.task_public, profile, space)
        model.observe(view)
        request = replace(context, message_id="request0")
        prediction = model.predict(request)
        validate_prediction(prediction, request, view, space, cap, profile)
        record.update(real_model_request=True, public_prediction=to_data(prediction),
                      model_diagnostics=model.diagnostics(), closed_loop=False, official_score=None)
        # Reuse the fixed native client's pure packing/threshold functions,
        # not another inference. Compare every cached row, not task success.
        import torch
        import torch.nn.functional as functional
        from check_mn_extension_native_mappings import native_class, same
        from embodied_eval.mn.droid_simulator import DroidSimulator
        native_source = Path(cfg.model.options["client_repo"]) / "policies/cosmos3/client.py"
        oracle = native_class(native_source, "Cosmos3Client", ("_pack_request", "_postprocess_chunk"),
            {"np": np, "torch": torch, "F": functional})
        oracle._image_h, oracle._image_w = 360, 640
        expected_input = oracle._pack_request({
            "left_image": model.images.resize_with_pad(np.asarray(view.sensors["exterior_left"].payload.data, np.uint8), 360, 640),
            "right_image": model.images.resize_with_pad(np.asarray(view.sensors["exterior_right"].payload.data, np.uint8), 360, 640),
            "wrist_image": model.images.resize_with_pad(np.asarray(view.sensors["wrist"].payload.data, np.uint8), 360, 640),
            "joint_position": np.asarray(view.states["joints"].payload.data, np.float32),
            "gripper_position": (1. - np.asarray(view.states["grip"].payload.data, np.float64)).astype(np.float32),
            "session_id": model.session_id}, view.task_public.instruction)
        expected_rows = oracle._postprocess_chunk(generated[0][:model.horizon])
        predictions = [prediction]
        for index in range(1, model.horizon):
            request = replace(context, message_id=f"request{index}")
            item = model.predict(request)
            validate_prediction(item, request, view, space, cap, profile)
            predictions.append(item)
        executor = DroidSimulator.__new__(DroidSimulator)
        executor.space = space
        actual_rows = np.stack([executor.native_action(item.action) for item in predictions])
        input_exact, actions_exact = same(expected_input, model.native_input(view)), same(expected_rows, actual_rows)
        record.update(native_generated_chunk={"shape": list(generated[0].shape), "dtype": str(generated[0].dtype), "values": generated[0].tolist()},
            cached_rows_compared=len(predictions), native_source_sha256=hashlib.sha256(native_source.read_bytes()).hexdigest(),
            native_source_comparison={"input_pixels_state_prompt_session_exact": input_exact,
                "all_cached_action_values_dtypes_exact": actions_exact, "independent_native_reinference": False,
                "new_simulator_execution_compared": False}, model_diagnostics=model.diagnostics())
        if not input_exact or not actions_exact or len(generated) != 1:
            raise RuntimeError("native_cosmos_conversion_comparison_failed")
    finally:
        model.close()
        import torch.distributed as distributed
        if distributed.is_initialized():
            distributed.destroy_process_group()


def child(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    record = {"component": args.component, "mode": args.mode, "started_at_utc": now(),
              "passed": False, "installed_versions": packages(), "model_loaded": False,
              "real_model_request": False, "simulator_started": False,
              "observation_synthetic": args.mode in ("openwam-request", "cosmos-request"), "official_score": None}
    try:
        record["gpu_probe"] = torch_probe()
        if args.mode == "import":
            record["native_import"] = native_import(args.component)
        elif args.mode == "openwam-request":
            if args.component != "openwam_arx":
                raise ValueError("OpenWAM-only source-defined request mode")
            openwam_request(record, args.openwam_input_episode)
        elif args.mode == "droid-reset":
            if args.component != "droid_sim" or not args.accept_runtime_eula:
                raise ValueError("DROID-only reset with explicit user EULA approval required")
            droid_reset(record, args)
        elif args.mode in ("cosmos-request", "cosmos-server"):
            if args.component != "cosmos3":
                raise ValueError("Cosmos-only source-defined native policy service")
            cosmos_request(record, args)
        record["passed"] = True
    except Exception as error:
        record["exception_chain"] = public_exception(error)
    record["finished_at_utc"] = now()
    write(args.output / "component-result.json", record)
    print("COMPONENT_PROBE " + json.dumps({k: record.get(k) for k in
        ("component", "mode", "passed", "model_loaded", "real_model_request", "exception_chain")}), flush=True)
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--mode", choices=("import", "openwam-request", "droid-reset", "cosmos-request", "cosmos-server"), default="import")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--accept-runtime-eula", action="store_true",
                        help="Use only after the user explicitly approves NVIDIA Omniverse terms")
    parser.add_argument("--cosmos-local-cuda-libraries", action="store_true",
                        help="Point native CUDA library discovery only at this component's installed wheel libraries")
    parser.add_argument("--isaac-local-libraries", type=Path,
                        help="Private verified GLU/OpenGL preparation directory, not a system library change")
    parser.add_argument("--cosmos-native-vae", type=Path,
                        help="Checksum-verified source-pinned native auxiliary VAE receipt directory")
    parser.add_argument("--droid-hold-steps", type=int, default=0,
                        help="Bounded real simulator holding commands from measured reset state, not model evaluation")
    parser.add_argument("--port", type=int, default=8940)
    parser.add_argument("--openwam-input-episode", type=Path,
                        help="Saved real ARX public input for native conversion comparison, not a new scene")
    parser.add_argument("--cosmos-input-episode", type=Path,
                        help="Saved real Cosmos public input for full native chunk conversion comparison")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.child:
        return child(args)
    if args.gpu < 0 or args.timeout_s < 1:
        raise ValueError("positive timeout and nonnegative GPU required")
    if not 0 <= args.droid_hold_steps <= 8 or (args.droid_hold_steps and args.mode != "droid-reset"):
        raise ValueError("DROID-only bounded hold test allows zero to eight actions")
    args.output.mkdir(parents=True, exist_ok=False)
    environment_name, repo_name, revision = COMPONENTS[args.component]
    from embodied_eval.mn.assets_common import _revision
    _revision(WORKSPACE / "upstream" / repo_name, revision)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    if args.accept_runtime_eula:
        if args.component != "droid_sim":
            raise ValueError("reviewed runtime EULA applies only to DROID probe")
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if args.cosmos_local_cuda_libraries:
        if args.component != "cosmos3":
            raise ValueError("source-defined local CUDA library discovery is Cosmos-only")
        libraries = WORKSPACE / "upstream/envs/cosmos3-mn-native/lib/python3.13/site-packages/nvidia"
        environment["CUDA_HOME"] = str(libraries)
        environment["NVTE_CUDA_INCLUDE_DIR"] = str(libraries / "cuda_runtime/include")
    if args.isaac_local_libraries:
        if args.component != "droid_sim":
            raise ValueError("reviewed native Kit library path is DROID-only")
        preparation = args.isaac_local_libraries.resolve()
        preparation.relative_to(ROOT / "outputs")
        lib_receipt = json.loads((preparation / "receipt.json").read_text())
        if not lib_receipt["passed"]:
            raise ValueError("local shared-library preparation incomplete")
        for item in lib_receipt["libraries"]:
            path = preparation / item["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("private shared library checksum changed")
        directory = preparation / "library-root/usr/lib/x86_64-linux-gnu"
        environment["LD_LIBRARY_PATH"] = str(directory) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    if args.cosmos_native_vae and (args.component != "cosmos3" or args.mode not in ("cosmos-request", "cosmos-server")):
        raise ValueError("native auxiliary VAE path applies only to Cosmos inference")
    if args.openwam_input_episode and (args.component != "openwam_arx" or args.mode != "openwam-request"):
        raise ValueError("saved ARX public input applies only to OpenWAM native request diagnostic")
    if args.cosmos_input_episode and (args.component != "cosmos3" or args.mode != "cosmos-request"):
        raise ValueError("saved Cosmos public input applies only to Cosmos native request diagnostic")
    # These diagnostics never fetch assets; no proxy/global state is changed.
    command = [str(WORKSPACE / "upstream/envs" / environment_name / "bin/python"), __file__,
               "--component", args.component, "--mode", args.mode, "--output", str(args.output), "--child"]
    if args.accept_runtime_eula:
        command.append("--accept-runtime-eula")
    if args.cosmos_native_vae:
        command.extend(["--cosmos-native-vae", str(args.cosmos_native_vae.resolve())])
    command.extend(["--droid-hold-steps", str(args.droid_hold_steps)])
    if args.openwam_input_episode:
        command.extend(["--openwam-input-episode", str(args.openwam_input_episode.resolve())])
    if args.cosmos_input_episode:
        command.extend(["--cosmos-input-episode", str(args.cosmos_input_episode.resolve())])
    record = {"kind": "supervised_native_component_probe_not_benchmark_acceptance", "component": args.component,
              "mode": args.mode, "source_revision": revision, "gpu": args.gpu, "started_at_utc": now(),
              "passed": False, "timeout_s": args.timeout_s, "global_proxy_modified": False,
              "shared_driver_modified": False, "historical_environments_modified": False,
              "raw_logs_persisted": False, "stage4_accepted": False}
    record["runtime_eula_accepted_for_this_process"] = args.accept_runtime_eula
    record["local_cuda_library_discovery"] = args.cosmos_local_cuda_libraries
    record["private_isaac_library_preparation"] = str(args.isaac_local_libraries) if args.isaac_local_libraries else None
    record["native_vae_preparation"] = str(args.cosmos_native_vae) if args.cosmos_native_vae else None
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=args.timeout_s)
        record["exit_code"] = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        record.update(exit_code=process.returncode, failure_class="bounded_component_timeout")
    result_path = args.output / "component-result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        record["passed"] = record["exit_code"] == 0 and result["passed"]
        record["component_result"] = str(result_path.relative_to(ROOT))
        record["model_loaded"] = result["model_loaded"]
        record["real_model_request"] = result["real_model_request"]
    elif args.mode == "droid-reset":
        record["application_created"] = (args.output / "application-created.json").is_file()
        interruption = args.output / "before-native-close.json"
        if interruption.is_file():
            record["native_close_diagnostic"] = str(interruption.relative_to(ROOT))
        validated = args.output / "validated-native-reset.json"
        if validated.is_file():
            reset = json.loads(validated.read_text())
            record.update(native_reset_validated_before_close=reset["native_reset_passed"],
                          reset_result=str(validated.relative_to(ROOT)))
            record["passed"] = record["exit_code"] == 0 and reset["passed"]
    # Do not classify our own JSON keys as upstream error messages (e.g. an
    # explicitly false runtime_eula_required key contains the word EULA).
    upstream_stdout = b"\n".join(line for line in stdout.splitlines()
                                   if not line.startswith(b"COMPONENT_PROBE "))
    record["public_log_categories"] = categories((upstream_stdout + stderr).decode("utf-8", errors="replace"))
    public_logs = (upstream_stdout + stderr).decode("utf-8", errors="replace")
    record["public_library_names"] = sorted(set(re.findall(r"\b(lib[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*)\b", public_logs)))
    record["finished_at_utc"] = now()
    _revision(WORKSPACE / "upstream" / repo_name, revision)
    write(args.output / "receipt.json", record)
    print(json.dumps(record), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
