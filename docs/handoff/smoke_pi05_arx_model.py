"""Actual native GPU inference with a SYNTHETIC observation, not a task score.

Loads checksum-verified ARX weights through the pinned native policy factory.
The independent Adapter handles public input/action/cache; no simulator runs.
Use an external, task-owned bounded diagnostic process, not Runner hard timeout.
"""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    receipt = {"kind": "real_gpu_model_synthetic_observation_smoke_not_benchmark_evaluation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False, "model_loaded": False,
        "synthetic_observation": True, "synthetic_model": False, "simulator_started": False,
        "official_score": None, "new_real_benchmark_trials": 0, "model_forward_count": None,
        "transport": "inprocess_native_policy_not_websocket_identity_attestation",
        "dependency_deviation": "native-supported legacy LeRobot 0cf8648; rerun-sdk 0.21.0; optax/chex native lock versions",
        "stage4_formally_accepted": False, "contract_frozen": False}
    phase = "imports"
    progress = args.output.with_suffix(".progress.json")
    if progress.exists():
        raise ValueError("new progress output required")
    def checkpoint():
        progress.write_text(json.dumps({"kind": "native_model_smoke_progress_not_result", "phase": phase,
            "model_loaded_at_last_checkpoint": receipt["model_loaded"], "at_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    checkpoint()
    try:
        import numpy as np
        from embodied_eval import _legacy_contracts as leaf
        from embodied_eval.mn import contracts as c
        from embodied_eval.mn.binding import resolve
        from embodied_eval.mn.native_models import NativeActionModel
        from embodied_eval.mn.validation import validate_prediction
        from embodied_eval.mn.assets import verify_model
        from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator
        from prepare_mn_extension import configurations
        cfg = configurations()["pi05_arx_robodojo"]
        phase = "assets"
        checkpoint()
        receipt["verified_model_assets"] = verify_model("openpi", cfg.model.options)
        receipt["installed_versions"] = {name: metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "chex", "orbax-checkpoint", "torch", "numpy", "lerobot")}
        native = WORKSPACE / "upstream/xpolicy-reference-mn/policy/Pi_05/openpi"
        sys.path[:0] = [str(native / "src"), str(native / "packages/openpi-client/src")]
        from openpi.training import config
        from openpi.policies import policy_config
        import jax
        devices = jax.devices()
        if not devices or any(d.platform != "gpu" for d in devices):
            raise RuntimeError("native_jax_gpu_required")
        receipt["jax_devices"] = [str(d) for d in devices]
        phase = "load_native_weights_and_normalization"
        checkpoint()
        selected = config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
        receipt["native_config"] = selected.name
        started = monotonic()
        policy = policy_config.create_trained_policy(selected, cfg.model.options["checkpoint_dir"])
        receipt["model_loaded"] = True
        checkpoint()
        receipt["load_elapsed_s"] = monotonic() - started
        model = NativeActionModel("openpi", cfg.model.options)
        benchmark = RoboDojoSimulator(cfg.benchmark.options)  # metadata only, no _load/reset
        profile = resolve(benchmark.space, model.cap, cfg.binding)
        ctx = c.Context(run_id="pi05-arx-gpu-smoke", env_id="synthetic", episode_id="ep-0", message_id="request")
        model.peer = SimpleNamespace(infer=policy.infer)
        public = leaf.PublicTask(instruction="stack the bowls")
        model.begin_episode(ctx, public, profile, benchmark.space)
        sensors = {slot: leaf.SensorSample(frame_id="world", captured_at_s=0., validity="valid", provenance="sensor",
            payload=leaf.Tensor(dtype="uint8", shape=(224, 224, 3), layout="HWC", data=np.zeros((224, 224, 3), np.uint8).tolist()))
            for slot in ("head", "left_wrist", "right_wrist")}
        states = {}
        for side in ("left", "right"):
            for suffix, data, commanded in (("_joints", [0.] * 6, False), ("_grip", [.5], True)):
                key = side + suffix
                states[key] = c.StateSample(captured_at_s=0., provenance="controller_target" if commanded else "measurement",
                    payload=leaf.Tensor(dtype="float64", shape=(len(data),), layout="vector", data=data))
        view = c.ModelInput(context=ctx, observation_id="synthetic-obs", sim_time_s=0., task_public=public,
            sensors=sensors, states=states, state_specs={slot: benchmark.space.state_fields[native_field]
                for slot, native_field in cfg.binding.inputs.items() if native_field in benchmark.space.state_fields})
        phase = "native_inference_and_adapter_decode"
        checkpoint()
        model.observe(view)
        started = monotonic()
        actions = []
        for index in range(50):
            from dataclasses import replace
            current = replace(ctx, message_id="request-" + str(index))
            prediction = model.predict(current)
            validate_prediction(prediction, current, model.latest, benchmark.space, model.cap, profile)
            actions.append({key: value.tolist() for key, value in benchmark.native_action(prediction.action).items()})
        receipt.update(passed=True, native_chunk_rows=len(actions), action_dimension=14,
            adapter_requests=model.requests, adapter_calls=model.calls, cache_remaining=len(model.cache),
            inference_and_50_mapping_elapsed_s=monotonic() - started, decoded_native_actions=actions)
    except Exception as error:
        receipt.update(error_type=type(error).__name__, missing_module=error.name if isinstance(error, ModuleNotFoundError) else None)
        # Persist only a known invariant code, never arbitrary native errors.
        if str(error) == "native_jax_gpu_required":
            receipt["error_code"] = "native_jax_gpu_required"
    finally:
        receipt["finished_phase"] = phase
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({key: receipt.get(key) for key in ("passed", "finished_phase", "model_loaded", "error_type", "missing_module", "native_chunk_rows", "adapter_requests")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
