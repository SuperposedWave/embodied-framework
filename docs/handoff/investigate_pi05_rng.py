"""Controlled native pi05 sampling diagnostic, never official evaluation.

Three requests use one recorded real input: first default RNG, same restored RNG,
then next RNG. This explicitly controlled experiment does not prove a historical
first-action difference had that cause. No persistent sampler setting changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
from importlib import metadata
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
from probe_mn_components import write, public_exception, categories
from prepare_robodojo_environment import bounded


def child(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = {"kind": "controlled_native_sampling_diagnostic_not_scoring_or_historical_cause_proof",
        "passed": False, "model_loaded": False, "actual_native_requests": 0,
        "same_recorded_real_input": True, "simulator_started": False,
        "historical_difference_cause_proven": False, "official_sampler_settings_changed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        import numpy as np
        import jax
        from embodied_eval.codec import decode
        from embodied_eval.mn import contracts as c
        from embodied_eval.mn.native_models import NativeActionModel
        from embodied_eval.mn.native_catalog import capability
        from embodied_eval.mn.assets import verify_model
        from embodied_eval.mn.transforms import prepare_input
        from embodied_eval.adapters.native_common import native_module
        manifest = json.loads((args.framework / "manifest.json").read_text())
        options = manifest["config"]["model"]["options"]
        if manifest["synthetic"] or options["io_profile"] != "state8_osc":
            raise ValueError("real pi05 state8 profile required")
        verified = verify_model("openpi", options)
        sys.path.insert(0, str(Path(options["repo"]) / "src"))
        sys.path.insert(0, str(Path(options["repo"]) / "packages/openpi-client/src"))
        from openpi.policies import policy_config
        from openpi.training import config
        policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), options["checkpoint_dir"])
        if policy._is_pytorch_model:
            raise ValueError("reviewed diagnostic controls native JAX RNG only")
        result["model_loaded"] = True
        space = decode(c.SpaceSpec, manifest["space"])
        profile = decode(c.BoundProfile, manifest["profile"])
        model = NativeActionModel("openpi", options)
        model.profile, model.space = profile, space
        model.images = native_module(options["repo"], "openpi_client.image_tools", "packages/openpi-client/src")
        episode = args.framework / "episodes" / manifest["tasks"][0]["episode_id"]
        with (episode / "model_input.public.jsonl").open() as stream:
            view = decode(c.ModelInput, json.loads(stream.readline()))
        wire = model.native_input(prepare_input(view, model.cap, profile, space))
        initial = policy._rng
        arrays, keys = [], []
        for index in range(3):
            if index == 1:
                # Diagnostic intervention ONLY; keep production policy untouched.
                policy._rng = initial
            keys.append(np.asarray(jax.random.key_data(policy._rng)).tolist())
            arrays.append(np.asarray(policy.infer(wire)["actions"]).copy())
            result["actual_native_requests"] += 1
        def compare(left, right):
            return {"exact_equal": left.dtype == right.dtype and bool(np.array_equal(left, right)),
                "maximum_absolute_difference": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))),
                "first_action_maximum_absolute_difference": float(np.max(np.abs(left[0].astype(np.float64) - right[0].astype(np.float64))))}
        result.update(passed=True, rng_keys_before_requests=keys,
            same_rng_repeated=compare(arrays[0], arrays[1]), next_rng=compare(arrays[1], arrays[2]),
            native_chunk_shape=list(arrays[0].shape), native_chunk_dtype=str(arrays[0].dtype),
            native_chunk_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
            first_native_actions=[a[0].tolist() for a in arrays],
            native_input_fingerprint={key: {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()} if isinstance(value, np.ndarray)
                else {"text_sha256": hashlib.sha256(value.encode()).hexdigest()} for key, value in wire.items()},
            native_package_versions={name: metadata.version(name) for name in ("jax", "jaxlib", "flax", "numpy", "orbax-checkpoint")},
            jax_devices=[str(device) for device in jax.devices()],
            source_sha256=hashlib.sha256((Path(options["repo"]) / "src/openpi/policies/policy.py").read_bytes()).hexdigest(),
            recorded_real_framework=str(args.framework.relative_to(ROOT)), assets_verified=True)
        del policy
    except Exception as error:
        result["exception_chain"] = public_exception(error)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "model-result.json", result)
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--dependency-overlay", type=Path)
    args = parser.parse_args()
    args.output, args.framework = args.output.resolve(), args.framework.resolve()
    args.framework.relative_to(ROOT / "outputs")
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1", XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    if args.dependency_overlay:
        overlay_path = args.dependency_overlay.resolve()
        overlay_path.relative_to(ROOT / "outputs")
        overlay = json.loads((overlay_path / "receipt.json").read_text())
        if not overlay["passed"] or overlay["native_environment_modified"]:
            raise ValueError("verified non-mutating diagnostic overlay required")
        env["PYTHONPATH"] = overlay["path"]
    command = [str(WORKSPACE / "upstream/envs/openpi-stage4/bin/python"), __file__, "--child",
        "--framework", str(args.framework), "--output", str(args.output)]
    code, stdout, stderr = bounded(command, env, 900)
    path = args.output / "model-result.json"
    completed = path.is_file() and json.loads(path.read_text())["passed"]
    receipt = {"kind": "supervised_pi05_rng_investigation_not_formal_acceptance", "passed": code == 0 and completed,
        "exit_code": code, "gpu": 2, "public_log_categories": categories((stdout + stderr).decode(errors="replace")),
        "raw_logs_saved": False, "historical_records_changed": False, "production_sampler_modified": False,
        "occupancy_process_changed": False, "global_proxy_modified": False, "shared_driver_modified": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
