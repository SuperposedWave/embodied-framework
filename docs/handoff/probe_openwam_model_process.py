"""Real ARX inference through isolated model runtime, with synthetic inputs.

Tests the registered independent Model Adapter, not task success or a scene.
The supervising helper alone has a diagnostic deadline; Runner is unchanged.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import sys
from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from prepare_robodojo_environment import bounded
sys.path.insert(0, str(ROOT))


def child(args):
    import numpy as np
    from embodied_eval import _legacy_contracts as leaf
    from embodied_eval.codec import to_data
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.registry import default_registry
    from embodied_eval.mn.binding import resolve
    from embodied_eval.mn.validation import validate_prediction
    from prepare_mn_extension import configurations
    from embodied_eval.mn.robodojo_simulator import RoboDojoSimulator
    record = {"kind": "real_model_isolated_runtime_synthetic_inputs_not_benchmark", "passed": False,
        "synthetic_observations": True, "closed_loop": False, "official_score": None}
    model = None
    try:
        cfg = configurations()["openwam_arx_robodojo"]
        options = dict(cfg.model.options, runtime_python=str(WORKSPACE / "upstream/envs/openwam_arx-mn-native/bin/python"), runtime_gpu=4)
        reg = default_registry().model(cfg.model.adapter)
        record["asset_verification"] = reg.verify_assets(options)
        cap = reg.describe(options)
        executor = RoboDojoSimulator(cfg.benchmark.options)  # No reset/Isaac.
        space = executor.describe()
        profile = resolve(space, cap, cfg.binding)
        context = c.Context(run_id="model-runtime-isolation-diagnostic", env_id="synthetic", episode_id="synthetic", message_id="obs0")
        sensors, states = {}, {}
        for req in cap.inputs:
            if req.kind == "sensor":
                array = np.full((480, 640, 3), 30 + 40 * len(sensors), np.uint8)
                sensors[req.key] = leaf.SensorSample(frame_id=cfg.binding.inputs[req.key], captured_at_s=0.,
                    validity="valid", provenance="sensor", payload=leaf.Tensor(dtype="uint8", shape=array.shape, layout="HWC", data=array.tolist()))
            elif req.kind == "state":
                array = np.zeros(req.tensor.shape, np.float64)
                if req.semantic == "eef_orientation":
                    array[-1] = 1.
                elif req.semantic == "eef_position":
                    array[:] = (.2, 0., .8)
                elif "gripper" in req.semantic:
                    array[:] = 1.
                states[req.key] = c.StateSample(captured_at_s=0.,
                    provenance="controller_target" if req.semantic == "gripper_drive_target" else "measurement",
                    payload=leaf.Tensor(dtype="float64", shape=array.shape, layout="vector", data=array.tolist()))
        view = c.ModelInput(context=context, observation_id="obs0", sim_time_s=0.,
            task_public=leaf.PublicTask(instruction="stack the bowls"), sensors=sensors, states=states)
        model = reg.factory(options, None)
        model.begin_episode(context, view.task_public, profile, space)
        record["model_loaded"] = True
        model.observe(view)
        predictions = []
        for index in range(32):
            request = replace(context, message_id=f"prediction:{index}")
            prediction = model.predict(request)
            validate_prediction(prediction, request, view, space, cap, profile)
            native = executor.native_action(prediction.action)
            if any(array.dtype != np.float32 for array in native.values()):
                raise ValueError("native dtype lost through model-local pipe")
            predictions.append(to_data(prediction))
        diagnostic = model.diagnostics()
        record.update(model_diagnostics=diagnostic, cached_predictions=len(predictions), public_predictions=predictions,
            parent_imported_torch="torch" in sys.modules, parent_imported_isaac="isaaclab" in sys.modules)
        if diagnostic["native_generation_requests"] != 1 or diagnostic["predict_calls"] != 32:
            raise ValueError("one native chunk/cache order expected")
        if not diagnostic["native_model_isolated"] or record["parent_imported_torch"] or record["parent_imported_isaac"]:
            raise ValueError("component environments not isolated")
        model.end_episode(context)
        model.close()
        model = None
        record["passed"] = True
    except Exception as error:
        record["exception_chain"] = public_exception(error)
    finally:
        if model is not None:
            try:
                model.close()
            except Exception as error:
                record["close_exception_chain"] = public_exception(error)
    write(args.output / "component-result.json", record)
    return 0 if record["passed"] else 1


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if args.child:
        return child(args)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1")
    code, out, err = bounded([str(ROOT / ".venv/bin/python"), __file__, "--child", "--output", str(args.output)], environment, 900)
    result = args.output / "component-result.json"
    record = {"kind": "supervised_isolated_real_model_synthetic_inputs_not_benchmark", "exit_code": code,
        "passed": code == 0 and result.is_file() and json.loads(result.read_text())["passed"],
        "public_log_categories": categories((out + err).decode(errors="replace")), "model_gpu": 4,
        "raw_logs_saved": False, "shared_driver_modified": False, "global_proxy_modified": False,
        "formal_acceptance": False}
    write(args.output / "receipt.json", record)
    print(json.dumps(record), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
