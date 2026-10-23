"""Investigate fresh-run differences separately from mapping/replay checks.

Compare actual first native input mappings and decoded execution parameters to
historical trials. No inference is repeated and no success substitutes for
equality. Projection/conversion helpers under test are used diagnostically;
independent fixture/native execution receipts are separate evidence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compare_mn_mappings import SOURCE


def first_input(root):
    manifest = json.loads((root / "manifest.json").read_text())
    episode = manifest["tasks"][0]["episode_id"]
    folder = root / "episodes" / episode
    with (folder / "model_input.public.jsonl").open() as stream:
        view = json.loads(stream.readline())
    prediction = None
    with (folder / "trace.internal.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "prediction":
                prediction = event["data"]
                break
    return manifest, view, prediction, json.loads((folder / "result.json").read_text())


def equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, str) or a is None:
        return a == b
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    return np.array_equal(np.asarray(a), np.asarray(b))


def compare(case, configs, trials):
    from dataclasses import replace
    from embodied_eval.core import types as old
    from embodied_eval.codec import decode
    from embodied_eval.config import load_config
    from embodied_eval.native import contracts as c
    from tests.legacy_support.frozen.mn.binding import model_input, resolve
    from embodied_eval.native.model_catalog import capability
    from embodied_eval.native.benchmark_catalog import space
    from embodied_eval.native.native_models import NativeActionModel
    from embodied_eval.native.native_benchmarks import NativeSimulator
    from embodied_eval.native.transforms import prepare_input
    from embodied_eval.adapters.native_common import native_module
    config = load_config(configs / (case + ".json"))
    before_manifest, before, before_pred, before_result = first_input(ROOT / "outputs" / SOURCE[case])
    after_manifest, after, after_pred, after_result = first_input(trials / case / "framework")
    cap, s = capability(config.model.adapter, config.model.options), space(config.benchmark.adapter, config.benchmark.options)
    bench = NativeSimulator.__new__(NativeSimulator)
    bench.family, bench.space = config.benchmark.adapter, s
    previous = decode(old.ModelInput, before)
    obs = bench._project(previous.observation_view)
    task = c.TaskSpec(task_id="audit", episode_id=obs.context.episode_id, public=previous.task_public,
        budgets=config.episodes[0].budgets, initialization_ref="historical", evaluation_config_ref="not-scoring")
    profile = resolve(s, cap, config.binding)
    projected = prepare_input(model_input(obs, task, cap, profile), cap, profile, s)
    current = prepare_input(decode(c.ModelInput, after), cap, profile, s)
    model = NativeActionModel(config.model.adapter, config.model.options)
    model.profile, model.space = profile, s
    if config.model.adapter == "openwam":
        model.utils = native_module(config.model.options["repo"], "benchmarks.utils")
        model.conversion = native_module(config.model.options["repo"], "benchmarks.utils.action_conversion")
    elif config.model.adapter == "openpi" and config.model.options["io_profile"] == "state8_osc":
        model.images = native_module(config.model.options["repo"], "openpi_client.image_tools", "packages/openpi-client/src")
    a, b = model.native_input(projected), model.native_input(current)
    differences = [key for key in a if not equal(a[key], b.get(key))]
    decision = before_pred["decision"]
    parameters = decision["commands"][0]["parameters"] if "commands" in decision else decision["steps"][0]["commands"][0]["parameters"]
    action = decode(c.Action, after_pred["action"])
    if config.benchmark.adapter == "vlabench":
        decoded = bench._cartesian_parameters(action)
    elif case == "lingbot_robotwin":
        commands = {cmd.target_id.split("/", 1)[1]: cmd for cmd in action.commands}
        decoded = np.asarray([v for side in ("left", "right") for v in
            [*[commands[side + f"_j{i}"].value for i in range(6)], commands[side + "_grip"].parameters["requested_open_fraction"]]], dtype=action.native_dtype)
        parameters = np.asarray(parameters["native_row"], dtype=parameters["native_dtype"])
    elif case == "openwam_robotwin":
        decoded = np.asarray(action.commands[0].parameters["native_row"], dtype=action.native_dtype)
        parameters = np.asarray(parameters["native_row"], dtype=parameters["native_dtype"])
    else:
        decoded = action.commands[0].parameters
    return {"case": case, "first_native_input_values_and_dtypes_exact": not differences,
        "different_first_input_keys": differences, "first_decoded_execution_parameters_exact": bool(equal(parameters, decoded)),
        "historical_actions": before_result["execution_statistics"]["env_steps"],
        "new_actions": after_result["execution_statistics"]["env_steps"],
        "source_revision_declared_same": cap.source_revision == before_manifest["capability"]["source_revision"],
        "source_checkpoint_note": "Local pinned assets verified separately; server identity still unverified.",
        "investigation": "Starting native inputs differ; investigate listed input fields before attributing output differences to randomness." if differences else
            "Starting native inputs match. Fresh decoded action may differ; this alone does not locate whether stochastic sampling, numerical/backend state or service state caused it.",
        "fresh_policy_bitwise_reproduction_proven": False, "not_mapping_or_execution_acceptance": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=list(SOURCE))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    rows = [compare(case, args.configs, args.trials) for case in args.cases]
    receipt = {"kind": "fresh_trial_difference_investigation_not_determinism_claim", "at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic": False, "new_model_requests_performed_by_this_script": 0, "formal_acceptance": False, "results": rows}
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
