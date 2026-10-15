"""Export all 55 original cells with honest implementation and protocol gates.

This is a coverage ledger, not a runtime registry or score report. Historical
stage-0 materials remain immutable. A native model bridge is not a runnable pair.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/handoff/source-materials/compatibility-draft.json"
CODE = {
    "AD18": ["embodied_eval/adapters/openwam_libero_model.py", "embodied_eval/adapters/libero_spatial_benchmark.py"],
    "AD14": ["embodied_eval/adapters/openpi_libero_spec.py", "embodied_eval/adapters/openpi_libero_model.py",
             "embodied_eval/adapters/libero_openpi_benchmark.py", "embodied_eval/stage4_pairs.py"],
    "AB18": ["embodied_eval/adapters/vlabench_spec.py", "embodied_eval/adapters/openwam_vlabench_model.py",
             "embodied_eval/adapters/vlabench_benchmark.py", "embodied_eval/stage4_pairs.py"],
    "AB14": ["embodied_eval/adapters/openpi_vlabench_spec.py", "embodied_eval/adapters/openpi_vlabench_model.py",
             "embodied_eval/adapters/vlabench_openpi_benchmark.py", "embodied_eval/stage4_pairs.py"],
    "AE4": ["embodied_eval/adapters/native_robotwin_models.py", "embodied_eval/adapters/native_robotwin_executor.py",
            "embodied_eval/adapters/robotwin_spec.py", "embodied_eval/adapters/robotwin_model.py",
            "embodied_eval/adapters/robotwin_benchmark.py", "embodied_eval/robotwin_pairs.py"],
    "AE18": ["embodied_eval/adapters/native_robotwin_models.py", "embodied_eval/adapters/native_robotwin_executor.py",
             "embodied_eval/adapters/robotwin_spec.py", "embodied_eval/adapters/robotwin_model.py",
             "embodied_eval/adapters/robotwin_benchmark.py", "embodied_eval/robotwin_pairs.py"],
}
DEFINITIONS = {
    "accepted_single_trial": "Previously accepted one-trial real/native comparison, not a full benchmark score",
    "implemented_awaiting_joint_real_test": "Registered code exists; real closed-loop and independent native-comparison gates are listed separately, not accepted",
    "native_model_bridge_only": "Source-defined model bridge and low-level execution/cache components exist; no full framework Benchmark registration",
    "blocked_protocol_or_matching_asset": "No runnable pair claimed; specific role/protocol/asset/runtime gap listed below",
}
MODEL_GAPS = {
    "Cosmos 3": ("model_variant_and_role", "Local Cosmos3-Nano is a reasoner; choose precise Agent/world-model/policy variant and matching task assets"),
    "LingBot-VA 2.0": ("matching_checkpoint_and_identity", "Only reviewed LingBot-VLA 2.0 RoboTwin assets selected; other benchmarks need dedicated v2 identity/weights/normalization"),
    "Hy-Embodied-VLM-1.0": ("agent_tool_protocol", "Local VLM weights exist; native closed-loop tool vocabulary, planner and scoring protocol are unselected"),
    "MiniMax H3": ("service_identity_and_endpoint", "Exact model ID/revision, accessible endpoint and evaluated role not supplied"),
    "Wan2.2": ("world_model_action_conditioning", "Exact video variant, action conditioning, subject role and future-prediction scoring unselected"),
    "GPT-6 Astra": ("service_identity_and_endpoint", "Reference label is not proof of an accessible inference endpoint or native tool protocol"),
    "Qwen3.8-27B": ("agent_tool_protocol", "Local visual-language weights exist, but no selected native tools/controller/planner protocol"),
    "Claude-5.5-Opus": ("service_identity_and_endpoint", "Exact model ID/revision and accessible visual/tool endpoint not supplied"),
    "LingBot-World 2.0": ("world_model_action_conditioning", "Exact v2 checkpoint, camera/action domains and prediction scoring not selected"),
    "DreamZero-DROID": ("droid_protocol_definition", "DROID sim_evals is only a candidate for undefined R021/R022; do not substitute AgiBot or real RoboArena"),
    "DreamDojo": ("world_model_action_conditioning", "Robot/action/camera-matched checkpoint and world-model evaluation role not selected"),
    "π0.5": ("matching_checkpoint_and_normalization", "Dedicated LIBERO and primitive-10task VLABench profiles now exist; pi05_base cannot substitute for other robot-specific weights and normalization"),
    "Gemini Robotics ER 2": ("service_identity_and_endpoint", "Exact ER version/endpoint and reasoning-versus-control/tool boundary not supplied"),
    "Seedance 2.5": ("service_identity_and_action_conditioning", "Exact version/endpoint, action-conditioned stepping and feedback protocol not supplied"),
    "Matrix-Game 3.5": ("world_model_action_conditioning", "Game actions/camera conditions have no selected mapping to this robot/backend protocol"),
    "OpenWAM-α": ("matching_checkpoint_and_protocol", "Four dedicated Sim releases exist; they do not define VidMan/DreamZero/World-in-World protocols"),
}
BENCHMARK_GAPS = {
    "vlabench": ("benchmark_subprotocol", "R054/R055 definitions are absent; only reviewed frozen track1/select_fruit EE-policy pilot is implemented"),
    "vidman": ("benchmark_protocol_undefined", "Define R009/R010 as CALVIN closed loop, OXE offline metric, or another explicit environment/protocol"),
    "libero": ("policy_vs_agent_protocol", "VLA spatial policy exists; Agent tool/controller interface must be specified separately"),
    "robotwin": ("policy_vs_agent_domain", "Two source-defined native dual-arm policy profiles registered; other models still need native policy weights or a separately defined Agent tool interface"),
    "robodojo": ("native_runtime_decoupling", "Official eval_env imports XPolicyLab WsModelClient; extract independent native reset/observation/control/score integration, with fixed Isaac/assets/task configuration"),
    "dreamzero_protocol": ("benchmark_protocol_undefined", "R021/R022 undefined; freeze DROID sim backend, suite, initial states and native scoring"),
    "wiw": ("world_model_role_and_backend", "Select ImageNav versus Manipulation, evaluated policy/world-model/system role and proposer/planner auxiliaries"),
}

REAL_LOOPS = {
    "AD14": ("openpi_libero", "outputs/stage4-closed-loop-pi05-libero-20261006T1436Z/receipt.json"),
    "AB18": ("openwam_vlabench", "outputs/stage4-closed-loop-openwam-vlabench-20261006T1450Z-optimized/receipt.json"),
    "AB14": ("openpi_vlabench", "outputs/stage4-closed-loop-pi05-vlabench-20261006T1618Z-native-f32/receipt.json"),
    "AE4": ("lingbot_robotwin", "outputs/stage4-closed-loop-lingbot-robotwin-20261006T1611Z-transport-fixed/receipt.json"),
    "AE18": ("openwam_robotwin", "outputs/stage4-closed-loop-openwam-robotwin-20261006T1618Z/receipt.json"),
}

RECORDED_ACTION_REPLAYS = {
    "AD14": "outputs/stage4-native-action-replay-pi05-libero-20261006T1511Z-final/receipt.json",
    "AB18": "outputs/stage4-native-action-replay-openwam-vlabench-20261006T1544Z/receipt.json",
    "AB14": "outputs/stage4-native-action-replay-pi05-vlabench-20261006T1716Z/receipt.json",
    "AE4": "outputs/stage4-native-action-replay-lingbot-robotwin-20261006T1633Z/receipt.json",
    "AE18": "outputs/stage4-native-action-replay-openwam-robotwin-20261006T1702Z-seed-search/receipt.json",
}

NATIVE_METER_CHECKS = {
    "AE18": "outputs/stage4-native-meter-crosscheck-openwam-robotwin-20261006T1655Z/receipt.json",
}

FRESH_NATIVE_ROLLOUTS = {
    "AD14": "outputs/stage4-fresh-native-policy-pi05-libero-20261006T1652Z-repeat/receipt.json",
    "AE4": "outputs/stage4-fresh-native-policy-lingbot-robotwin-20261006T1730Z-repeat/native/receipt.json",
}


def separate_check(cell, mapping, key):
    relative = mapping.get(cell)
    if not relative or not (ROOT / relative).is_file():
        return None
    value = json.loads((ROOT / relative).read_text())
    return {"receipt": relative, "receipt_sha256": digest(ROOT / relative),
            "passed": value.get(key, False), "formal_acceptance": False,
            "first_divergence": value.get("first_divergence"),
            "first_meter_divergence": next((row["env_step"] for row in value.get("meter_cross_check_rows", [])
                                              if not row["exact_equal"]), None),
            "source_trial_reproduction_passed": value.get("passed", False),
            "native_policy_actions": value.get("native_policy_actions"),
            "native_success": value.get("native_success"),
            "native_physx_steps": value.get("native_physx_steps"),
            "framework_meter_physics_steps_this_check": value.get("framework_meter_physics_steps_this_check")}


def recorded_action_replay(cell):
    relative = RECORDED_ACTION_REPLAYS.get(cell)
    if not relative or not (ROOT / relative).is_file():
        return None
    value = json.loads((ROOT / relative).read_text())
    return {"receipt": relative, "receipt_sha256": digest(ROOT / relative),
            "passed": value.get("passed", False), "fresh_model_inference": False,
            "compared_snapshots": value.get("compared_snapshots"),
            "first_divergence": value.get("first_divergence"),
            "framework_physics_steps": value.get("framework_physics_steps"),
            "native_physx_steps": value.get("native_physx_steps"),
            "physics_counter_backend": value.get("physics_counter_backend"),
            "recorded_seed_search_reproduced": value.get("recorded_seed_search_reproduced", False),
            "formal_acceptance": False}


def completed_real_loop(cell):
    if cell not in REAL_LOOPS:
        return None
    pair, relative = REAL_LOOPS[cell]
    path = ROOT / relative
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text())
    records = [item for item in receipt.get("results", []) if item.get("pair") == pair]
    if len(records) != 1:
        raise ValueError("ambiguous real-loop evidence")
    record = records[0]
    complete = bool(record.get("passed") and record.get("closed_loop") and
                    record.get("synthetic") is False and record.get("exit_code") == 0 and
                    record.get("scoring_performed") and record.get("results") and
                    all(item.get("run_status") == "completed" for item in record["results"]))
    return {"receipt": relative, "receipt_sha256": digest(path), "passed": complete,
            "real_model_and_benchmark": record.get("synthetic") is False,
            "native_stepwise_comparison_performed": record.get("native_stepwise_comparison_performed", False),
            "formal_acceptance": False,
            "results": [{"episode_id": item["episode_id"], "metrics": item["metrics"],
                         "execution_statistics": item["execution_statistics"]}
                        for item in record.get("results", [])]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    source = json.loads(SOURCE.read_text())
    originals = source["combinations"]
    cells = [item["source_cell"] for item in originals]
    if len(cells) != 55 or len(set(cells)) != 55:
        raise ValueError("expected 55 unique original cells")
    rows = []
    for item in originals:
        cell, model, benchmark = item["source_cell"], item["model_name"], item["benchmark_id"]
        status = "blocked_protocol_or_matching_asset"
        gaps = [MODEL_GAPS[model], BENCHMARK_GAPS[benchmark]]
        runtime_config = None
        accepted_evidence = None
        real_loop = completed_real_loop(cell)
        if cell == "AD18":
            status, gaps = "accepted_single_trial", []
            runtime_config = "examples/stage3-openwam-libero-spatial-pilot.json"
            accepted_evidence = "outputs/stage3-compare-20261005T1734Z.json"
        elif cell in ("AD14", "AB18", "AB14"):
            status = "implemented_awaiting_joint_real_test"
            gaps = [("native_stepwise_comparison_pending", "Full independent native policy rollout comparison and acceptance remain pending")]
            if not real_loop or not real_loop["passed"]:
                gaps.insert(0, ("joint_real_integration_pending", "Source-defined real full trial is pending or failed; no completed score claimed"))
            fresh = separate_check(cell, FRESH_NATIVE_ROLLOUTS, "passed")
            if fresh and not fresh["passed"]:
                gaps.insert(0, ("fresh_native_policy_rollout_divergence", "Native fresh inference completed successfully but strict stepwise actions/trajectory differ; input mapping/codec were independently exact"))
            runtime_config = {"AD14": "examples/stage4-pi05-libero-spatial.json",
                              "AB18": "examples/stage4-openwam-vlabench-select-fruit.json",
                              "AB14": "examples/stage4-pi05-vlabench-select-fruit.json"}[cell]
        elif cell in ("AE4", "AE18"):
            status = "implemented_awaiting_joint_real_test"
            gaps = [("native_policy_comparison_and_acceptance_pending", "Independent recorded-action replay is separate from a fresh native policy rollout and formal acceptance")]
            fresh = separate_check(cell, FRESH_NATIVE_ROLLOUTS, "passed")
            if fresh and fresh["passed"]:
                gaps = [("formal_acceptance_and_broader_suite_pending", "Exact fresh native model/environment one-trial comparison passed; stage4 formal acceptance and broader task statistics remain separate")]
            elif fresh:
                gaps.insert(0, ("fresh_native_policy_rollout_divergence", "Independent native fresh inference or strict comparison failed; inspect the separate receipt"))
            if not real_loop or not real_loop["passed"]:
                gaps.insert(0, ("joint_real_integration_pending", "Full native model/controller trial must complete after asset and runtime preflight"))
            replay = recorded_action_replay(cell)
            counter = separate_check(cell, NATIVE_METER_CHECKS, "simultaneous_meter_cross_check_passed")
            if not ((counter and counter["passed"]) or (replay and replay["passed"] and replay["physics_counter_backend"] == "PhysxCpuSystem"
                    and replay["framework_physics_steps"] == replay["native_physx_steps"]
                    and replay["native_physx_steps"] is not None)):
                gaps.insert(0, ("native_scene_counter_real_validation_pending", "Independent native PhysX step counter and raw snapshots must match the recorded real trial"))
            if replay and not replay["passed"]:
                gaps.insert(0, ("native_recorded_action_replay_divergence", "Real direct replay has a first-step trajectory/physics-time difference; simultaneous meter validation does not establish exact source-trial reproduction"))
            runtime_config = "examples/stage4-lingbot-robotwin-adjust-bottle.json" if cell == "AE4" else "examples/stage4-openwam-robotwin-adjust-bottle.json"
        if cell == "AF18":
            gaps = [BENCHMARK_GAPS[benchmark],
                    ("robodojo_independent_model_bridge", "Matching published checkpoint exists, but calibrated dual-X5 runtime frame/image/client protocol is not independently implemented")]
        rows.append({"source_cell": cell, "model": model, "benchmark": benchmark,
            "stage0_status": item["status"], "implementation_status": status,
            "framework_registered": runtime_config is not None,
            "native_execution_component_implemented": cell in ("AE4", "AE18"),
            "joint_real_test_this_batch": bool(real_loop and real_loop["passed"]),
            "real_closed_loop_evidence": real_loop, "accepted_evidence": accepted_evidence,
            "independent_native_recorded_action_replay": recorded_action_replay(cell),
            "separate_simultaneous_real_meter_check": separate_check(cell, NATIVE_METER_CHECKS, "simultaneous_meter_cross_check_passed"),
            "fresh_native_policy_rollout_comparison": separate_check(cell, FRESH_NATIVE_ROLLOUTS, "passed"),
            "runtime_config": runtime_config, "source_files": CODE.get(cell, []),
            "blocking_items": [{"code": code, "meaning": meaning} for code, meaning in gaps],
            "identity_note": "Reviewed implementation uses LingBot-VLA 2.0, not old LingBot-VA weights" if cell == "AE4" else None})
    counts = dict(Counter(row["implementation_status"] for row in rows))
    expected = {"accepted_single_trial": 1, "implemented_awaiting_joint_real_test": 5,
                "blocked_protocol_or_matching_asset": 49}
    if counts != expected:
        raise ValueError(f"unexpected implementation counts: {counts}")
    paths = sorted({path for group in CODE.values() for path in group} | {
        "embodied_eval/config.py", "embodied_eval/contracts.py", "embodied_eval/interfaces.py",
        "embodied_eval/runner.py", "embodied_eval/validation.py", "embodied_eval/compatibility.py",
        "embodied_eval/codec.py", "embodied_eval/visibility.py", "embodied_eval/artifacts.py",
        "tests/test_stage4_adapters.py", "tests/test_native_timing.py", "tests/test_robotwin_executor.py", "tests/test_robotwin_pairs.py",
        "tests/test_inline_tensor_paths.py", "docs/handoff/replay_libero_native_actions.py",
        "docs/handoff/replay_vlabench_native_actions.py", "docs/handoff/replay_robotwin_native_actions.py",
        "docs/handoff/download_hf_ranges.py",
        "docs/handoff/compare_libero_fresh_native_policy.py", "docs/handoff/check_pi05_libero_native_input.py",
        "docs/handoff/compare_lingbot_fresh_native_policy.py",
        "docs/handoff/build_delivery_receipt.py", "docs/handoff/record_stage4_preparation.py",
        "docs/handoff/probe_robotwin_models.py", "docs/handoff/serve_lingbot_native.py",
        "docs/handoff/probe_lingbot_checkpoint_startup.py",
        "embodied_eval/adapters/native_common.py", "embodied_eval/adapters/native_policy_transport.py",
        "embodied_eval/adapters/robotwin_identity.py", "embodied_eval/adapters/openwam_robotwin_objects.json",
        "embodied_eval/adapters/native_robotwin_runtime.py", "embodied_eval/adapters/qwen3_vl_processor_objects.json",
        "embodied_eval/adapters/lingbot_robotwin_objects.json", "embodied_eval/adapters/robotwin_assets_objects.json",
        "embodied_eval/adapters/pi05_libero_objects.json", "embodied_eval/adapters/pi05_vlabench_objects.json"})
    return {"document_kind": "implementation_coverage_not_acceptance_or_score",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": str(SOURCE.relative_to(ROOT)), "source_snapshot_sha256": digest(SOURCE),
        "status_definitions": DEFINITIONS, "planned_cells": 55, "counts": counts,
        "measured_native_core_implemented": True, "robotwin_full_pair_registered": True,
        "models": [{"name": name, "remaining_gap": {"code": gap[0], "meaning": gap[1]}}
                   for name, gap in MODEL_GAPS.items()],
        "benchmark_groups": [{"id": name, "remaining_gap": {"code": gap[0], "meaning": gap[1]}}
                             for name, gap in BENCHMARK_GAPS.items()],
        "code_sha256": {path: digest(ROOT / path) for path in paths}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite historical evidence")
    value = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.output}: {value['counts']}")


if __name__ == "__main__":
    main()
