"""Reconcile existing fixed-native reference clocks with audited framework logs.

The native results are historical component-reference evidence, not new runs,
not an unmodified official all-task evaluator, and not MassWorld bridge runs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
from docs.handoff.execution_audit import EPS_S, digest_file, summary


def resolve_binding(manifest_path, native, native_directory):
    """Bind by recorded hashes, never by folder names or an equality pass flag."""
    actual = digest_file(manifest_path)
    if native.get("framework_manifest_sha256") is not None:
        if native["framework_manifest_sha256"] != actual:
            raise ValueError("native_framework_manifest_hash_mismatch")
        return {"basis": "native_result_manifest_sha256", "files": {}}
    companion_path = native_directory / "first-native-request-check.json"
    if not companion_path.is_file():
        raise ValueError("native_reference_binding_missing")
    companion = json.loads(companion_path.read_text())
    if companion.get("framework_manifest_sha256") != actual:
        raise ValueError("companion_framework_manifest_hash_mismatch")
    requests = list(native_directory.glob("native-exchanges/request-0001.*"))
    if len(requests) != 1 or digest_file(requests[0]) != companion.get("native_request_sha256"):
        raise ValueError("companion_actual_native_request_hash_mismatch")
    manifest = json.loads(manifest_path.read_text())
    view_path = manifest_path.parent / "episodes" / manifest["config"]["episodes"][0]["episode_id"] / "model_input.public.jsonl"
    import hashlib
    with view_path.open("rb") as f:
        first_sha = hashlib.sha256(f.readline()).hexdigest()
    if first_sha != companion.get("framework_public_input_first_line_sha256"):
        raise ValueError("companion_framework_first_input_hash_mismatch")
    return {"basis": "companion_first_request_check_and_actual_native_request_hashes",
            "companion_input_equality_check_passed": companion.get("passed"),
            "input_equality_is_not_a_source_identity_gate": True,
            "files": {str(p): digest_file(p) for p in (companion_path, requests[0])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.execution_audit.resolve().relative_to(ROOT / "outputs")
    args.output.resolve().relative_to(ROOT / "outputs")
    if args.output.exists():
        raise ValueError("new_native_clock_receipt_required")
    completion = json.loads((args.execution_audit / "receipt.json").read_text())
    if not completion["completed"]:
        raise ValueError("wait_for_complete_execution_audit_before_native_clock_review")
    from build_native_comparison_report import cases
    selected = []
    protected = {}
    for case in cases():
        route = case["id"].replace("-", "_")
        audit_path = args.execution_audit / ("historical-" + route + ".json")
        audit = json.loads(audit_path.read_text())
        path = case["native"] / "native-result.json"
        protected[str(path)] = digest_file(path)
        native = json.loads(path.read_text())
        manifest_path = case["framework"] / "manifest.json"
        actual_manifest = digest_file(manifest_path)
        binding = resolve_binding(manifest_path, native, case["native"])
        protected.update(binding["files"])
        if audit["protected_file_sha256"][str(manifest_path)] != actual_manifest:
            raise ValueError("native_reference_is_not_bound_to_audited_framework_manifest:" + route)
        if not native["completed"] or native["synthetic"] or not audit["passed"]:
            raise ValueError("completed_real_reference_and_execution_audit_required:" + route)
        by_index = {row["action_index"]: row for row in audit["transitions"]}
        if len({row["env_step"] for row in native["rows"]}) != len(native["rows"]):
            raise ValueError("duplicate_native_action_index:" + route)
        comparisons = []
        previous = None
        for row in native["rows"]:
            index = row["env_step"]
            if index == 0:
                previous = row
                continue
            if index in by_index:
                ref = by_index[index]
                elapsed = row.get("native_elapsed_s")
                if elapsed is None and previous is not None:
                    elapsed = row["native_sim_time_s"] - previous["native_sim_time_s"]
                physics = row.get("native_physics_delta")
                comparisons.append({"action_index": index, "framework_elapsed_s": ref["elapsed_s"],
                    "native_elapsed_s": elapsed,
                    "elapsed_equal_within_1e_8_s": elapsed is not None and abs(elapsed - ref["elapsed_s"]) <= EPS_S,
                    "framework_physics_steps": ref["physics_steps"], "native_physics_steps": physics,
                    "physics_equal": None if physics is None or ref["physics_steps"] is None else physics == ref["physics_steps"],
                    "original_action_comparison": row.get("action_exact", row.get("fields", {}).get("action", {}).get("exact_equal"))})
            previous = row
        initial = native["rows"][0]
        initial_differences = [key for key, value in initial["fields"].items()
                               if value.get("exact_equal") is False or value.get("dtype_equal") is False]
        first_time = next((x["action_index"] for x in comparisons if not x["elapsed_equal_within_1e_8_s"]), None)
        first_physics = next((x["action_index"] for x in comparisons if x["physics_equal"] is False), None)
        first_action = next((x["action_index"] for x in comparisons if x["original_action_comparison"] is False), None)
        selected.append({"route": route, "framework_run": audit["directory"],
            "original_reference": str(path), "reference_sha256": protected[str(path)],
            "framework_manifest_identity_matches": True,
            "source_binding": binding,
            "framework_actions": audit["executed_actions"],
            "native_actions": native.get("native_actions", native.get("native_policy_actions")),
            "framework_native_requests": audit["recorded_native_requests"],
            "native_reference_requests": native.get("native_requests", native.get("native_policy_requests")),
            "compared_common_action_prefix": len(comparisons),
            "first_elapsed_difference": first_time, "first_physics_count_difference": first_physics,
            "first_recorded_action_difference": first_action,
            "original_strict_equal": native["strict_equal"],
            "original_first_divergence": native["first_divergence"],
            "original_initial_difference_fields": initial_differences,
            "framework_elapsed_s": audit["execution_elapsed_s"],
            "native_prefix_elapsed_s": summary(x["native_elapsed_s"] for x in comparisons if x["native_elapsed_s"] is not None),
            "execution_comparisons": comparisons,
            "unequal_episode_lengths_not_used_as_frequency_mismatch": True,
            "root_cause_proven": False})
    if any(digest_file(path) != sha for path, sha in protected.items()):
        raise ValueError("historical_native_reference_modified")
    receipt = {"kind": "saved_native_component_reference_clock_review_not_new_inference",
        "completed": True, "source_identity_checks_passed": True,
        "at_utc": datetime.now(timezone.utc).isoformat(), "cases": selected,
        "protected_file_sha256": protected, "historical_files_unchanged": True,
        "new_inference": False, "new_simulation": False, "formal_acceptance": False,
        "strict_equal_references": sum(x["original_strict_equal"] for x in selected),
        "strict_reproduction_gaps_closed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"completed": True, "native_references": len(selected),
        "strict_equal": receipt["strict_equal_references"], "new_inference": False}))


if __name__ == "__main__":
    main()
