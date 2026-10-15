"""Verify a prepared report's source, evidence links and honest state boundaries.

Integrity pass is NOT stage acceptance, task success, or a repaired historical
failure. GPU jobs and user-deferred validation remain separately visible.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from prepare_mn_extension import ROOT, read, save, digest, historical_integrity


def verify(directory):
    from serve_report import catalog
    report = directory / "M+N-acceptance-preparation-and-evidence.html"
    expected = read(directory / "evidence-sha256.json")
    evidence = {}
    for relative, value in expected.items():
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        actual = digest(path)
        if actual != value:
            raise ValueError("report_or_linked_evidence_changed")
        evidence[relative] = actual
    route, exposed, hashes = catalog(report)
    if set(hashes.values()) != set(evidence.values()):
        raise ValueError("served_catalog_and_report_evidence_inventory_differ")
    validator = Path("/root/.codex/skills/html-report/scripts/validate_report.py")
    validation = subprocess.run([sys.executable, str(validator), str(report)], capture_output=True, text=True, timeout=30)
    if validation.returncode:
        raise ValueError("HTML_skill_validation_failed")
    snapshot, ledger = read(directory / "snapshot.json"), read(directory / "coverage-ledger.json")
    original = read(ROOT / "outputs/mn-extension-20261007-final2/coverage-ledger.json")
    if (snapshot["stage4_accepted"] or snapshot["contract_frozen"] or ledger.get("formal_acceptance", False)
            or len(ledger["rows"]) != 55 or len(original["rows"]) != 55
            or sum(row["was_one_of_original_49_gaps"] for row in ledger["rows"]) != 49):
        raise ValueError("coverage_or_acceptance_boundary_changed")
    for before, after in zip(original["rows"], ledger["rows"]):
        for key, value in before.items():
            if after[key] != value:
                raise ValueError("historical_coverage_lineage_rewritten")
    text = report.read_text()
    if any(f'id="gap-{index:02d}"' not in text for index in range(1, 50)):
        raise ValueError("49_individual_gap_explanations_required")
    history = historical_integrity()
    saved_history = read(directory / "historical-integrity.json")
    if any(saved_history[key] != value for key, value in history.items()):
        raise ValueError("historical_snapshot_differs_from_current_read_only_check")
    public = saved_history["public_v02_core_and_runner_unchanged"]
    if len(public) != 4 or any(not value["unchanged"] or digest(ROOT / path) != value["sha256"] for path, value in public.items()):
        raise ValueError("four_public_core_and_runner_seals_required")
    source = {}
    for folder in ("embodied_eval", "tests", "docs/handoff"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                ast.parse(path.read_text())
                source[str(path.relative_to(ROOT))] = digest(path)
    jobs = snapshot["jobs"]
    required = ["OpenWAM_RoboDojo_full_limit", "pi05_RoboDojo_full_limit", "pi05_RoboTwin_full_native_limit"]
    pending = [name for name in required if not jobs[name]["terminal_receipt_available"]]
    failed = [name for name, value in jobs.items() if value["terminal_receipt_available"] and not value["receipt"].get("passed", False)]
    state = read(directory / "stage-state-snapshot.json")["mn_acceptance_preparation_20261008"]
    dream_deferred = state["dreamzero"]["user_deferred_large_memory_gpu_validation"]
    if not dream_deferred or state["dreamzero"]["real_model_request_passed"]:
        raise ValueError("user_deferred_unverified_DreamZero_must_remain_explicit")
    return {"kind": "acceptance_preparation_integrity_not_stage_pass", "at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_integrity_passed": True, "HTML_skill_validation_passed": True,
        "local_links_and_evidence_hashes_passed": True, "exposed_file_count": len(exposed),
        "linked_evidence_sha256": evidence, "source_python_files_parsed": len(source), "source_python_sha256": source,
        **history, "public_v02_core_and_runner_unchanged": public,
        "current_software_tests_per_environment": snapshot["tests_in_each_current_completed_environment"],
        "completed_baseline_environments": snapshot["completed_baseline_environments"],
        "CLI_exit_codes": snapshot["CLI_exit_codes"], "original_49_gaps_individually_explained": True,
        "pending_real_trials": pending, "failed_receipts_preserved": failed,
        "DreamZero_H100_validation_explicitly_user_deferred": dream_deferred,
        "all_current_terminal_receipts_are_not_all_passed": bool(failed),
        "report_route": route, "formal_stage4_accepted": False, "contract_frozen": False,
        "real_task_success_or_historical_pixel_reproduction_inferred_from_integrity": False,
        "framework_git_commit_created": False, "external_upload_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    directory = args.delivery.resolve()
    directory.relative_to(ROOT / "outputs")
    value = verify(directory)
    save(directory / "verification.json", value)
    print(json.dumps({key: value[key] for key in ("delivery_integrity_passed", "exposed_file_count", "source_python_files_parsed",
        "pending_real_trials", "failed_receipts_preserved")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
