"""Read-only attempt ledger: failed diagnostics are not silently discarded."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from probe_mn_components import ROOT, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.resolve().relative_to(ROOT / "outputs")
    if args.output.exists():
        raise FileExistsError("new_attempt_ledger_required")
    attempts = []
    for family in ("pi05-robodojo", "openwam-robodojo", "cosmos3-droid", "robodojo-startup-isolation"):
        runs = sorted((ROOT / "outputs").glob(f"native-comparison-{family}-20261009-run*"), key=lambda p: int(p.name.rsplit("run", 1)[1]))
        for run in runs:
            row = {"directory": str(run.relative_to(ROOT)), "completed": False, "formal_acceptance": False}
            for name in ("receipt.json", "native-result.json", "cancellation.json", "private-runtime-paths.json"):
                path = run / name
                if not path.is_file():
                    continue
                value = json.loads(path.read_text())
                keys = ("kind", "completed", "passed", "child_exit_code", "close_completed", "native_actions", "native_policy_actions",
                    "native_requests", "native_metrics", "strict_equal", "fresh_native_policy_inference", "fresh_policy_inference",
                    "diagnostic_safety_refusal", "exception_chain", "public_failure_categories", "public_log_categories",
                    "fresh_policy_actions_executed", "native_cleanup_acknowledged", "forced_kill_required",
                    "private_graphics_runtime_path_at_cancellation", "private_library_directory", "absolute_and_exists_before_child_changes_directory")
                row[name] = {key: value[key] for key in keys if key in value}
            receipt = row.get("receipt.json", {}); native = row.get("native-result.json", {})
            row["completed"] = bool(receipt.get("completed") and native.get("completed") and native.get("close_completed"))
            row["is_startup_recorded_action_probe_not_fresh_policy"] = family == "robodojo-startup-isolation"
            attempts.append(row)
    result = {"kind": "preserved_native_comparison_attempt_ledger_not_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
        "attempts": attempts, "attempt_count": len(attempts), "raw_stdout_stderr_headers_or_credentials_saved": False,
        "incomplete_or_failed_attempts_counted_as_pass": False, "model_or_environment_calls": 0,
        "confirmed_new_diagnostic_issue": "Relative private graphics library path did not exist after child cwd change. Absolute path normalization restored both native resets; model/controller/source/public contract unchanged.",
        "additional_unique_render_or_model_nondeterminism_root_cause_claimed": False}
    write(args.output, result)
    print(json.dumps({"attempts_recorded": len(attempts)}))


if __name__ == "__main__":
    main()
