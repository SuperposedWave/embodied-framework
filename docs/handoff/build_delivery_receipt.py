"""Index final evidence and hash the small project files, not a stage acceptance.

Read-only checks except exclusive new output files. Never inspect credentials,
proxy values, native weights, or model server environments.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if tag == "a" and name == "href":
                self.targets.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads((ROOT / "docs/stage-state.json").read_text())
    current = state["stage4_implementation_20261006"]
    evidence = {}
    paths = [current[k] for k in ("coverage_ledger", "preparation_inventory", "architecture_report",
        "baseline_replay_vlabench_environment", "baseline_replay_libero_environment",
        "baseline_replay_robotwin_environment", "gpu_evidence")]
    for record in current["real_closed_loop_trials"]:
        if record.get("receipt"):
            paths.append(record["receipt"])
    for record in current["independent_native_recorded_action_replays"]:
        paths.append(record["receipt"])
    for record in current.get("fresh_native_policy_rollout_comparisons", []):
        paths.append(record["receipt"])
    for record in current.get("native_repeatability_diagnostics", []):
        paths.append(record["receipt"])
    paths.extend(current.get("gpu_other_completed_windows", []))
    for key in ("simultaneous_real_meter_check", "pi05_libero_native_first_input_and_codec"):
        if current.get(key):
            paths.append(current[key]["receipt"])
    for relative in sorted(set(paths)):
        path = ROOT / relative
        evidence[relative] = {"exists": path.is_file(), "sha256": digest(path) if path.is_file() else None}
    checks = {}
    for key in ("baseline_replay_vlabench_environment", "baseline_replay_libero_environment", "baseline_replay_robotwin_environment"):
        receipt = json.loads((ROOT / current[key]).read_text())
        cases = {row["case"]: row for row in receipt["checks"]}
        codes = [cases[name]["actual_exit_code"] for name in ("chunk", "tool", "cache", "score-error", "rejected")]
        checks[key] = {"passed": bool(receipt["passed"] and cases["unittest"]["tests_run"] == 202 and codes == [0,0,0,1,2]),
                       "tests": cases["unittest"]["tests_run"], "cli_exit_codes": codes, "synthetic": receipt["synthetic"]}
    coverage = json.loads((ROOT / current["coverage_ledger"]).read_text())
    cells = [row["source_cell"] for row in coverage["rows"]]
    registered = sum(row["framework_registered"] for row in coverage["rows"])
    checks["original_coverage"] = {"passed": len(cells) == len(set(cells)) == 55 and registered == 6,
        "registered": registered, "counts": coverage["counts"]}
    trials = current["real_closed_loop_trials"]
    checks["source_defined_real_trial_receipts_complete"] = {
        "passed": len(trials) == 5 and all(record.get("completed") and record.get("official_success")
            and json.loads((ROOT / record["receipt"]).read_text()).get("passed") for record in trials),
        "trials": len(trials), "not_formal_acceptance_or_full_suite": True}
    inventory = json.loads((ROOT / current["preparation_inventory"]).read_text())
    checks["native_tracked_sources_clean"] = {"passed": all(s["tracked_source_clean"] for s in inventory["sources"].values()),
                                             "sources": len(inventory["sources"])}
    historical = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if relative.startswith("outputs/stage2-20261005T063033Z-final/") or relative in (
                "docs/stage2-acceptance.md", "docs/stage2-evidence.json"):
            historical.append({"path": relative, "unchanged": digest(ROOT / relative) == expected})
    checks["historical_stage2_sha256"] = {"passed": len(historical) == 35 and all(v["unchanged"] for v in historical),
                                         "files_checked": len(historical), "files": historical}
    links = Links()
    links.feed((ROOT / current["architecture_report"]).read_text())
    local_links = [unquote(urlsplit(v).path) for v in links.targets if v.startswith("/")]
    missing = sorted({v for v in local_links if not Path(v).exists()})
    checks["report_local_links"] = {"passed": not missing, "links_checked": len(local_links), "missing": missing}
    small_paths = {ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "uv.lock",
                   ROOT / current["architecture_report"]}
    for name in ("embodied_eval", "tests", "docs", "examples"):
        small_paths.update(p for p in (ROOT / name).rglob("*")
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
            and p.suffix in (".py", ".md", ".json", ".txt", ".toml"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(small_paths) if p.is_file()}
    with (args.output / "SHA256SUMS").open("x") as stream:
        stream.write("".join(f"{value}  {path}\n" for path, value in hashes.items()))
    receipt = {"kind": "delivery_integrity_and_evidence_index_not_formal_acceptance",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_integrity_checks_passed": all(c["passed"] for c in checks.values()),
        "checks": checks, "evidence": evidence, "small_project_file_count": len(hashes),
        "source_file_hashes": hashes, "stage4_formally_accepted": False, "contract_frozen": False,
        "all_55_pairs_implemented": False,
        "strict_native_validation_complete_for_all_registered_pairs": False,
        "no_framework_git_commit_or_remote": True}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("source_file_hashes", "evidence")}))
    return 0 if receipt["delivery_integrity_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
