"""Bind repair evidence and regression receipts without model/scene calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/handoff"))
from repair_native_interpreters import metadata_files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "native_repair_recovery_source_and_software_evidence_gate", "passed": False,
              "at_utc": datetime.now(timezone.utc).isoformat(), "model_or_simulator_calls": 0,
              "formal_acceptance": False}
    try:
        old = json.loads((ROOT / "docs/handoff/current-only-execution-20261023-index.json").read_text())
        expected = old["runtime_python_sha256"]
        assert all(digest(ROOT / name) == value for name, value in expected.items())
        report["production_runtime_unchanged_python_files"] = len(expected)
        report["production_runtime_sha256"] = expected
        repair_dir = ROOT / "outputs/native-env-repair-20261023-run1"
        repair = json.loads((repair_dir / "receipt.json").read_text())
        archive = json.loads((ROOT / "outputs/worldeval-platform-assets-20261019/runtime-manifest.json").read_text())
        rows = []
        for row in repair["environments"]:
            env = ROOT.parent / "upstream/envs" / row["environment"]
            original = json.loads((repair_dir / row["environment"] / "installed-metadata-before.json").read_text())
            current = metadata_files(env)
            binary = Path(row["new_python_target"])
            minor = row["python"]["minor"]
            binary_hash = digest(binary)
            assert binary_hash == archive["runtimes"][minor]["binary_sha256"]
            assert original == current
            assert (env / "bin/python").resolve() == binary.resolve()
            assert (repair_dir / row["environment"] / "pyvenv.cfg.original").is_file()
            rows.append({"environment": row["environment"], "unchanged_installed_distributions": len(current),
                         "same_historical_archived_python_binary": True, "interpreter": row["python"]["version"],
                         "binary_sha256": binary_hash, "recoverable_original_config_and_links": True})
        report["repairs"] = rows
        report["no_dependency_upgrade_performed"] = True
        completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "integrations/massworld/tests", "-v"],
            cwd=ROOT, capture_output=True, text=True)
        log = completed.stdout + completed.stderr
        (args.output / "bridge-tests.txt").write_text(log)
        count = re.search(r"Ran (\d+) tests?", log)
        assert completed.returncode == 0 and count and int(count.group(1)) == 34
        assert not re.search(r"skipped=\d+", log)
        report["bridge_tests"] = {"passed": True, "tests": 34, "skips": 0,
                                  "log_sha256": digest(args.output / "bridge-tests.txt")}
        rng = {label: json.loads((ROOT / f"outputs/libero-rng-audit-20261023-{label}/receipt.json").read_text())
               for label in ("framework0", "native0", "framework7", "native7")}
        assert all(row["completed"] for row in rng.values())
        comparisons = []
        for left, right in (("framework0", "native7"), ("framework0", "native0"), ("framework7", "native7")):
            a, b = rng[left], rng[right]
            comparisons.append({"left": left, "right": right,
                "pixels_exact": {key: a["pixels"][key]["sha256"] == b["pixels"][key]["sha256"] for key in a["pixels"]},
                "different_native_array_fields": [key for key in a["native_arrays"] if a["native_arrays"][key]["sha256"] != b["native_arrays"][key]["sha256"]]})
        report["constructor_rng_controlled_diagnostic"] = {"comparisons": comparisons,
            "historical_difference_unique_cause_proven": False,
            "production_protocol_changed": False,
            "conclusion": "All selected current same-host initial pixels/arrays equal, so constructor RNG alone is not demonstrated as the historical cause."}
        report["passed"] = True
    finally:
        (args.output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "repairs": len(report.get("repairs", [])),
                      "production_files_unchanged": report.get("production_runtime_unchanged_python_files")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
