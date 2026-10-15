"""Replay the existing synthetic acceptance baseline in a relocated checkout.

Uses the invoking interpreter, creates a new output directory and never rewrites
historical evidence. This checks framework migration, not GPU/model readiness.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_command(args, output):
    result = subprocess.run([sys.executable, *args], cwd=ROOT,
                            capture_output=True, text=True)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="New directory; existing directories are refused")
    parser.add_argument("--additional-tests", type=int, default=0,
                        help="Expected tests added after the fixed 85-test handoff baseline")
    args = parser.parse_args()
    require(args.additional_tests >= 0, "additional test count cannot be negative")
    target = args.output or ROOT / "outputs" / (
        "handoff-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid4().hex[:8])
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=False)
    receipt = {"synthetic": True, "framework_root": str(ROOT),
               "checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "platform": platform.platform(),
               "interpreter": sys.executable, "checks": [], "passed": False}
    try:
        test = run_command(["-m", "unittest", "discover", "-s", "tests", "-v"],
                           target / "unittest.txt")
        match = re.search(r"Ran (\d+) tests?", test.stdout + test.stderr)
        count = int(match.group(1)) if match else 0
        receipt["checks"].append({"case": "unittest", "exit_code": test.returncode,
                                  "tests_run": count})
        expected = 85 + args.additional_tests
        receipt["handoff_tests_expected"] = 85
        receipt["additional_tests_expected"] = args.additional_tests
        require(test.returncode == 0 and count == expected,
                f"Expected {expected} passing tests (85 handoff + {args.additional_tests} additional); "
                f"got {count}, exit {test.returncode}")

        # Build and validate examples without the module's file-writing entrypoint.
        wire = run_command(["-c", "from examples.validate_contracts import build_examples; "
                            "assert len(build_examples()['prediction_cases']) == 8; "
                            "print('8 contract examples validated')"], target / "contracts.txt")
        receipt["checks"].append({"case": "contract_examples", "exit_code": wire.returncode})
        require(wire.returncode == 0, "Contract examples failed; see contracts.txt")

        cases = [("chunk", "mock_chunk_run.json", 0, 1, 0),
                 ("tool", "mock_tool_run.json", 0, 1, 0),
                 ("cache", "mock_cache_run.json", 0, 2, 0),
                 ("score-error", "mock_score_error_run.json", 1, 0, 1),
                 ("rejected", "mock_rejected_run.json", 2, None, None)]
        stats = {"chunk": (6, .30, 3, 0, 3, 7),
                 "tool": (5, .25, 2, 2, 2, 6),
                 "cache": (3, .15, 3, 0, 1, 4),
                 "score-error": (3, .15, 3, 0, 3, 4)}
        for name, config, expected_code, completed, errors in cases:
            destination = target / name
            result = run_command(["-m", "embodied_eval", "run", "--config",
                                  str(ROOT / "examples" / config), "--output", str(destination)],
                                 target / f"{name}.txt")
            check = {"case": name, "expected_exit_code": expected_code,
                     "actual_exit_code": result.returncode, "output": str(destination)}
            receipt["checks"].append(check)
            require(result.returncode == expected_code, f"{name}: unexpected exit {result.returncode}")
            if name == "rejected":
                require(not destination.exists(), "Preflight rejection unexpectedly created output")
                require("profile_mismatch" in result.stderr, "Expected chunk ownership rejection")
                print("rejected: expected exit 2, no output created")
                continue
            summary = json.loads((destination / "summary.json").read_text())
            require(summary["synthetic"] is True and summary["completed"] == completed
                    and summary["errors"] == errors, f"{name}: unexpected summary")
            check["episode_statistics"] = []
            for episode in summary["episodes"]:
                episode_dir = destination / "episodes" / episode["episode_id"]
                value = json.loads((episode_dir / "result.json").read_text())
                diagnostic = json.loads((episode_dir / "diagnostics.internal.json").read_text())
                actual = value["execution_statistics"]
                ticks, sim_time, calls, tools, forwards, observes = stats[name]
                require(actual["env_steps"] == ticks and math.isclose(actual["sim_time_s"], sim_time)
                        and actual["model_calls"] == calls and actual["tool_calls"] == tools
                        and actual["model_forward_count"] == forwards
                        and diagnostic["model"]["observe_count"] == observes,
                        f"{name}/{episode['episode_id']}: unexpected execution statistics")
                metric = value["metrics"]["success"]
                if name == "score-error":
                    require(metric["value"] is None and metric["valid"] is False
                            and metric["unavailable_reason"] == "scorer_failed", "Scoring failure semantics changed")
                else:
                    require(metric["value"] is False and metric["valid"] is True
                            and value["termination_reason"] == "budget_exhausted", "Mock outcome semantics changed")
                if name == "cache":
                    begin = next(event for event in diagnostic["model"]["events"]
                                 if event["event"] == "begin" and event["episode_id"] == episode["episode_id"])
                    require(begin["cache_at_begin"] == 0 and begin["history_at_begin"] == 0,
                            "Episode inherited model state")
                check["episode_statistics"].append({"episode_id": episode["episode_id"],
                                                     "statistics": actual})
            print(f"{name}: expected exit {expected_code}, baseline verified")
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        (target / "baseline-check.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(target)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
