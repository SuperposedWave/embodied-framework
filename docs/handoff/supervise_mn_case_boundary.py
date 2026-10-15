"""Stop THIS diagnostic supervisor after a completed case, not in an action.

Caller must supply its known owned supervisor PID and exact output argument.
No process search, occupancy killing, environment inspection or Runner changes.
The existing supervisor's finally block cleans only its launched process groups.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--supervisor-output-argument", required=True)
    parser.add_argument("--completed-summary", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.supervisor_pid <= 1:
        raise ValueError("known owned PID and new receipt required")
    proc = Path(f"/proc/{args.supervisor_pid}/cmdline")
    receipt = {"owned_supervisor_pid": args.supervisor_pid, "occupancy_process_changed": False,
               "runner_concurrency_or_recovery_added": False, "signal_sent": False}
    while proc.exists():
        argv = proc.read_bytes().decode().split("\0")
        if (not any(value.endswith("run_mn_trials.py") for value in argv)
                or "--child" in argv or "--output" not in argv
                or argv[argv.index("--output") + 1] != args.supervisor_output_argument):
            raise ValueError("PID no longer matches the explicitly owned diagnostic")
        if args.completed_summary.is_file():
            summary = json.loads(args.completed_summary.read_text())
            # Summary is written after both adapters close. Allow child stdout
            # and the supervisor's already-completed case receipt to flush.
            time.sleep(2)
            if proc.exists():
                latest = proc.read_bytes().decode().split("\0")
                if latest != argv:
                    raise ValueError("owned PID changed before signaling")
                os.kill(args.supervisor_pid, signal.SIGINT)
                receipt["signal_sent"] = True
            receipt.update(case_summary=str(args.completed_summary), completed=summary["completed"], errors=summary["errors"],
                           outcome="case_finished_then_owned_supervision_stopped_remaining_cases_validated_elsewhere")
            break
        time.sleep(2)
    receipt.setdefault("outcome", "owned_supervisor_already_finished")
    receipt["at_utc"] = datetime.now(timezone.utc).isoformat()
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
