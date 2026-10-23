"""Nonblocking known-child stack sample; no locals, arguments or raw logs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    selector = p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--port")
    selector.add_argument("--required-arg", help="Exact owned config argument for a supervised runner, not arbitrary processes")
    p.add_argument("--expected-script", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    if args.output.exists():
        raise ValueError("new_capture_path_required")
    proc = Path("/proc") / str(args.pid)
    command = (proc / "cmdline").read_bytes().split(b"\0")
    required = args.port or args.required_arg
    if str(args.expected_script.absolute()).encode() not in command or required.encode() not in command:
        raise ValueError("exact_owned_script_and_port_required")
    binary = ROOT / "outputs/native-diagnostic-tools-20261023/py-spy/bin/py-spy"
    result = subprocess.run([str(binary), "dump", "--pid", str(args.pid), "--nonblocking", "--json"], capture_output=True, check=True, text=True, timeout=20)
    frames = json.loads(result.stdout)
    stacks = [{"pid": row["pid"], "thread_id": row["thread_id"], "active": row["active"],
               "frames": [{key: frame.get(key) for key in ("name", "filename", "line")} for frame in row["frames"]]} for row in frames]
    report = {"kind": "nonblocking_owned_startup_stack_sample_not_root_cause_proof", "pid": args.pid,
              "at_utc": datetime.now(timezone.utc).isoformat(), "locals_or_arguments_saved": False,
              "process_paused": False, "stacks": stacks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pid": args.pid, "top_frames": [row["frames"][0] if row["frames"] else None for row in stacks]}))


if __name__ == "__main__":
    main()
