"""Cancel one explicit, verified owned diagnostic simulator, not occupancy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
from probe_mn_components import ROOT, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.relative_to(ROOT / "outputs")
    if args.pid <= 1 or os.getpgid(args.pid) != args.pid:
        raise ValueError("independent_owned_process_group_required")
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes().split(b"\0")
    parts = [p.decode() for p in command if p]
    scripts = [str(ROOT / "docs/handoff" / name) for name in
        ("compare_native_robodojo_closedloop.py", "native_robodojo_reference_entry.py")]
    if not any(script in parts for script in scripts) or "--child" not in parts or "--output" not in parts or parts[parts.index("--output") + 1] != str(output):
        raise ValueError("exact_owned_simulator_and_output_required")
    if (output / "frames").exists() or (output / "native-exchanges").exists():
        raise ValueError("this_cancellation_helper_only_handles_verified_pre_policy_startup")
    evidence = {"kind": "owned_diagnostic_cancellation", "pid": args.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "reason": "native_reset_stalled_at_first_camera_CPU_Warp_launch",
        "fresh_policy_actions_executed": 0, "native_cleanup_acknowledged": False,
        "other_models_occupancy_driver_or_global_environment_changed": False, "formal_acceptance": False}
    environment = Path(f"/proc/{args.pid}/environ").read_bytes().split(b"\0")
    library = next((value.split(b"=", 1)[1].decode().split(":")[0] for value in environment
        if value.startswith(b"LD_LIBRARY_PATH=")), None)
    if library is not None:
        directory = Path(f"/proc/{args.pid}/cwd").resolve()
        evidence["private_graphics_runtime_path_at_cancellation"] = {
            "prefix": library, "is_absolute": Path(library).is_absolute(),
            "exists_relative_to_child_working_directory": (directory / library).is_dir()}
    if (output / "cancellation.json").exists():
        raise FileExistsError("cancellation_evidence_already_exists")
    os.killpg(args.pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while Path(f"/proc/{args.pid}").exists() and time.monotonic() < deadline:
        time.sleep(.25)
    evidence["forced_kill_required"] = Path(f"/proc/{args.pid}").exists()
    if evidence["forced_kill_required"]:
        os.killpg(args.pid, signal.SIGKILL)
    evidence["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "cancellation.json", evidence)
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
