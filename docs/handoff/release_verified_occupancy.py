"""Stop only selected, revalidated GPU-usage workers under existing authority.

Keep the controller and unselected workers alive. Never stop model/simulation
processes, infer ownership from process name alone, or persist command lines.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from prepare_mn_native_environment import write

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = Path("/inspire/hdd3/project/agent-driven-world-model/zhouzhixiang-240107010008/Misc/miaomiaoxiaogongjv/gpu_usage.py")


def query(arguments):
    value = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [tuple(part.strip() for part in line.split(",")) for line in value.stdout.splitlines()]


def parent(pid):
    return int(next(line.split()[1] for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("PPid:")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--gpu", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    if args.output.exists() or len(set(args.gpu)) != len(args.gpu) or not set(args.gpu) <= set(range(8)) or len(args.gpu) > 4:
        raise ValueError("new receipt and at most four explicit GPU workers required")
    command = Path(f"/proc/{args.parent_pid}/cmdline").read_bytes().split(b"\0")
    cwd = Path(f"/proc/{args.parent_pid}/cwd").resolve()
    scripts = [(cwd / value.decode()).resolve() for value in command if value.endswith(b".py")]
    if scripts != [EXPECTED] or hashlib.sha256(EXPECTED.read_bytes()).hexdigest() != "9f05dc4ff6391bd0714b08a2d2e542206fcf6834a5949d9422531337daa04ad9":
        raise ValueError("reviewed occupancy-controller source required")
    by_uuid = {uuid: int(index) for index, uuid in query(["--query-gpu=index,uuid"])}
    applications = {(by_uuid[uuid], int(pid)) for uuid, pid in query(["--query-compute-apps=gpu_uuid,pid"])}
    selected = [(gpu, pid) for gpu, pid in applications if gpu in args.gpu]
    if len(selected) != len(args.gpu) or set(gpu for gpu, _ in selected) != set(args.gpu) or any(parent(pid) != args.parent_pid for _, pid in selected):
        raise ValueError("selected GPUs must contain only one verified occupancy child each")
    identities = {pid: Path(f"/proc/{pid}/stat").read_text().split()[21] for _, pid in selected}
    record = {"kind": "authorized_selected_occupancy_workers_released_not_evaluation_work", "at_utc": datetime.now(timezone.utc).isoformat(),
        "controller_pid": args.parent_pid, "source_sha256": hashlib.sha256(EXPECTED.read_bytes()).hexdigest(),
        "selected": [{"gpu": gpu, "pid": pid, "signal": "SIGTERM"} for gpu, pid in sorted(selected)],
        "controller_or_unselected_workers_signalled": False, "raw_command_lines_saved": False}
    for _, pid in selected:
        if parent(pid) != args.parent_pid or Path(f"/proc/{pid}/stat").read_text().split()[21] != identities[pid]:
            raise ValueError("process identity changed before signalling")
        os.kill(pid, signal.SIGTERM)
    time.sleep(2)
    after = {(by_uuid[uuid], int(pid)) for uuid, pid in query(["--query-compute-apps=gpu_uuid,pid"])}
    record.update(selected_workers_absent_from_gpu=not any(item in after for item in selected),
        controller_still_alive=Path(f"/proc/{args.parent_pid}").is_dir(), remaining_gpu_processes=sorted(after))
    write(args.output, record)
    print(json.dumps(record))
    return 0 if record["selected_workers_absent_from_gpu"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
