"""Capture live owned comparison GPU identities, never arbitrary command/env logs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from probe_mn_components import ROOT, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    if args.output.exists():
        raise FileExistsError("new_hardware_receipt_required")
    gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30)
    gpu = {}
    for line in gpu_query.stdout.splitlines():
        index, uuid, name, driver, memory = (value.strip() for value in line.split(","))
        gpu[int(index)] = {"index": int(index), "uuid": uuid, "name": name, "driver": driver, "memory_total_MiB": int(memory)}
    compute_query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30)
    contexts = {}
    for line in compute_query.stdout.splitlines():
        pid, uuid, memory = (value.strip() for value in line.split(","))
        contexts.setdefault(int(pid), []).append({"gpu_uuid": uuid, "memory_used_MiB": int(memory)})
    processes = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            parts = [value.decode() for value in (item / "cmdline").read_bytes().split(b"\0") if value]
        except (OSError, UnicodeError):
            continue
        # Only three exact scripts owned by this workflow are retained.
        scripts = {"simulator": "native_robodojo_reference_entry.py", "OpenWAM_model": "native_openwam_arx_worker.py", "pi05_model": "serve_pi05_arx_native.py"}
        for role, filename in scripts.items():
            if str(ROOT / "docs/handoff" / filename) in parts:
                processes.append((int(item.name), role, parts))
    rows = []
    for run in args.runs:
        run = run.resolve(); run.relative_to(ROOT / "outputs")
        launch = json.loads((run / "native-launch-config.json").read_text())
        # GPU for rendering is taken from the exact owned simulator command.
        sim = [(pid, parts) for pid, role, parts in processes if role == "simulator"
            and "--output" in parts and parts[parts.index("--output")+1] == str(run)]
        if len(sim) != 1:
            raise ValueError("one_live_owned_simulator_required")
        sim_pid, sim_args = sim[0]; render_index = int(sim_args[sim_args.index("--gpu")+1])
        owned = [{"pid": sim_pid, "role": "simulator", "compute_contexts": contexts.get(sim_pid, [])}]
        for pid, role, parts in processes:
            if role == "OpenWAM_model" and "--output" in parts and parts[parts.index("--output")+1] == str(run / "native-exchanges"):
                owned.append({"pid": pid, "role": role, "compute_contexts": contexts.get(pid, [])})
            if role == "pi05_model" and "--port" in parts and int(parts[parts.index("--port")+1]) == launch["port"]:
                owned.append({"pid": pid, "role": role, "compute_contexts": contexts.get(pid, [])})
        rows.append({"run": str(run.relative_to(ROOT)), "model_gpu_identity": gpu[launch["model_gpu"]],
            "render_gpu_identity": gpu[render_index], "owned_live_processes": owned,
            "private_runtime_paths": json.loads((run / "private-runtime-paths.json").read_text()),
            "model_asset_verification_sha256": hashlib.sha256((run / "model-assets-verified.json").read_bytes()).hexdigest()})
    result = {"kind": "live_owned_comparison_hardware_identity_capture", "at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_is_after_startup_not_retroactive_launch_metadata": True, "comparisons": rows,
        "unrelated_process_arguments_or_environment_saved": False, "driver_proxy_or_occupancy_changed": False,
        "model_or_simulator_extra_calls": 0, "formal_acceptance": False}
    write(args.output, result)
    print(json.dumps({"runs": len(rows), "hardware_evidence_saved": True}))


if __name__ == "__main__":
    main()
