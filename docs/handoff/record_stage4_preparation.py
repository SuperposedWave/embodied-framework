"""Read-only inventory of source revisions, isolated environments and receipts.

No model import, GPU workload change, credentials, or global proxy inspection.
This evidence is not a benchmark evaluation or a lifetime GPU average.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(name):
    python = WORKSPACE / "upstream/envs" / name / "bin/python"
    code = ("import importlib.metadata as m,json,sys; names="
            "['numpy','torch','jax','jaxlib','orbax-checkpoint','websockets','msgpack',"
            "'mujoco','dm-control','open3d','scipy','robosuite','gym','openpi','openpi-client','google-crc32c',"
            "'sapien','mplib','warp-lang','nvidia-curobo','torchvision','transformers','flash-attn']; "
            "d={x.metadata['Name'].lower():x.version for x in m.distributions()}; "
            "print(json.dumps({'python':sys.version.split()[0],'package_count':len(d),"
            "'versions':{x:d.get(x) for x in names}}))")
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=30)
    return {"interpreter": str(python), "diagnostic_exit_code": result.returncode,
            "details": json.loads(result.stdout) if result.returncode == 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite historical evidence")
    sources = {}
    for name in ("OpenWAM", "openpi", "openpi-vlabench", "VLABench", "RoboTwin-openwam",
                 "RoboTwin-lingbot", "lingbot-vla-v2", "RoboDojo", "rrt-algorithms-stage4", "curobo-stage4"):
        path = WORKSPACE / "upstream" / name
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        clean = subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD"], capture_output=True)
        sources[name] = {"path": str(path), "revision": head.stdout.strip() if head.returncode == 0 else None,
                         "tracked_source_clean": clean.returncode == 0}
        if (path / "uv.lock").is_file():
            sources[name]["uv_lock_sha256"] = digest(path / "uv.lock")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    devices = []
    for row in query.stdout.splitlines():
        index, name, total, used, utilization, driver = [x.strip() for x in row.split(",")]
        devices.append({"index": int(index), "name": name, "total_memory_mib": int(total),
                        "used_memory_mib": int(used), "utilization_percent": int(utilization), "driver": driver})
    disk = shutil.disk_usage(WORKSPACE)
    receipts = {}
    for name in ("stage4-download-pi05_libero-20261006T0735Z.json",
                 "stage4-download-openwam_vlabench-20261006T0757Z-retry.json",
                 "stage4-download-pi05_vlabench-20261006T0823Z-retry.json",
                 "stage4-download-openwam_robotwin-20261006T1406Z.json",
                 "stage4-download-lingbot_robotwin-20261006T1406Z.json",
                 "stage4-download-robotwin-assets-20261006T1415Z.json",
                 "stage4-vlabench-mirror-20261006T1040Z-resume.json",
                 "stage4-download-robotwin-assets-20261006T1452Z-resume.json",
                 "stage4-download-lingbot-robotwin-20261006T1452Z-resume.json",
                 "stage4-download-openwam-robotwin-20261006T1452Z-resume.json",
                 "stage4-download-openwam-robotwin-20261006T1541Z-ranges.json",
                 "stage4-download-lingbot-robotwin-20261006T1514Z-xet.json",
                 "stage4-download-lingbot-robotwin-20261006T1541Z-ranges.json",
                 "stage4-download-robotwin-assets-20261006T1514Z-xet.json",
                 "stage4-download-robotwin-assets-20261006T1541Z-ranges.json",
                 "stage4-robotwin-objects-20261006T1547Z.json",
                 "stage4-robotwin-background-20261006T1554Z.json",
                 "stage4-download-qwen3-vl-processor-20261006T1505Z.json",
                 "stage4-cuda121-toolchain-20261006T1507Z.json",
                 "stage4-vlabench-mirror-20261006T0806Z.json",
                 "stage4-vlabench-mirror-20261006T0823Z-git-assisted.json"):
        path = ROOT / "outputs" / name
        record = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            value = json.loads(path.read_text())
            record.update(passed=value.get("passed"), sha256=digest(path),
                          expected_total_bytes=value.get("expected_total_bytes"),
                          finished_at_utc=value.get("finished_at_utc"))
        receipts[name] = record
    report = {"kind": "preparation_inventory_not_evaluation", "at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {"platform": platform.platform(), "architecture": platform.machine(), "cpu_count": os.cpu_count(),
                   "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")},
        "shared_disk": {"path": str(WORKSPACE), "total_bytes": disk.total, "free_bytes": disk.free,
                        "user_quota_verified": False},
        "gpus_snapshot_not_time_mean": devices, "gpu_workload_modified_this_inventory": False,
        "environments": {name: environment(name) for name in ("openpi-stage4", "libero-openpi-stage4", "openpi-vlabench-stage4",
                                                              "vlabench-stage4", "robotwin-stage4", "lingbot-stage4", "downloader")},
        "sources": sources, "download_receipts": receipts,
        "real_model_imported_or_loaded": False, "real_benchmark_ran": False, "xpolicylab_modified": False,
        "global_proxy_modified": False, "mac_historical_paths_modified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
