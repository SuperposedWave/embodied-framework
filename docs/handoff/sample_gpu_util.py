"""Sample all local GPU utilization values without modifying GPU settings."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--target-percent", type=float, default=15.0)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1 or args.interval_s <= 0 or not 0 <= args.target_percent <= 100:
        raise SystemExit("output must be new; samples and interval must be positive; target must be 0..100")
    rows = []
    for _ in range(args.samples):
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        devices = []
        for line in result.stdout.splitlines():
            index, util, memory = (int(part.strip()) for part in line.split(","))
            devices.append({"index": index, "utilization_percent": util, "memory_mib": memory})
        rows.append({"at_utc": datetime.now(timezone.utc).isoformat(), "devices": devices,
                     "eight_gpu_mean_percent": sum(item["utilization_percent"] for item in devices) / len(devices)})
        time.sleep(args.interval_s)
    means = {str(index): sum(row["devices"][index]["utilization_percent"] for row in rows) / len(rows)
             for index in range(len(rows[0]["devices"]))}
    aggregate = sum(row["eight_gpu_mean_percent"] for row in rows) / len(rows)
    result = {"metric": "time-mean of per-sample arithmetic mean over all 8 GPU utilization readings",
              "samples": len(rows), "interval_s": args.interval_s, "aggregate_mean_percent": aggregate,
              "aggregate_target_percent": args.target_percent,
              "aggregate_target_met": aggregate >= args.target_percent,
              "per_gpu_mean_percent": means, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"samples={len(rows)} aggregate_mean={aggregate:.2f}% target_met={aggregate >= args.target_percent}")


if __name__ == "__main__":
    main()
