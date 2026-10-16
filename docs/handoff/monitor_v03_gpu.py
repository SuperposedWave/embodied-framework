"""Finite workflow GPU sampling; stop only via a scoped stop marker or interrupt.

Does not start/stop workloads or save process commands/environment variables.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.stop_file.exists():
        raise ValueError('new output and absent stop marker required')
    rows = []
    try:
        while not args.stop_file.exists():
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            devices = [list(map(int, line.split(','))) for line in result.stdout.splitlines()]
            rows.append(dict(at_utc=datetime.now(timezone.utc).isoformat(), devices=devices, all_GPU_mean_percent=sum(v[1] for v in devices) / len(devices)))
            time.sleep(5)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mean = sum(row['all_GPU_mean_percent'] for row in rows) / len(rows) if rows else None
        with args.output.open('x') as stream:
            json.dump(dict(samples=rows, aggregate_mean_percent=mean, target_percent=20, target_met=mean is not None and mean >= 20,
                includes_preexisting_occupancy_not_evaluation_work=True, whole_project_lifetime_average=False), stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
