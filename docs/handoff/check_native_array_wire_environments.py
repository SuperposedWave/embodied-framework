"""Synthetic lossless roundtrip in the actual RoboDojo/OpenWAM interpreters."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys

from probe_mn_components import ROOT, WORKSPACE, write
from native_array_wire import pack, unpack


def main():
    import numpy as np
    from importlib.metadata import version
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    if args.output.exists():
        raise ValueError("new_synthetic_transport_receipt_required")
    values = (np.arange(42, dtype=">f8").reshape(6, 7), np.arange(90, dtype=np.uint8).reshape(5, 6, 3)[::-1], np.array([1.01], np.float32))
    program = '''import sys,pickle
from importlib.metadata import version
sys.path.insert(0,sys.argv[1])
from native_array_wire import pack,unpack
received=pickle.loads(sys.stdin.buffer.read())
reply={"numpy":version("numpy"),"python":sys.version.split()[0],"arrays":[pack(unpack(x)) for x in received]}
sys.stdout.buffer.write(pickle.dumps(reply,protocol=4))
'''
    checks = []
    payload = pickle.dumps([pack(value) for value in values], protocol=4)
    for environment in ("robodojo-mn-native", "openwam_arx-mn-native"):
        python = WORKSPACE / "upstream/envs" / environment / "bin/python"
        reply = subprocess.run([str(python), "-c", program, str(ROOT / "docs/handoff")], input=payload,
            capture_output=True, timeout=60, check=True)
        parsed = pickle.loads(reply.stdout)  # Only our fixed trusted child code.
        actual = [unpack(value) for value in parsed["arrays"]]
        exact = all(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b) and a.tobytes() == b.tobytes() for a, b in zip(values, actual))
        checks.append({"environment": environment, "numpy": parsed["numpy"], "python": parsed["python"],
            "dtype_shape_byte_order_values_and_bytes_exact": exact, "array_count": len(actual)})
    record = {"kind": "synthetic_private_transport_not_real_inference", "at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(row["dtype_shape_byte_order_values_and_bytes_exact"] and row["array_count"] == len(values) for row in checks),
        "invoking_numpy": version("numpy"), "checks": checks, "public_contract_changed": False,
        "arrays_are_synthetic": True, "credentials_or_raw_child_logs_saved": False,
        "wire_helper_sha256": hashlib.sha256((ROOT / "docs/handoff/native_array_wire.py").read_bytes()).hexdigest()}
    write(args.output, record)
    print(json.dumps(record), flush=True)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
