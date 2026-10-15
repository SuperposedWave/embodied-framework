"""Add only framework diagnostic dependencies to a NEW Python path overlay.

The historical native model environment is NOT modified. Native model, sampler
and CUDA packages still come from that pinned environment. No source is edited.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from prepare_robodojo_environment import bounded
from prepare_mn_native_environment import write, public_details
from download_stage4_assets import PROXY_KEYS

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    target = output / "python-dependencies"
    interpreter = WORKSPACE / "upstream/envs/openpi-stage4/bin/python"
    uv = WORKSPACE / "upstream/envs/downloader/bin/uv"
    receipt = {"kind": "framework_diagnostic_dependency_overlay_not_native_environment_edit",
        "passed": False, "path": str(target), "interpreter": str(interpreter), "attempts": [],
        "native_environment_modified": False, "global_proxy_modified": False, "model_loaded": False}
    for route in ("temporary_direct", "original_proxy_environment"):
        env = os.environ.copy()
        if route == "temporary_direct":
            for key in PROXY_KEYS:
                env.pop(key, None)
        code, _, stderr = bounded([str(uv), "pip", "install", "--python", str(interpreter), "--target", str(target),
            "--index-url", "https://pypi.org/simple", "jsonschema>=4.26,<5", "google-crc32c>=1.6"], env, 120)
        receipt["attempts"].append({"route": route, "exit_code": code,
            "public_details": public_details(stderr) if code != 0 else {}})
        if code == 0:
            receipt["passed"] = True
            break
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "native_environment_modified": False}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
