"""Fetch the official source-specified FlashAttention wheel with release digest."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from download_stage4_assets import transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory.name != "lingbot-native-wheels" or args.receipt.exists():
        raise ValueError("dedicated wheel directory and new receipt required")
    directory.mkdir(parents=True, exist_ok=True)
    name = "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    item = {"relative": name, "size": 256040057,
            "sha256": "f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb",
            "url": "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/" + name.replace("+", "%2B")}
    record = transfer(directory, item, 300)
    report = {"kind": "official_native_dependency_download_and_integrity_not_model_validation",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "release_metadata": "https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3",
              "passed": record["status"] != "failed", "global_proxy_modified": False, "record": record}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
