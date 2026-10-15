"""Offline native LingBot import/processor gate, not checkpoint inference."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent / "upstream/lingbot-vla-v2"
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import flash_attn
    from transformers import AutoConfig, AutoProcessor
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    processor_root = ROOT.parent.parent.parent / "Model/Qwen3-VL-4B-Instruct-processor"
    config = AutoConfig.from_pretrained(processor_root, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(processor_root, local_files_only=True, trust_remote_code=False)
    uv = ROOT.parent / "upstream/envs/downloader/bin/uv"
    check = subprocess.run([str(uv), "pip", "check", "--python", sys.executable], capture_output=True, text=True)
    versions = {name: metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "tokenizers", "numpy", "flash-attn", "lerobot", "draccus")}
    report = {"kind": "native_lingbot_import_and_processor_only_not_model_forward",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "python": sys.version.split()[0], "versions": versions, "torch_cuda": torch.version.cuda,
              "torch_cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
              "native_policy_class_imported": LingbotVLAv2Server.__name__, "processor": type(processor).__name__,
              "config": type(config).__name__, "processor_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
              "dependency_metadata_check_passed": check.returncode == 0,
              "dependency_metadata_diagnostics": check.stdout + check.stderr,
              "metadata_note": "Native guide intentionally installs lerobot v0.4.2 --no-deps with the newer native Torch stack; upper bounds conflict. Preserve pinned native core and report conflicts, not a clean pip-check claim.",
              "checkpoint_loaded": False, "model_forward_performed": False, "scoring_performed": False}
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "dependency_metadata_diagnostics"}))


if __name__ == "__main__":
    main()
