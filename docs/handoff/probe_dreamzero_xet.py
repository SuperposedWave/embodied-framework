"""Bounded alternate transport for ONE fixed DreamZero auxiliary object.

No model runs, implicit HF token use, global proxy changes or raw logs. Both
routes reuse the same fixed revision/hash; partial caches are never readiness.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import quote
from download_stage4_assets import transfer
from prepare_dreamzero_backbone import MODEL_ROOT, REPO
from prepare_mn_native_environment import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.manifest_from / "manifest.json").read_text())
    revision = "6b73f84e66371cdfe870c72acd6826e1d61cf279"
    if manifest["repository"] != REPO or manifest["revision"] != revision:
        raise ValueError("fixed native auxiliary identity required")
    item = next(i for i in manifest["objects"] if i["relative"] == "diffusion_pytorch_model-00001-of-00007.safetensors")
    request = dict(item, url=f"https://huggingface.co/{REPO}/resolve/{revision}/{quote(item['relative'], safe='/')}")
    result = transfer(MODEL_ROOT, request, 120, "hf_xet")
    receipt = {"kind": "single_fixed_auxiliary_transport_probe_not_complete_backbone_or_runtime",
        "passed": result["status"] != "failed", "record": result, "global_proxy_modified": False,
        "model_loaded": False, "implicit_hf_token_use": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "status": result["status"]}), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
