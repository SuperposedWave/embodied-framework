"""One bounded, fixed-revision persistent range; never an asset readiness claim."""
import argparse
from datetime import datetime, timezone
import json
from urllib.parse import quote
from pathlib import Path
from download_hf_ranges import chunk
from prepare_dreamzero_backbone import MODEL_ROOT, REPO
from probe_mn_components import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.manifest_from / "manifest.json").read_text())
    if manifest["repository"] != REPO or manifest["revision"] != "6b73f84e66371cdfe870c72acd6826e1d61cf279":
        raise ValueError("fixed native runtime manifest required")
    item = next(row for row in manifest["objects"] if row["relative"] == "diffusion_pytorch_model-00001-of-00007.safetensors")
    size = 8 << 20
    cache = MODEL_ROOT / ".persistent-http-ranges" / (item["sha256"] + f"-{size}")
    cache.mkdir(parents=True, exist_ok=True)
    index = next(i for i in range((item["size"] + size - 1) // size) if not (cache / f"{i:06d}.chunk").exists())
    request = dict(item, url=f"https://huggingface.co/{REPO}/resolve/{manifest['revision']}/{quote(item['relative'], safe='/')}")
    started = datetime.now(timezone.utc).isoformat()
    result = chunk(cache, request, index, size, 300)
    record = {"kind": "one_complete_http_range_not_complete_auxiliary_asset", "range_completed": result["status"] != "failed",
        "file": item["relative"], "revision": manifest["revision"], "result": result, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "global_proxy_modified": False,
        "native_model_loaded": False, "full_object_checksum_verified": False, "raw_headers_or_logs_saved": False}
    write(args.output / "receipt.json", record)
    print(json.dumps(record), flush=True)
    return 0 if record["range_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
