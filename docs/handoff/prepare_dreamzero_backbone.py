"""Download fixed native DreamZero auxiliary assets, not training data.

The pinned constructor still loads Wan2.1 components before overwriting with
DreamZero weights. Preserve that order rather than patching the native loader.
Publisher metadata is fetched direct-first, then proxy fallback. No raw logs,
credentials or redirect headers are saved. Existing final files never replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from download_stage4_assets import transfer, verify, PROXY_KEYS
from download_hf_ranges import transfer as ranged_transfer
from prepare_mn_native_environment import write

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
MODEL_ROOT = WORKSPACE.parent.parent / "Model/Wan2.1-I2V-14B-480P-DreamZero-runtime"


def metadata(url):
    attempts = []
    for route in ("temporary_direct", "original_proxy_environment"):
        env = os.environ.copy()
        if route == "temporary_direct":
            for key in PROXY_KEYS:
                env.pop(key, None)
        result = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
            "--connect-timeout", "15", "--max-time", "60", url], env=env, capture_output=True, timeout=70)
        attempts.append({"route": route, "exit_code": result.returncode})
        if result.returncode == 0:
            return json.loads(result.stdout), attempts
    raise RuntimeError("public_metadata_fetch_failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--manifest-from", type=Path, help="Resume the already-fixed manifest, never resolve a new main")
    parser.add_argument("--transport", choices=("ranges", "whole", "xet"), default="ranges")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--range-workers", type=int, default=4)
    parser.add_argument("--max-seconds-per-route", type=int, default=900)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    fixed = None
    if args.manifest_from:
        preparation = args.manifest_from.resolve()
        preparation.relative_to(ROOT / "outputs")
        fixed = json.loads((preparation / "manifest.json").read_text())
        if fixed["repository"] != REPO or fixed["revision"] != "6b73f84e66371cdfe870c72acd6826e1d61cf279":
            raise ValueError("source-fixed native backbone identity mismatch")
        head, first = {"sha": fixed["revision"]}, []
    else:
        head, first = metadata(f"https://huggingface.co/api/models/{REPO}")
    revision = head["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("fixed publisher revision required")
    info, second = metadata(f"https://huggingface.co/api/models/{REPO}/revision/{revision}?blobs=true")
    if info["sha"] != revision:
        raise ValueError("metadata revision mismatch")
    items = []
    for row in info["siblings"]:
        name = row["rfilename"]
        if not (name.startswith("diffusion_pytorch_model") or name.startswith("google/umt5-xxl/")
                or name in ("models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                            "Wan2.1_VAE.pth")):
            continue
        item = {"relative": name, "size": row["size"]}
        if "lfs" in row:
            item["sha256"] = row["lfs"]["sha256"]
        else:
            item["git_blob"] = row["blobId"]
        items.append(item)
    required = {"models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "Wan2.1_VAE.pth", "diffusion_pytorch_model.safetensors.index.json"}
    if not required <= {item["relative"] for item in items}:
        raise ValueError("source-required native assets absent from publisher metadata")
    if fixed and items != fixed["objects"]:
        raise ValueError("fixed publisher manifest changed")
    if args.max_seconds_per_route < 1 or not 1 <= args.workers <= 4 or not 1 <= args.range_workers <= 16:
        raise ValueError("bounded download duration required")
    manifest = {"kind": "models", "repository": REPO, "revision": revision, "objects": items,
                "reason": "source-pinned WANPolicyHead constructor loads components before full DreamZero checkpoint",
                "training_data_downloaded": False}
    write(args.output / "manifest.json", manifest)
    print(json.dumps({"repository": REPO, "revision": revision, "selected_files": len(items),
        "selected_bytes": sum(item["size"] for item in items)}), flush=True)
    if args.metadata_only:
        return 0
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (args.output / "object-results").mkdir()
    started = datetime.now(timezone.utc).isoformat()
    def download(item):
        request = dict(item, url=f"https://huggingface.co/{REPO}/resolve/{revision}/{quote(item['relative'], safe='/')}")
        try:
            if args.transport == "ranges" and item["size"] > 32 << 20 and "sha256" in item:
                result = ranged_transfer(MODEL_ROOT, request, args.range_workers, 8 << 20, args.max_seconds_per_route)
            else:
                result = transfer(MODEL_ROOT, request, args.max_seconds_per_route,
                    "hf_xet" if args.transport == "xet" and item["size"] > 32 << 20 else "curl")
        except Exception as error:
            # A single failure must not discard other completed transfers or
            # expose signed URLs/proxy credentials via arbitrary error text.
            result = {"file": item["relative"], "expected_bytes": item["size"],
                      "status": "failed", "error_type": type(error).__name__}
        # Save each completed object's bounded attempt immediately; a long
        # auxiliary download must not hide all evidence until the last shard.
        write(args.output / "object-results" / (item["relative"].replace("/", "__") + ".json"), result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(download, items))
    passed = all(row["status"] != "failed" for row in records)
    cache = args.output / "offline-hf-cache"
    if passed:
        # A private offline cache resolves the native constructor's unqualified
        # main lookup to this explicitly recorded fixed revision, never to a
        # changing remote main. Only checksum-verified selected files are linked.
        repo_cache = cache / "models--Wan-AI--Wan2.1-I2V-14B-480P"
        for item in items:
            source = MODEL_ROOT / item["relative"]
            verify(source, item)
            dest = repo_cache / "snapshots" / revision / item["relative"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(source)
        ref = repo_cache / "refs/main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        with ref.open("x") as stream:
            stream.write(revision)
    receipt = {"kind": "fixed_native_runtime_component_download_not_model_validation", "repository": REPO,
        "revision": revision, "directory": str(MODEL_ROOT), "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "selected_files": len(items), "selected_bytes": sum(item["size"] for item in items),
        "metadata_routes": first + second, "records": records, "global_proxy_modified": False,
        "model_loaded": False, "training_data_downloaded": False,
        "transport": args.transport, "max_seconds_per_route": args.max_seconds_per_route,
        "file_workers": args.workers, "range_workers_per_file": args.range_workers,
        "private_offline_cache": str(cache) if passed else None,
        "offline_main_alias_to_fixed_revision": revision if passed else None}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "selected_files", "selected_bytes", "private_offline_cache")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
