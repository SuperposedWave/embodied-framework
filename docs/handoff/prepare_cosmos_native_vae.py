"""Fetch only the extra VAE pinned by the read-only Cosmos checkpoint registry."""
import argparse
from pathlib import Path
from urllib.parse import quote
from prepare_dreamzero_backbone import metadata
from download_hf_ranges import transfer as ranged_transfer
from download_stage4_assets import transfer
from prepare_mn_native_environment import write

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
NAME = "Wan2.2_VAE.pth"
DESTINATION = WORKSPACE.parent.parent / "Model/Cosmos3-native-Wan2.2-VAE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("ranges", "whole"), default="ranges")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from embodied_eval.native.assets_common import _revision
    _revision(WORKSPACE / "upstream/cosmos-framework-mn", "cf5d68c00d97ccd2480a2320ed652b92dec63102")
    source = (WORKSPACE / "upstream/cosmos-framework-mn/cosmos_framework/inference/common/checkpoints.py").read_text()
    if not all(value in source for value in (REPOSITORY, REVISION, NAME)):
        raise ValueError("pinned source asset declaration changed")
    info, routes = metadata(f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true")
    if info["sha"] != REVISION:
        raise ValueError("fixed revision mismatch")
    row = next(row for row in info["siblings"] if row["rfilename"] == NAME)
    item = {"relative": NAME, "size": row["size"], "sha256": row["lfs"]["sha256"]}
    write(args.output / "manifest.json", {"repository": REPOSITORY, "revision": REVISION, "objects": [item]})
    DESTINATION.mkdir(parents=True, exist_ok=True)
    print(f"selected native VAE: {item['size']} bytes", flush=True)
    request = dict(item, url=f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{quote(NAME)}")
    record = (ranged_transfer(DESTINATION, request, 4, 8 << 20, 120) if args.transport == "ranges"
              else transfer(DESTINATION, request, 180))
    passed = record["status"] != "failed"
    write(args.output / "receipt.json", {"kind": "fixed_native_auxiliary_asset_not_model_validation",
        "passed": passed, "directory": str(DESTINATION), "repository": REPOSITORY, "revision": REVISION,
        "metadata_routes": routes, "records": [record], "global_proxy_modified": False,
        "model_loaded": False, "shared_driver_modified": False})
    print(f"verified native VAE: {passed}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
