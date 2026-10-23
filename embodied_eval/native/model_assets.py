"""Model-owned verification: no simulator/task metadata imports."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from ..core.validation import require
from .assets_common import _revision, _file, _hf_objects, _sha
from types import SimpleNamespace
from .model_catalog import model_options, PROFILES
ADAPTERS = Path(__file__).resolve().parents[1] / "adapters"
WEIGHT_FILES = {'eef10_osc': {'CHECKPOINT_SHA256': '34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a', 'NORMALIZATION_SHA256': 'f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e'}, 'eef10_absolute': {'CHECKPOINT_SHA256': '6352f34db37f0a44ff8262f4d401d85d2a7b036799d00c6402ff81363684eb5e', 'NORMALIZATION_SHA256': '6434b2f23e8406d224424b57cca5e13527777dc260de446ec2bdf4876c29af8f', 'CONFIG_SHA256': '6a70a66d7e6c0c360910ba60f47eae858cb25b6bb32d0c196c3e1fae8244ebc8'}, 'state8_osc': {'NORMALIZATION_FILE': 'assets/physical-intelligence/libero/norm_stats.json', 'NORMALIZATION_SHA256': 'b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'}, 'state7_absolute': {'NORMALIZATION_FILE': 'assets/vlabench/vlabench_ft_primitive/norm_stats.json', 'NORMALIZATION_SHA256': 'bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f'}}

def verify_model(family, options):
    cfg = model_options(family,options)
    profile = cfg.io_profile
    _revision(cfg.repo,PROFILES[profile][2])
    root = Path(cfg.checkpoint_dir).resolve()
    require(root.is_dir(),"asset_missing",str(root),"checkpoint directory absent")
    digests = {}
    if profile in ("eef10_osc","eef10_absolute"):
        spec = SimpleNamespace(**WEIGHT_FILES[profile])
        name = "checkpoint_step_10690.safetensors" if profile == "eef10_osc" else "checkpoint_step_6000.safetensors"
        digests[name] = _file(root/name,spec.CHECKPOINT_SHA256)
        digests["normalization_stats.npy"] = _file(root/"normalization_stats.npy",spec.NORMALIZATION_SHA256)
        if profile == "eef10_absolute":
            digests["config.yaml"] = _file(root/"config.yaml",spec.CONFIG_SHA256)
            for filename,digest in (("spiece.model","e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
                    ("tokenizer.json","6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b")):
                digests[filename]=_file(root/"tokenizer/google/umt5-xxl"/filename,digest)
    elif profile == "state8_osc":
        import google_crc32c
        manifest = ADAPTERS/"pi05_libero_objects.json"
        for item in json.loads(manifest.read_text())["objects"]:
            rel = item["name"].removeprefix("checkpoints/pi05_libero/")
            path = root/rel
            require(path.resolve().is_relative_to(root) and path.is_file() and path.stat().st_size == int(item["size"]),"asset_missing",rel,"missing GCS object")
            checksum,sha=google_crc32c.Checksum(),hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b""):
                    checksum.update(block); sha.update(block)
            require(base64.b64encode(checksum.digest()).decode()==item["crc32c"],"asset_hash_mismatch",rel,"pinned GCS checksum differs")
            digests[rel]=sha.hexdigest()
        spec = SimpleNamespace(**WEIGHT_FILES[profile])
        require(digests[spec.NORMALIZATION_FILE]==spec.NORMALIZATION_SHA256,"asset_hash_mismatch","normalization","matching normalization required")
        digests["manifest"]=_sha(manifest)
    elif profile == "arx_joint14":
        digests.update(_hf_objects(root, Path(__file__).parent / "asset_manifests/pi05_arx.json"))
        require(digests["assets/arx_x5_sim/norm_stats.json"] == PROFILES[profile][4], "asset_hash_mismatch", "normalization",
                "released ARX checkpoint and normalization must match")
    elif profile == "eef20_arx_absolute":
        digests.update(_hf_objects(root, Path(__file__).parent / "asset_manifests/openwam_arx.json"))
        require(digests["normalization_stats.npy"] == PROFILES[profile][4], "asset_hash_mismatch", "normalization",
                "released ARX step60000 checkpoint and normalization must match")
    else:
        manifests={"state7_absolute":"pi05_vlabench_objects.json","eef20_dual_legacy":"openwam_robotwin_objects.json",
            "eef20_dual_wxyz_encoded_absolute":"openwam_robotwin_objects.json","joint14":"lingbot_robotwin_objects.json"}
        digests.update(_hf_objects(root,ADAPTERS/manifests[profile]))
        if profile == "state7_absolute":
            spec = SimpleNamespace(**WEIGHT_FILES[profile])
            require(digests[spec.NORMALIZATION_FILE]==spec.NORMALIZATION_SHA256,"asset_hash_mismatch","normalization","matching normalization required")
        elif profile == "joint14":
            digests["normalization"]=_file(Path(cfg.repo)/"assets/norm_stats/robotwin.json",PROFILES[profile][4])
            digests["robot_mapping"]=_file(Path(cfg.repo)/"configs/robot_configs/robotwin.yaml","2c2a26509fc9662a8ada64adaa7fa959627dd6dadc6b48b680c4285da2d8ffd8")
        else:
            digests["normalization"]=_file(root/"normalization_stats.npy",PROFILES[profile][4])
    return {"source_revision":PROFILES[profile][2],"sha256":digests}
