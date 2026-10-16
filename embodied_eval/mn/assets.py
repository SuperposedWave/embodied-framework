"""Independent local source/weight/scene verification, never pair factories."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from .._legacy_validation import require
from .assets_common import _revision, _file, _hf_objects, _sha
from .native_catalog import model_options, benchmark_options, PROFILES

ADAPTERS = Path(__file__).resolve().parents[1]/"adapters"


def verify_model(family, options):
    cfg = model_options(family,options)
    profile = cfg.io_profile
    _revision(cfg.repo,PROFILES[profile][2])
    root = Path(cfg.checkpoint_dir).resolve()
    require(root.is_dir(),"asset_missing",str(root),"checkpoint directory absent")
    digests = {}
    if profile in ("eef10_osc","eef10_absolute"):
        from ..adapters import openwam_libero_spec as ow, vlabench_spec as vla
        spec = ow if profile == "eef10_osc" else vla
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
        from ..adapters import openpi_libero_spec as spec
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
            from ..adapters import openpi_vlabench_spec as spec
            require(digests[spec.NORMALIZATION_FILE]==spec.NORMALIZATION_SHA256,"asset_hash_mismatch","normalization","matching normalization required")
        elif profile == "joint14":
            digests["normalization"]=_file(Path(cfg.repo)/"assets/norm_stats/robotwin.json",PROFILES[profile][4])
            digests["robot_mapping"]=_file(Path(cfg.repo)/"configs/robot_configs/robotwin.yaml","2c2a26509fc9662a8ada64adaa7fa959627dd6dadc6b48b680c4285da2d8ffd8")
        else:
            digests["normalization"]=_file(root/"normalization_stats.npy",PROFILES[profile][4])
    return {"source_revision":PROFILES[profile][2],"sha256":digests}


def verify_benchmark(family,options):
    cfg=benchmark_options(family,options)
    from .native_catalog import space
    spec=space(family,options)
    root=Path(cfg.repo).resolve()
    _revision(root,spec.revision)
    digests={}
    if family=="libero":
        from ..adapters.openwam_libero_spec import TASK_STEM
        for folder,suffix in (("bddl_files",".bddl"),("init_files",".pruned_init")):
            relative=f"libero/libero/{folder}/libero_spatial/{TASK_STEM}{suffix}"
            blob=subprocess.run(["git","-C",str(root),"show",f"{spec.revision}:{relative}"],capture_output=True)
            require(blob.returncode==0,"asset_missing",relative,"pinned task missing")
            digests[relative]=_file(root/relative,hashlib.sha256(blob.stdout).hexdigest())
    elif family=="vlabench":
        from ..adapters.vlabench_spec import TRACK_SHA256
        digests["track"]=_file(root/"VLABench/configs/evaluation/tracks/track_1_in_distribution.json",TRACK_SHA256)
        for name in ("assets/base/default.xml","assets/obj","assets/scenes"):
            require((root/"VLABench"/name).exists(),"asset_missing",name,"scene asset missing")
    else:
        for relative in ("task_config/demo_clean.yml","task_config/_embodiment_config.yml","task_config/_camera_config.yml"):
            blob=subprocess.run(["git","-C",str(root),"show",f"{spec.revision}:{relative}"],capture_output=True)
            require(blob.returncode==0,"asset_missing",relative,"pinned task config missing")
            digests[relative]=_file(root/relative,hashlib.sha256(blob.stdout).hexdigest())
        require((root/"assets/embodiments/aloha-agilex").is_dir(),"asset_missing","robotwin embodiments","robot asset missing")
    return {"source_revision":spec.revision,"sha256":digests}
