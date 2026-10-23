"""Environment-owned verification: no model/weight metadata imports."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from ..core.validation import require
from .assets_common import _revision, _file, _hf_objects, _sha
from .benchmark_catalog import benchmark_options

def verify_benchmark(family,options):
    cfg=benchmark_options(family,options)
    from .benchmark_catalog import space
    spec=space(family,options)
    root=Path(cfg.repo).resolve()
    _revision(root,spec.revision)
    digests={}
    if family=="libero":
        from .environment_specs.libero_30 import TASK_STEM
        for folder,suffix in (("bddl_files",".bddl"),("init_files",".pruned_init")):
            relative=f"libero/libero/{folder}/libero_spatial/{TASK_STEM}{suffix}"
            blob=subprocess.run(["git","-C",str(root),"show",f"{spec.revision}:{relative}"],capture_output=True)
            require(blob.returncode==0,"asset_missing",relative,"pinned task missing")
            digests[relative]=_file(root/relative,hashlib.sha256(blob.stdout).hexdigest())
    elif family=="vlabench":
        from .environment_specs.vlabench import TRACK_SHA256
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
