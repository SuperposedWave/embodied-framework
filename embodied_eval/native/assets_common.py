"""Local integrity primitives; no model, simulator or pair dispatch."""
import hashlib
import json
from pathlib import Path
import subprocess
from ..core.validation import require


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(root,expected):
    root=Path(root).resolve()
    result=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True)
    require(result.returncode==0 and result.stdout.strip()==expected,"source_revision_mismatch",str(root),f"expected {expected}")
    clean=subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD"],capture_output=True)
    require(clean.returncode==0,"source_tree_modified",str(root),"pinned native sources must not be modified")


def _file(path,expected):
    require(path.is_file(),"asset_missing",str(path),"required native asset missing")
    actual=_sha(path)
    require(actual==expected,"asset_hash_mismatch",str(path),"asset differs from pinned SHA256")
    return actual


def _hf_objects(checkpoint,manifest):
    digests={}
    for item in json.loads(manifest.read_text())["objects"]:
        path=checkpoint/item["relative"]
        require(path.resolve().is_relative_to(checkpoint) and path.is_file() and path.stat().st_size==item["size"],
                "asset_missing",item["relative"],"pinned object missing or wrong size")
        if "sha256" in item:
            digests[item["relative"]]=_file(path,item["sha256"])
        else:
            blob=hashlib.sha1(f"blob {item['size']}\0".encode())
            sha=hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b""):
                    blob.update(block); sha.update(block)
            require(blob.hexdigest()==item["git_blob"],"asset_hash_mismatch",item["relative"],"pinned metadata git blob differs")
            digests[item["relative"]]=sha.hexdigest()
    digests["pinned_hf_object_manifest"]=_sha(manifest)
    return digests
