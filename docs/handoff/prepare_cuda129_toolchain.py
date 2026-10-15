"""Prepare a checksum-pinned PRIVATE CUDA 12.9 compiler, not a driver.

Only NVIDIA's NVCC, runtime headers and CCCL archives are downloaded. Official
release metadata and each complete archive must match the reviewed checksums.
No system package, global environment, historical toolchain or GPU execution.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import urllib.request
from download_stage4_assets import transfer
from probe_mn_components import ROOT, write, public_exception

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
METADATA = BASE + "redistrib_12.9.1.json"
PARTS = {
    "cuda_nvcc": ("12.9.86", 81100244, "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35"),
    "cuda_cudart": ("12.9.79", 1514676, "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99"),
    "cuda_cccl": ("12.9.27", 997888, "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa"),
}


def private_toolchain(path):
    """Require the helper-owned receipt and the entire published file inventory."""
    path = path.resolve()
    path.relative_to(ROOT / "outputs")
    proof = json.loads((path / "receipt.json").read_text())
    if not proof["passed"] or proof["metadata_source"] != METADATA or proof["cuda_version"] != "12.9":
        raise ValueError("verified_private_cuda129_toolchain_required")
    toolkit = path / "toolkit"
    if Path(proof["toolkit"]).resolve() != toolkit.resolve():
        raise ValueError("private_toolchain_location_mismatch")
    for item in proof["published_files"]:
        relative = PurePosixPath(item["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe_private_toolchain_inventory")
        candidate = toolkit / relative
        candidate.resolve().relative_to(path)
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("private_toolchain_inventory_hash_mismatch")
    return toolkit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    proof = {"kind": "private_cuda129_compile_only_toolchain_not_runtime_validation", "passed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "metadata_source": METADATA, "cuda_version": "12.9",
        "global_proxy_modified": False, "system_cuda_or_driver_modified": False, "gpu_execution_performed": False,
        "raw_logs_saved": False, "records": [], "metadata_routes": [], "published_files": []}
    try:
        data = None
        for route in ("temporary_direct", "original_proxy_environment"):
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if route == "temporary_direct" else urllib.request.build_opener()
                with opener.open(METADATA, timeout=30) as response:
                    raw = response.read()
                data = json.loads(raw)
                write(output / "publisher-metadata.json", data)
                proof["publisher_metadata_sha256"] = hashlib.sha256(raw).hexdigest()
                proof["metadata_routes"].append({"route": route, "passed": True})
                break
            except Exception as error:
                proof["metadata_routes"].append({"route": route, "exception_chain": public_exception(error)})
        if data is None:
            raise ValueError("official_cuda_release_metadata_unavailable")
        downloads = output / "archives"
        downloads.mkdir()
        toolkit = output / "toolkit"
        toolkit.mkdir()
        extracted = output / "extracted"
        extracted.mkdir()
        for name, (version, size, digest) in PARTS.items():
            part = data[name]
            item = part["linux-x86_64"]
            if part["version"] != version or item["sha256"] != digest or int(item["size"]) != size:
                raise ValueError("fixed_cuda_release_metadata_mismatch")
            relative = PurePosixPath(item["relative_path"])
            if relative.parts[:2] != (name, "linux-x86_64") or relative.name != f"{name}-linux-x86_64-{version}-archive.tar.xz":
                raise ValueError("reviewed_cuda_archive_path_required")
            record = transfer(downloads, {"relative": relative.name, "size": size, "sha256": digest, "url": BASE + str(relative)}, 300)
            proof["records"].append(dict(record, part=name, version=version))
            if record["status"] == "failed":
                raise ValueError("private_cuda_archive_download_or_hash_failed")
            archive_root = relative.name.removesuffix(".tar.xz")
            with tarfile.open(downloads / relative.name, "r:xz") as stream:
                for member in stream.getmembers():
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts or path.parts[0] != archive_root:
                        raise ValueError("unsafe_private_cuda_archive_member")
                stream.extractall(extracted, filter="data")
            source_root = extracted / archive_root
            # Licenses stay separately in their original extracted components;
            # only toolkit trees are assembled and no file is overwritten.
            for tree in ("bin", "include", "lib", "lib64", "nvvm", "targets"):
                source = source_root / tree
                if not source.is_dir():
                    continue
                for candidate in sorted(source.rglob("*")):
                    candidate.resolve().relative_to(source_root)
                    target = toolkit / candidate.relative_to(source_root)
                    if candidate.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif candidate.is_file():
                        digest_file = hashlib.sha256(candidate.read_bytes()).hexdigest()
                        if target.exists():
                            if hashlib.sha256(target.read_bytes()).hexdigest() != digest_file:
                                raise ValueError("conflicting_private_cuda_component_file")
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.symlink_to(candidate)
                        proof["published_files"].append({"file": str(target.relative_to(toolkit)), "sha256": digest_file})
        nvcc = toolkit / "bin/nvcc"
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, timeout=30, check=True)
        if b"release 12.9, V12.9.86" not in version.stdout:
            raise ValueError("private_cuda_compiler_version_mismatch")
        source = Path(__file__).with_name("cuda_toolchain_probe.cu")
        compile_result = subprocess.run([str(nvcc), "--cubin", "--gpu-architecture=sm_89", "-I", str(toolkit / "include"),
            str(source), "-o", str(output / "compile-only-probe.cubin")], capture_output=True, timeout=120)
        proof.update(toolkit=str(toolkit), nvcc_version="12.9.86", compile_only_probe_exit_code=compile_result.returncode,
            probe_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            passed=compile_result.returncode == 0)
        if compile_result.returncode == 0:
            proof["probe_cubin_sha256"] = hashlib.sha256((output / "compile-only-probe.cubin").read_bytes()).hexdigest()
    except Exception as error:
        proof["exception_chain"] = public_exception(error)
    proof["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "receipt.json", proof)
    print(json.dumps({key: proof.get(key) for key in ("passed", "nvcc_version", "toolkit", "compile_only_probe_exit_code")}))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
