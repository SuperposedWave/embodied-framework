"""Prepare an isolated, checksummed NVIDIA CUDA compiler; no global CUDA edit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

PARTS = (
    ("cuda_nvcc", "12.1.105", 44459684, "0b85f7eee17788abbd170b0b493c74ce2e9fd5a9604461b99c2c378165e1083b"),
    ("cuda_cudart", "12.1.105", 984372, "6096ec878c8c443258d39c6e9cf2decef127f8aa8da594fdc5a336d047ab6bd9"),
    ("cuda_cccl", "12.1.109", 1024940, "b84ef3ec3dc1b4891267be25846f0c3ed7f9fa84154d59eba805402b86991baa"),
)
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    if root.name != "cuda-12.1-stage4" or args.receipt.exists():
        raise ValueError("expected dedicated new receipt and toolchain directory")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for name, version, size, digest in PARTS:
            filename = f"{name}-linux-x86_64-{version}-archive.tar.xz"
            target = root / filename
            routes = []
            if not target.exists():
                partial = root / (filename + ".partial")
                url = f"https://developer.download.nvidia.com/compute/cuda/redist/{name}/linux-x86_64/{filename}"
                for route in ("temporary_direct", "original_proxy_environment"):
                    environment = os.environ.copy()
                    if route == "temporary_direct":
                        for key in PROXIES:
                            environment.pop(key, None)
                    result = subprocess.run(["curl", "-sS", "-fL", "--http1.1", "--connect-timeout", "20",
                        "--max-time", "120", "--continue-at", "-", "--output", str(partial), url],
                        env=environment, capture_output=True)
                    routes.append({"route": route, "exit_code": result.returncode})
                    if result.returncode == 0:
                        break
                else:
                    raise RuntimeError("bounded_toolchain_transfer_failed_partial_preserved")
                if partial.stat().st_size != size or sha(partial) != digest:
                    raise ValueError("toolchain_checksum_mismatch")
                os.link(partial, target)
                partial.unlink()
            if target.stat().st_size != size or sha(target) != digest:
                raise ValueError("toolchain_checksum_mismatch")
            unpacked = root / filename.removesuffix(".tar.xz")
            if unpacked.exists():
                raise ValueError("refusing_to_reuse_unverified_extraction")
            with tarfile.open(target) as archive:
                archive.extractall(root, filter="data")  # Python3.13 safe path/link/device filter.
            if not unpacked.is_dir():
                raise ValueError("unexpected_toolchain_archive_root")
            published = 0
            for source in sorted(unpacked.rglob("*")):
                if source.is_dir():
                    continue
                resolved = source.resolve()
                if not resolved.is_relative_to(unpacked):
                    raise ValueError("toolchain_link_escapes_archive")
                destination = root / "toolkit" / source.relative_to(unpacked)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if sha(destination) != sha(source):
                        raise ValueError("conflicting_toolchain_component")
                    continue
                destination.symlink_to(resolved)
                published += 1
            records.append({"part": name, "version": version, "size": size, "sha256": digest,
                            "routes": routes, "published_files": published})
        version = subprocess.run([str(root / "toolkit/bin/nvcc"), "--version"],
                                 capture_output=True, text=True, check=True)
        report = {"passed": True, "nvcc_version_output": version.stdout, "toolkit": str(root / "toolkit")}
    except Exception as error:
        report = {"passed": False, "error_type": type(error).__name__,
                  "error_code": str(error) if isinstance(error, (ValueError, RuntimeError)) else "toolchain_operation_failed"}
    report.update(kind="isolated_cuda_build_toolchain_not_simulator_validation", records=records,
                  started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  global_proxy_modified=False, system_cuda_modified=False,
                  metadata_source="https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.1.1.json")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
