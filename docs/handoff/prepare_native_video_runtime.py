"""Expose a RECORD-verified local FFmpeg for unchanged native video writers.

No system install or global PATH change. The binary is an already installed
imageio-ffmpeg dependency of our own DreamZero environment, not an arbitrary
download. Its x264 encoder is tested before creating an explicit private PATH.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from probe_mn_components import ROOT, WORKSPACE, write, public_exception


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    proof = {"kind": "private_native_ffmpeg_runtime_not_evaluation", "passed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "global_path_modified": False,
        "shared_packages_modified": False, "native_video_writer_modified": False, "raw_logs_saved": False}
    try:
        site = WORKSPACE / "upstream/envs/dreamzero-mn-native/lib/python3.11/site-packages"
        records = list(site.glob("imageio_ffmpeg-*.dist-info/RECORD"))
        if len(records) != 1:
            raise ValueError("one_installed_local_ffmpeg_distribution_required")
        candidates = [row for row in csv.reader(records[0].open())
            if row[0].startswith("imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-")]
        if len(candidates) != 1:
            raise ValueError("one_recorded_native_ffmpeg_binary_required")
        relative, digest, size = candidates[0]
        binary = (site / relative).resolve()
        binary.relative_to(site.resolve())
        actual = hashlib.sha256(binary.read_bytes()).digest()
        expected = "sha256=" + base64.urlsafe_b64encode(actual).decode().rstrip("=")
        if digest != expected or binary.stat().st_size != int(size):
            raise ValueError("installed_ffmpeg_record_hash_mismatch")
        version = subprocess.run([str(binary), "-version"], capture_output=True, timeout=30, check=True)
        if not version.stdout.startswith(b"ffmpeg version 7.0.2"):
            raise ValueError("reviewed_ffmpeg_version_required")
        encode = subprocess.run([str(binary), "-v", "error", "-f", "lavfi", "-i", "color=size=16x16:rate=1",
            "-frames:v", "1", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-f", "null", "-"],
            capture_output=True, timeout=30, check=True)
        directory = output / "bin"
        directory.mkdir()
        (directory / "ffmpeg").symlink_to(binary)
        proof.update(passed=True, binary=str(binary), binary_sha256=actual.hex(), binary_bytes=int(size),
            package_record=str(records[0]), package_record_sha256=hashlib.sha256(records[0].read_bytes()).hexdigest(),
            ffmpeg_version="7.0.2", libx264_encode_probe_exit_code=encode.returncode, private_bin=str(directory))
    except Exception as error:
        proof["exception_chain"] = public_exception(error)
    proof["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "receipt.json", proof)
    print(json.dumps({key: proof.get(key) for key in ("passed", "ffmpeg_version", "private_bin")}))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
