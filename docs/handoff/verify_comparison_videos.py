"""Read-only decode and original-pixel checks; never call a policy/simulator."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, nargs="+", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from PIL import Image
    runtime = json.loads((args.runtime / "receipt.json").read_text())
    binary = Path(runtime["binary"])
    if not runtime["passed"] or digest(binary) != runtime["binary_sha256"] or args.output.exists():
        raise ValueError("verified_private_decoder_and_new_output_required")
    proof = {"kind": "read_only_original_PNG_and_full_MP4_decode_verification", "passed": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": False,
        "model_or_environment_calls": 0, "formal_acceptance": False, "directories": []}
    for directory in args.directories:
        directory = directory.resolve(); directory.relative_to(ROOT / "outputs")
        receipt = json.loads((directory / "receipt.json").read_text())
        native = Path(receipt["native_frames"])
        if not native.is_absolute():
            native = ROOT / native
        record = {"directory": str(directory.relative_to(ROOT)), "PNG_files_verified": 0, "videos": []}
        for frames in (directory / "framework-frames", native):
            frames = frames.resolve(); frames.relative_to(ROOT / "outputs")
            inventory = json.loads((frames / "frames.json").read_text())
            last = -1.
            for row in inventory["frames"]:
                if row["sim_time_s"] < last or set(row["cameras"]) != set(inventory["cameras"]):
                    raise ValueError("complete_cameras_and_monotone_time_required")
                last = row["sim_time_s"]
                for camera in row["cameras"].values():
                    path = (frames / camera["file"]).resolve(); path.relative_to(frames)
                    with Image.open(path) as pixels:
                        if pixels.mode != "RGB" or list(pixels.size[::-1]) + [3] != camera["shape"]:
                            raise ValueError("original_RGB_shape_mismatch")
                        raw = pixels.tobytes()
                    if digest(path) != camera["file_sha256"] or hashlib.sha256(raw).hexdigest() != camera["pixel_sha256"]:
                        raise ValueError("original_PNG_or_pixels_hash_mismatch")
                    record["PNG_files_verified"] += 1
        for video in receipt["videos"]:
            path = (directory / video["file"]).resolve(); path.relative_to(directory)
            if digest(path) != video["sha256"] or path.stat().st_size != video["bytes"]:
                raise ValueError("MP4_file_hash_or_size_mismatch")
            result = subprocess.run([str(binary), "-v", "error", "-nostats", "-i", str(path),
                "-map", "0:v:0", "-progress", "pipe:1", "-f", "null", "-"], capture_output=True, text=True, timeout=300)
            counts = [int(line.partition("=")[2]) for line in result.stdout.splitlines() if line.startswith("frame=")]
            count = counts[-1] if counts else None
            if result.returncode or result.stderr.strip() or count != video["encoded_frames"]:
                raise ValueError("MP4_full_decode_or_frame_count_failed")
            record["videos"].append({"file": video["file"], "sha256": video["sha256"],
                "decoded_frames": count, "decoder_exit_code": result.returncode})
        proof["directories"].append(record)
        print(f"verified {directory.name}: {record['PNG_files_verified']} PNGs, {len(record['videos'])} videos", flush=True)
    proof["PNG_files_verified"] = sum(row["PNG_files_verified"] for row in proof["directories"])
    proof["MP4_files_fully_decoded"] = sum(len(row["videos"]) for row in proof["directories"])
    with args.output.open("x") as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2); stream.write("\n")


if __name__ == "__main__":
    main()
