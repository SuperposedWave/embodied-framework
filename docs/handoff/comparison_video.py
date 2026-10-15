"""Passive observation videos: no render, inference or simulation calls.

PNG originals and acquisition clocks remain inspectable. Encoded video holds
the last available observation at 30 fps on the actual simulation-time axis;
it is not a continuously sampled physics camera or an equality test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Frames:
    def __init__(self, directory, cameras):
        self.directory = Path(directory)
        self.cameras = tuple(cameras)
        if (not self.cameras or len(set(self.cameras)) != len(self.cameras)
                or any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in self.cameras)):
            raise ValueError("unique_safe_camera_names_required")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows = []

    def append(self, images, step, sim_time, acquired_at=None):
        import numpy as np
        from PIL import Image
        if set(images) != set(self.cameras) or type(step) is not int or step < 0:
            raise ValueError("complete_camera_inventory_and_nonnegative_step_required")
        if not math.isfinite(sim_time) or sim_time < 0 or self.rows and sim_time < self.rows[-1]["sim_time_s"]:
            raise ValueError("finite_monotone_simulation_clock_required")
        if acquired_at is not None and set(acquired_at) != set(self.cameras):
            raise ValueError("complete_acquisition_clock_inventory_required")
        row = {"env_step": step, "sim_time_s": float(sim_time), "cameras": {}}
        validated = []
        for name in self.cameras:
            pixels = np.asarray(images[name])
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                raise ValueError("original_uint8_HWC_RGB_required")
            timestamp = sim_time if acquired_at is None else acquired_at[name]
            if not math.isfinite(timestamp) or not 0 <= timestamp <= sim_time + 1e-9:
                raise ValueError("camera_acquisition_clock_outside_episode")
            validated.append((name, pixels, timestamp))
        # Validate every camera before writing any part of the observation.
        for name, pixels, timestamp in validated:
            filename = f"frame-{len(self.rows):05d}-{name}.png"
            path = self.directory / filename
            Image.fromarray(pixels).save(path)
            row["cameras"][name] = {"file": filename, "shape": list(pixels.shape),
                "captured_at_s": float(timestamp), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "file_sha256": digest(path)}
        self.rows.append(row)

    def close(self):
        if not self.rows:
            raise ValueError("no_real_observations_recorded")
        path = self.directory / "frames.json"
        with path.open("x") as stream:
            json.dump({"kind": "passive_original_observation_frames", "cameras": self.cameras,
                "additional_get_obs_render_or_simulation_calls": 0, "frames": self.rows}, stream, indent=2)
            stream.write("\n")
        return path


def extract_framework(framework, output):
    import numpy as np
    manifest = json.loads((framework / "manifest.json").read_text())
    if manifest["synthetic"]:
        raise ValueError("real_framework_episode_required")
    episode = framework / "episodes" / manifest["tasks"][0]["episode_id"]
    recorder = None
    with (episode / "trace.internal.jsonl").open() as stream:
        for line in stream:
            if not line.startswith(('{"event":"reset",', '{"event":"step_result",')):
                continue
            event = json.loads(line)
            obs = event["data"] if event["event"] == "reset" else event["data"]["observation"]
            if recorder is None:
                recorder = Frames(output, tuple(obs["sensors"]))
            images = {key: np.asarray(value["payload"]["data"], np.uint8) for key, value in obs["sensors"].items()}
            recorder.append(images, obs["env_step"], obs["sim_time_s"],
                {key: value["captured_at_s"] for key, value in obs["sensors"].items()})
    if recorder is None:
        raise ValueError("real_reset_and_step_observations_missing")
    return recorder.close()


def load_frames(directory):
    data = json.loads((directory / "frames.json").read_text())
    if not data["frames"]:
        raise ValueError("empty_video_source")
    return data


def frame_at(rows, time_s):
    # A final frame may be held in the other side's longer video, explicitly
    # labeled FINISHED. Never stretch either episode to an equal action count.
    import bisect
    index = bisect.bisect_right([row["sim_time_s"] for row in rows], time_s) - 1
    return max(index, 0)


def encode(directories, output, ffmpeg, labels, fps=30):
    from PIL import Image, ImageDraw
    if output.exists() or len(directories) != len(labels) or not 1 <= fps <= 60:
        raise ValueError("new_video_and_explicit_labels_required")
    sets = [load_frames(directory) for directory in directories]
    tile_w, tile_h = 320, 240
    width = tile_w * max(len(data["cameras"]) for data in sets)
    panel_h = tile_h + 64
    height = panel_h * len(sets)
    end = max(data["frames"][-1]["sim_time_s"] for data in sets)
    count = max(1, math.ceil(end * fps) + 1)
    command = [str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
        f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    cached = [None] * len(sets)
    try:
        for number in range(count):
            time_s = number / fps
            canvas = Image.new("RGB", (width, height), (19, 23, 31))
            draw = ImageDraw.Draw(canvas)
            for side, (directory, data, label) in enumerate(zip(directories, sets, labels)):
                index = frame_at(data["frames"], time_s)
                row = data["frames"][index]
                if cached[side] is None or cached[side][0] != index:
                    tiles = []
                    for name in data["cameras"]:
                        info = row["cameras"][name]
                        path = directory / info["file"]
                        if digest(path) != info["file_sha256"]:
                            raise ValueError("original_video_frame_hash_changed")
                        with Image.open(path) as image:
                            tiles.append(image.convert("RGB").resize((tile_w, tile_h)))
                    cached[side] = (index, tiles)
                top = side * panel_h
                finished = " FINISHED (last frame held)" if time_s > data["frames"][-1]["sim_time_s"] else ""
                draw.text((8, top + 4), f"{label} | sim={time_s:.3f}s | action={row['env_step']}{finished}", fill="white")
                for column, (name, tile) in enumerate(zip(data["cameras"], cached[side][1])):
                    canvas.paste(tile, (column * tile_w, top + 64))
                    captured = row["cameras"][name]["captured_at_s"]
                    draw.text((column * tile_w + 8, top + 22), f"{name} | captured={captured:.3f}s", fill="white")
                draw.text((8, top + 42), "Observation recording; held frames are not new camera captures.", fill=(255, 204, 116))
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        errors = process.stderr.read()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("verified_local_ffmpeg_encode_failed")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    return {"file": output.name, "sha256": digest(output), "bytes": output.stat().st_size,
        "fps": fps, "encoded_frames": count, "simulation_time_axis_end_s": end,
        "native_pixels_compared_using_lossless_PNG_not_lossy_MP4": True,
        "continuous_physics_camera_capture": False, "frame_selection": "last_observation_at_actual_simulation_time"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--native-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    framework_frames = args.output / "framework-frames"
    extract_framework(args.framework, framework_frames)
    results = []
    for directories, filename, labels in (([framework_frames], "framework.mp4", ["EmbodiedEval"]),
            ([args.native_frames], "native.mp4", ["Native reference"]),
            ([framework_frames, args.native_frames], "comparison.mp4", ["EmbodiedEval", "Native reference"])):
        results.append(encode(directories, args.output / filename, args.ffmpeg, labels))
    with (args.output / "receipt.json").open("x") as stream:
        json.dump({"kind": "recorded_real_observation_comparison_videos_not_new_rollouts", "videos": results,
            "framework": str(args.framework), "native_frames": str(args.native_frames),
            "additional_model_or_environment_calls": 0}, stream, indent=2)
        stream.write("\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
