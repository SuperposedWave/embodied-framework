"""Independent native LIBERO action replay, NOT fresh policy inference or acceptance.

The real framework's recorded actions are applied directly through the native
environment. Compare all raw state/image snapshots and model input mappings.
No BenchmarkAdapter or ModelAdapter reset/advance/predict code is reused.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def compare_snapshot(recorded, native):
    extension = recorded["extensions"]["libero.native_proprio.v1"]
    fields = {}
    for name, key in (("eef_position_m", "robot0_eef_pos"),
                      ("eef_quaternion_xyzw", "robot0_eef_quat"),
                      ("gripper_qpos_m", "robot0_gripper_qpos")):
        left, right = np.asarray(extension[name]), np.asarray(native[key])
        fields[name] = {"exact_equal": np.array_equal(left, right),
                        "max_abs_diff": float(np.max(np.abs(left - right)))}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        left = np.asarray(recorded["sensors"][key]["payload"]["data"], dtype=np.uint8)
        right = np.asarray(native[key])
        fields[key] = {"exact_equal": np.array_equal(left, right),
                      "different_scalar_values": int(np.count_nonzero(left != right)),
                      "native_pixel_sha256": hashlib.sha256(right.tobytes()).hexdigest()}
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from embodied_eval.config import load_config
    from embodied_eval.codec import to_data
    from embodied_eval.stage4_pairs import verify
    config = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
    verify(config)  # Source/blob verification before the native legacy pickle.
    manifest_path = args.framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["synthetic"] or manifest["config"] != to_data(config):
        raise ValueError("expected real dedicated pi05 LIBERO run with exact expanded configuration")
    source_config = manifest["config"]
    for section, keys in (("benchmark", ("adapter", "libero_repo")),
                          ("model", ("adapter", "openpi_repo", "checkpoint_dir"))):
        for key in keys:
            if source_config[section][key] != getattr(getattr(config, section), key):
                raise ValueError("source manifest identity differs from verified configuration")
    episode = config.episodes[0]
    if source_config["episodes"][0]["seed"] != episode.seed:
        raise ValueError("source manifest seed differs")
    package = Path(config.benchmark.libero_repo) / "libero/libero"
    rows = []
    with tempfile.TemporaryDirectory(prefix="embodied-eval-native-replay-") as setup_dir:
        paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                 "init_states": str(package / "init_files"), "datasets": str(package.parent.parent / "datasets"),
                 "assets": str(package / "assets")}
        (Path(setup_dir) / "config.yaml").write_text(yaml.safe_dump(paths))
        os.environ.update(LIBERO_CONFIG_PATH=setup_dir, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        sys.path.insert(0, config.benchmark.libero_repo)
        sys.path.insert(0, str(Path(config.model.openpi_repo) / "packages/openpi-client/src"))
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        from embodied_eval.adapters.openpi_libero_model import quat_to_axisangle
        import torch
        native_source = Path(config.model.openpi_repo) / "examples/libero/main.py"
        module = ast.parse(native_source.read_text())
        formula = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_quat2axisangle")
        namespace = {"np": np, "math": math}
        exec(compile(ast.Module(body=[formula], type_ignores=[]), str(native_source), "exec"), namespace)
        native_formula = namespace["_quat2axisangle"]
        np.random.seed(episode.seed)
        task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
        init_path = package / "init_files" / task.problem_folder / task.init_states_file
        states = torch.load(init_path, weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=str(package / "bddl_files" / task.problem_folder / task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        try:
            env.seed(episode.seed)
            env.reset()
            raw = env.set_init_state(states[0])
            for _ in range(10):
                raw, _, _, _ = env.step([0.0] * 6 + [-1.0])
            pending = None
            done, policy_steps = False, 0
            episode_dir = args.framework / "episodes" / episode.episode_id
            trace_path = episode_dir / "trace.internal.jsonl"
            with trace_path.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["event"] == "prediction":
                        pending = event["data"]["decision"]["steps"][0]["commands"][0]["parameters"]["osc_action7"]
                    elif event["event"] in ("reset", "step_result"):
                        observation = event["data"] if event["event"] == "reset" else event["data"]["observation"]
                        if event["event"] == "step_result":
                            if pending is None or done:
                                raise ValueError("invalid native replay command sequence")
                            raw, _, done, _ = env.step(pending)
                            pending = None
                            policy_steps += 1
                            if bool(event["data"]["terminated"]) != bool(done):
                                raise ValueError("native termination differs from framework")
                        fields = compare_snapshot(observation, raw)
                        quaternion = np.asarray(raw["robot0_eef_quat"]).copy()
                        fields["native_axisangle_formula"] = {"exact_equal": np.array_equal(
                            native_formula(quaternion.copy()), quat_to_axisangle(quaternion.copy()))}
                        # Public raw images to native model input: exact flip/pad/uint8.
                        for key in ("agentview_image", "robot0_eye_in_hand_image"):
                            left = np.asarray(observation["sensors"][key]["payload"]["data"], dtype=np.uint8)
                            native_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                                np.ascontiguousarray(raw[key][::-1, ::-1]), 224, 224))
                            mapped_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                                np.ascontiguousarray(left[::-1, ::-1]), 224, 224))
                            fields[key + "_model_input224"] = {"exact_equal": np.array_equal(mapped_image, native_image)}
                        rows.append({"env_step": policy_steps, "fields": fields})
            if pending is not None:
                raise ValueError("unexecuted command at end of source trace")
            result_path = episode_dir / "result.json"
            result = json.loads(result_path.read_text())
            expected_steps = result["execution_statistics"]["env_steps"]
            passed = (policy_steps == expected_steps and len(rows) == policy_steps + 1 and
                      bool(done) == result["metrics"]["success"]["value"] and
                      all(field["exact_equal"] for row in rows for field in row["fields"].values()))
            report = {"kind": "independent_native_recorded_action_replay_not_fresh_model_or_full_native_rollout",
                "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
                "synthetic": False, "model_server_called": False, "formal_acceptance": False,
                "native_benchmark_revision": manifest["space"]["revision"], "seed": episode.seed,
                "native_settle_steps": 10, "framework_policy_steps": expected_steps,
                "native_replayed_policy_steps": policy_steps, "compared_snapshots": len(rows),
                "native_success": bool(done), "framework_success": result["metrics"]["success"]["value"],
                "manifest_sha256": digest(manifest_path), "trace_sha256": digest(trace_path),
                "result_sha256": digest(result_path), "native_main_sha256": digest(native_source),
                "first_divergence": next((row["env_step"] for row in rows if not all(
                    value["exact_equal"] for value in row["fields"].values())), None), "rows": rows}
            (args.output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
            return 0 if passed else 1
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
