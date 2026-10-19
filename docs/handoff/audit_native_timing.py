"""Offline fixed-source timing audit; no weights, GPU, model or environment.

Only small selected metadata are read. Native DreamZero image assembly is
replayed on visibly synthetic images, not a model forward or real history.
Never copy full checkpoint YAML: it may contain irrelevant credential fields.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def checked_source(repo, revision, relative, markers=()):
    """Compare only the cited file to the pinned Git blob, not just HEAD."""
    repo = Path(repo)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError("source_revision_mismatch:" + str(repo))
    expected = subprocess.check_output(["git", "-C", str(repo), "show", revision + ":" + relative])
    actual = (repo / relative).read_bytes()
    if actual != expected:
        raise ValueError("selected_source_modified:" + relative)
    source = actual.decode("utf-8")
    lines = source.splitlines()
    locations = []
    for marker in markers:
        if marker not in source:
            raise ValueError("reviewed_source_anchor_missing:" + relative + ":" + marker)
        locations.append({"anchor": marker, "line": next(i for i, line in enumerate(lines, 1) if marker in line)})
    return {"repo": str(repo), "revision": revision, "relative": relative,
            "sha256": sha(actual), "pinned_git_blob_bytes_equal": True, "anchors": locations}


def source_files(family, profile):
    """Model-owned source only. No model/environment pair selection."""
    if family == "openwam":
        prefix = "policy/OpenWAM/OpenWAM/" if profile == "eef20_arx_absolute" else ""
        files = {
            "openwam/deploy/policy.py": ("conditions[\"first_frame_image\"] = [img]",),
            "openwam/deploy/executors/sync_executor.py": ("len(self._action_buffer) == 0", "actions[:inference_horizon]"),
            "configs/deploy.yaml": ("inference_horizon: null",),
        }
        if profile in ("eef10_osc", "eef10_absolute"):
            files["openwam/dataloader/bases/lerobot_v3_reader.py"] = (
                "np.arange(0, self._num_frames, self._video_stride", "T_action = self._num_frames - 1", "action_20d[:n_valid]")
            files["openwam/dataloader/" + ("libero.py" if profile == "eef10_osc" else "vlabench.py")] = ()
        elif profile == "eef20_arx_absolute":
            files["openwam/dataloader/robodojo.py"] = ("self.num_action_steps = self.num_frames - 1",)
        else:
            files["openwam/dataloader/robotwin.py"] = ("raw_actions[1 : self.num_frames]", "raw_frames[i] for i in self._video_sample_indices")
        return {prefix + name: markers for name, markers in files.items()}
    if family == "openpi":
        prefix = "policy/Pi_05/openpi/" if profile == "arx_joint14" else ""
        files = {"src/openpi/training/data_loader.py": ("t / dataset_meta.fps",)}
        if profile == "state8_osc":
            files["examples/libero/convert_libero_data_to_lerobot.py"] = ("fps=10", "for step in episode[\"steps\"]")
            files["examples/libero/main.py"] = ("replan_steps: int = 5",)
        elif profile == "state7_absolute":
            files["examples/vlabench/convert_to_lerobot.py"] = ("fps=10", "vlabench/vlabench_pretrain_primitive")
            files["src/openpi/training/config.py"] = ("vlabench/vlabench_ft_primitive", "pi05_ft_vlabench_primitive")
        return {prefix + name: markers for name, markers in files.items()}
    if family == "lingbot_vla":
        return {
            "lingbotvla/data/vla_data/base_dataset.py": ("offsets = [0, (self.chunk_size - 1) / fps]", "float(self.dataset_meta.fps) / float(max(1, self.chunk_size - 1))"),
            "tasks/vla/train_lingbotvla.py": ("future_video_effective_fps", "get_video_target("),
            "deploy/lingbot_vla_v2_policy.py": ("chunk_ret",),
        }
    if family == "cosmos3":
        return {
            "cosmos_framework/scripts/action_policy_server_robolab.py": ("_DEFAULT_CONDITIONING_FPS = 15.0", "video[:, 0]", "num_history_rows = self.cfg.history_length - use_state_rows"),
            "cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py": ("fps=15.0", "chunk_length=32"),
            "cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py": ("observation_ts = [i * self._dt", "action_ts = [i * self._dt"),
        }
    if family == "dreamzero":
        return {"socket_test_optimized_AR.py": ("FRAMES_PER_CHUNK = 4", "frames_to_use.insert(0, buffer[0])", "lazy_joint_forward_causal(batch)")}
    raise ValueError("unreviewed_family")


def metadata_check(path, manifest=None, expected_sha=None):
    payload = path.read_bytes()
    result = {"path": str(path), "sha256": sha(payload), "bytes": len(payload),
              "pin_status": "new_read_only_hash_only_not_preexisting_training_proof"}
    if manifest:
        entries = json.loads(manifest.read_text())["objects"]
        matches = [x for x in entries if path.as_posix().endswith("/" + x["relative"])]
        if len(matches) != 1:
            raise ValueError("metadata_manifest_path_missing_or_ambiguous")
        item = matches[0]
        if item["size"] != len(payload):
            raise ValueError("metadata_size_mismatch")
        if "sha256" in item:
            if sha(payload) != item["sha256"]:
                raise ValueError("metadata_manifest_sha256_mismatch")
        else:
            blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if blob != item["git_blob"]:
                raise ValueError("metadata_git_blob_mismatch")
        result.update(pin_status="existing_asset_manifest_matches", manifest=str(manifest), manifest_sha256=sha(manifest.read_bytes()))
    if expected_sha:
        if sha(payload) != expected_sha:
            raise ValueError("metadata_sha256_mismatch")
        result["pin_status"] = "existing_asset_pin_matches"
    return result


def replay_dreamzero_assembly(source):
    """Execute exactly one reviewed pure method, without upstream imports."""
    import numpy as np
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ARDroidRoboarenaPolicy")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_convert_observation")
    constant = next(n for n in cls.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FRAMES_PER_CHUNK" for t in n.targets))
    frame_count = ast.literal_eval(constant.value)
    if frame_count != 4 or method.decorator_list:
        raise ValueError("native_assembly_requires_new_review")
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<pinned-native-image-assembly>", "exec"), scope)
    keys = ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left")
    state = SimpleNamespace(_frame_buffers={key: [] for key in keys}, _is_first_call=True, FRAMES_PER_CHUNK=frame_count)
    records = []
    for index in range(1, 6):
        # Each view/time has a unique, obviously synthetic constant pixel ID.
        obs = {key: np.full((2, 3, 3), 10 * index + camera, dtype=np.uint8)
               for camera, key in enumerate(("observation/exterior_image_0_left", "observation/exterior_image_1_left", "observation/wrist_image_left"))}
        obs.update({"observation/joint_position": np.arange(7, dtype=np.float64),
                    "observation/gripper_position": np.array([.5]), "prompt": "synthetic source replay only"})
        converted = scope["_convert_observation"](state, obs)
        views = {key: {"pixel_ids": converted[key][:, 0, 0, 0].tolist(), "dtype": str(converted[key].dtype)} for key in keys}
        expected = [10 * index] if index == 1 else ([10] * max(0, 4 - index) + [10 * t for t in range(max(1, index - 3), index + 1)])
        for camera, key in enumerate(keys):
            if views[key]["pixel_ids"] != [x + camera for x in expected]:
                raise ValueError("native_assembly_replay_differs_from_review")
        records.append({"RPC_call": index, "views": views, "distinct_input_calls_in_external_0": len(set(expected))})
        state._is_first_call = False  # Native infer flips this AFTER forward.
    return {"kind": "pinned_native_pure_method_with_synthetic_images", "passed": True,
            "source_method_sha256": sha(ast.get_source_segment(source, method).encode()),
            "method": "ARDroidRoboarenaPolicy._convert_observation", "calls": records,
            "GPU_inference": False, "actual_model_forward": False, "timestamps_calibrated": False,
            "native_padding_observed_not_public_history_padding": True}


def audit_case(name, cfg):
    from embodied_eval.codec import to_data
    from embodied_eval.v03.registry import default_registry
    import yaml
    family, options = cfg.model.adapter, cfg.model.options
    cap = default_registry().model(family).describe(options)
    profile, repo, checkpoint = options["io_profile"], Path(options["repo"]), Path(options["checkpoint_dir"])
    sources = [checked_source(repo, cap.source_revision, path, markers)
               for path, markers in source_files(family, profile).items()]
    metadata, selected = [], {}
    manifests = ROOT / "embodied_eval/native/asset_manifests"
    adapters = ROOT / "embodied_eval/adapters"
    if family == "openwam":
        manifest = (manifests / "openwam_arx.json" if profile == "eef20_arx_absolute" else
                    adapters / "openwam_robotwin_objects.json" if profile.startswith("eef20_dual") else None)
        expected = None
        if profile == "eef10_absolute":
            from embodied_eval.adapters.vlabench_spec import CONFIG_SHA256
            expected = CONFIG_SHA256
        metadata.append(metadata_check(checkpoint / "config.yaml", manifest, expected))
        config = yaml.safe_load((checkpoint / "config.yaml").read_text())
        selected = {k: config["dataloader"][k] for k in ("num_frames", "video_stride")}
        if selected != {"num_frames": 33, "video_stride": 4}:
            raise ValueError("openwam_training_window_changed")
    elif family == "lingbot_vla":
        path = checkpoint / "lingbotvla_cli.yaml"
        metadata.append(metadata_check(path, adapters / "lingbot_robotwin_objects.json"))
        config = yaml.safe_load(path.read_text())
        selected = {"chunk_size": config["train"]["chunk_size"],
                    "teacher_effective_fps_default": config["train"]["align_params"]["video"]["effective_fps"]}
        if selected != {"chunk_size": 50, "teacher_effective_fps_default": 1.0}:
            raise ValueError("lingbot_training_config_changed")
    elif family == "cosmos3":
        path = checkpoint / "checkpoint.json"
        metadata.append(metadata_check(path, manifests / "cosmos3.json"))
        selected = {"checkpoint_json_empty": json.loads(path.read_text()) == {}}
        if not selected["checkpoint_json_empty"]:
            raise ValueError("new_training_metadata_requires_review")
    elif family == "dreamzero":
        path = checkpoint / "experiment_cfg/conf.yaml"
        metadata.append(metadata_check(path, manifests / "dreamzero.json"))
        config = yaml.safe_load(path.read_text())
        modality = config["train_dataset"]["all_modality_configs"]["oxe_droid"]
        selected = {"training_video_indices": modality["video"]["delta_indices"],
                    "eval_video_indices": modality["video"]["eval_delta_indices"],
                    "action_indices": modality["action"]["delta_indices"]}
        if selected != {"training_video_indices": list(range(25)), "eval_video_indices": [0], "action_indices": list(range(24))}:
            raise ValueError("dreamzero_modality_indices_changed")
    if cap.temporal.native_action_period_s is not None or any(r.history is not None for r in cap.inputs):
        raise ValueError("unreviewed_new_physical_timing_or_history")
    return {"configuration": name, "family": family, "profile": profile,
            "checkpoint_ref": cap.checkpoint_ref, "source_revision": cap.source_revision,
            "timing_declaration": to_data(cap.temporal), "sources": sources,
            "small_weight_metadata": metadata, "selected_metadata": selected,
            "matching_training_dataset_verified": False, "live_server_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"kind": "offline_native_timing_source_and_metadata_audit", "passed": False,
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
               "new_GPU_inference": False, "new_real_simulation": False, "platform_jobs": 0,
               "weight_tensors_rehashed": False, "action_resampling": False,
               "inference_schedule_changed": False, "formal_acceptance": False}
    receipt["framework_source_sha256"] = {name: sha((ROOT / name).read_bytes()) for name in (
        "embodied_eval/native/timing_profiles.py", "embodied_eval/v03/models.py",
        "docs/handoff/audit_native_timing.py", "tests/test_native_timing_profiles.py")}
    try:
        from tests.v03_helpers import cases
        configs = cases()
        for name, cfg in configs.items():
            receipt["cases"].append(audit_case(name, cfg))
        dream = next(cfg for cfg in configs.values() if cfg.model.adapter == "dreamzero")
        source = (Path(dream.model.options["repo"]) / "socket_test_optimized_AR.py").read_bytes()
        checked = next(x for x in receipt["cases"] if x["family"] == "dreamzero")["sources"][0]
        if sha(source) != checked["sha256"]:
            raise ValueError("source_changed_during_audit")
        result = replay_dreamzero_assembly(source.decode())
        (output / "dreamzero-native-assembly-synthetic.json").write_text(json.dumps(result, indent=2) + "\n")
        receipt["dreamzero_pure_assembly_replay"] = "dreamzero-native-assembly-synthetic.json"
        receipt["unique_weight_io_profiles"] = len({(x["family"], x["profile"]) for x in receipt["cases"]})
        receipt["all_native_action_periods_still_unknown"] = True
        if any(sha((ROOT / name).read_bytes()) != digest for name, digest in receipt["framework_source_sha256"].items()):
            raise ValueError("framework_source_changed_during_audit")
        receipt["passed"] = len(receipt["cases"]) == 13
    except Exception as error:
        receipt["failure_type"] = type(error).__name__
        # Controlled audit failures only; no raw upstream logs/config content.
        if isinstance(error, ValueError):
            receipt["failure"] = str(error)
        raise
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"passed": receipt["passed"], "cases": len(receipt["cases"]), "receipt": str(output / "receipt.json")}))


if __name__ == "__main__":
    main()
