"""CPU-only pinned native-function comparisons, using synthetic public inputs.

No checkpoint loaded, policy inference, simulator, or official score. Extract
reviewed pure functions/class methods without executing upstream policy imports.
"""
import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def functions(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("native_function_set_changed")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {n.name: hashlib.sha256(ast.get_source_segment(path.read_text(), n).encode()).hexdigest() for n in nodes}


def native_class(path, classname, names, scope):
    source = path.read_text()
    cls = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == classname)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != set(names):
        raise ValueError("native_method_set_changed")
    shell = ast.ClassDef(name=classname, bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[])), str(path), "exec"), scope)
    return scope[classname].__new__(scope[classname])


def same(expected, actual):
    import numpy as np
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(expected) == set(actual) and all(same(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, np.ndarray):
        return isinstance(actual, np.ndarray) and expected.dtype == actual.dtype and np.array_equal(expected, actual)
    return expected == actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    import numpy as np
    import torch
    import torch.nn.functional as F
    from embodied_eval.core import types as leaf
    from embodied_eval.native import contracts as c
    from embodied_eval.native.additional_catalog import POLICIES, OPENPI_REVISION, ROBOLAB_REVISION
    from embodied_eval.native.additional_models import JointPolicyModel
    from embodied_eval.native.assets_common import _revision
    from tests.legacy_support.frozen.mn.binding import resolve
    from embodied_eval.native.droid_simulator import space as droid_space, DroidSimulator
    from embodied_eval.native.model_catalog import PROFILES
    from embodied_eval.native.native_models import NativeActionModel
    from embodied_eval.native.robodojo_simulator import space as dojo_space, RoboDojoSimulator, extract_environment_class, REVISION
    from embodied_eval.adapters.native_common import native_module
    upstream = ROOT.parent / "upstream"
    repos = {"cosmos3": upstream / "cosmos-framework-mn", "dreamzero": upstream / "dreamzero-mn",
        "robolab": upstream / "robolab-mn", "openpi": upstream / "openpi", "pi05_arx": upstream / "xpolicy-reference-mn",
        "robodojo": upstream / "RoboDojo"}
    revisions = {"cosmos3": POLICIES["cosmos3"]["source_revision"], "dreamzero": POLICIES["dreamzero"]["source_revision"],
        "robolab": ROBOLAB_REVISION, "openpi": OPENPI_REVISION, "pi05_arx": PROFILES["arx_joint14"][2], "robodojo": REVISION}
    for name, repo in repos.items():
        _revision(repo, revisions[name])
    images = native_module(str(repos["openpi"]), "openpi_client.image_tools", "packages/openpi-client/src")
    robolab_images = native_module(str(repos["robolab"]), "robolab.core.utils.image_utils")
    context = c.Context(run_id="cpu-source-probe", env_id="synthetic", episode_id="ep", message_id="request")
    rng = np.random.default_rng(317)
    rgb = {key: rng.integers(0, 256, (91, 157, 3), dtype=np.uint8) for key in ("exterior_right", "exterior_left", "wrist")}
    joints = np.linspace(-.3, .4, 7, dtype=np.float32)
    closed = np.array([.8], np.float32)
    def sample(array, sensor=False):
        tensor = leaf.Tensor(dtype=str(array.dtype), shape=array.shape, layout="HWC" if sensor else "vector", data=array.tolist())
        return leaf.SensorSample(frame_id="world", captured_at_s=0., validity="valid", provenance="sensor", payload=tensor) if sensor else c.StateSample(captured_at_s=0., payload=tensor)
    view = c.ModelInput(context=context, observation_id="obs", sim_time_s=0., task_public=leaf.PublicTask(instruction="synthetic source comparison"),
        sensors={k: sample(v, True) for k, v in rgb.items()}, states={"joints": sample(joints.astype(np.float64)), "grip": sample(1. - closed.astype(np.float64))})
    droid = droid_space({"repo": "/absent", "assets_dir": "/absent"})
    binding = c.BindingConfig(inputs={"exterior_right": "external_cam", "exterior_left": "external_cam_2", "wrist": "wrist_cam",
        "joints": "arm_joint_positions", "grip": "gripper_open_fraction", "instruction": "instruction"},
        targets={**{f"j{i}": f"robot0/j{i}" for i in range(7)}, "grip": "robot0/grip"}, action_spec_ref="joint7-binary-gripper@1")
    models = {}
    for family in POLICIES:
        model = JointPolicyModel(family, {"repo": str(repos[family]), "codec_repo": str(repos["openpi"]), "checkpoint_dir": "/absent",
            "client_repo": str(repos["robolab"]), "io_profile": POLICIES[family]["profile"]})
        model.profile, model.space = resolve(droid, model.cap, binding), droid
        model.images, model.session_id = (robolab_images if family == "cosmos3" else images), "fixed-source-probe"
        models[family] = model
    scope = {"np": np, "torch": torch, "F": F}
    cosmos = native_class(repos["robolab"] / "policies/cosmos3/client.py", "Cosmos3Client", ("_pack_request", "_postprocess_chunk"), scope)
    cosmos._image_h, cosmos._image_w = 360, 640
    expected = cosmos._pack_request({"left_image": robolab_images.resize_with_pad(rgb["exterior_left"], 360, 640),
        "right_image": robolab_images.resize_with_pad(rgb["exterior_right"], 360, 640), "wrist_image": robolab_images.resize_with_pad(rgb["wrist"], 360, 640),
        "joint_position": joints, "gripper_position": closed, "session_id": "fixed-source-probe"}, view.task_public.instruction)
    results = [{"component": "cosmos3", "check": "native_input_pixels_order_state_dtype", "passed": same(expected, models["cosmos3"].native_input(view))}]
    rows = rng.normal(size=(32, 8)).astype(np.float32)
    rows[:4, -1] = [.5, .50001, -.1, 1.1]
    executor = DroidSimulator.__new__(DroidSimulator)
    executor.space = droid
    native_rows = cosmos._postprocess_chunk(rows)
    actual = np.stack([executor.native_action(models["cosmos3"].decode_action(row, context)) for row in rows])
    results.append({"component": "cosmos3", "check": "32_native_actions_and_gripper_threshold_dtype", "passed": same(native_rows, actual)})
    dream = native_class(repos["dreamzero"] / "eval_utils/run_sim_eval.py", "DreamZeroJointPosClient", ("infer", "_extract_observation"),
        {"np": np, "image_tools": images})
    dream.actions_from_chunk_completed, dream.open_loop_horizon, dream.session_id = 0, 8, "fixed-source-probe"
    sent = []
    def native_request(value):
        value["endpoint"] = "infer"
        sent.append(value)
        return {"actions": rows}
    dream.client = SimpleNamespace(infer=native_request)
    raw = {"policy": {**{key: torch.from_numpy(rgb[slot])[None] for slot, key in (
        ("exterior_right", "external_cam"), ("exterior_left", "external_cam_2"), ("wrist", "wrist_cam"))},
        "arm_joint_pos": torch.from_numpy(joints), "gripper_pos": torch.from_numpy(closed)}}
    with redirect_stdout(io.StringIO()):
        native_rows = np.stack([dream.infer(raw, view.task_public.instruction)["action"] for _ in range(8)])
    results.append({"component": "dreamzero", "check": "native_input_image_order_pixels_dtype_endpoint", "passed": same(sent[0], models["dreamzero"].native_input(view))})
    from prepare_mn_extension import configurations
    examples = configurations()
    # Carry distinct fixture pixels all the way from environment camera IDs
    # through emitted wiring. The previous comparison tested model slots only.
    assert examples["dreamzero_droid_sim"].binding.inputs["exterior_right"] == "external_cam"
    assert examples["dreamzero_droid_sim"].binding.inputs["exterior_left"] == "external_cam_2"
    cosmos_inputs = examples["cosmos3_droid_sim"].binding.inputs
    assert cosmos_inputs["exterior_right"] == "external_cam_2" and cosmos_inputs["exterior_left"] == "external_cam"
    actual = np.stack([executor.native_action(models["dreamzero"].decode_action(row, context)) for row in rows[:8]])
    results.append({"component": "dreamzero", "check": "8_native_actions_float64_promotion_and_threshold", "passed": same(native_rows, actual)})
    scope = {"np": np, "Any": Any}
    hashes = functions(repos["pi05_arx"] / "utils/process_data.py", ("_validate_config", "_get_state_keys", "_ensure_valid_state_array", "pack_robot_state", "unpack_robot_state"), scope)
    hashes.update(functions(repos["pi05_arx"] / "policy/Pi_05/model.py", ("encode_obs", "extract_image", "ensure_chw_uint8"), scope))
    dims = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    dual_rgb = dict(zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), rgb.values()))
    dual_state = {side + suffix: rng.normal(size=width) if width == 6 else np.array([.5]) for side in ("left", "right")
        for suffix, width in (("_arm_joint_state", 6), ("_ee_joint_state", 1))}
    raw = {"vision": {key: {"color": img} for key, img in dual_rgb.items()}, "state": dual_state, "instruction": view.task_public.instruction}
    dual_view = c.ModelInput(context=context, observation_id="obs", sim_time_s=0., task_public=view.task_public,
        sensors={key: sample(img, True) for key, img in zip(("head", "left_wrist", "right_wrist"), dual_rgb.values())},
        states={side + suffix: sample(dual_state[side + source]) for side in ("left", "right")
            for suffix, source in (("_joints", "_arm_joint_state"), ("_grip", "_ee_joint_state"))})
    pi = NativeActionModel("openpi", {"repo": str(repos["pi05_arx"]), "checkpoint_dir": "/absent", "io_profile": "arx_joint14"})
    dojo = RoboDojoSimulator({"repo": str(repos["robodojo"])})
    pi.space = dojo.space
    pi.profile = resolve(dojo.space, pi.cap, c.BindingConfig(inputs={"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist", "instruction": "instruction",
        **{side + suffix: side + source for side in ("left", "right") for suffix, source in (("_joints", "_joint_positions"), ("_grip", "_gripper_target"))}},
        targets={key: "robot0/" + key for key in pi.cap.output.target_slots}, action_spec_ref="dual-joint-raw-gripper@1"))
    results.append({"component": "pi05_arx", "check": "native_encode_obs_state_arm_order_CHW_dtype", "passed": same(scope["encode_obs"](raw, "joint", dims), pi.native_input(dual_view))})
    dual_rows = rng.normal(size=(50, 14)).astype(np.float32)
    expected = scope["unpack_robot_state"](dual_rows, "joint", dims, source_type="obs")
    actual = [dojo.native_action(pi.decode_action(row, context, dual_view)) for row in dual_rows]
    results.append({"component": "pi05_arx", "check": "all_50_native_dual_action_dicts_dtype_and_raw_grippers", "passed": all(same(e, a) for e, a in zip(expected, actual))})
    cls, dojo_hashes = extract_environment_class((repos["robodojo"] / "src/eval_client/eval_env.py").read_text(), object,
        {"np": np, "deepcopy": deepcopy, "UnStableError": RuntimeError})
    results.append({"component": "robodojo", "check": "pinned_native_environment_methods_compile_without_policy_client", "passed": not hasattr(cls, "model_client")})
    # The new OpenWAM physical profile is compared with the native client,
    # NOT with the older ambiguous RoboTwin quaternion conversion.
    from collections.abc import Mapping
    from PIL import Image
    from embodied_eval.native.openwam_arx import OpenWAMArxModel
    wamroot = repos["pi05_arx"] / "policy/OpenWAM/OpenWAM"
    geom = {"np": np, "_QUATERNION_ATOL": 1e-6, "_ROT6D_DEGENERACY_EPS": 1e-8}
    functions(wamroot / "openwam/dataloader/utils/eef.py", ("quat_xyzw_to_rot6d", "quat_wxyz_to_rot6d"), geom)
    geom["shared_quat_wxyz_to_rot6d"] = geom["quat_wxyz_to_rot6d"]
    posefile = wamroot / "openwam/dataloader/utils/poses.py"
    functions(posefile, tuple(n.name for n in ast.parse(posefile.read_text()).body if isinstance(n, ast.FunctionDef)), geom)
    contract_scope = {"np": np, "Mapping": Mapping, "Path": Path, "Any": Any}
    contractfile = wamroot / "openwam/dataloader/robodojo_contract.py"
    nodes = [n for n in ast.parse(contractfile.read_text()).body if isinstance(n, (ast.Assign, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(contractfile), "exec"), contract_scope)
    prompt_scope = {}
    functions(wamroot / "openwam/dataloader/transforms/multiview.py", ("format_prompt_for_inference",), prompt_scope)
    source_model = repos["pi05_arx"] / "policy/OpenWAM/model.py"
    scope = {"np": np, "Any": Any, "Mapping": Mapping, "_GRIPPER_ATOL": 1e-8, "_EEF20_DIM": 20,
        "_CAMERA_TO_PAYLOAD": {"cam_head": "head_camera", "cam_left_wrist": "left_wrist_camera", "cam_right_wrist": "right_wrist_camera"}}
    functions(source_model, ("_validated_pose", "_validated_gripper", "_to_pil", "_decode_instruction"), scope)
    native_wam = native_class(source_model, "Model", ("_state_to_eef20", "_encode_obs", "_eef20_chunk_to_native"), scope)
    native_wam._poses = SimpleNamespace(**geom)
    native_wam._calibration = contract_scope["arx_x5_calibration"]()
    native_wam._format_prompt = prompt_scope["format_prompt_for_inference"]
    native_wam.default_instruction = "unused"
    wam = OpenWAMArxModel({"repo": str(repos["pi05_arx"]), "checkpoint_dir": "/absent", "io_profile": "eef20_arx_absolute"})
    wam.poses, wam.calibration, wam.format_prompt = native_wam._poses, native_wam._calibration, native_wam._format_prompt
    states = {}
    native_states = {}
    for side, offset in (("left", -.3), ("right", .3)):
        pos = np.array([offset + .1, -.15, .85])
        quat = np.array([0., 0., 0., 1.])
        grip = np.array([.8])
        native_states[side + "_ee_pose"] = np.concatenate((pos, quat[[3, 0, 1, 2]]))
        native_states[side + "_ee_joint_state"] = grip
        states.update({side + "_position": sample(pos), side + "_quaternion": sample(quat), side + "_grip": sample(grip)})
    wam_view = c.ModelInput(context=context, observation_id="obs", sim_time_s=0., task_public=view.task_public,
        sensors=dual_view.sensors, states=states)
    expected = native_wam._encode_obs({"vision": raw["vision"], "state": native_states, "instruction": view.task_public.instruction})
    actual = wam.native_input(wam_view)
    matched = expected["state"] == actual["state"] and expected["prompt"] == actual["prompt"] and all(
        same(np.asarray(expected["images"][key]), np.asarray(actual["images"][key])) for key in expected["images"])
    results.append({"component": "openwam_arx", "check": "native_input_model_base_calibration_rot6d_pixels_prompt", "passed": matched})
    # Build valid rotations from native observed EEF20, then vary positions and
    # deliberately exceed gripper bounds to test native legality projection.
    chunk = np.tile(np.asarray(actual["state"], np.float32), (32, 1))
    chunk[:, 0] += np.linspace(0., .05, 32, dtype=np.float32)
    chunk[0, 9], chunk[1, 19] = 1.1, -.1
    expected = native_wam._eef20_chunk_to_native(chunk)
    actual = wam.native_rows(chunk)
    wam.profile = resolve(dojo.space, wam.cap, c.BindingConfig(inputs={"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist", "instruction": "instruction",
        **{side + suffix: side + source for side in ("left", "right") for suffix, source in (("_position", "_eef_position"), ("_quaternion", "_eef_quaternion"), ("_grip", "_gripper_target"))}},
        targets={key: "robot0/" + key for key in wam.cap.output.target_slots}, action_spec_ref="dual-pose-absolute@1"))
    executed = [dojo.native_action(wam.decode_action(row, context)) for row in actual]
    results.append({"component": "openwam_arx", "check": "all_32_native_absolute_pose_and_gripper_dicts_dtype", "passed": all(same(e, a) and same(e, r) for e, a, r in zip(expected, actual, executed))})
    receipt = {"kind": "synthetic_input_pinned_native_function_comparison_not_real_evaluation", "passed": all(r["passed"] for r in results),
        "at_utc": datetime.now(timezone.utc).isoformat(), "results": results, "source_revisions": revisions,
        "pi_native_function_sha256": hashes, "robodojo_method_sha256": dojo_hashes,
        "emitted_camera_wiring_checked": {"cosmos3": {slot: examples["cosmos3_droid_sim"].binding.inputs[slot]
            for slot in ("exterior_right", "exterior_left", "wrist")},
            "dreamzero": {slot: examples["dreamzero_droid_sim"].binding.inputs[slot]
            for slot in ("exterior_right", "exterior_left", "wrist")}},
        "torch": torch.__version__, "gpu_inference": False, "model_loaded": False, "simulator_started": False, "official_score": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("kind", "passed", "results")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
