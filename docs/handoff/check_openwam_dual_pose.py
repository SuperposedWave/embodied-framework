"""Offline gate: fixed native sources + saved real inputs/actions; no GPU calls.

AST means Python's parsed syntax tree. Selected unchanged pure source functions
are executed without importing the training dataset or constructing a simulator.
Saved data are not new inference or task-success evidence. Output must be new.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(path, names, namespace):
    nodes = [node for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names):
        raise ValueError("exact_fixed_source_definitions_required")
    exec(compile(ast.Module(body=sorted(nodes, key=lambda x: x.lineno), type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def pose_api(args):
    """Actual installed SAPIEN value API; no Scene, rendering, IK or inference."""
    import numpy as np
    import sapien
    import transforms3d as t3d
    cases=[("identity",np.array([1.,0.,0.,0.]))]
    for axis in range(3):
        for degrees in (90,180):
            q=np.zeros(4); q[0]=np.cos(np.deg2rad(degrees)/2); q[axis+1]=np.sin(np.deg2rad(degrees)/2)
            cases.append(("xyz"[axis]+str(degrees),q))
    q=np.array([.7,.2,-.3,.4]); cases.append(("general",q/np.linalg.norm(q)))
    rows=[]
    for name,q in cases:
        pose=sapien.Pose([.1,.2,.3],q)
        matrix=pose.to_transformation_matrix()
        error=float(np.max(np.abs(matrix[:3,:3].astype(np.float64)-t3d.quaternions.quat2mat(q))))
        rows.append({"case":name,"wxyz_rotation_matrix_max_abs_diff":error,
                     "q_accessor_wxyz_max_abs_diff":float(np.max(np.abs(np.asarray(pose.q,np.float64)-q)))})
    report={"kind":"actual_installed_SAPIEN_Pose_wxyz_value_API_not_scene_execution","passed":all(x["wxyz_rotation_matrix_max_abs_diff"]<3e-7 for x in rows),
        "new_scene_or_rendering_or_inference":False,"cases":rows,"formal_acceptance":False}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


def run(args):
    import numpy as np
    import transforms3d as t3d
    from embodied_eval import _legacy_contracts as leaf
    from embodied_eval.codec import decode, to_data
    from embodied_eval.config import load_config
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.assets import verify_model, verify_benchmark
    from embodied_eval.mn.binding import resolve
    from embodied_eval.mn.native_catalog import capability, space
    from embodied_eval.mn.native_models import NativeActionModel
    from embodied_eval.mn.native_benchmarks import NativeSimulator
    from embodied_eval.mn.openwam_dual_pose import PROFILE

    cfg = load_config(args.config)
    if cfg.model.options["io_profile"] != PROFILE:
        raise ValueError("explicit_new_profile_required")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(cfg.model.options["repo"])
    simulator = Path(cfg.benchmark.options["repo"])
    files = [source/"openwam/dataloader/robotwin.py", source/"openwam/dataloader/transforms/rotation.py",
        source/"openwam/dataloader/utils/stats_computation/robotwin_stats_computation.py",
        source/"benchmarks/utils/action_conversion.py", source/"benchmarks/robotwin/openwam2robotwin_interface.py",
        simulator/"envs/_base_task.py", simulator/"envs/robot/robot.py", simulator/"envs/robot/planner.py",
        simulator/"envs/utils/pkl2hdf5.py", simulator/"assets/embodiments/aloha-agilex/config.yml",
        ROOT/"examples/stage4-openwam-robotwin-adjust-bottle.json", ROOT/"docs/contract-v0.2.md",
        args.saved_inputs, args.exchanges.parent/"native-result.json"]
    protected = {str(path): sha(path) for path in files}
    report = {"kind": "saved_real_data_and_fixed_source_mapping_gate_not_new_inference", "passed": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "new_gpu_inference": False,
        "new_simulation_execution": False, "formal_acceptance": False, "source_and_history_sha256_before": protected,
        "verified_assets": {"model": verify_model("openwam", cfg.model.options),
                            "benchmark": verify_benchmark("robotwin", cfg.benchmark.options)},
        "profile": PROFILE, "saved_inputs": str(args.saved_inputs), "saved_actions": str(args.exchanges),
        "training_HDF5_files_present": False,
        "training_evidence_scope": "released checkpoint config + pinned dataset/statistics reader + native HDF5 writer; original training jobs/data are not independently attested",
        "observations": [], "actions": [], "rotation_cases": []}
    try:
        conversion = native_module(source, "benchmarks.utils.action_conversion")
        utils = native_module(source, "benchmarks.utils")
        prompt = native_module(source, "benchmarks.robotwin.prompt_template")
        rotation = selected(files[1], {"RotationType", "_quaternion_to_matrix", "_matrix_to_rotation_6d", "convert_rotation", "quat_xyzw_to_rotation_6d"}, {"np": np, "Enum": Enum})
        types = rotation["RotationType"]
        rotation.update(_REPR_DIM={types.QUATERNION:4, types.ROTATION_6D:6},
            _TO_MATRIX={types.QUATERNION:rotation["_quaternion_to_matrix"]},
            _FROM_MATRIX={types.ROTATION_6D:rotation["_matrix_to_rotation_6d"]})
        dataset = selected(files[0], {"_read_eef_actions"}, {"np":np, "quat_xyzw_to_rotation_6d": rotation["quat_xyzw_to_rotation_6d"]})
        native_client = selected(files[4], {"_extract_eef_proprio"}, {"np":np, "action_conversion":conversion})
        controller = selected(files[6], {"_trans_from_gripper_to_endlink"},
            {"np":np, "deepcopy":copy.deepcopy, "t3d":t3d,
             "sapien":SimpleNamespace(Pose=lambda p,q: SimpleNamespace(p=np.asarray(p), q=np.asarray(q)))})
        # This is the exact native pure controller conversion, with a value-only
        # Pose stand-in; no claim of actual SAPIEN/IK execution in this gate.
        controller_config = SimpleNamespace(left_gripper_bias=.12, right_gripper_bias=.12,
            left_inv_delta_matrix=np.eye(3), right_inv_delta_matrix=np.eye(3))
        s, cap = space("robotwin", cfg.benchmark.options), capability("openwam", cfg.model.options)
        model = NativeActionModel("openwam", cfg.model.options)
        model.profile, model.space = resolve(s, cap, cfg.binding), s
        model.conversion, model.utils = conversion, utils
        context = c.Context(run_id="offline-mapping", env_id="saved", episode_id="saved", message_id="mapping")

        def view_from_raw(ext, public):
            values = {}
            for side in ("left", "right"):
                raw = np.asarray(ext[side+"_endpose_world_wxyz"], np.float64)
                values[side+"_position"] = raw[:3]
                values[side+"_quaternion"] = raw[[4,5,6,3]]
                values[side+"_grip"] = np.asarray([ext[side+"_gripper_open_fraction"]], np.float64)
            samples = {key:c.StateSample(captured_at_s=ext["captured_at_sim_time_s"],
                provenance="controller_target" if key.endswith("grip") else "measurement",
                payload=leaf.Tensor(dtype="float64", shape=val.shape, layout="vector", data=val.tolist())) for key,val in values.items()}
            cameras = {key:public.observation_view.sensors[name] for key,name in
                (("head","head_camera"),("left_wrist","left_camera"),("right_wrist","right_camera"))}
            return c.ModelInput(context=context, observation_id="saved-observation", sim_time_s=public.observation_view.sim_time_s,
                               task_public=public.task_public, sensors=cameras, states=samples)

        with args.saved_inputs.open() as stream:
            for index, line in enumerate(stream):
                public = decode(leaf.ModelInput, json.loads(line))
                ext = public.observation_view.extensions["robotwin.native_proprio.v1"]
                view = view_from_raw(ext, public)
                actual = model.native_input(view)
                endpose = {side+"_endpose":ext[side+"_endpose_world_wxyz"] for side in ("left","right")}
                endpose.update({side+"_gripper":ext[side+"_gripper_open_fraction"] for side in ("left","right")})
                expected_state = native_client["_extract_eef_proprio"]({"endpose":endpose})
                expected = utils.build_payload(**{slot:utils.encode_numpy_b64(np.ascontiguousarray(public.observation_view.sensors[name].payload.data,np.uint8))
                    for slot,name in (("head","head_camera"),("left_wrist","left_camera"),("right_wrist","right_camera"))},
                    prompt=prompt.format_prompt_for_inference(public.task_public.instruction), state=expected_state.tolist())
                hdf5 = {"endpose/"+key:np.asarray(value,np.float64)[None,...] for key,value in endpose.items()}
                trained = dataset["_read_eef_actions"](None, hdf5, 0, 1)[0]
                difference = float(np.max(np.abs(trained.astype(np.float64)-expected_state.astype(np.float64))))
                row = {"sample":index, "native_payload_exact":actual==expected,
                    "training_vs_native_max_abs_diff":difference,
                    "training_vs_native_source_float32_bound":3e-7,
                    "training_comparison_is_not_byte_equality":not np.array_equal(trained,expected_state)}
                if not row["native_payload_exact"] or difference > 3e-7:
                    raise ValueError("saved_input_mapping_differs")
                report["observations"].append(row)
                if index == 0:
                    (args.output/"first-public-input.json").write_text(json.dumps(to_data(view),ensure_ascii=False)+"\n")
                if args.max_observations and len(report["observations"]) >= args.max_observations:
                    break

        def check_row(raw, name):
            native = conversion.eef20d_to_ee16d(raw)
            action = model.decode_action(raw, context, None)
            actual, mode = NativeSimulator.robotwin_native_action(action)
            errors = []
            for offset, side in ((0,"left"),(8,"right")):
                old_pose = controller["_trans_from_gripper_to_endlink"](controller_config, native[offset:offset+7], side)
                new_pose = controller["_trans_from_gripper_to_endlink"](controller_config, actual[offset:offset+7], side)
                errors.append(float(np.max(np.abs(t3d.quaternions.quat2mat(old_pose.q)-t3d.quaternions.quat2mat(new_pose.q)))))
                public_q = np.asarray(action.commands[0 if side=="left" else 2].pose.quaternion_xyzw)
                physical = public_q[[3,0,1,2]]
                # Compare mathematical orientations in the SAME arithmetic
                # precision. Actual native controller rows are separately exact.
                errors.append(float(np.max(np.abs(t3d.quaternions.quat2mat(physical.astype(np.float64))-
                    t3d.quaternions.quat2mat(native[offset+3:offset+7].astype(np.float64))))))
            row = {"sample":name, "native_row_exact":bool(np.array_equal(native,actual)), "dtype_equal":native.dtype==actual.dtype,
                   "mode":mode, "controller_rotation_matrix_max_abs_diff":max(errors),
                   "raw_grippers":[float(raw[9]),float(raw[19])], "public_pose_count":sum(isinstance(x,leaf.PoseTarget) for x in action.commands)}
            if not row["native_row_exact"] or not row["dtype_equal"] or max(errors)>1e-12 or mode!="ee" or row["public_pose_count"]!=2:
                report["failed_mapping_row"]=row
                raise ValueError("saved_action_physical_mapping_differs")
            return row, action

        for path in sorted(args.exchanges.glob("action-*.json")):
            raw = np.asarray(json.loads(path.read_text())["action"], np.float32)
            row, action = check_row(raw, path.name)
            row["source_sha256"] = sha(path)
            report["actions"].append(row)
            if len(report["actions"])==1:
                (args.output/"first-public-action.json").write_text(json.dumps(to_data(action),ensure_ascii=False,indent=2)+"\n")

        quaternions = [("identity",np.array([0.,0.,0.,1.]))]
        for axis in range(3):
            for degrees in (90,180):
                q=np.zeros(4); q[axis]=np.sin(np.deg2rad(degrees)/2.); q[3]=np.cos(np.deg2rad(degrees)/2.)
                quaternions.append(("xyz"[axis]+str(degrees),q))
        q=np.array([.2,-.3,.4,.7]); quaternions.append(("general",q/np.linalg.norm(q)))
        for name,q in quaternions:
            raw=conversion.robotwin_endpose_to_eef20d(np.r_[.1,.2,.3,q[[3,0,1,2]]],np.r_[-.1,-.2,.4,-q[[3,0,1,2]]],1.00390625,-.00390625)
            row,action=check_row(raw,name)
            expected=t3d.quaternions.quat2mat(q[[3,0,1,2]])
            differences=[]
            for command in (action.commands[0],action.commands[2]):
                actual=np.array(command.pose.quaternion_xyzw)[[3,0,1,2]]
                differences.append(float(np.max(np.abs(t3d.quaternions.quat2mat(actual)-expected))))
            row["known_physical_rotation_matrix_max_abs_diff"]=max(differences)
            if max(differences)>3e-7:
                raise ValueError("known_physical_rotation_differs")
            report["rotation_cases"].append(row)
        if not report["observations"] or not report["actions"]:
            raise ValueError("saved_real_inputs_and_actions_required")
        report["source_and_history_sha256_after"]={str(path):sha(path) for path in files}
        report["protected_files_unchanged"]=report["source_and_history_sha256_after"]==protected
        report["passed"]=report["protected_files_unchanged"]
    except Exception as error:
        from probe_mn_components import public_exception
        report["exception_chain"]=public_exception(error)
    finally:
        report["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
        (args.output/"receipt.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"observations":len(report["observations"]),
                      "saved_actions":len(report["actions"]),"known_rotations":len(report["rotation_cases"])}))
    return 0 if report["passed"] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"examples/mn_openwam_dual_wxyz_encoded_absolute.json")
    parser.add_argument("--saved-inputs",type=Path,default=ROOT/"outputs/stage4-closed-loop-openwam-robotwin-20261006T1618Z/openwam_robotwin/framework/episodes/adjust-bottle-clean-episode0/model_input.public.jsonl")
    parser.add_argument("--exchanges",type=Path,default=ROOT/"outputs/native-comparison-openwam_robotwin-20261009-run1/native/native-exchanges")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-observations",type=int,default=0,help="0 checks every saved input")
    parser.add_argument("--pose-api-only",action="store_true",help="Verify actual installed SAPIEN Pose quaternion ordering without a scene")
    args=parser.parse_args()
    return pose_api(args) if args.pose_api_only else run(args)


if __name__=="__main__":
    raise SystemExit(main())
