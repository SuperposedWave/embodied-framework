"""Compare 0.2 mappings to fixed native clients using historical public samples.

Input samples are real historical observations; output rows are deterministic
SYNTHETIC fixtures. No server request, simulator execution or benchmark score.
Every case is isolated in a process to keep conflicting native imports apart.
"""
import argparse
import ast
import collections
import math
from types import SimpleNamespace
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
SOURCE={
 "openwam_libero":"stage3-framework-20261005T1729Z-final",
 "openpi_libero":"stage4-closed-loop-pi05-libero-20261006T1436Z/openpi_libero/framework",
 "openwam_vlabench":"stage4-closed-loop-openwam-vlabench-20261006T1450Z-optimized/openwam_vlabench/framework",
 "openpi_vlabench":"stage4-closed-loop-pi05-vlabench-20261006T1618Z-native-f32/openpi_vlabench/framework",
 "openwam_robotwin":"stage4-closed-loop-openwam-robotwin-20261006T1618Z/openwam_robotwin/framework",
 "lingbot_robotwin":"stage4-closed-loop-lingbot-robotwin-20261006T1611Z-transport-fixed/lingbot_robotwin/framework",
}


def source_nodes(path, predicate, namespace):
    """Execute selected unchanged source nodes, not copied adapter formulas.

    AST is Python's parsed syntax tree. Selecting pure definitions/assignments
    avoids importing the upstream CLI's simulator or launching a service.
    The upstream file is never edited and its byte hash is recorded below.
    """
    nodes=[node for node in ast.walk(ast.parse(path.read_text())) if predicate(node)]
    if not nodes:
        raise ValueError("selected_native_source_nodes_absent")
    nodes.sort(key=lambda node:node.lineno)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def perform(case,configs):
    import numpy as np
    from embodied_eval import _legacy_contracts as old
    from embodied_eval.codec import decode
    from embodied_eval.config import load_config
    from embodied_eval.mn import contracts as c
    from embodied_eval.mn.binding import model_input,resolve
    from embodied_eval.mn.native_catalog import capability,space
    from embodied_eval.mn.native_models import NativeActionModel
    from embodied_eval.mn.native_benchmarks import NativeSimulator
    from embodied_eval.mn.transforms import prepare_input
    from embodied_eval.adapters.native_common import native_module
    cfg=load_config(configs/(case+".json"))
    root=ROOT/"outputs"/SOURCE[case]
    manifest=json.loads((root/"manifest.json").read_text())
    episode=manifest["tasks"][0]["episode_id"]
    input_path=root/"episodes"/episode/"model_input.public.jsonl"
    with input_path.open() as stream:
        public=decode(old.ModelInput,json.loads(stream.readline()))
    cap=capability(cfg.model.adapter,cfg.model.options)
    s=space(cfg.benchmark.adapter,cfg.benchmark.options)
    if cfg.benchmark.adapter=="vlabench":
        s=replace(s,control_period_s=manifest["space"]["control_period_s"])
    bench=NativeSimulator.__new__(NativeSimulator); bench.family=cfg.benchmark.adapter; bench.space=s
    obs=bench._project(public.observation_view)
    task=c.TaskSpec(task_id="mapping",episode_id=episode,public=public.task_public,budgets=cfg.episodes[0].budgets,
                    initialization_ref="historical-public-input",evaluation_config_ref="no-score")
    profile=resolve(s,cap,cfg.binding)
    view=prepare_input(model_input(obs,task,cap,profile),cap,profile,s)
    model=NativeActionModel(cfg.model.adapter,cfg.model.options)
    model.profile,model.space=profile,s
    oldcfg=decode(__import__("embodied_eval.config",fromlist=["RunConfig"]).RunConfig,manifest["config"])
    oldprofile=decode(old.AdaptationProfile,manifest["profile"])
    request=replace(public.context,message_id="mapping-request")
    new_request=c.Context(run_id=request.run_id,env_id=request.env_id,episode_id=request.episode_id,message_id=request.message_id)
    rng=np.random.RandomState(17)
    reference_source=None
    if cfg.model.adapter=="openwam":
        model.utils=native_module(cfg.model.options["repo"],"benchmarks.utils")
        model.conversion=native_module(cfg.model.options["repo"],"benchmarks.utils.action_conversion")
        width=20 if case=="openwam_robotwin" else 10
        raw=rng.uniform(-.1,.1,(width,)).astype(np.float32)
        for offset in (0,10) if width==20 else (0,):
            raw[offset+3:offset+9]=[1,0,0,0,1,0]
        class Capture:
            payload=None
            def ping(self): return {"type":"pong","representation":"eef"}
            def reset(self): return {"type":"reset_ack"}
            def predict(self,payload): self.payload=payload; return {"action":raw}
            predict_once=predict
            def close(self): pass
        peer=Capture()
        if case=="openwam_libero":
            native=native_module(cfg.model.options["repo"],"benchmarks.libero.openwam2libero_interface")
            reference_source=Path(native.__file__)
            policy=native.OpenWAMLiberoPolicy(_client=peer)
            ext=public.observation_view.extensions["libero.native_proprio.v1"]
            native_obs={key:np.asarray(public.observation_view.sensors[key].payload.data,np.uint8) for key in ("agentview_image","robot0_eye_in_hand_image")}
            native_obs.update(robot0_eef_pos=ext["eef_position_m"],robot0_eef_quat=ext["eef_quaternion_xyzw"],robot0_gripper_qpos=ext["gripper_qpos_m"])
            expected=policy.act(native_obs,public.task_public.instruction)
            native_payload=peer.payload
        elif case=="openwam_vlabench":
            from embodied_eval.adapters.openwam_vlabench_model import OpenWAMVLABenchModel
            reference=OpenWAMVLABenchModel(oldcfg.model,Path("/unused"))
            native=native_module(cfg.model.options["repo"],"benchmarks.vlabench.openwam2vlabench_interface")
            reference_source=Path(native.__file__)
            with patch.object(native,"WSPolicyClient",return_value=peer):
                policy=native.OpenWAMVLABenchPolicy(head_camera_index=0,left_wrist_camera_index=1,right_wrist_camera_index=2)
            position,euler,fingers=policy.predict(reference.native_input(public))
            expected={"position_world_m":position.tolist(),"euler_xyz_rad":euler.tolist(),"finger_targets_m":fingers.tolist()}
            native_payload=peer.payload
        else:
            native=native_module(cfg.model.options["repo"],"benchmarks.robotwin.openwam2robotwin_interface")
            reference_source=Path(native.__file__)
            reference=native.ModelClient.__new__(native.ModelClient)
            reference._client=peer
            reference._task_description=public.task_public.instruction
            reference._send_state=True; reference._state_dim=20
            reference._action_type="ee"; reference._debug=False
            reference._action_indices=None; reference._step=0
            ext=public.observation_view.extensions["robotwin.native_proprio.v1"]
            native_obs={"observation":{key:{"rgb":np.asarray(value.payload.data,np.uint8)} for key,value in public.observation_view.sensors.items()},
                "endpose":{side+suffix:ext[side+source] for side in ("left","right") for suffix,source in
                    (("_endpose","_endpose_world_wxyz"),("_gripper","_gripper_open_fraction"))}}
            captured=[]
            env=SimpleNamespace(take_action_cnt=1,get_instruction=lambda:public.task_public.instruction,
                take_action=lambda row,**kwargs:captured.append(row.copy()))
            native.eval(env,reference,native_obs)
            expected=captured[0]; native_payload=peer.payload
    elif cfg.model.adapter=="openpi":
        raw=rng.uniform(-.1,.1,(5,7)).astype(np.float32)
        class Peer:
            payload=None
            def infer(self,payload): self.payload=payload; return {"actions":raw.copy()}
        if case=="openpi_libero":
            model.images=native_module(cfg.model.options["repo"],"openpi_client.image_tools","packages/openpi-client/src")
            reference_source=Path(cfg.model.options["repo"])/"examples/libero/main.py"
            ext=public.observation_view.extensions["libero.native_proprio.v1"]
            native_obs={key:np.asarray(public.observation_view.sensors[key].payload.data,np.uint8) for key in ("agentview_image","robot0_eye_in_hand_image")}
            native_obs.update(robot0_eef_pos=np.asarray(ext["eef_position_m"]),robot0_eef_quat=np.asarray(ext["eef_quaternion_xyzw"]),robot0_gripper_qpos=np.asarray(ext["gripper_qpos_m"]))
            namespace={"np":np,"math":math,"image_tools":model.images,"obs":native_obs,"args":SimpleNamespace(resize_size=224),"task_description":public.task_public.instruction}
            source_nodes(reference_source,lambda node:isinstance(node,ast.FunctionDef) and node.name=="_quat2axisangle",namespace)
            source_nodes(reference_source,lambda node:isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ("img","wrist_img","element") for target in node.targets),namespace)
            native_payload=namespace["element"]; expected={"osc_action7":raw[0].tolist()}
        else:
            utils=native_module(oldcfg.benchmark.vlabench_repo,"VLABench.utils.utils")
            reference_source=Path(cfg.model.options["repo"])/"examples/vlabench/eval.py"
            namespace={"Policy":object,"np":np,"collections":collections,"quaternion_to_euler":utils.quaternion_to_euler}
            source_nodes(reference_source,lambda node:isinstance(node,ast.ClassDef) and node.name=="Pi0",namespace)
            peer=Peer(); policy=namespace["Pi0"](peer)
            ext=public.observation_view.extensions["vlabench.native_proprio.v1"]
            images={key:np.asarray(value.payload.data,np.uint8) for key,value in public.observation_view.sensors.items()}
            native_obs={"rgb":[images["right_rgb"],None,images["forward_rgb"],images["franka_wrist_rgb"]],
                "ee_state":np.asarray(ext["ee_state_world_wxyz"],np.float64),"last_action":np.zeros(8),
                "robot_frame":np.asarray(ext["robot_base_world_m"],np.float64),"instruction":ext["instruction"]}
            position,euler,fingers=policy.predict(native_obs)
            native_payload=peer.payload
            expected={"position_world_m":position.tolist(),"euler_xyz_rad":euler.tolist(),"finger_targets_m":fingers.tolist()}
        raw=raw[0]
    else:
        reference_source=Path(cfg.model.options["repo"])/"experiment/robotwin/eval_policy_client_lingbotvla.py"
        ext=public.observation_view.extensions["robotwin.native_proprio.v1"]
        native_obs={"observation":{key:{"rgb":np.asarray(value.payload.data,np.uint8)} for key,value in public.observation_view.sensors.items()},
            "joint_action":{"vector":np.asarray(ext["joint_drive_targets_and_grippers"],np.float64)}}
        namespace={"observation":native_obs,"TASK_ENV":SimpleNamespace(get_instruction=lambda:public.task_public.instruction)}
        source_nodes(reference_source,lambda node:isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=="formatted_observation" for target in node.targets),namespace)
        native_payload=namespace["formatted_observation"]
        raw=rng.uniform(-.1,.1,(14,)).astype(np.float32); raw[6]=raw[13]=.5
        expected=raw.copy()
    mapped=model.native_input(view)
    def equality(a,b):
        if isinstance(a,dict):
            return set(a)==set(b) and all(equality(v,b[k]) for k,v in a.items())
        if isinstance(a,str) or a is None:
            return a==b
        if isinstance(a,np.ndarray):
            return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
        return np.array_equal(np.asarray(a),np.asarray(b))
    inputs_equal=equality(native_payload,mapped)
    action=model.decode_action(raw,new_request,view)
    if case in ("openwam_libero","openpi_libero"):
        actual=action.commands[0].parameters
        expected={"osc_action7":expected.tolist()} if isinstance(expected,np.ndarray) else expected
    elif cfg.benchmark.adapter=="vlabench":
        actual=bench._cartesian_parameters(action)
    elif case=="openwam_robotwin":
        actual=np.asarray(action.commands[0].parameters["native_row"],dtype=action.commands[0].parameters["native_dtype"])
    else:
        commands={x.target_id.split("/",1)[1]:x for x in action.commands}
        actual=np.asarray([v for side in ("left","right") for v in [*[commands[side+f"_j{i}"].value for i in range(6)],commands[side+"_grip"].parameters["requested_open_fraction"]]],dtype=action.native_dtype)
    outputs_equal=equality(expected,actual)
    return {"case":case,"passed":bool(inputs_equal and outputs_equal),"input_values_and_numpy_dtypes_exact":bool(inputs_equal),
        "native_execution_parameters_exact":bool(outputs_equal),"synthetic_output_fixture":True,"fresh_model_inference":False,
        "historical_input_ref":str(input_path),"native_source_revision":cap.source_revision,
        "additional_native_source_file":str(reference_source) if reference_source else None,
        "additional_native_source_sha256":hashlib.sha256(reference_source.read_bytes()).hexdigest() if reference_source else None,
        "not_benchmark_score":True,"failure_keys":[k for k in native_payload if not equality(native_payload[k],mapped.get(k))]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--child",choices=tuple(SOURCE))
    args=parser.parse_args()
    if args.child:
        result=perform(args.child,args.configs)
        print("MN_MAPPING "+json.dumps(result),flush=True)
        return 0 if result["passed"] else 1
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for key in SOURCE:
        completed=subprocess.run([sys.executable,__file__,"--configs",str(args.configs.resolve()),"--output",str(args.output.resolve()),"--child",key],cwd=ROOT,capture_output=True,text=True)
        (args.output/(key+".txt")).write_text(completed.stdout+completed.stderr)
        lines=[line for line in completed.stdout.splitlines() if line.startswith("MN_MAPPING ")]
        row=json.loads(lines[-1].removeprefix("MN_MAPPING ")) if lines else {"case":key,"passed":False,"exit_code":completed.returncode}
        rows.append(row); print(json.dumps(row),flush=True)
    receipt={"kind":"historical_public_input_and_synthetic_output_native_mapping_comparison","synthetic":True,
             "at_utc":datetime.now(timezone.utc).isoformat(),"passed":all(r["passed"] for r in rows),"results":rows}
    (args.output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    return 0 if receipt["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
