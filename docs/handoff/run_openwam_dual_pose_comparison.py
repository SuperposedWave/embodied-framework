"""Gated four-action real comparison with own freshly launched policy servers.

Preserves source/checkpoint/controller/scorer and native sampling cadence.
Raw upstream logs stay in memory, never saved; only safe failure categories
are persisted. Only subprocess groups created here are stopped. Diagnostic
process time bounds are not new Runner hard timeouts or an acceptance protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

from probe_mn_components import ROOT, WORKSPACE, write, categories, public_exception
from run_mn_trials import server_spec, stop_owned
sys.path.insert(0,str(ROOT))


def canonical_config(config):
    """Materialize optional defaults exactly as Runner's immutable manifest."""
    from embodied_eval.codec import to_data
    return to_data(config)


def child(args):
    import numpy as np
    from embodied_eval.config import load_config
    from embodied_eval.native.registry import default_registry, AdapterRegistry
    from tests.legacy_support.frozen.mn.runner import run
    from embodied_eval.native.native_models import NativeActionModel
    from embodied_eval.native.native_benchmarks import NativeSimulator
    from comparison_video import Frames
    args.output.mkdir(parents=True,exist_ok=False)
    framework=args.output/"framework"
    exchanges=args.output/"model-exchanges"
    exchanges.mkdir()
    frames=Frames(args.output/"frames",("head_camera","left_camera","right_camera"))
    class CapturedModel(NativeActionModel):
        def _connect(self):
            if self.peer is not None:
                return
            super()._connect()
            original=self.peer.predict_once
            self.recorded_requests=0
            def observed(payload):
                self.recorded_requests+=1
                write(exchanges/f"request-{self.recorded_requests:04d}.json",payload)
                reply=original(payload)
                write(exchanges/f"action-{self.recorded_requests:04d}.json",{"action":np.asarray(reply["action"],np.float32).tolist()})
                return reply
            self.peer.predict_once=observed
    class CapturedBenchmark(NativeSimulator):
        def record(self,observation):
            frames.append({name:np.asarray(sample.payload.data,np.uint8) for name,sample in observation.sensors.items()},
                          observation.env_step,observation.sim_time_s,
                          {name:sample.captured_at_s for name,sample in observation.sensors.items()})
        def reset(self,task,context):
            obs=super().reset(task,context); self.record(obs); return obs
        def advance(self,action,context,remaining):
            reply=super().advance(action,context,remaining); self.record(reply.observation); return reply
    registry=AdapterRegistry()
    base=default_registry()
    registry.models.update(base.models); registry.benchmarks.update(base.benchmarks)
    registry.models["openwam"]=replace(base.models["openwam"],factory=lambda o,out:CapturedModel("openwam",o,out))
    registry.benchmarks["robotwin"]=replace(base.benchmarks["robotwin"],factory=lambda o,out:CapturedBenchmark("robotwin",o,out))
    try:
        result=run(load_config(args.config),framework,registry)
        return result.exit_code
    finally:
        if frames.rows:
            frames.close()


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"examples/mn_openwam_dual_wxyz_encoded_absolute.json")
    parser.add_argument("--mapping-receipt",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model-gpu",type=int,default=6)
    parser.add_argument("--simulator-gpu",type=int,default=0)
    parser.add_argument("--framework-recording",type=Path,help="Reuse a completed, matching recorded-framework read-only; run only a fresh native side")
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    args.output,args.config=args.output.resolve(),args.config.resolve()
    args.output.relative_to(ROOT/"outputs")
    if args.child:
        return child(args)
    if args.mapping_receipt is None or json.loads(args.mapping_receipt.read_text()).get("passed") is not True:
        raise ValueError("completed_offline_mapping_gate_required")
    from embodied_eval.config import load_config
    from embodied_eval.native.openwam_dual_pose import PROFILE
    cfg=load_config(args.config)
    if cfg.model.options["io_profile"]!=PROFILE or cfg.episodes[0].budgets.env_steps!=4:
        raise ValueError("named_profile_and_reviewed_four_action_bound_required")
    if args.model_gpu==args.simulator_gpu or not all(0<=x<=7 for x in (args.model_gpu,args.simulator_gpu)):
        raise ValueError("distinct_available_local_gpu_ordinals_required")
    recording=args.framework_recording.resolve() if args.framework_recording else args.output/"recorded-framework"
    if args.framework_recording:
        recording.relative_to(ROOT/"outputs")
        manifest=json.loads((recording/"framework/manifest.json").read_text())
        summary=json.loads((recording/"framework/summary.json").read_text())
        if manifest["synthetic"] or manifest["config"]!=canonical_config(cfg) or summary["errors"]!=0 or summary["completed"]!=1:
            raise ValueError("completed_real_same_canonical_framework_config_required")
    args.output.mkdir(parents=True,exist_ok=False)
    config_path=args.output/"config.json"
    write(config_path,canonical_config(cfg))
    command,repo,client=server_spec(cfg)
    port=cfg.model.options["port"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",port))
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    for key in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
        env.pop(key,None)
    model_env=dict(env,CUDA_VISIBLE_DEVICES=str(args.model_gpu))
    sim_env=dict(env,CUDA_VISIBLE_DEVICES=str(args.simulator_gpu),CUDA_HOME=str(WORKSPACE/"upstream/toolchains/cuda-12.1-stage4/toolkit"),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(WORKSPACE/"upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"))
    receipt={"kind":"bounded_real_explicit_encoding_and_native_component_comparison_not_acceptance","completed":False,
        "started_at_utc":datetime.now(timezone.utc).isoformat(),"mapping_gate":str(args.mapping_receipt.resolve()),
        "config":str(config_path),"model_gpu":args.model_gpu,"simulator_gpu":args.simulator_gpu,
        "checkpoint_normalization_or_controller_changed":False,"public_contract_fields_or_Runner_changed":False,
        "global_proxy_or_occupancy_changed":False,"raw_native_logs_or_credentials_saved":False,
        "server_identity_verified":False,"formal_acceptance":False,"sides":[],
        "framework_recording":str(recording),"framework_recording_reused_read_only":args.framework_recording is not None}
    receipt["hardware"]=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip().splitlines()
    try:
        sides=("native",) if args.framework_recording else ("framework","native")
        for side in sides:
            server=worker=None
            with ThreadPoolExecutor(max_workers=2) as pool:
                logs=None
                row={"side":side,"fresh_owned_policy_server":True,"completed":False}
                try:
                    server=subprocess.Popen(command,cwd=repo,env=model_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
                    logs=pool.submit(server.communicate)
                    deadline=time.monotonic()+900
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError("owned_policy_server_start_failed")
                        try:
                            with socket.create_connection(("127.0.0.1",port),timeout=.5):
                                break
                        except OSError:
                            if time.monotonic()>deadline:
                                raise TimeoutError("diagnostic_server_start_bound")
                            time.sleep(1)
                    print(json.dumps({"side":side,"policy_server_ready":True}),flush=True)
                    if side=="framework":
                        invocation=[str(client),__file__,"--child","--config",str(config_path),"--output",str(args.output/"recorded-framework")]
                    else:
                        invocation=[str(client),str(ROOT/"docs/handoff/compare_native_robotwin_closedloop.py"),"--config",str(config_path),
                            "--framework",str(recording/"framework"),"--output",str(args.output/"native")]
                    worker=subprocess.Popen(invocation,cwd=ROOT,env=sim_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
                    worker_logs=pool.submit(worker.communicate)
                    try:
                        out,err=worker_logs.result(timeout=1800)
                    except Exception:
                        stop_owned(worker)
                        raise
                    row["exit_code"]=worker.returncode
                    row["public_failure_categories"]=categories((out+err).decode(errors="replace"))
                    row["completed"]=worker.returncode==0
                except Exception as error:
                    row["exception_chain"]=public_exception(error)
                finally:
                    stop_owned(worker); stop_owned(server)
                    if logs is not None:
                        out,err=logs.result(timeout=30)
                        row["server_public_failure_categories"]=categories((out+err).decode(errors="replace"))
                    receipt["sides"].append(row)
                    write(args.output/(side+"-supervision.json"),row)
                    print(json.dumps({"side":side,"completed":row["completed"],"exit_code":row.get("exit_code")}),flush=True)
            if not row["completed"]:
                break
        receipt["completed"]=len(receipt["sides"])==len(sides) and all(x["completed"] for x in receipt["sides"])
    finally:
        receipt["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
        write(args.output/"receipt.json",receipt)
    return 0 if receipt["completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
