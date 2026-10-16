"""Supervised real single trials on independent 0.3 registrations.

This diagnostic owns only its launched process groups. No GPU occupancy process
is killed, no global proxy is changed, and no environment/secrets are serialized.
Timeouts here supervise diagnostic child processes, not a new Runner protocol.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parent
sys.path.insert(0,str(ROOT))


def server_spec(config):
    o=config.model.options
    repo=Path(o["repo"])
    profile=o["io_profile"]
    envs=WORKSPACE/"upstream/envs"
    port=str(o["port"])
    if config.model.adapter=="openwam":
        if profile not in ("eef10_osc", "eef10_absolute", "eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute"):
            raise ValueError("this historical server supervisor does not launch the in-process ARX OpenWAM engine")
        command=[str(envs/"openwam-py310/bin/python"),str(repo/"scripts/deploy.py"),"--ckpt-dir",o["checkpoint_dir"],
                 "--host","127.0.0.1","--port",port,"--device","cuda:0","--denoise-steps","10","--denoise-mode","sync","--inference-mode","sync"]
        if profile not in ("eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute"):
            command.extend(("--compile-enabled","false"))
    elif config.model.adapter=="openpi":
        if profile=="arx_joint14":
            command=[str(envs/"pi05-arx-mn-inference/bin/python"),str(ROOT/"docs/handoff/serve_pi05_arx_native.py"),
                "--port",port,"--checkpoint",o["checkpoint_dir"]]
        elif profile in ("state8_osc", "state7_absolute"):
            state8=profile=="state8_osc"
            command=[str(envs/("openpi-stage4" if state8 else "openpi-vlabench-stage4")/"bin/python"),str(repo/"scripts/serve_policy.py"),
                     "--port",port,"policy:checkpoint","--policy.config","pi05_libero" if state8 else "pi05_ft_vlabench_primitive","--policy.dir",o["checkpoint_dir"]]
        else:
            raise ValueError("named native model server launch not implemented for this profile")
    elif config.model.adapter=="lingbot_vla":
        command=[str(envs/"lingbot-stage4/bin/python"),str(ROOT/"docs/handoff/serve_lingbot_native.py"),
                 "--model-path",str(Path(o["checkpoint_dir"])/"checkpoints/global_step_50000/hf_ckpt"),"--port",port]
    else:
        raise ValueError("this diagnostic supervisor has no reviewed server launch for this model family")
    # Client dependency stack is selected by SIMULATOR only.
    environment={"libero":"libero-openpi-stage4","vlabench":"vlabench-stage4","robotwin":"robotwin-stage4"}[config.benchmark.adapter]
    if config.benchmark.adapter=="libero" and config.benchmark.options["protocol"]=="spatial-30settle":
        environment="libero"
    return command,repo,envs/environment/"bin/python"


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def trial(path,target,server_gpu,simulator_gpu):
    from embodied_eval.config import load_config
    from embodied_eval.v03.preflight import check
    config=load_config(path)
    checked=check(config)
    if not checked["structurally_compatible"]:
        return {"passed":False,"gate":"structure","issues":checked["issues"]}
    command,repo,client=server_spec(config)
    target.mkdir(parents=True,exist_ok=False)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1",config.model.options["port"]))
    environment=os.environ.copy()
    for key in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
        environment.pop(key,None)
    models=WORKSPACE.parent.parent/"Model"
    environment.update(CUDA_VISIBLE_DEVICES=server_gpu,HF_HUB_OFFLINE="1",HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.6",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
        QWEN3VL_PATH=str(models/"Qwen3-VL-4B-Instruct-processor"))
    (target/"launch.json").write_text(json.dumps({"command":command,"source_cwd":str(repo),"server_gpu":server_gpu,
        "simulator_gpu":simulator_gpu,"client_interpreter":str(client),"global_proxy_modified":False,
        "occupancy_process_modified":False,"formal_acceptance":False},indent=2)+"\n")
    server=child=None
    try:
        with (target/"server.txt").open("x") as server_log,(target/"client.txt").open("x") as client_log:
            server=subprocess.Popen(command,cwd=repo,env=environment,stdout=server_log,stderr=server_log,start_new_session=True)
            deadline=time.monotonic()+900
            while True:
                if server.poll() is not None:
                    return {"passed":False,"gate":"server_start","exit_code":server.returncode}
                try:
                    with socket.create_connection(("127.0.0.1",config.model.options["port"]),timeout=.5):
                        break
                except OSError:
                    if time.monotonic()>deadline:
                        return {"passed":False,"gate":"bounded_startup_timeout"}
                    time.sleep(1)
            client_env=environment.copy()
            client_env.update(CUDA_VISIBLE_DEVICES=simulator_gpu,CUDA_HOME=str(WORKSPACE/"upstream/toolchains/cuda-12.1-stage4/toolkit"))
            egl=WORKSPACE/"upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
            if config.benchmark.adapter=="robotwin" and egl.is_file():
                client_env["__EGL_VENDOR_LIBRARY_FILENAMES"]=str(egl)
            child=subprocess.Popen([str(client),__file__,"--child",str(path.resolve()),"--output",str(target.resolve()/"framework")],
                cwd=ROOT,env=client_env,stdout=client_log,stderr=client_log,start_new_session=True)
            code=child.wait(timeout=5400)
        framework=target/"framework"
        summary=json.loads((framework/"summary.json").read_text()) if (framework/"summary.json").is_file() else None
        return {"passed":code==0,"gate":"closed_loop" if summary else "client_start","exit_code":code,"synthetic":False,
            "summary_ref":str(framework/"summary.json") if summary else None,"scoring_performed":summary is not None,
            "completed":summary["completed"] if summary else None,"formal_acceptance":False}
    except subprocess.TimeoutExpired:
        return {"passed":False,"gate":"bounded_diagnostic_timeout","execution_state":"unknown"}
    finally:
        stop_owned(child); stop_owned(server)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs",type=Path)
    parser.add_argument("--cases",nargs="+",default=["openpi_libero","openwam_libero","openwam_vlabench","openpi_vlabench","openwam_robotwin","lingbot_robotwin"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--child",type=Path)
    parser.add_argument("--server-gpu",default="3")
    parser.add_argument("--simulator-gpu",default="0")
    args=parser.parse_args()
    if args.child:
        from embodied_eval.config import load_config
        from embodied_eval.v03.runner import run
        try:
            result=run(load_config(args.child),args.output)
            print("MN_TRIAL "+json.dumps({"exit_code":result.exit_code,"summary":str(args.output/"summary.json")}),flush=True)
            return result.exit_code
        except Exception as error:
            from embodied_eval.errors import ContractError
            print("MN_TRIAL_ERROR "+json.dumps({"type":type(error).__name__,"code":error.issue.code if isinstance(error,ContractError) else None}),flush=True)
            raise
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for key in args.cases:
        try:
            value=trial(args.configs/(key+".json"),args.output/key,args.server_gpu,args.simulator_gpu)
        except Exception as error:
            value={"passed":False,"gate":"supervisor","error_type":type(error).__name__}
        value["case"]=key
        results.append(value)
        (args.output/"progress.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(value),flush=True)
    receipt={"kind":"mn_real_single_trials_not_formal_acceptance_or_strict_native_comparison","at_utc":datetime.now(timezone.utc).isoformat(),
             "passed":all(r["passed"] for r in results),"results":results}
    (args.output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    return 0 if receipt["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
