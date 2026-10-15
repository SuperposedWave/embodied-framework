"""Read-only evidence reconciliation and a new, sealed delivery index.

Run in the existing robotwin-stage4 interpreter to use the native NumPy codec.
Does not infer, simulate, edit old evidence, or declare stage acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from probe_mn_components import ROOT, write
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import transforms3d as t3d
    from embodied_eval.adapters.native_common import native_module
    from compare_native_robotwin_closedloop import recorded_dual_pose_row
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--live",type=Path,required=True)
    parser.add_argument("--gpu",type=Path,nargs="+",required=True)
    parser.add_argument("--pose-api",type=Path,required=True)
    parser.add_argument("--video-verification",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    mapping=json.loads((args.mapping/"receipt.json").read_text())
    baseline=json.loads((args.baseline/"baseline-check.json").read_text())
    live=json.loads((args.live/"receipt.json").read_text())
    pose_api=json.loads(args.pose_api.read_text())
    videos=json.loads(args.video_verification.read_text())
    if not mapping["passed"] or not baseline["passed"] or not live["completed"] or not pose_api["passed"] or not videos["passed"]:
        raise ValueError("mapping_software_and_bounded_runtime_receipts_required")
    recording=Path(live.get("framework_recording",str(args.live/"recorded-framework")))
    framework=recording/"framework"
    manifest=json.loads((framework/"manifest.json").read_text())
    episode=framework/"episodes"/manifest["tasks"][0]["episode_id"]
    result=json.loads((episode/"result.json").read_text())
    native=json.loads((args.live/"native/native-result.json").read_text())
    if manifest["synthetic"] or not native["completed"] or mapping["verified_assets"]!=manifest["provenance"]["verified_assets"]:
        raise ValueError("same_verified_assets_and_completed_real_components_required")
    conversion=native_module(manifest["config"]["model"]["options"]["repo"],"benchmarks.utils.action_conversion")
    actions=[]
    with (episode/"trace.internal.jsonl").open() as stream:
        for line in stream:
            event=json.loads(line)
            if event["event"]=="prediction":
                actions.append(event["data"]["action"])
    rows=[]
    for index,action in enumerate(actions,1):
        source=recording/"model-exchanges"/f"action-{index:04d}.json"
        raw=np.asarray(json.loads(source.read_text())["action"],np.float32)
        expected=conversion.eef20d_to_ee16d(raw)
        actual=recorded_dual_pose_row(action)
        rows.append({"env_action":index,"native_row_exact":bool(np.array_equal(expected,actual)),
            "native_dtype_equal":expected.dtype==actual.dtype,"native_bytes_exact":expected.tobytes()==actual.tobytes(),
            "raw_grippers":[float(expected[7]),float(expected[15])],
            "public_pose_count":sum(x["kind"]=="pose_target" for x in action["commands"]),
            "acquire_observation_after":action["acquire_observation_after"]})
    mapping_passed=len(rows)==4 and all(x["native_row_exact"] and x["native_dtype_equal"] and x["native_bytes_exact"] and
        x["public_pose_count"]==2 and x["acquire_observation_after"] for x in rows)
    framework_first=json.loads((recording/"model-exchanges/request-0001.json").read_text())
    native_first=json.loads((args.live/"native/native-exchanges/request-0001.json").read_text())
    fresh_rows=[]
    for index in range(1,min(len(actions),native["native_actions"])+1):
        fw=recording/"model-exchanges"
        ref=args.live/"native/native-exchanges"
        a=json.loads((fw/f"request-{index:04d}.json").read_text())
        b=json.loads((ref/f"request-{index:04d}.json").read_text())
        ar=np.asarray(json.loads((fw/f"action-{index:04d}.json").read_text())["action"],np.float32)
        br=np.asarray(json.loads((ref/f"action-{index:04d}.json").read_text())["action"],np.float32)
        an,bn=conversion.eef20d_to_ee16d(ar),conversion.eef20d_to_ee16d(br)
        rotation_difference=max(float(np.max(np.abs(t3d.quaternions.quat2mat(an[j:j+4].astype(np.float64))-
            t3d.quaternions.quat2mat(bn[j:j+4].astype(np.float64))))) for j in (3,11))
        fresh_rows.append({"request":index,"native_input_exact":a==b,"different_input_fields":[k for k in a if a[k]!=b.get(k)],
            "raw_model_output_exact":bool(np.array_equal(ar,br)),"raw_output_max_abs_diff":float(np.max(np.abs(ar.astype(np.float64)-br.astype(np.float64)))),
            "decoded_controller_position_max_abs_diff_m":max(float(np.max(np.abs(an[j:j+3].astype(np.float64)-bn[j:j+3].astype(np.float64)))) for j in (0,8)),
            "decoded_controller_rotation_matrix_max_abs_diff_sign_invariant":rotation_difference})
    core=json.loads((ROOT/"outputs/native-comparison-integrity-20261009-final/receipt.json").read_text())["source_hashes"]
    protected={name:expected for name,expected in core.items() if name.startswith("outputs/") or name in
        ("embodied_eval/mn/runner.py","embodied_eval/mn/contracts.py","embodied_eval/contracts.py","docs/contract-v0.2.md")}
    protection={name:sha(ROOT/name)==expected for name,expected in protected.items()}
    saved_actions_protection={row["sample"]:sha(Path(mapping["saved_actions"])/row["sample"])==row["source_sha256"] for row in mapping["actions"]}
    saved_inputs_protection=sha(mapping["saved_inputs"])==mapping["source_and_history_sha256_before"][mapping["saved_inputs"]]
    gpu=[json.loads(path.read_text()) for path in args.gpu]
    gpu_samples=sum(x["samples"] for x in gpu)
    gpu_mean=sum(x["aggregate_mean_percent"]*x["samples"] for x in gpu)/gpu_samples
    codes=[x["actual_exit_code"] for x in baseline["checks"] if "actual_exit_code" in x]
    software_ok=baseline["checks"][0]["tests_run"]==416 and codes==[0,0,0,1,2]
    framework_summary=json.loads((framework/"summary.json").read_text())
    bounded_ok=native["native_actions"]==4 and result["execution_statistics"]["env_steps"]==4 and framework_summary["errors"]==0
    report={"kind":"OpenWAM_old_dual_explicit_encoding_public_pose_delivery_not_stage_acceptance",
        "at_utc":datetime.now(timezone.utc).isoformat(),"implementation_scope":"one existing checkpoint I/O gap only",
        "framework_recording":str(recording),"native_supervision":str(args.live/"receipt.json"),
        "actual_installed_SAPIEN_Pose_API":{"receipt":str(args.pose_api),"passed":pose_api["passed"],"cases":len(pose_api["cases"])},
        "comparison_videos":{"verification":str(args.video_verification),"PNG_files_verified":videos["PNG_files_verified"],
            "MP4_files_fully_decoded":videos["MP4_files_fully_decoded"],"additional_model_or_environment_calls":0},
        "passed_mapping_software_and_bounded_execution":mapping_passed and software_ok and bounded_ok and all(protection.values()) and
            all(saved_actions_protection.values()) and saved_inputs_protection,
        "public_contract_fields_and_Runner_unchanged":all(v for k,v in protection.items() if not k.startswith("outputs/")),
        "source_history_and_saved_data_protection":protection,"saved_actions_unchanged":all(saved_actions_protection.values()),
        "saved_inputs_unchanged":saved_inputs_protection,"verified_assets":mapping["verified_assets"],
        "offline_gate":{"receipt":str(args.mapping/"receipt.json"),"real_saved_observations":len(mapping["observations"]),
            "real_saved_actions":len(mapping["actions"]),"known_source_rotation_cases":len(mapping["rotation_cases"]),
            "maximum_controller_rotation_matrix_difference":max(x["controller_rotation_matrix_max_abs_diff"] for x in mapping["actions"]),
            "source_training_vs_inference_float32_difference":max(x["training_vs_native_max_abs_diff"] for x in mapping["observations"])},
        "software":{"receipt":str(args.baseline/"baseline-check.json"),"tests":baseline["checks"][0]["tests_run"],"cli_exit_codes":codes},
        "real_boundary_mapping":{"passed":mapping_passed,"rows":rows},
        "fresh_comparison":{"native_result":str(args.live/"native/native-result.json"),
            "first_model_input_exact":framework_first==native_first,"initial_observation_exact":native["initial_observation_exact"],
            "strict_full_trajectory_equal":native["strict_equal"],"first_divergence":native["first_divergence"],
            "framework_statistics":result["execution_statistics"],"native_actions":native["native_actions"],
            "native_physics_steps":native["native_physics_steps"],"native_sim_time_s":native["native_sim_time_s"],
            "framework_metrics":result["metrics"],"native_success":native["native_success"],
            "per_request_inference_investigation":fresh_rows,
            "fresh_differences_reported_separately_from_same_row_mapping_gate":True,"full_task_or_suite_success_rate":False},
        "gpu":{"receipts":[str(x) for x in args.gpu],"samples":gpu_samples,"sample_weighted_eight_card_mean_percent":gpu_mean,
            "target_percent":20,"target_met":gpu_mean>=20,"includes_preexisting_occupancy":True,
            "occupancy_not_evaluation_work_or_score":True,"not_unique_entire_run_lifetime_average":True},
        "remaining_gaps":["original training job/full training HDF5 not independently attested",
            "native gripper requests intentionally retained; poses are physical xyzw",
            "trusted server weight-identity handshake unavailable","no all-task/multi-seed evaluation or cross-robot TCP retargeting"]+
            ([f"strict trajectory differs at action {native['first_divergence']}; all raw model outputs equal: {all(x['raw_model_output_exact'] for x in fresh_rows)}; planning/physics cause not isolated"] if not native["strict_equal"] else []),
        "preserved_diagnostic_attempts":[
            {"receipt":"outputs/openwam-dual-pose-mapping-20261015-run1/receipt.json","cause":"diagnostic matrix comparison mixed float32 and float64 arithmetic; production native rows unchanged"},
            {"receipt":"outputs/openwam-dual-pose-live-20261015-run1/receipt.json","cause":"native comparison rejected raw JSON omitting null budget fields; canonical config matches existing framework manifest"}],
        "all_strict_reproduction_gates_passed":native["strict_equal"],"requires_user_review":True,
        "stage4_formally_accepted":False,"contract_frozen":False,"ARX_or_other_models_used_as_substitute":False,
        "framework_git_commit_created":False,"old_reports_or_results_rewritten":False}
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/"receipt.json",report)
    changed=("embodied_eval/mn/native_catalog.py","embodied_eval/mn/assets.py","embodied_eval/mn/native_models.py",
        "embodied_eval/mn/native_benchmarks.py","embodied_eval/mn/openwam_dual_pose.py","tests/test_mn_openwam_dual_pose.py",
        "examples/mn_openwam_dual_wxyz_encoded_absolute.json","docs/handoff/run_mn_trials.py",
        "docs/handoff/compare_native_robotwin_closedloop.py","docs/handoff/check_openwam_dual_pose.py",
        "docs/handoff/run_openwam_dual_pose_comparison.py","docs/handoff/summarize_openwam_dual_pose.py",
        "docs/openwam-dual-pose-20261015.md")
    hashes={name:sha(ROOT/name) for name in changed}
    write(args.output/"source-sha256.json",hashes)
    print(json.dumps({"passed":report["passed_mapping_software_and_bounded_execution"],"real_actions":len(rows),
                      "fresh_strict_equal":native["strict_equal"],"first_divergence":native["first_divergence"]}))
    return 0 if report["passed_mapping_software_and_bounded_execution"] else 1


if __name__=="__main__":
    raise SystemExit(main())
