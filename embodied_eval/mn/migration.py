"""Offline 0.1 -> 0.2 config translation. No runtime pair dispatch/factories.

Model checkpoint profiles and simulator protocols translate independently.
Bindings are explicit migration data and are subsequently semantically checked.
"""
from pathlib import Path
from ..codec import to_data
from . import contracts as c
from .config import AdapterConfig, EpisodeConfig, RunConfig

MODELS={
    "openwam_libero":("openwam","eef10_osc","openwam_repo"),
    "openwam_vlabench":("openwam","eef10_absolute","openwam_repo"),
    "openwam_robotwin":("openwam","eef20_dual_legacy","openwam_repo"),
    "openpi_libero":("openpi","state8_osc","openpi_repo"),
    "openpi_vlabench":("openpi","state7_absolute","openpi_repo"),
    "lingbot_robotwin":("lingbot_vla","joint14","lingbot_repo"),
}
BENCHMARKS={
    "libero_spatial":("libero","spatial-30settle","libero_repo"),
    "libero_openpi":("libero","spatial-10settle","libero_repo"),
    "vlabench":("vlabench","frozen-track1","vlabench_repo"),
    "vlabench_openpi":("vlabench","frozen-track1","vlabench_repo"),
    "robotwin_openwam":("robotwin","sealed-instruction","robotwin_repo"),
    "robotwin_lingbot":("robotwin","seeded-instruction","robotwin_repo"),
}


def translate(old):
    family,profile,repo=MODELS[old.model.adapter]
    model=AdapterConfig(adapter=family,options={"io_profile":profile,"repo":getattr(old.model,repo),
        "checkpoint_dir":old.model.checkpoint_dir,"host":old.model.host,"port":old.model.port,"request_timeout_s":old.model.request_timeout_s})
    bench_family,protocol,repo=BENCHMARKS[old.benchmark.adapter]
    benchmark=AdapterConfig(adapter=bench_family,options={"repo":getattr(old.benchmark,repo),"protocol":protocol,
        **{key:getattr(old.benchmark,key) for key in ("task_name","task_config","instruction_type","max_seed_candidates","track","trial_index")}})
    from .native_catalog import capability,space
    cap,spec=capability(family,model.options),space(bench_family,benchmark.options)
    cameras={"libero":{"head":"agentview_image","left_wrist":"robot0_eye_in_hand_image"},
             "vlabench":{"head":"forward_rgb","left_wrist":"franka_wrist_rgb","right_wrist":"right_rgb"},
             "robotwin":{"head":"head_camera","left_wrist":"left_camera","right_wrist":"right_camera"}}
    state_aliases={"position":"eef_position","quaternion":"eef_quaternion","fingers":"gripper_positions",
        "grip":"gripper_open_fraction","base":"robot_base_position","drive_targets":"joint_drive_targets"}
    for side in ("left","right"):
        state_aliases.update({side+"_position":side+"_eef_position",side+"_quaternion":side+"_eef_quaternion",side+"_grip":side+"_gripper_open_fraction"})
    inputs={req.key:("instruction" if req.kind=="task" else cameras[bench_family][req.key]
                      if req.kind=="sensor" else state_aliases[req.key]) for req in cap.inputs}
    aliases={"eef":"robot0/eef","grip":"robot0/grip","dual":"robot0/base"}
    aliases.update({side+f"_j{i}":"robot0/"+side+f"_j{i}" for side in ("left","right") for i in range(6)})
    aliases.update({side+"_grip":"robot0/"+side+"_grip" for side in ("left","right")})
    targets={key:aliases[key] for key in cap.output.target_slots}
    actions={"eef10_osc":"osc-pose@1","state8_osc":"osc-pose@1","eef10_absolute":"cartesian-absolute@1",
             "state7_absolute":"cartesian-absolute@1","eef20_dual_legacy":"dual-native-legacy@1","joint14":"joint-absolute-native-gripper@1"}
    return RunConfig(run_id=old.run_id+"-mn",model=model,benchmark=benchmark,
        binding=c.BindingConfig(inputs=inputs,targets=targets,action_spec_ref=actions[profile]),
        episodes=tuple(EpisodeConfig(episode_id=e.episode_id,seed=e.seed,budgets=e.budgets) for e in old.episodes))
