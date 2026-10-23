"""Benchmark-only protocol catalogue; no model catalogue or checkpoint metadata."""
from dataclasses import dataclass, replace
from pathlib import Path
from ..core import types as leaf
from ..codec import decode
from ..core.validation import require
from . import contracts as c
from ..core.registry import ModelRegistration, BenchmarkRegistration
from .protocols import OSC_PROTOCOL, OSC_SIGNATURE, DUAL_PROTOCOL, DUAL_SIGNATURE, GRIPPER_PROTOCOL, GRIPPER_SIGNATURE, GRIPPER_CONTROLLER
from .joint_native import RAW_GRIPPER_PROTOCOL, RAW_GRIPPER_SIGNATURE
from .environment_specs import libero_30 as ow, libero_10 as pi, vlabench as vla, robotwin as rw
@dataclass(frozen=True, kw_only=True)
class BenchmarkOptions:
    repo: str
    protocol: str
    task_name: str = "select_fruit"
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    max_seed_candidates: int = 100
    track: str = "track_1_in_distribution"
    trial_index: int = 0

def benchmark_options(family, options):
    cfg = decode(BenchmarkOptions, options)
    known = {"libero": ("spatial-30settle", "spatial-10settle"), "vlabench": ("frozen-track1",),
             "robotwin": ("sealed-instruction", "seeded-instruction")}
    require(cfg.protocol in known[family], "unknown_protocol", "benchmark.protocol", "unknown simulator protocol")
    require(cfg.max_seed_candidates > 0 and cfg.trial_index == 0, "pilot_scope", "benchmark", "native trial0 and bounded seed search")
    if family == "vlabench":
        require(cfg.task_name == "select_fruit" and cfg.track == "track_1_in_distribution", "pilot_scope", "benchmark", "only the reviewed frozen scene is implemented")
    if family == "robotwin":
        require(cfg.task_name == "adjust_bottle" and cfg.task_config == "demo_clean" and cfg.instruction_type in ("seen", "unseen"),
                "pilot_scope", "benchmark", "only reviewed adjust_bottle/demo_clean is implemented")
    return cfg

def state_spec(semantic, unit, width, origin, frame="world"):
    return c.StateSpec(semantic=semantic, unit=unit, frame_id=frame,
                       tensor=leaf.TensorSpec(dtype="float64", shape=(width,), layout="vector"), origin=origin)

def space(family, options):
    cfg = benchmark_options(family, options)
    if family == "libero":
        old = ow.space() if cfg.protocol == "spatial-30settle" else pi.space()
        controller = "osc-pose-normalized7@1"
        robot = old.robots["robot0"]
        return c.SpaceSpec(spec_id="libero-"+cfg.protocol+"@"+old.revision[:7], revision=old.revision,
            protocol_id="libero_spatial", protocol_version=cfg.protocol+"-mn1", timing_mode="fixed_period", control_period_s=.05,
            frames=old.frames, robots={"robot0": robot}, sensors=old.sensors,
            state_fields={"eef_position": state_spec("eef_position", "m", 3, "grip_site achieved position"),
                          "eef_quaternion": state_spec("eef_orientation", "quaternion_xyzw", 4, "right_hand achieved body orientation, NOT grip_site orientation"),
                          "gripper_positions": state_spec("gripper_position", "m", 2, "measured finger qpos", None)},
            controllers={controller: leaf.ControllerSpec(frame_id="world", parameter_schema={"type":"object","properties":{
                "osc_action7":{"type":"array","items":{"type":"number"},"minItems":7,"maxItems":7}},
                "required":["osc_action7"],"additionalProperties":False}, parameter_units={"osc_action7":"controller_native"})},
            action_specs={"osc-pose@1": leaf.ActionSpec(command_kinds=("controller_command",), target_ids=("robot0/eef",), controller_refs=(controller,))},
            native_protocols={OSC_PROTOCOL: OSC_SIGNATURE})
    if family == "vlabench":
        old = vla.space(.002)
        robot = leaf.RobotSpec(embodiment_id="franka-panda", base_frame_id="world",
            end_effectors={"eef": leaf.EndEffectorSpec(frame_id="world", tcp_id="native-ee")},
            grippers={"grip": leaf.GripperSpec(modes=("width_m",), max_width_m=.08)})
        return c.SpaceSpec(spec_id="vlabench-frozen-track1@"+old.revision[:7], revision=old.revision,
            protocol_id=old.protocol_id, protocol_version="frozen-track1-mn1", timing_mode="fixed_period", control_period_s=.002,
            frames=old.frames, robots={"robot0":robot}, sensors=old.sensors,
            state_fields={"eef_position":state_spec("eef_position","m",3,"native ee_state position"),
                "eef_quaternion":state_spec("eef_orientation","quaternion_xyzw",4,"native ee_state orientation converted from wxyz"),
                "gripper_open_fraction":state_spec("gripper_open_fraction","open_fraction",1,"1 minus native inverted closed-flag",None),
                "robot_base_position":state_spec("base_position","m",3,"live robot base world position")},
            action_specs={"cartesian-absolute@1":leaf.ActionSpec(command_kinds=("pose_target","gripper_command"), target_ids=("robot0/eef","robot0/grip"))},
            required_action_targets={"cartesian-absolute@1":("robot0/eef","robot0/grip")})
    old = rw.space(cfg.protocol)
    joints = {side+f"_j{i}":leaf.JointSpec(joint_type="revolute", position_unit="rad", velocity_unit="rad/s", effort_unit="Nm") for side in ("left","right") for i in range(6)}
    robot = leaf.RobotSpec(embodiment_id="aloha-agilex-dual",base_frame_id="world",joints=joints,
        end_effectors={side+"_eef":leaf.EndEffectorSpec(frame_id="world",tcp_id=side+"_native_tcp") for side in ("left","right")},
        grippers={side+"_grip":leaf.GripperSpec(modes=("open_fraction",),max_width_m=.1) for side in ("left","right")})
    states = {"joint_drive_targets":state_spec("joint_drive_target","rad_and_open_fraction",14,
              "left drive targets rad[6], left gripper fraction[1], right equivalent; NOT measured qpos",None)}
    for side in ("left","right"):
        states.update({side+"_eef_position":state_spec("eef_position","m",3,
                side+" get_arm_pose/get_ee_pose world position, is_endpose=False; NOT get_tcp_pose gripper center"),
            side+"_eef_quaternion":state_spec("eef_orientation","quaternion_xyzw",4,
                side+" get_arm_pose/get_ee_pose world orientation with native global/delta calibration; wxyz reordered to xyzw"),
            side+"_gripper_open_fraction":state_spec("gripper_drive_target","open_fraction",1,side+" clipped commanded gripper fraction; NOT measured finger qpos",None)})
        states[side+"_joint_positions"] = state_spec("joint_position", "rad", 6,
            side+" get_arm_real_jointState arm part, measured SAPIEN articulation qpos (gripper tail excluded)", None)
    ctrl = "dual-eef16-legacy@1"
    controller = leaf.ControllerSpec(frame_id="world", parameter_schema={"type":"object","properties":{
        "native_row":{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},
        "native_dtype":{"type":"string","enum":["float32","float64","int32","int64"]}},
        "required":["native_row","native_dtype"],"additionalProperties":False},parameter_units={"native_row":"controller_native","native_dtype":"metadata"})
    allj = tuple("robot0/"+j for j in joints)+("robot0/left_grip","robot0/right_grip")
    raw_gripper = leaf.ControllerSpec(frame_id="world",parameter_schema={"type":"object",
        "properties":{"requested_open_fraction":{"type":"number"}},"required":["requested_open_fraction"],"additionalProperties":False},
        parameter_units={"requested_open_fraction":"native_unbounded_open_fraction_request_interpolate_then_clamp"})
    return c.SpaceSpec(spec_id="robotwin-"+cfg.protocol+"@"+old.revision[:7],revision=old.revision,
        protocol_id=old.protocol_id,protocol_version=cfg.protocol+"-mn1",timing_mode="measured_native",control_period_s=None,
        physics_period_s=.004,frames=old.frames,robots={"robot0":robot},sensors=old.sensors,state_fields=states,
        action_specs={"joint-absolute@1":leaf.ActionSpec(command_kinds=("joint_target","gripper_command"),target_ids=allj),
            "joint-absolute-native-gripper@1":leaf.ActionSpec(command_kinds=("joint_target","controller_command"),target_ids=allj,controller_refs=(GRIPPER_CONTROLLER,)),
            "dual-cartesian-absolute@1":leaf.ActionSpec(command_kinds=("pose_target","gripper_command"),target_ids=("robot0/left_eef","robot0/right_eef","robot0/left_grip","robot0/right_grip")),
            "dual-cartesian-native-gripper@1":leaf.ActionSpec(command_kinds=("pose_target","controller_command"),
                target_ids=("robot0/left_eef","robot0/right_eef","robot0/left_grip","robot0/right_grip"),controller_refs=(GRIPPER_CONTROLLER,)),
            "dual-native-legacy@1":leaf.ActionSpec(command_kinds=("controller_command",),target_ids=("robot0/base",),controller_refs=(ctrl,))},
        controllers={ctrl:controller,GRIPPER_CONTROLLER:raw_gripper},native_protocols={DUAL_PROTOCOL:DUAL_SIGNATURE,GRIPPER_PROTOCOL:GRIPPER_SIGNATURE,
            RAW_GRIPPER_PROTOCOL:RAW_GRIPPER_SIGNATURE},
        required_action_targets={"joint-absolute@1":allj,"joint-absolute-native-gripper@1":allj,
            "dual-cartesian-absolute@1":("robot0/left_eef","robot0/right_eef","robot0/left_grip","robot0/right_grip"),
            "dual-cartesian-native-gripper@1":("robot0/left_eef","robot0/right_eef","robot0/left_grip","robot0/right_grip")},
        action_joint_modes={"joint-absolute@1":("position",),"joint-absolute-native-gripper@1":("position",)})
def readiness(options, model=False):
    paths = ("repo", "checkpoint_dir") if model else ("repo",)
    missing = [key for key in paths if not isinstance(options.get(key),str) or not Path(options[key]).is_dir()]
    if model and options.get("runtime_python") is not None:
        import os
        path = Path(options["runtime_python"])
        if not path.is_file() or not os.access(path, os.X_OK):
            missing.append("runtime_python")
    return {"local_paths_available":not missing,"missing_paths":missing,"hashes_verified":False,
            "endpoint_verified":False,"dependencies_verified":False}

def register(registry):
    from .benchmark_assets import verify_benchmark
    for family in ("libero","vlabench","robotwin"):
        def factory(o,out,family=family):
            from .native_benchmarks import NativeSimulator
            return NativeSimulator(family,o,out)
        def metrics(o,family=family):
            names = {"success":"boolean"}
            if family == "vlabench":
                names.update(progress_score="score",intention_score="score")
            return {k:leaf.Metric(value=None,unit=u,source=family+":official-native",valid=False,unavailable_reason="not_scored") for k,u in names.items()}
        registry.register_benchmark(family, BenchmarkRegistration(lambda o,f=family:space(f,o),factory,
            lambda o,f=family:benchmark_options(f,o),lambda o,f=family:verify_benchmark(f,o),
            readiness,lambda o,f=family:{"family":f,"protocol":o["protocol"],"scene_asset_origin_verified":f!="vlabench"},metrics))
