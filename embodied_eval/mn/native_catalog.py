"""Pinned model I/O profiles and simulator protocols, registered independently.

Imports here are metadata/standard-library only. Native packages and GPU state
are touched by factories during run, never by describe or structural preflight.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from .. import contracts as leaf
from ..codec import decode
from ..validation import require
from ..adapters import openwam_libero_spec as ow, openpi_libero_spec as pi
from ..adapters import vlabench_spec as vla, openpi_vlabench_spec as pivla, robotwin_spec as rw
from . import contracts as c
from .registry import ModelRegistration, BenchmarkRegistration
from .joint_native import RAW_GRIPPER_PROTOCOL, RAW_GRIPPER_SIGNATURE

OSC_PROTOCOL = "osc-pose-normalized7@1"
OSC_SIGNATURE = "xyz3+axisangle3+grip_close1;normalized[-1,1];environment-scales;world-delta;20Hz"
DUAL_PROTOCOL = "dual-eef16-legacy-quaternion@1"
DUAL_SIGNATURE = "left+right:xyz-m3+helper-xyzw4-interpreted-as-native-wxyz+open-fraction1;dtype-preserved"
GRIPPER_PROTOCOL = "gripper-open-fraction-interpolate-then-clamp@1"
GRIPPER_SIGNATURE = "finite-unbounded-open-fraction-request;interpolate-from-current-per-native-physics-path;clamp-each-step[0,1];dtype-preserved"
GRIPPER_CONTROLLER = "gripper-open-fraction-raw-request@1"
PROFILES = {
    "eef10_osc": ("openwam", ow.MODEL_ID, ow.MODEL_REVISION, ow.CHECKPOINT_SHA256, ow.NORMALIZATION_SHA256),
    "eef10_absolute": ("openwam", "openwam-alpha-sim-vlabench@480d3bc", vla.MODEL_REVISION, vla.CHECKPOINT_SHA256, vla.NORMALIZATION_SHA256),
    "eef20_dual_legacy": ("openwam", "openwam-alpha-dual", rw.OPENWAM_REVISION, rw.VARIANTS["openwam_robotwin"][2], rw.VARIANTS["openwam_robotwin"][5]),
    # Same released weights/statistics, with their scalar-first-as-xyzw
    # training encoding explicitly inverted before exposing physical poses.
    "eef20_dual_wxyz_encoded_absolute": ("openwam", "openwam-alpha-dual-wxyz-encoded-absolute@1", rw.OPENWAM_REVISION,
        rw.VARIANTS["openwam_robotwin"][2], rw.VARIANTS["openwam_robotwin"][5]),
    "state8_osc": ("openpi", pi.MODEL_ID, pi.MODEL_REVISION, pi.CHECKPOINT_REF, pi.NORMALIZATION_SHA256),
    "state7_absolute": ("openpi", "pi05-vlabench-primitive10@f658a8b", pivla.MODEL_REVISION, pivla.HF_REVISION, pivla.NORMALIZATION_SHA256),
    "joint14": ("lingbot_vla", "lingbot-vla-v2-6b", rw.LINGBOT_REVISION, rw.VARIANTS["lingbot_robotwin"][2], rw.VARIANTS["lingbot_robotwin"][5]),
    "arx_joint14": ("openpi", "pi05-arx-x5-sim-seed0-step59999", "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2",
        "hf:datasets/RoboDojo-Benchmark/RoboDojo@35efbc7dedfdbeeb6e95fb749bd885d73d483e41/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999",
        "ad7dea3e3d2bcdb348945fe03422ab1adccd03baf67318b1a1d153dfe8694db5"),
    "eef20_arx_absolute": ("openwam", "openwam-alpha-sim-arx-x5-step60000", "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2",
        "hf:OpenWAM/OpenWAM-Alpha-Sim-RoboDojo@2c1302294e3ba8319bbdb2c803b7a27de9292d03",
        "48bc266e0390f341c86dc3712501788c2bfdb75e44f005e41cc3e5820ace36a3"),
}


@dataclass(frozen=True, kw_only=True)
class ModelOptions:
    io_profile: str
    repo: str
    checkpoint_dir: str
    host: str = "127.0.0.1"
    port: int = 8920
    request_timeout_s: float = 300.
    runtime_python: str | None = None
    runtime_gpu: int | None = None


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


def model_options(family, options):
    cfg = decode(ModelOptions, options)
    require(cfg.io_profile in PROFILES and PROFILES[cfg.io_profile][0] == family,
            "unknown_io_profile", "model.io_profile", "profile belongs to a different model family")
    require(cfg.host in ("127.0.0.1", "localhost") and 1 <= cfg.port <= 65535 and cfg.request_timeout_s > 0,
            "invalid_endpoint", "model", "bounded local endpoint required")
    require(bool(cfg.repo and cfg.checkpoint_dir), "missing_source", "model", "explicit source/checkpoint paths required")
    require(cfg.runtime_python is None or (cfg.io_profile == "eef20_arx_absolute" and
            isinstance(cfg.runtime_python, str) and Path(cfg.runtime_python).is_absolute()),
            "invalid_model_runtime", "model.runtime_python", "explicit absolute native Python path supported only by ARX OpenWAM")
    require(cfg.runtime_gpu is None or (cfg.runtime_python is not None and
            type(cfg.runtime_gpu) is int and cfg.runtime_gpu >= 0),
            "invalid_model_runtime", "model.runtime_gpu", "native model runtime required for an explicit nonnegative GPU ordinal")
    return cfg


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


def _state_input(key, semantic, size, unit):
    return c.InputSpec(key=key, kind="state", semantic=semantic, unit=unit,
                       tensor=leaf.TensorSpec(dtype="float64", shape=(size,), layout="vector"),
                       frame_id="world" if semantic in ("eef_position","eef_orientation","base_position") else None)


def capability(family, options):
    cfg = model_options(family, options)
    _, model_id, revision, checkpoint, norm = PROFILES[cfg.io_profile]
    profile = cfg.io_profile
    camera_slots = ("head", "left_wrist") if profile in ("eef10_osc", "state8_osc") else ("head", "left_wrist", "right_wrist")
    inputs = tuple(c.InputSpec(key=key, kind="sensor", semantic="rgb",
        tensor=leaf.TensorSpec(dtype="uint8", shape=(-1,-1,3), layout="HWC")) for key in camera_slots) + (c.InputSpec(key="instruction", kind="task"),)
    if profile in ("eef10_osc", "state8_osc", "eef10_absolute", "state7_absolute"):
        inputs += (_state_input("position", "eef_position", 3, "m"),
                   _state_input("quaternion", "eef_orientation", 4, "quaternion_xyzw"))
        if profile in ("eef10_osc", "state8_osc"):
            inputs += (_state_input("fingers", "gripper_position", 2, "m"),)
        else:
            inputs += (_state_input("grip", "gripper_open_fraction", 1, "open_fraction"),
                       _state_input("base", "base_position", 3, "m"))
    elif profile in ("eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute", "eef20_arx_absolute"):
        for side in ("left", "right"):
            inputs += (_state_input(side+"_position", "eef_position", 3, "m"),
                       _state_input(side+"_quaternion", "eef_orientation", 4, "quaternion_xyzw"),
                       _state_input(side+"_grip", "gripper_drive_target", 1, "open_fraction"))
    elif profile == "arx_joint14":
        for side in ("left", "right"):
            inputs += (_state_input(side+"_joints", "joint_position", 6, "rad"),
                       _state_input(side+"_grip", "gripper_drive_target", 1, "open_fraction"))
    else:
        inputs += (_state_input("drive_targets", "joint_drive_target", 14, "rad_and_open_fraction"),)
    if profile in ("eef10_osc", "state8_osc"):
        output = c.OutputSpec(command_kinds=("controller_command",), target_slots={"eef": "end_effector"},
                              native_protocol=OSC_PROTOCOL, native_signature=OSC_SIGNATURE)
        embodiments = ("libero-panda-osc-pose",)
    elif profile == "eef20_dual_legacy":
        output = c.OutputSpec(command_kinds=("controller_command",), target_slots={"dual": "base"},
                              native_protocol=DUAL_PROTOCOL, native_signature=DUAL_SIGNATURE)
        embodiments = ("aloha-agilex-dual",)
    elif profile == "eef20_dual_wxyz_encoded_absolute":
        targets = {side+"_eef": "end_effector" for side in ("left", "right")}
        targets.update({side+"_grip": "gripper" for side in ("left", "right")})
        # Only the gripper remains native: clipping the request before the
        # simulator's interpolation changes actual execution. Poses are SI/xyzw.
        output = c.OutputSpec(command_kinds=("pose_target", "controller_command"), target_slots=targets,
            native_protocol=GRIPPER_PROTOCOL, native_signature=GRIPPER_SIGNATURE)
        embodiments = ("aloha-agilex-dual",)
    elif profile == "eef20_arx_absolute":
        targets = {side+"_eef": "end_effector" for side in ("left", "right")}
        targets.update({side+"_grip": "gripper" for side in ("left", "right")})
        output = c.OutputSpec(command_kinds=("pose_target", "gripper_command"), target_slots=targets,
            gripper_modes={side+"_grip": ("open_fraction",) for side in ("left", "right")})
        embodiments = ("arx-x5-dual",)
    elif profile in ("joint14", "arx_joint14"):
        targets = {side+f"_j{i}": "joint" for side in ("left", "right") for i in range(6)}
        targets.update({side+"_grip": "gripper" for side in ("left", "right")})
        # Native diffusion output can exceed [0,1]. The simulator interpolates
        # before per-physics-step clipping; model-side clipping changes its
        # trajectory. Joints remain physical; ONLY grippers use this extension.
        output = c.OutputSpec(command_kinds=("joint_target", "controller_command"), target_slots=targets,joint_modes=("position",),
            native_protocol=RAW_GRIPPER_PROTOCOL if profile == "arx_joint14" else GRIPPER_PROTOCOL,
            native_signature=RAW_GRIPPER_SIGNATURE if profile == "arx_joint14" else GRIPPER_SIGNATURE)
        embodiments = ("arx-x5-dual",) if profile == "arx_joint14" else ("aloha-agilex-dual",)
    else:
        output = c.OutputSpec(command_kinds=("pose_target", "gripper_command"), target_slots={"eef": "end_effector", "grip": "gripper"},
                              gripper_modes={"grip":("width_m",)})
        embodiments = ("franka-panda",)
    return c.CapabilitySpec(model_id=model_id, source_revision=revision, checkpoint_ref=checkpoint,
        normalization_ref="sha256:"+norm, inputs=inputs, output=output, training_embodiments=embodiments,
        supported_timing_modes=("measured_native",) if profile in ("joint14", "arx_joint14", "eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute", "eef20_arx_absolute") else ("fixed_period",),
        observation_update="decision_boundary" if profile == "joint14" else "every_env_step")


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
    legacy = "openwam_robotwin" if cfg.protocol == "sealed-instruction" else "lingbot_robotwin"
    old = rw.space(legacy)
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
    from .assets import verify_model, verify_benchmark
    for family in ("openwam","openpi","lingbot_vla"):
        def factory(o, out, family=family):
            if o.get("io_profile") == "eef20_arx_absolute":
                if o.get("runtime_python") is not None:
                    from .native_model_process import NativeModelProcess
                    return NativeModelProcess(family, o, out)
                from .openwam_arx import OpenWAMArxModel
                return OpenWAMArxModel(o, out)
            from .native_models import NativeActionModel
            return NativeActionModel(family,o,out)
        registry.register_model(family, ModelRegistration(lambda o,f=family:capability(f,o),factory,
            lambda o,f=family:model_options(f,o),lambda o,f=family:verify_model(f,o),
            lambda o:readiness(o,True),lambda o,f=family:{"family":f,"io_profile":o["io_profile"],"server_identity_verified":False}))
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
