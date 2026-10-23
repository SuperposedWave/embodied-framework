"""Model-only weight metadata and registrations; no environment catalogue."""
from dataclasses import dataclass, replace
from pathlib import Path
from ..core import types as leaf
from ..codec import decode
from ..core.validation import require
from . import contracts as c
from ..core.registry import ModelRegistration, BenchmarkRegistration
from .protocols import OSC_PROTOCOL, OSC_SIGNATURE, DUAL_PROTOCOL, DUAL_SIGNATURE, GRIPPER_PROTOCOL, GRIPPER_SIGNATURE, GRIPPER_CONTROLLER
from .joint_native import RAW_GRIPPER_PROTOCOL, RAW_GRIPPER_SIGNATURE

# This is a checkpoint's training-normalization selector in its native wire
# protocol. It is NOT the identity of the benchmark selected for this run.
WEIGHT_WIRE_PROFILES = {"joint14": {"training_normalization_selector": "robotwin"}}
PROFILES = {'eef10_osc': ('openwam', 'openwam-alpha-sim-libero@110b07e4d558', '48bd67b89d489b14d03b8d92bc66e65d306df32e', '34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a', 'f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e'), 'eef10_absolute': ('openwam', 'openwam-alpha-sim-vlabench@480d3bc', '48bd67b89d489b14d03b8d92bc66e65d306df32e', '6352f34db37f0a44ff8262f4d401d85d2a7b036799d00c6402ff81363684eb5e', '6434b2f23e8406d224424b57cca5e13527777dc260de446ec2bdf4876c29af8f'), 'eef20_dual_legacy': ('openwam', 'openwam-alpha-dual', '48bd67b89d489b14d03b8d92bc66e65d306df32e', '04b96af53eeeb111c64631f822efcbb82f4b186e', '3eb52f759ebd004feda2772925d587054f9c2f05e1b979c3ff8c94fdd40e5db5'), 'eef20_dual_wxyz_encoded_absolute': ('openwam', 'openwam-alpha-dual-wxyz-encoded-absolute@1', '48bd67b89d489b14d03b8d92bc66e65d306df32e', '04b96af53eeeb111c64631f822efcbb82f4b186e', '3eb52f759ebd004feda2772925d587054f9c2f05e1b979c3ff8c94fdd40e5db5'), 'state8_osc': ('openpi', 'pi05-libero@openpi-215abfb', '215abfb217dbac7d5f1273282331b9b1866c0479', 'gs://openpi-assets/checkpoints/pi05_libero/', 'b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84'), 'state7_absolute': ('openpi', 'pi05-vlabench-primitive10@f658a8b', '788ef1f20e655de5bb40869fb7b7fcdd49732387', 'f658a8bec801aab960fe0ceab370a123b31c1837', 'bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f'), 'joint14': ('lingbot_vla', 'lingbot-vla-v2-6b', 'be969b8fd117fb70550c5d4bf4bc328211b5b1b6', '0451855729ec904f970600e0aec8b84661423afe', '488ec6569c30dd69419358241efdb2e6c3488117768e7028a1e891be83e10db8'), 'arx_joint14': ('openpi', 'pi05-arx-x5-sim-seed0-step59999', '53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2', 'hf:datasets/RoboDojo-Benchmark/RoboDojo@35efbc7dedfdbeeb6e95fb749bd885d73d483e41/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999', 'ad7dea3e3d2bcdb348945fe03422ab1adccd03baf67318b1a1d153dfe8694db5'), 'eef20_arx_absolute': ('openwam', 'openwam-alpha-sim-arx-x5-step60000', '53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2', 'hf:OpenWAM/OpenWAM-Alpha-Sim-RoboDojo@2c1302294e3ba8319bbdb2c803b7a27de9292d03', '48bc266e0390f341c86dc3712501788c2bfdb75e44f005e41cc3e5820ace36a3')}

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
def readiness(options, model=True):
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
    from .model_assets import verify_model
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
