"""Additional source-defined action policies; no pair factories or GPU imports.

Checkpoint identity is separate from environment identity. These registrations
do not imply that missing weights/runtime assets have been installed or tested.
"""
from dataclasses import dataclass
from pathlib import Path
from ..core import types as leaf
from ..codec import decode
from ..core.validation import require
from . import contracts as c
from .registry import ModelRegistration, BenchmarkRegistration
from .assets_common import _revision, _hf_objects

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
ROBOLAB_REVISION = "ad45d4f974725d020f82c2b0d77d78533aeba2b3"
POLICIES = {
    "cosmos3": {
        "profile": "nano-policy-droid-joint8", "model_id": "Cosmos3-Nano-Policy-DROID",
        "repo_id": "nvidia/Cosmos3-Nano-Policy-DROID",
        "source_revision": "cf5d68c00d97ccd2480a2320ed652b92dec63102",
        "checkpoint_revision": "805c0d6d46196ecdc789213a6de72303c383895b",
        "normalization": "server-owned:pinned-action-transform-pipeline",
        "horizon": 32,
    },
    "dreamzero": {
        "profile": "droid-joint8-causal", "model_id": "DreamZero-DROID",
        "repo_id": "GEAR-Dreams/DreamZero-DROID",
        "source_revision": "ab790c198fbce33503358efbbd4187ce9a89adf3",
        "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942",
        "normalization": "server-owned:experiment_cfg/metadata.json",
        "horizon": 8,
    },
}


@dataclass(frozen=True, kw_only=True)
class PolicyOptions:
    repo: str
    codec_repo: str
    checkpoint_dir: str
    io_profile: str
    client_repo: str = ""
    host: str = "127.0.0.1"
    port: int = 8940
    request_timeout_s: float = 300.
    open_loop_horizon: int | None = None


def policy_options(family, options):
    cfg = decode(PolicyOptions, options)
    require(family in POLICIES and cfg.io_profile == POLICIES[family]["profile"],
            "unknown_io_profile", "model", "an exact source-defined policy profile is required")
    require(cfg.host in ("127.0.0.1", "localhost") and 1 <= cfg.port <= 65535
            and cfg.request_timeout_s > 0, "invalid_endpoint", "model", "bounded local policy endpoint required")
    require(all((cfg.repo, cfg.codec_repo, cfg.checkpoint_dir)), "missing_source", "model", "explicit local asset paths required")
    require(family != "cosmos3" or bool(cfg.client_repo), "missing_source", "model.client_repo", "pinned RoboLab image client source required")
    require(cfg.open_loop_horizon is None or 1 <= cfg.open_loop_horizon <= POLICIES[family]["horizon"],
            "invalid_horizon", "model", "execution horizon must fit the source-defined chunk")
    return cfg


def _input(key, semantic, width, unit):
    return c.InputSpec(key=key, kind="state", semantic=semantic, unit=unit,
                       tensor=leaf.TensorSpec(dtype="float64", shape=(width,), layout="vector"))


def policy_capability(family, options):
    policy_options(family, options)
    item = POLICIES[family]
    inputs = tuple(c.InputSpec(key=key, kind="sensor", semantic="rgb",
        tensor=leaf.TensorSpec(dtype="uint8", shape=(-1, -1, 3), layout="HWC"))
        for key in ("exterior_right", "exterior_left", "wrist"))
    inputs += (_input("joints", "joint_position", 7, "rad"),
               _input("grip", "gripper_open_fraction", 1, "open_fraction"),
               c.InputSpec(key="instruction", kind="task"))
    targets = {f"j{i}": "joint" for i in range(7)}
    targets["grip"] = "gripper"
    return c.CapabilitySpec(model_id=item["model_id"], source_revision=item["source_revision"],
        checkpoint_ref=f"hf:{item['repo_id']}@{item['checkpoint_revision']}",
        normalization_ref=item["normalization"], inputs=inputs,
        output=c.OutputSpec(command_kinds=("joint_target", "gripper_command"), target_slots=targets,
            joint_modes=("position",), gripper_modes={"grip": ("open_fraction",)}),
        supported_timing_modes=("fixed_period", "measured_native"),
        training_embodiments=("droid-franka-robotiq",))


def policy_readiness(options):
    missing = [key for key in ("repo", "codec_repo", "checkpoint_dir")
               if not Path(options.get(key, "__absent__")).is_dir()]
    if options.get("io_profile") == POLICIES["cosmos3"]["profile"] and not Path(options.get("client_repo", "__absent__")).is_dir():
        missing.append("client_repo")
    return {"local_paths_available": not missing, "missing_paths": missing,
            "hashes_verified": False, "endpoint_verified": False, "dependencies_verified": False}


def verify_policy(family, options):
    cfg = policy_options(family, options)
    item = POLICIES[family]
    _revision(cfg.repo, item["source_revision"])
    _revision(cfg.codec_repo, OPENPI_REVISION)
    if family == "cosmos3":
        _revision(cfg.client_repo, ROBOLAB_REVISION)
    manifest = Path(__file__).parent / "asset_manifests" / (family + ".json")
    return {"source_revision": item["source_revision"], "codec_revision": OPENPI_REVISION,
            "sha256": _hf_objects(Path(cfg.checkpoint_dir).resolve(), manifest),
            "server_identity_verified": False}


def register(registry):
    for family in POLICIES:
        def factory(options, output, family=family):
            from .additional_models import JointPolicyModel
            return JointPolicyModel(family, options, output)
        registry.register_model(family, ModelRegistration(
            lambda o, f=family: policy_capability(f, o), factory,
            lambda o, f=family: policy_options(f, o), lambda o, f=family: verify_policy(f, o),
            policy_readiness, lambda o, f=family: {**POLICIES[f], "server_identity_verified": False}))
