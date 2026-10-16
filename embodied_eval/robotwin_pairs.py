"""Pinned RoboTwin pair registration and fail-closed preflight, not acceptance."""
from pathlib import Path

from . import _legacy_contracts as c
from ._legacy_validation import require
from .adapters import robotwin_spec as spec

PAIRS = {("openwam_robotwin", "robotwin_openwam"), ("lingbot_robotwin", "robotwin_lingbot")}
_VERIFIED = {}


def key(config):
    return (config.model.adapter, config.model.checkpoint_dir, config.model.openwam_repo,
            config.model.lingbot_repo, config.benchmark.robotwin_repo)


def validate(config):
    require(len(config.episodes) == 1, "pilot_scope", "episodes", "one fixed RoboTwin episode required")
    model_repo = config.model.openwam_repo if config.model.adapter == "openwam_robotwin" else config.model.lingbot_repo
    require(model_repo is not None and config.benchmark.robotwin_repo is not None and config.model.checkpoint_dir is not None,
            "missing_source", "adapters", "explicit source/checkpoint/RoboTwin roots required")
    require(config.model.host in ("127.0.0.1", "localhost") and 1 <= config.model.port <= 65535 and config.model.request_timeout_s > 0,
            "invalid_endpoint", "model", "local policy endpoint required")
    require(config.profile.chunk_mode == "adapter_owned" and config.profile.execution_horizon_steps == 1,
            "pilot_profile_mismatch", "profile", "native adapter owns cache, one native row per decision")
    require(config.benchmark.task_name == "adjust_bottle" and config.benchmark.task_config == "demo_clean"
            and config.benchmark.instruction_type == "unseen" and config.benchmark.trial_index == 0
            and 1 <= config.benchmark.max_seed_candidates <= 100,
            "pilot_scope", "benchmark", "reviewed profile: demo_clean/adjust_bottle/unseen/episode0; bounded expert scan")
    ep = config.episodes[0]
    require(ep.seed == 0 and ep.budgets.env_steps == 400 and ep.budgets.model_calls == 400
            and ep.budgets.sim_time_s is None, "pilot_scope", "episode", "seed-index0, 400 native rows; no hard simulation-time limit")


def verify(config):
    from .stage4_pairs import _revision, _file, _hf_objects, _sha
    _VERIFIED.pop(key(config), None)
    variant = config.model.adapter
    model_rev, bench_rev, _, _, _, normalization = spec.VARIANTS[variant]
    root = Path(config.benchmark.robotwin_repo).resolve()
    model_root = Path(config.model.openwam_repo if variant == "openwam_robotwin" else config.model.lingbot_repo).resolve()
    _revision(model_root, model_rev)
    _revision(root, bench_rev)
    digests = _hf_objects(Path(config.model.checkpoint_dir).resolve(),
                          Path(__file__).parent / f"adapters/{variant}_objects.json")
    if variant == "openwam_robotwin":
        require(digests["normalization_stats.npy"] == normalization, "normalization_mismatch", variant, "matching RoboTwin normalization required")
    else:
        digests["native_robotwin_norm"] = _file(model_root / "assets/norm_stats/robotwin.json", normalization)
        digests["native_robot_mapping"] = _file(model_root / "configs/robot_configs/robotwin.yaml",
            "2c2a26509fc9662a8ada64adaa7fa959627dd6dadc6b48b680c4285da2d8ffd8")
    for relative in ("task_config/demo_clean.yml", "task_config/_eval_step_limit.yml", "task_config/_camera_config.yml",
                     "task_config/_embodiment_config.yml", "envs/adjust_bottle.py", "description/task_instruction/adjust_bottle.json"):
        path = root / relative
        require(path.is_file(), "asset_missing", relative, "native fixed trial source asset required")
        digests[relative] = _sha(path)
    # Presence is not archive integrity or proof of the loaded server weights.
    for relative in ("assets/embodiments/aloha-agilex/config.yml", "assets/objects", "assets/background_texture"):
        require((root / relative).exists(), "asset_missing", relative, "RoboTwin native assets not ready")
    _VERIFIED[key(config)] = digests
    return digests


def make_adapters(config, output):
    verify(config)
    from .adapters.robotwin_model import RobotwinModel
    from .adapters.robotwin_benchmark import RobotwinBenchmark
    return RobotwinModel(config.model, output), RobotwinBenchmark(
        config.benchmark, config.model.adapter, output, model_repo=config.model.openwam_repo)


def make_profile(config, space, cap):
    return spec.profile(config.model.adapter, space, cap)


def fallback_metrics(config):
    return {"success": c.Metric(value=None, unit="boolean", source="RoboTwin native eval_success@" + spec.VARIANTS[config.model.adapter][1][:7],
                                valid=False, unavailable_reason="not_scored")}


def provenance(config):
    model, bench, hf, _, _, norm = spec.VARIANTS[config.model.adapter]
    return {"synthetic": False, "model_adapter": config.model.adapter, "benchmark_adapter": config.benchmark.adapter,
            "model_source_revision": model, "benchmark_source_revision": bench, "model_asset_revision": hf,
            "normalization_sha256": norm, "server_identity_verified": False,
            "asset_checks_completed": key(config) in _VERIFIED, "verified_asset_sha256": _VERIFIED.get(key(config), {}),
            "scene_asset_origin_verified": False, "native_scene_meter_real_verified": False,
            "protocol_version": spec.space(config.model.adapter).protocol_version,
            "upstream_quaternion_convention_conflict_retained": config.model.adapter == "openwam_robotwin"}
