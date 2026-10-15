"""Strict CLI configuration and the built-in adapter registry."""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal
import hashlib
import re
import subprocess

from . import contracts as c
from .codec import decode, loads, to_data
from .validation import require


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    adapter: Literal["mock_model", "openwam_libero", "openpi_libero", "openwam_vlabench", "openpi_vlabench", "openwam_robotwin", "lingbot_robotwin"] = "mock_model"
    openwam_repo: str | None = None
    openpi_repo: str | None = None
    lingbot_repo: str | None = None
    checkpoint_dir: str | None = None
    host: str = "127.0.0.1"
    port: int = 8920
    request_timeout_s: int = 300
    scenario: Literal["single", "chunk", "adapter_cache", "dual_arm", "pose_delta", "discrete", "tool", "stop"] = "chunk"
    chunk_length: int = 8
    observation_update: Literal["every_env_step", "decision_boundary"] = "every_env_step"
    delta_reference: Literal["chunk_start", "step_observed", "previous_command"] = "step_observed"
    delta_composition: Literal["pre_multiply", "post_multiply"] = "pre_multiply"
    fault: Literal["none", "stale_observation", "cross_episode", "wrong_request", "invalid_unit", "service_error", "begin_error", "end_error", "close_error"] = "none"
    fault_episode_id: str | None = None
    report_forward_count: bool = True


@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    adapter: Literal["mock_benchmark", "libero_spatial", "libero_openpi", "vlabench", "vlabench_openpi", "robotwin_openwam", "robotwin_lingbot"] = "mock_benchmark"
    libero_repo: str | None = None
    vlabench_repo: str | None = None
    robotwin_repo: str | None = None
    task_config: str = "demo_clean"
    instruction_type: Literal["seen", "unseen"] = "unseen"
    max_seed_candidates: int = 100
    task_name: str = "select_fruit"
    track: str = "track_1_in_distribution"
    trial_index: int = 0
    control_period_s: float = .05
    tool_ticks: int = 3
    intermediate_observations_supported: bool = True
    success_at_step: int | None = None
    fault: Literal["none", "partial_execution_error", "score_error", "missing_history", "reset_error", "close_error", "planning_failed"] = "none"
    fault_episode_id: str | None = None
    fault_after_step: int = 1


@dataclass(frozen=True, kw_only=True)
class ProfileConfig:
    chunk_mode: Literal["runner_owned", "adapter_owned"] = "runner_owned"
    execution_horizon_steps: int = 2


@dataclass(frozen=True, kw_only=True)
class EpisodeConfig:
    episode_id: str
    seed: int = 0
    budgets: c.Budgets = field(default_factory=lambda: c.Budgets(env_steps=6, model_calls=6, tool_calls=3))


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    run_id: str
    episodes: tuple[EpisodeConfig, ...]
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


def validate_config(config: RunConfig):
    decode(RunConfig, config)
    pattern = r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}"
    require(re.fullmatch(pattern, config.run_id) is not None, "invalid_id", "run_id", "use a safe, nonempty run ID")
    require(bool(config.episodes), "missing_episode", "episodes", "at least one episode required")
    pair = (config.model.adapter, config.benchmark.adapter)
    from . import stage4_pairs
    require(pair in (("mock_model", "mock_benchmark"), ("openwam_libero", "libero_spatial")) or pair in stage4_pairs.PAIRS,
            "adapter_pair_mismatch", "adapters", "only registered model/benchmark pairs may run")
    if pair in stage4_pairs.PAIRS:
        stage4_pairs.validate(config)
        for name in ("scenario", "chunk_length", "observation_update", "delta_reference", "delta_composition",
                     "fault", "fault_episode_id", "report_forward_count"):
            require(getattr(config.model, name) == getattr(ModelConfig(), name),
                    "mock_option_on_real_adapter", "model." + name, "mock-only option cannot configure a native adapter")
        for name in ("control_period_s", "tool_ticks", "intermediate_observations_supported", "success_at_step",
                     "fault", "fault_episode_id", "fault_after_step"):
            require(getattr(config.benchmark, name) == getattr(BenchmarkConfig(), name),
                    "mock_option_on_real_adapter", "benchmark." + name, "mock-only option cannot configure a native benchmark")
    if pair == ("openwam_libero", "libero_spatial"):
        mock_model_defaults = ModelConfig()
        mock_benchmark_defaults = BenchmarkConfig()
        for name in ("scenario", "chunk_length", "observation_update", "delta_reference",
                     "delta_composition", "fault", "fault_episode_id", "report_forward_count"):
            require(getattr(config.model, name) == getattr(mock_model_defaults, name),
                    "mock_option_on_real_adapter", "model." + name, "mock-only option cannot configure OpenWAM")
        for name in ("control_period_s", "tool_ticks", "intermediate_observations_supported",
                     "success_at_step", "fault", "fault_episode_id", "fault_after_step"):
            require(getattr(config.benchmark, name) == getattr(mock_benchmark_defaults, name),
                    "mock_option_on_real_adapter", "benchmark." + name, "mock-only option cannot configure LIBERO")
        require(len(config.episodes) == 1, "pilot_scope", "episodes", "pilot supports exactly one episode")
        require(config.model.openwam_repo is not None and config.benchmark.libero_repo is not None
                and config.model.checkpoint_dir is not None,
                "missing_source", "adapters", "explicit OpenWAM, LIBERO and checkpoint paths required")
        require(config.model.host in ("127.0.0.1", "localhost") and 1 <= config.model.port <= 65535
                and config.model.request_timeout_s > 0, "invalid_endpoint", "model", "local endpoint required")
        require(config.profile.chunk_mode == "adapter_owned" and config.profile.execution_horizon_steps == 1,
                "pilot_profile_mismatch", "profile", "native policy owns its action cache")
        ep = config.episodes[0]
        require(ep.seed == 42 and ep.budgets.env_steps == 600 and ep.budgets.sim_time_s == 30
                and ep.budgets.model_calls == 600, "pilot_scope", "episode",
                "pilot fixes seed=42 and 600 steps / 30 simulation seconds / 600 model calls")
    ids = [ep.episode_id for ep in config.episodes]
    require(len(ids) == len(set(ids)), "duplicate_id", "episodes", "episode IDs must be unique")
    require(config.benchmark.control_period_s > 0, "invalid_timing", "control_period_s", "positive control period required")
    require(config.benchmark.tool_ticks > 0 and config.model.chunk_length > 0
            and config.profile.execution_horizon_steps > 0, "invalid_horizon", "config", "positive lengths required")
    require(config.benchmark.success_at_step is None or config.benchmark.success_at_step > 0,
            "invalid_success_condition", "success_at_step", "explicit success step must be positive")
    require(config.benchmark.fault_after_step > 0, "invalid_fault", "fault_after_step", "positive failure step required")
    for selector in (config.model.fault_episode_id, config.benchmark.fault_episode_id):
        require(selector is None or selector in ids, "unknown_episode", "fault_episode_id", "fault episode not selected")
    for ep in config.episodes:
        require(re.fullmatch(pattern, ep.episode_id) is not None, "invalid_id", "episode_id", "use a safe episode ID")
        require(0 <= ep.seed < 2**32, "invalid_seed", ep.episode_id, "use a uint32 seed")
        require(all(value is None or value >= 0 for value in to_data(ep.budgets).values()),
                "invalid_budget", ep.episode_id, "budget cannot be negative")
        bounded = any(value is not None for value in (ep.budgets.env_steps, ep.budgets.sim_time_s,
                      ep.budgets.model_calls, ep.budgets.wall_time_s))
        bounded |= config.model.scenario == "tool" and ep.budgets.tool_calls is not None
        require(bounded or config.model.scenario == "stop", "unbounded_mock_run", ep.episode_id,
                "synthetic nonterminal policies require a finite relevant budget")
    return config


def load_config(path: Path):
    import json
    if json.loads(path.read_text()).get("contract_version") == "0.2":
        from .mn.config import load_config as load_mn
        return load_mn(path)
    return validate_config(loads(RunConfig, path.read_text()))


def make_adapters(config, output=None):
    from . import stage4_pairs
    if (config.model.adapter, config.benchmark.adapter) in stage4_pairs.PAIRS:
        return stage4_pairs.make_adapters(config, output)
    if config.model.adapter == "openwam_libero":
        verify_real_assets(config)
        from .adapters.openwam_libero_model import OpenWAMLiberoModel
        from .adapters.libero_spatial_benchmark import LiberoSpatialBenchmark
        return OpenWAMLiberoModel(config.model, output), LiberoSpatialBenchmark(config.benchmark)
    from .adapters.mock_model import MockModel
    from .adapters.mock_benchmark import MockBenchmark
    return MockModel(config.model, config.benchmark.tool_ticks), MockBenchmark(config.benchmark)


def verify_real_assets(config):
    """Fail closed on source and checkpoint mismatch before claiming real provenance."""
    from .adapters.openwam_libero_spec import (MODEL_REVISION, BENCHMARK_REVISION, CHECKPOINT_SHA256,
        NORMALIZATION_SHA256, TASK_STEM, BDDL_SHA256, INIT_SHA256)
    for path, expected in ((config.model.openwam_repo, MODEL_REVISION),
                           (config.benchmark.libero_repo, BENCHMARK_REVISION)):
        root = Path(path).resolve()
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        require(result.returncode == 0 and result.stdout.strip() == expected,
                "source_revision_mismatch", str(root), f"expected source revision {expected}")
    directory = Path(config.model.checkpoint_dir).resolve()
    benchmark_root = Path(config.benchmark.libero_repo).resolve() / "libero" / "libero"
    assets = (
        (directory / "checkpoint_step_10690.safetensors", CHECKPOINT_SHA256),
        (directory / "normalization_stats.npy", NORMALIZATION_SHA256),
        (benchmark_root / "bddl_files" / "libero_spatial" / (TASK_STEM + ".bddl"), BDDL_SHA256),
        (benchmark_root / "init_files" / "libero_spatial" / (TASK_STEM + ".pruned_init"), INIT_SHA256),
    )
    for path, expected in assets:
        require(path.is_file(), "asset_missing", str(path), "pilot model asset missing")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        require(digest.hexdigest() == expected, "asset_hash_mismatch", str(path), "pilot asset differs from reviewed SHA256")


def make_profile(config, space, cap):
    from . import stage4_pairs
    if (config.model.adapter, config.benchmark.adapter) in stage4_pairs.PAIRS:
        return stage4_pairs.make_profile(config, space, cap)
    if config.model.adapter == "openwam_libero":
        from .adapters.openwam_libero_spec import profile
        return profile(space, cap)
    from .adapters.mock_spec import make_case
    _, _, base, _, _ = make_case()
    scenario = config.model.scenario
    bindings = dict(base.input_bindings)
    if scenario == "dual_arm":
        bindings |= {key: c.InputBinding(kind="sensor", source_id=key) for key in ("left_wrist_rgb", "right_wrist_rgb")}
    return replace(base, profile_id="mock-pair@1", revision="mock-profile-1", benchmark_spec_ref=space.spec_id,
        model_id=cap.model_id, model_source_revision=cap.source_revision, checkpoint_ref=cap.checkpoint_ref,
        normalization_ref=cap.normalization_ref, role="agent" if scenario == "tool" else "action_policy",
        input_bindings=bindings, action_spec_refs=() if scenario in ("tool", "discrete") else cap.produced_action_specs,
        discrete_spec_refs=cap.produced_discrete_specs if scenario == "discrete" else (),
        tool_refs=cap.produced_tools if scenario == "tool" else (),
        chunk_mode=config.profile.chunk_mode, execution_horizon_steps=config.profile.execution_horizon_steps,
        view_policy=replace(base.view_policy, robot_fields=base.view_policy.robot_fields + ("robot0/base_pose",)))


def fallback_metrics(config=None):
    # Registry metadata, not a benchmark-specific assumption inside the runner.
    from . import stage4_pairs
    if config is not None and (config.model.adapter, config.benchmark.adapter) in stage4_pairs.PAIRS:
        return stage4_pairs.fallback_metrics(config)
    if config is not None and config.benchmark.adapter == "libero_spatial":
        return {"success": c.Metric(value=None, unit="boolean", source="LIBERO official environment done@8f1084e",
                                    valid=False, unavailable_reason="not_scored")}
    return {"success": c.Metric(value=None, unit="boolean", source="mock-scorer@1", valid=False,
                               unavailable_reason="not_scored")}


def run_provenance(config):
    """Registry-owned origin marker, independent of episode success."""
    from . import stage4_pairs
    if (config.model.adapter, config.benchmark.adapter) in stage4_pairs.PAIRS:
        return stage4_pairs.provenance(config)
    if config.model.adapter == "openwam_libero":
        from .adapters.openwam_libero_spec import (MODEL_REVISION, BENCHMARK_REVISION, CHECKPOINT_SHA256,
            NORMALIZATION_SHA256, HF_REVISION, BDDL_SHA256, INIT_SHA256)
        return {"synthetic": False, "model_adapter": "openwam_libero", "benchmark_adapter": "libero_spatial",
                "model_source_revision": MODEL_REVISION, "benchmark_source_revision": BENCHMARK_REVISION,
                "model_asset_revision": HF_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256, "normalization_sha256": NORMALIZATION_SHA256,
                "bddl_sha256": BDDL_SHA256, "init_state_sha256": INIT_SHA256,
                "native_reference_policy_config_sha256": "d4211e25b46ec5e164b3662f4ed06743bef2be9ca8243f7c41b14e631f1a89cf",
                "server_identity_verified": False}
    return {"synthetic": True, "model_adapter": "mock_model", "benchmark_adapter": "mock_benchmark"}


def adapter_diagnostics(model, benchmark):
    return {"model": model.diagnostics(), "benchmark": benchmark.diagnostics()}
