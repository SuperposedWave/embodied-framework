"""Only explicit current-version configurations; no implicit legacy fallback."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import re
from ..codec import decode, loads, to_data
from ..core.types import Budgets
from ..core.validation import require


@dataclass(frozen=True, kw_only=True)
class AdapterConfig:
    adapter: str
    options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class EpisodeConfig:
    episode_id: str
    seed: int = 0
    budgets: Budgets = field(default_factory=lambda: Budgets(env_steps=6, model_calls=6))


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    run_id: str
    model: AdapterConfig
    benchmark: AdapterConfig
    episodes: tuple[EpisodeConfig, ...]
    control_mode: str | None = None
    contract_version: Literal["0.3"] = "0.3"


def validate_config(config, registry=None):
    require(isinstance(config, RunConfig), "unsupported_config_version", "contract_version",
            "current run/check only accept 0.3 configurations; migrate old files explicitly")
    decode(RunConfig, config)
    pattern = r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}"
    require(re.fullmatch(pattern, config.run_id) is not None, "invalid_id", "run_id", "safe run ID required")
    require(bool(config.episodes), "missing_episode", "episodes", "select at least one episode")
    ids = [ep.episode_id for ep in config.episodes]
    require(len(ids) == len(set(ids)), "duplicate_id", "episodes", "unique episode IDs required")
    for ep in config.episodes:
        require(re.fullmatch(pattern, ep.episode_id) is not None, "invalid_id", "episode_id", "safe episode ID required")
        require(0 <= ep.seed < 2**32, "invalid_seed", ep.episode_id, "uint32 seed required")
        require(all(v is None or v >= 0 for v in to_data(ep.budgets).values()), "invalid_budget", ep.episode_id, "nonnegative budgets")
        require(any(v is not None for v in (ep.budgets.env_steps, ep.budgets.sim_time_s,
                    ep.budgets.model_calls, ep.budgets.wall_time_s)), "unbounded_run", ep.episode_id, "finite execution budget required")
    from .registry import default_registry
    registry = registry or default_registry()
    registry.model(config.model.adapter).validate_options(config.model.options)
    registry.benchmark(config.benchmark.adapter).validate_options(config.benchmark.options)
    return config


def load_config(path: Path):
    raw = loads(object, Path(path).read_text(encoding="utf-8"))
    require(isinstance(raw, dict) and raw.get("contract_version") == "0.3",
            "unsupported_config_version", "contract_version",
            "only explicit contract_version=0.3 is runnable; old/missing versions require offline migration")
    return validate_config(decode(RunConfig, raw))
