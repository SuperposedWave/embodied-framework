"""Internal parameters for reused environment loaders, NOT a run config."""
from dataclasses import dataclass, field
from typing import Literal
from ..native import environment_records as c

@dataclass(frozen=True, kw_only=True)
class BenchmarkConfig:
    libero_repo: str | None = None
    vlabench_repo: str | None = None
    robotwin_repo: str | None = None
    task_config: str = "demo_clean"
    instruction_type: Literal["seen", "unseen"] = "unseen"
    max_seed_candidates: int = 100
    task_name: str = "select_fruit"
    track: str = "track_1_in_distribution"
    trial_index: int = 0


@dataclass(frozen=True, kw_only=True)
class EpisodeConfig:
    episode_id: str
    seed: int = 0
    budgets: c.Budgets = field(default_factory=lambda: c.Budgets(env_steps=6, model_calls=6, tool_calls=3))
