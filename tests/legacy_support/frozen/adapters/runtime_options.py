"""Internal parameters for reused environment loaders, NOT a run config."""
from dataclasses import dataclass, field
from typing import Literal
import tests.legacy_support.frozen.environment_records as c

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
class EpisodeConfig:
    episode_id: str
    seed: int = 0
    budgets: c.Budgets = field(default_factory=lambda: c.Budgets(env_steps=6, model_calls=6, tool_calls=3))
