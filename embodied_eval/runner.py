"""Current sequential entry point; adapter-specific work stays in adapters."""
from dataclasses import dataclass
from pathlib import Path
from .errors import AdapterError, ContractError


@dataclass(frozen=True)
class RunOutcome:
    output: Path
    results: tuple
    exit_code: int
    summary: dict


def _error(error, phase):
    if isinstance(error, AdapterError):
        return {"category": error.category, "code": error.code, "message": str(error), "phase": phase,
                "execution_started": error.execution_started, "state_known": error.state_known,
                "retryability": error.retryability}
    categories = {"predict": "model_output", "observe": "model_input", "begin": "model_service",
                  "reset": "benchmark", "execute": "execution", "score": "scoring",
                  "end": "model_service", "close": "infrastructure", "diagnostics": "infrastructure"}
    return {"category": categories.get(phase, "contract"),
            "code": error.issue.code if isinstance(error, ContractError) else type(error).__name__,
            "message": str(error), "phase": phase, "execution_started": phase == "execute",
            "state_known": phase != "execute" or isinstance(error, ContractError), "retryability": "never"}


def run(config, output, registry=None):
    from .v03.runner import run as current_run
    return current_run(config, output, registry=registry)
