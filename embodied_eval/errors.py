"""Errors remain separate from benchmark task failure."""
from dataclasses import dataclass
from typing import Literal

Category = Literal[
    "incompatibility", "contract", "model_input", "model_output", "model_service",
    "tool_argument", "planning", "execution", "benchmark", "scoring",
    "infrastructure", "budget",
]


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str


class ContractError(ValueError):
    def __init__(self, code: str, path: str, message: str):
        self.issue = Issue(code, path, message)
        super().__init__(f"{code} at {path}: {message}")


class AdapterError(RuntimeError):
    """Do not forward the diagnostic message to the model."""
    def __init__(
        self, category: Category, code: str, message: str, *,
        request_id: str | None = None, execution_started: bool = False,
        state_known: bool = True, retryability: Literal["never", "idempotent_only"] = "never",
    ):
        self.category = category
        self.code = code
        self.request_id = request_id
        self.execution_started = execution_started
        self.state_known = state_known
        self.retryability = retryability
        super().__init__(message)
