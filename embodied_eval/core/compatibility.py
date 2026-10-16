"""Version-neutral compatibility result, without a binding implementation."""
from dataclasses import dataclass
from ..errors import Issue


@dataclass(frozen=True)
class CompatibilityReport:
    issues: tuple[Issue, ...]
    warnings: tuple[str, ...]

    @property
    def compatible(self):
        return not self.issues

    def require_compatible(self):
        if self.issues:
            from ..errors import ContractError
            i = self.issues[0]
            raise ContractError(i.code, i.path, i.message)
