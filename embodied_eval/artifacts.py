"""Local run artifacts; public inputs and internal traces are stored separately."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from .codec import to_data
from .errors import ContractError


class ArtifactWriter:
    def __init__(self, output):
        self.root = Path(output).resolve()
        try:
            self.root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ContractError("output_exists", "--output", "refusing to overwrite an existing output directory") from error

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(to_data(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        temporary.replace(path)
        return str(path)

    def append(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(to_data(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    def manifest(self, config, space, cap, profile, tasks, fallback_metrics, provenance):
        source = Path(__file__).parent
        digests = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(source.rglob("*.py"))}
        lock = source.parent / "uv.lock"
        return self.write("manifest.json", {
            "synthetic": provenance["synthetic"], "provenance": provenance,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config, "space": space, "capability": cap, "profile": profile, "tasks": tasks,
            "fallback_metric_definitions": fallback_metrics, "source_sha256": digests,
            "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
            "runtime": {"python": platform.python_version(), "package": importlib.metadata.version("embodied-eval"),
                        "jsonschema": importlib.metadata.version("jsonschema")},
            "execution": {"workers": 1, "environments_per_worker": 1, "automatic_retry": False,
                          "resume": False, "wall_timeout": "cooperative_checks_only"},
        })
