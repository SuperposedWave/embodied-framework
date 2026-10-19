"""Opt-in whole-run backends, selected BEFORE legacy ModelAdapter binding.

This is an additive MassWorld integration API, not a robot observation contract.
Normal model/benchmark runners do not enter this path.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from importlib.metadata import entry_points
import json
import math
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from .contracts import Artifact, Cache, EvaluationResult, Plan, Retry, Stage
from .execution import ExecutionContext, ExecutionEngine, StagePayload
from .io import stable_hash, write_json
from .planner import PlanDecision
from .stores import EvidenceStore

GROUP = "worldevalkit.execution_backends"
_SECRET_KEYS = {"api_key", "apikey", "access_token", "refresh_token", "token", "password", "secret", "authorization"}


def guard_config(value):
    if isinstance(value, Mapping):
        if any(str(key).lower().replace("-", "_") in _SECRET_KEYS for key in value):
            raise ValueError("credentials must not be placed in backend configuration")
        for item in value.values():
            guard_config(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            guard_config(item)


def load_config(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate configuration key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("non-finite configuration number")
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    guard_config(value)
    return value


def backend_type(name):
    matches = [ep for ep in entry_points(group=GROUP) if ep.name == name]
    if len(matches) != 1:
        raise ValueError("execution backend must have exactly one installed entry point: " + str(name))
    value = matches[0].load()
    if not callable(value) or not callable(getattr(value, "catalog", None)):
        raise TypeError("execution backend must be a factory with catalog()")
    return value


def backend_catalog(name):
    return backend_type(name).catalog()


class BackendEvaluationRunner:
    def __init__(self, *, execution_backend, backend_config, output_dir, record_dir=None):
        guard_config(backend_config)
        self.config = copy.deepcopy(backend_config)
        self.backend_id = execution_backend
        self.backend = backend_type(execution_backend)(self.config)
        for method in ("preflight", "execute"):
            if not callable(getattr(self.backend, method, None)):
                raise TypeError("execution backend missing " + method)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.record_dir = Path(record_dir).expanduser().resolve() if record_dir else None
        identity = stable_hash({"backend": execution_backend, "config": self.config})
        self.run_dir = self.output_dir / ".backend-plans" / identity
        self._outcome = None

    def plan(self, *, write=True, inspect_readiness=True):
        report = self.backend.preflight(verify_assets=inspect_readiness)
        readiness = tuple(report.get('readiness', {}).values())
        runtime_verified = bool(readiness) and all(
            item.get('dependencies_verified') is True or
            (item.get('synthetic') is True and item.get('available') is True)
            for item in readiness)
        status = ("not_applicable" if not report["structurally_compatible"]
                  else "blocked" if report.get("asset_errors")
                  else "runnable" if report.get("assets_verified") and runtime_verified else "needs_setup")
        reasons = tuple(i["code"] for i in report["issues"]) + tuple(
            "asset verification failed: " + side for side in report.get("asset_errors", {}))
        if status == 'needs_setup':
            reasons += tuple(reason for condition, reason in (
                (not report.get('assets_verified'), 'assets_not_verified'),
                (not runtime_verified, 'runtime_dependencies_unverified')) if condition)
        plan = Plan(
            id="execution-backend",
            stages=(Stage(id="backend-run", uses="execution_backend.run",
                          outputs={"result": "core.metrics"},
                          retry=Retry(attempts=1), cache=Cache(enabled=False)),),
            metadata={"execution_backend": self.backend_id, "preflight": report,
                      "runtime_dependencies_verified": runtime_verified,
                      "cache": False, "resume": False, "automatic_retry": False,
                      "execution_loop_owner": "backend"},
        )
        decision = PlanDecision(status, reasons, plan=plan)
        if write:
            write_json(self.run_dir / "plan.json", decision.to_dict())
        return decision

    def doctor(self):
        decision = self.plan(write=False)
        return {"ok": decision.runnable, "planning": decision.to_dict(),
                "read_only_asset_verification": True}

    def setup(self):
        # Deliberately no install/download/provision operation in the first bridge.
        report = self.doctor()
        return {"ready": report["ok"], **report,
                "provisioning": "not_implemented; prepare native assets/environments separately"}

    def _execute_stage(self, stage, inputs, dependencies, context):
        outcome = self.backend.execute(context.run_dir / "backend")
        guard_config(outcome)
        if type(outcome.get("success")) is not bool or type(outcome.get("execution_state_known")) is not bool:
            raise ValueError("backend outcome must declare success and execution state")
        numeric = outcome.get("numeric_metrics", {})
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in numeric.values()):
            raise ValueError("numeric projection must exclude bool, null and non-finite values")
        result = EvaluationResult(metrics=numeric, samples=tuple(outcome["samples"]),
                                  metadata={**outcome["metadata"], "execution_state_known": outcome["execution_state_known"],
                                            "evidence": outcome.get("evidence", {})})
        write_json(context.run_dir / "backend-result.json", outcome)
        write_json(context.run_dir / "result.json", result.to_dict())
        self._outcome = outcome
        if not outcome["success"]:
            raise RuntimeError("backend run failed; inspect backend-result.json and native evidence; automatic retry disabled")
        return StagePayload(outputs={"result": Artifact.value("core.metrics", result.to_dict())})

    def run(self):
        from .runner import _termination_guard
        # Validate before allocating any native model/environment factories.
        decision = self.plan(write=False, inspect_readiness=False)
        if decision.status in {"not_applicable", "blocked"}:
            raise RuntimeError("backend preflight rejected: " + "; ".join(decision.reasons))
        target = self.record_dir or self.output_dir / (
            "backend-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12])
        # Refuse reuse even if an earlier execution was interrupted or incomplete.
        target.mkdir(parents=True, exist_ok=False)
        self.run_dir = target
        self._outcome = None
        write_json(target / "plan.json", decision.to_dict())
        evidence = EvidenceStore(target / "evidence", target.name)
        context = ExecutionContext(
            plan=decision.plan, run_dir=target,
            handlers={"execution_backend.run": self._execute_stage},
            evidence=evidence, resume=False, history=None,
            fingerprint=stable_hash({"backend": self.backend_id, "config": self.config}),
        )
        write_json(target / "evidence/provenance.json", {
            "execution_backend": self.backend_id, "config": self.config,
            "cache": False, "resume": False, "automatic_retry": False,
            "execution_loop_owner": "backend",
        })
        with _termination_guard(target):
            stages = ExecutionEngine(context).run()
        outcome = self._outcome
        success = all(s.successful for s in stages.values()) and outcome is not None and outcome["success"]
        scorecard = {
            "success": success, "execution_backend": self.backend_id,
            "success_meaning": "framework completed without errors, not task success",
            "execution_state_known": outcome["execution_state_known"] if outcome else None,
            "synthetic": outcome["metadata"].get("synthetic") if outcome else None,
            "metrics": outcome["numeric_metrics"] if outcome else {},
            "result_ref": str(target / "result.json") if outcome else None,
            "backend_output_dir": str(target / 'backend'),
            "backend_evidence": outcome.get('evidence', {}) if outcome else {},
            "stages": {k: v.to_dict() for k, v in stages.items()},
        }
        write_json(target / "scorecard.json", scorecard)
        return scorecard
