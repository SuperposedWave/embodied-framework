"""Generic sequential evaluation. Model and robot-specific logic stays in adapters."""
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time

from . import contracts as c
from .artifacts import ArtifactWriter
from .codec import decode, to_data
from .compatibility import check_compatibility, validate_model_input
from .errors import AdapterError, ContractError
from .interfaces import ModelAdapter, BenchmarkAdapter
from .validation import (
    require, same_episode, validate_observation, validate_task,
    validate_prediction, validate_step_result, validate_episode_result,
)
from .visibility import build_model_input, public_space, public_task


@dataclass
class Ledger:
    env_steps: int = 0
    sim_time_s: float = 0.
    model_calls: int = 0
    tool_calls: int = 0
    forward_count: int = 0
    forward_known: bool = True
    physics_steps: int | None = None

    def statistics(self):
        return c.ExecutionStatistics(env_steps=self.env_steps, sim_time_s=self.sim_time_s,
            model_calls=self.model_calls, tool_calls=self.tool_calls,
            model_forward_count=self.forward_count if self.forward_known else None,
            physics_steps=self.physics_steps)

    def remaining(self, limits, started):
        values = {}
        for field in ("env_steps", "sim_time_s", "model_calls", "tool_calls", "wall_time_s"):
            limit = getattr(limits, field)
            used = time.monotonic()-started if field == "wall_time_s" else getattr(self, field)
            values[field] = None if limit is None else max(0, limit-used)
        return c.Budgets(**values)


@dataclass
class EpisodeOutcome:
    result: c.EpisodeResult
    errors: list
    statistics_complete: bool


@dataclass(frozen=True)
class RunOutcome:
    output: Path
    results: tuple[c.EpisodeResult, ...]
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


def _execution_budget_exhausted(remaining, period):
    return ((remaining.env_steps is not None and remaining.env_steps < 1)
        or (remaining.sim_time_s is not None and period is not None and remaining.sim_time_s + 1e-9 < period)
        or (remaining.wall_time_s is not None and remaining.wall_time_s <= 0))


def _prediction_budget_exhausted(remaining, period, profile):
    tools_only = bool(profile.tool_refs) and not (profile.action_spec_refs or profile.discrete_spec_refs)
    return (_execution_budget_exhausted(remaining, period)
        or (remaining.model_calls is not None and remaining.model_calls <= 0)
        or (tools_only and remaining.tool_calls is not None and remaining.tool_calls <= 0))


def _record_progress(reply, previous, ledger, request, space):
    # Record consistent executed counts before checking the full sensor/history payload.
    same_episode(reply.context, previous.context)
    same_episode(reply.observation.context, previous.context)
    feedback = decode(c.ExecutionFeedback, reply.execution_feedback)
    require(feedback.request_id == request.message_id, "request_mismatch", "feedback", "response is not for the pending execution")
    step = decode(int, reply.observation.env_step)
    sim_time = decode(float, reply.observation.sim_time_s)
    require(step >= previous.env_step and step-previous.env_step == feedback.consumed_env_steps,
            "execution_count_mismatch", "feedback", "actual tick delta differs")
    require(sim_time >= previous.sim_time_s and math.isclose(sim_time-previous.sim_time_s, feedback.elapsed_sim_time_s, abs_tol=1e-9),
            "execution_time_mismatch", "feedback", "actual simulation time delta differs")
    if space.timing_mode == "measured_native":
        require(feedback.consumed_physics_steps is not None and feedback.consumed_physics_steps >= 0,
                "invalid_execution_count", "feedback.consumed_physics_steps", "native physical progress must be measured")
        require(math.isclose(feedback.elapsed_sim_time_s, feedback.consumed_physics_steps * space.physics_period_s, abs_tol=1e-9),
                "execution_time_mismatch", "feedback", "measured physical steps and simulation time disagree")
        ledger.physics_steps += feedback.consumed_physics_steps
    ledger.env_steps, ledger.sim_time_s = step, sim_time


def _unavailable(templates, code):
    return {key: replace(value, value=None, valid=False, unavailable_reason=code) for key, value in templates.items()}


def run_episode(model: ModelAdapter, benchmark: BenchmarkAdapter, profile: c.AdaptationProfile,
                task: c.TaskSpec, space: c.SpaceSpec, cap: c.CapabilitySpec, writer: ArtifactWriter,
                run_id: str, metric_templates: dict[str, c.Metric]) -> EpisodeOutcome:
    prefix = f"episodes/{task.episode_id}"
    context = c.Context(run_id=run_id, env_id="env-0", episode_id=task.episode_id, message_id="episode-start")
    ledger, errors, seen = Ledger(physics_steps=0 if space.timing_mode == "measured_native" else None), [], set()
    received_ids = set()
    started = time.monotonic()
    phase, reason, status = "reset", "run_error", "completed"
    reset_done = begin_attempted = False
    statistics_complete = True
    execution_in_flight = False
    feedback = None
    writer.write(prefix + "/task.internal.json", task)

    def trace(event, data):
        writer.append(prefix + "/trace.internal.jsonl", {"event": event, "data": data})

    def observe(obs, latest_feedback=None):
        if obs.observation_id in seen:
            return
        view = build_model_input(task, obs, profile, latest_feedback)
        validate_model_input(view, space, cap, profile)
        writer.append(prefix + "/model_input.public.jsonl", view)
        model.observe(view)
        seen.add(obs.observation_id)
        trace("observe", {"observation_id": obs.observation_id, "env_step": obs.env_step})

    try:
        validate_task(task, space)
        obs = benchmark.reset(task, context)
        validate_observation(obs, space)
        same_episode(obs.context, context)
        require(obs.env_step == 0 and obs.sim_time_s == 0, "invalid_reset_clock", "reset", "episode control clock starts at zero")
        reset_done = True
        received_ids.add(obs.observation_id)
        trace("reset", obs)
        phase, begin_attempted = "begin", True
        task_view = public_task(task, profile.view_policy)
        space_view = public_space(space, profile)
        trace("begin_public", {"context": context, "task": task_view, "space": space_view, "profile": profile})
        model.begin_episode(context, task_view, space_view, profile)
        phase = "observe"
        observe(obs)
        while True:
            remaining = ledger.remaining(task.budgets, started)
            if _prediction_budget_exhausted(remaining, space.control_period_s, profile):
                reason = "budget_exhausted"
                break
            phase = "observe"
            observe(obs, feedback)
            phase = "predict"
            request = replace(context, message_id=f"{task.episode_id}:request:{ledger.model_calls+1}")
            ledger.model_calls += 1
            forward_was_known = ledger.forward_known
            ledger.forward_known = False  # A thrown/malformed call leaves the actual forward count unknown.
            pred = model.predict(request)
            pred = decode(c.Prediction, pred)
            count = pred.diagnostics.model_forward_count
            # Retain knowledge across responses; a single unknown report stays unknown.
            if count is not None and count >= 0 and forward_was_known:
                ledger.forward_count += count
                ledger.forward_known = True
            trace("prediction", pred)
            validate_prediction(pred, obs, space, cap, profile, request_context=request)
            remaining = ledger.remaining(task.budgets, started)
            if remaining.wall_time_s is not None and remaining.wall_time_s <= 0:
                reason = "budget_exhausted"
                break
            if isinstance(pred.decision, c.StopRequest):
                reason = "model_stop"
                break
            decision = pred.decision
            operations = decision.steps[:decision.execution_horizon_steps] if isinstance(decision, c.ActionSequence) else (decision,)
            done = False
            for operation in operations:
                remaining = ledger.remaining(task.budgets, started)
                if _execution_budget_exhausted(remaining, space.control_period_s):
                    reason, done = "budget_exhausted", True
                    break
                if isinstance(operation, c.ToolCall):
                    if remaining.tool_calls is not None and remaining.tool_calls <= 0:
                        reason, done = "budget_exhausted", True
                        break
                    ledger.tool_calls += 1
                phase = "execute"
                execution_in_flight = True
                previous = obs
                reply = (benchmark.execute_tool(operation, request, remaining) if isinstance(operation, c.ToolCall)
                         else benchmark.advance(operation, request, remaining))
                trace("step_result", reply)
                _record_progress(reply, previous, ledger, request, space)
                execution_in_flight = False
                require(task.budgets.env_steps is None or ledger.env_steps <= task.budgets.env_steps,
                        "budget_overrun", "env_steps", "adapter exceeded environment budget")
                require(task.budgets.sim_time_s is None or ledger.sim_time_s <= task.budgets.sim_time_s + 1e-9,
                        "budget_overrun", "sim_time_s", "adapter exceeded simulation time budget")
                validate_step_result(reply, previous, space, expected_prediction=pred, capability=cap)
                for snapshot in (*reply.intermediate_observations, reply.observation):
                    require(snapshot.observation_id not in received_ids, "observation_id_reused", "observation_id",
                            "each returned snapshot needs a unique ID")
                    received_ids.add(snapshot.observation_id)
                if isinstance(operation, c.ActionStep) and reply.execution_feedback.status == "executed":
                    require(reply.execution_feedback.consumed_env_steps == 1, "invalid_execution_count", "advance",
                            "an executed ActionStep consumes exactly one tick")
                obs, feedback = reply.observation, reply.execution_feedback
                phase = "observe"
                if cap.observation_update == "every_env_step":
                    for intermediate in reply.intermediate_observations:
                        observe(intermediate)
                    observe(obs, feedback)
                if feedback.status == "execution_failed":
                    phase = "execute"
                    raise AdapterError("execution", feedback.code, "adapter reported partial execution failure",
                                       execution_started=feedback.consumed_env_steps > 0, state_known=True)
                if reply.terminated or reply.truncated:
                    reason = "environment_terminated" if reply.terminated else "budget_exhausted"
                    done = True
                    break
                if isinstance(operation, c.ActionStep) and feedback.status != "executed":
                    # Re-plan from the returned state; discard the rest of this chunk.
                    break
            if done:
                break
    except Exception as error:
        status, reason = "error", "run_error"
        item = _error(error, phase)
        if execution_in_flight and not (isinstance(error, AdapterError) and error.state_known and not error.execution_started):
            item["state_known"] = False
        errors.append(item)
        statistics_complete = item["state_known"]
        if not statistics_complete:
            reason = "execution_state_unknown"
        trace("error", item)
    finally:
        metrics = _unavailable(metric_templates, "episode_not_initialized")
        if reset_done and statistics_complete:
            try:
                phase = "score"
                metrics = benchmark.score(replace(context, message_id="score"))
                require(set(metric_templates) <= metrics.keys(), "missing_metric", "score", "required metric missing")
                probe = c.EpisodeResult(episode_id=task.episode_id, run_manifest_ref=str(writer.root/"manifest.json"),
                    run_status=status, termination_reason=reason, metrics=metrics, execution_statistics=ledger.statistics())
                validate_episode_result(probe)
                trace("score", metrics)
            except Exception as error:
                status, reason = "error", "run_error"
                errors.append(_error(error, "score"))
                trace("error", errors[-1])
                metrics = _unavailable(metric_templates, errors[-1]["code"])
        elif not statistics_complete:
            metrics = _unavailable(metric_templates, "execution_state_unknown")
        if begin_attempted:
            public_end = c.ModelEndContext(context=replace(context, message_id="episode-end"),
                reason=reason if reason in ("model_stop", "budget_exhausted", "environment_terminated") else "run_error")
            trace("end_public", public_end)
            try:
                model.end_episode(public_end)
            except Exception as error:
                status = "error"
                if statistics_complete:
                    reason = "run_error"
                errors.append(_error(error, "end"))
                trace("error", errors[-1])
    refs = tuple(str(writer.root/prefix/name) for name in (
        "task.internal.json", "trace.internal.jsonl", "model_input.public.jsonl", "diagnostics.internal.json")
        if name != "model_input.public.jsonl" or (writer.root/prefix/name).exists())
    result = c.EpisodeResult(episode_id=task.episode_id, run_manifest_ref=str(writer.root/"manifest.json"),
        run_status=status, termination_reason=reason, metrics=metrics,
        execution_statistics=ledger.statistics(), artifact_refs=refs)
    validate_episode_result(result)
    return EpisodeOutcome(result=result, errors=errors, statistics_complete=statistics_complete)


def _close_pair(model, benchmark):
    errors = []
    for adapter in (model, benchmark):
        try:
            adapter.close()
        except Exception as error:
            errors.append(_error(error, "close"))
    return errors


def run(config, output):
    """CLI registry bootstrap, followed by a model/benchmark-neutral episode loop."""
    from .mn.config import RunConfig as IndependentConfig
    if isinstance(config, IndependentConfig):
        from .mn.runner import run as independent_run
        return independent_run(config, output)
    from .config import validate_config, make_adapters, make_profile, fallback_metrics, adapter_diagnostics, run_provenance
    validate_config(config)
    model, benchmark = make_adapters(config, output)
    pair_open = True
    try:
        space, cap = benchmark.describe(), model.describe()
        profile = make_profile(config, space, cap)
        check_compatibility(space, cap, profile).require_compatible()
        selection = {"episodes": to_data(config.episodes)}
        tasks = benchmark.enumerate_tasks(selection)
        require([task.episode_id for task in tasks] == [ep.episode_id for ep in config.episodes],
                "task_selection_mismatch", "tasks", "benchmark returned different episodes")
        for task in tasks:
            validate_task(task, space)
        metrics = fallback_metrics(config)
        provenance = run_provenance(config)
        writer = ArtifactWriter(output)
        writer.manifest(config, space, cap, profile, tasks, metrics, provenance)
        results, rows = [], []
        for index, task in enumerate(tasks):
            if not pair_open:
                model, benchmark = make_adapters(config, output)
                pair_open = True
                require(model.describe() == cap and benchmark.describe() == space, "adapter_identity_changed",
                        "rebuild", "recreated adapters must keep the recorded identity")
                benchmark.enumerate_tasks(selection)
            outcome = run_episode(model, benchmark, profile, task, space, cap, writer, config.run_id, metrics)
            if outcome.result.run_status != "completed" or index == len(tasks)-1:
                close_errors = _close_pair(model, benchmark)
                pair_open = False
                for error in close_errors:
                    writer.append(f"episodes/{task.episode_id}/trace.internal.jsonl", {"event": "error", "data": error})
                if close_errors:
                    outcome.errors.extend(close_errors)
                    outcome.result = replace(outcome.result, run_status="error",
                        termination_reason="run_error" if outcome.statistics_complete else "execution_state_unknown")
            try:
                diagnostics = adapter_diagnostics(model, benchmark)
                writer.write(f"episodes/{task.episode_id}/diagnostics.internal.json", diagnostics)
            except Exception as error:
                outcome.errors.append(_error(error, "diagnostics"))
                outcome.result = replace(outcome.result, run_status="error",
                    termination_reason="run_error" if outcome.statistics_complete else "execution_state_unknown")
                if pair_open:
                    outcome.errors.extend(_close_pair(model, benchmark))
                    pair_open = False
            validate_episode_result(outcome.result)
            result_path = writer.write(f"episodes/{task.episode_id}/result.json", outcome.result)
            results.append(outcome.result)
            rows.append({"episode_id": task.episode_id, "run_status": outcome.result.run_status,
                "termination_reason": outcome.result.termination_reason, "result_ref": result_path,
                "errors": outcome.errors, "execution_statistics_complete": outcome.statistics_complete,
                "execution_statistics_scope": "full" if outcome.statistics_complete else "last_confirmed"})
        completed = sum(row["run_status"] == "completed" for row in rows)
        summary = {"run_id": config.run_id, "synthetic": provenance["synthetic"], "provenance": provenance,
                   "planned": len(tasks),
                   "completed": completed, "errors": len(rows)-completed, "episodes": rows}
        writer.write("summary.json", summary)
        return RunOutcome(output=writer.root, results=tuple(results), exit_code=0 if completed == len(tasks) else 1, summary=summary)
    finally:
        if pair_open:
            _close_pair(model, benchmark)
