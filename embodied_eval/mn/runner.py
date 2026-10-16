"""Sequential semantic runner: independent factories, no pair-specific branches."""
from dataclasses import replace
import copy
import time
from ..artifacts import ArtifactWriter
from ..codec import to_data, decode
from .._legacy_contracts import Budgets, Metric
from ..errors import ContractError
from .._legacy_validation import require
from ..runner import RunOutcome, _error
from . import contracts as c
from .binding import resolve, model_input
from .preflight import check
from .registry import default_registry
from .validation import validate_space, validate_observation, validate_prediction, validate_step, same_episode


def run(config, output, registry=None):
    registry = registry or default_registry()
    checked = check(config, registry)
    if checked["issues"]:
        i = checked["issues"][0]
        raise ContractError(i["code"], i["path"], i["message"])
    m, b = registry.model(config.model.adapter), registry.benchmark(config.benchmark.adapter)
    # Asset identity is never inferred from passing structural checks or success.
    assets = {"model": m.verify_assets(config.model.options), "benchmark": b.verify_assets(config.benchmark.options)}
    cap = m.describe(config.model.options)
    benchmark = model = None
    writer = None
    try:
        benchmark = b.factory(config.benchmark.options, output)
        space = validate_space(benchmark.describe())
        profile = resolve(space, cap, config.binding)
        tasks = benchmark.enumerate_tasks(config.episodes)
        require([t.episode_id for t in tasks] == [e.episode_id for e in config.episodes], "task_selection_mismatch", "tasks", "different selected episodes")
        model = m.factory(config.model.options, output)
        provenance = {"synthetic": m.synthetic or b.synthetic, "contract_version": "0.2",
                      "model": m.provenance(config.model.options), "benchmark": b.provenance(config.benchmark.options),
                      "asset_checks_completed": True, "verified_assets": assets, "server_identity_verified": False}
        writer = ArtifactWriter(output)
        metrics = b.metric_templates(config.benchmark.options)
        writer.manifest(config, space, cap, profile, tasks, metrics, provenance)
        rows, results = [], []
        for task in tasks:
            decode(c.TaskSpec, task)
            prefix = f"episodes/{task.episode_id}"
            writer.write(prefix + "/task.internal.json", task)
            context = c.Context(run_id=config.run_id, env_id="env-0", episode_id=task.episode_id, message_id="start")
            stats = dict(env_steps=0, sim_time_s=0., model_calls=0, tool_calls=0,
                         model_forward_count=0, physics_steps=0 if space.timing_mode == "measured_native" else None)
            errors, begun, reset, in_flight, state_known = [], False, False, False, True
            prediction_in_flight=False
            reason, phase, status = "budget_exhausted", "reset", "completed"
            started = time.monotonic()
            seen = set()
            def trace(event, data):
                writer.append(prefix + "/trace.internal.jsonl", {"event": event, "data": data})
            def remaining():
                return Budgets(**{key: None if value is None else max(0, value -
                    (time.monotonic() - started if key == "wall_time_s" else stats[key]))
                    for key, value in to_data(task.budgets).items()})
            def exhausted(left):
                return any(getattr(left, k) is not None and getattr(left, k) <= 0 for k in ("env_steps", "model_calls", "wall_time_s")) or (
                    left.sim_time_s is not None and left.sim_time_s + 1e-9 < space.control_period_s)
            try:
                obs = validate_observation(benchmark.reset(task, context), space)
                same_episode(obs.context, context)
                require(obs.env_step == 0 and obs.sim_time_s == 0, "invalid_reset_clock", "reset", "control clock starts at zero")
                reset = True
                seen.add(obs.observation_id)
                trace("reset", obs)
                phase, begun = "begin", True
                # No hidden task fields or benchmark object may enter the model.
                model.begin_episode(context, copy.deepcopy(task.public), copy.deepcopy(profile), copy.deepcopy(space))
                while not exhausted(remaining()):
                    phase = "observe"
                    view = model_input(obs, task, cap, profile)
                    writer.append(prefix + "/model_input.public.jsonl", view)
                    model.observe(view)
                    phase = "predict"
                    stats["model_calls"] += 1
                    prediction_in_flight=True
                    request = replace(context, message_id=f"request:{stats['model_calls']}")
                    pred = model.predict(request)
                    validate_prediction(pred, request, obs, space, cap, profile)
                    prediction_in_flight=False
                    trace("prediction", pred)
                    if stats["model_forward_count"] is not None:
                        stats["model_forward_count"] = None if pred.model_forward_count is None else stats["model_forward_count"] + pred.model_forward_count
                    left = remaining()
                    if left.wall_time_s is not None and left.wall_time_s <= 0:
                        break
                    phase, in_flight = "execute", True
                    reply = benchmark.advance(pred.action, request, left)
                    trace("step_result", reply)
                    validate_step(reply, obs, request, pred.action, space)
                    require(reply.observation.observation_id not in seen, "observation_id_reused", "observation_id", "new snapshot identity required")
                    in_flight = False
                    seen.add(reply.observation.observation_id)
                    feedback = reply.execution_feedback
                    stats["env_steps"] += feedback.consumed_env_steps
                    stats["sim_time_s"] += feedback.elapsed_sim_time_s
                    if stats["physics_steps"] is not None:
                        stats["physics_steps"] += feedback.consumed_physics_steps
                    require(task.budgets.env_steps is None or stats["env_steps"] <= task.budgets.env_steps,
                            "budget_overrun", "env_steps", "adapter exceeded budget")
                    require(task.budgets.sim_time_s is None or stats["sim_time_s"] <= task.budgets.sim_time_s + 1e-9,
                            "budget_overrun", "sim_time_s", "adapter exceeded budget")
                    obs = reply.observation
                    if feedback.status != "executed":
                        # No retry/recovery protocol exists. In particular a
                        # zero-step rejection must not loop forever under an
                        # environment-only budget or silently drop a command.
                        raise ContractError("action_not_executed", "advance", feedback.status)
                    if reply.terminated or reply.truncated:
                        reason = "environment_terminated" if reply.terminated else "budget_exhausted"
                        break
            except Exception as error:
                item = _error(error, phase)
                if prediction_in_flight:
                    stats["model_forward_count"]=None
                state_known = item["state_known"] and not (in_flight and item.get("execution_started",True))
                item["state_known"] = state_known
                errors.append(item)
                trace("error", item)
                status, reason = "error", "run_error" if state_known else "execution_state_unknown"
            scored = {k: replace(v, value=None, valid=False, unavailable_reason="not_initialized" if not reset else "execution_state_unknown") for k, v in metrics.items()}
            if reset and state_known:
                try:
                    scored = benchmark.score(replace(context, message_id="score"))
                    require(set(metrics) <= set(scored), "missing_metric", "score", "required metric missing")
                    for value in scored.values():
                        decode(Metric, value)
                        require(value.valid == (value.value is not None) and (value.valid or bool(value.unavailable_reason)),
                                "invalid_metric", "score", "invalid metrics need null and reason")
                    trace("score", scored)
                except Exception as error:
                    errors.append(_error(error, "score"))
                    status, reason = "error", "run_error"
                    scored = {k: replace(v, value=None, valid=False, unavailable_reason="scorer_failed") for k, v in metrics.items()}
            if begun:
                try:
                    model.end_episode(replace(context, message_id="end"))
                except Exception as error:
                    errors.append(_error(error, "end"))
                    status, reason = "error", "run_error" if state_known else "execution_state_unknown"
            result = dict(contract_version="0.2", episode_id=task.episode_id, run_status=status,
                          termination_reason=reason, metrics=scored, execution_statistics=stats)
            path = writer.write(prefix + "/result.json", result)
            try:
                writer.write(prefix + "/diagnostics.internal.json", {"model": model.diagnostics(), "benchmark": benchmark.diagnostics()})
            except Exception as error:
                errors.append(_error(error,"diagnostics"))
                status="error"
                reason="run_error" if state_known else "execution_state_unknown"
                result["run_status"]=status
                result["termination_reason"]=reason
                writer.write(prefix+"/result.json",result)
            results.append(result)
            rows.append(dict(episode_id=task.episode_id, run_status=status, termination_reason=reason,
                             result_ref=path, errors=errors, execution_statistics_complete=state_known))
            if status == "error":
                # Native stateful services are never automatically retried/reused after errors.
                break
        close_errors = []
        for adapter in (model, benchmark):
            try:
                adapter.close()
            except Exception as error:
                close_errors.append(_error(error, "close"))
        model = benchmark = None
        if close_errors and rows:
            rows[-1]["run_status"]="error"
            rows[-1]["errors"].extend(close_errors)
            rows[-1]["termination_reason"]="run_error" if rows[-1]["execution_statistics_complete"] else "execution_state_unknown"
            results[-1]["run_status"]="error"
            results[-1]["termination_reason"]=rows[-1]["termination_reason"]
            writer.write(f"episodes/{rows[-1]['episode_id']}/result.json",results[-1])
        completed = sum(r["run_status"] == "completed" for r in rows)
        summary = dict(contract_version="0.2", run_id=config.run_id, synthetic=provenance["synthetic"], provenance=provenance,
                       planned=len(tasks), completed=completed, errors=len(tasks)-completed,
                       episodes=rows, close_errors=close_errors, compatibility_warnings=list(profile.warnings))
        writer.write("summary.json", summary)
        return RunOutcome(output=writer.root, results=tuple(results), exit_code=0 if summary["errors"] == 0 else 1, summary=summary)
    finally:
        for adapter in (model, benchmark):
            if adapter is not None:
                adapter.close()
