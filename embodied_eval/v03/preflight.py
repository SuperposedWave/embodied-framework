"""GPU-free 0.3 structure, resolved public paths and independent asset status."""
from ..codec import to_data
from ..errors import ContractError
from .config import validate_config
from .matching import compatibility, resolve
from .registry import default_registry
from .validation import validate_space


def check(config, registry=None, verify_assets=False):
    # Custom registries are supported for independent plugin architecture tests.
    registry = registry or default_registry()
    validate_config(config, registry)
    model, benchmark = registry.model(config.model.adapter), registry.benchmark(config.benchmark.adapter)
    model.validate_options(config.model.options)
    benchmark.validate_options(config.benchmark.options)
    cap, space = model.describe(config.model.options), validate_space(benchmark.describe(config.benchmark.options))
    report, paths, _, _ = compatibility(space, cap, config.control_mode, config.temporal)
    from .temporal import assess
    _, _, temporal_report = assess(space, cap, paths, config.temporal)
    issues = [to_data(issue) for issue in report.issues]
    from .execution import assess as assess_execution
    execution_plan, execution_issues = assess_execution(space, cap, config.execution)
    issues.extend(execution_issues)
    if space.timing_mode == "measured_native" and any(ep.budgets.sim_time_s is not None for ep in config.episodes):
        issues.append({"code": "unsupported_native_budget", "path": "budgets.sim_time_s",
                       "message": "native calls cannot enforce a hard simulation-time cap"})
    readiness = {"model": model.readiness(config.model.options), "benchmark": benchmark.readiness(config.benchmark.options)}
    verified, asset_errors = {}, {}
    if verify_assets:
        for key, entry, options in (("model", model, config.model.options), ("benchmark", benchmark, config.benchmark.options)):
            try:
                verified[key] = entry.verify_assets(options)
            except (ContractError, OSError, ValueError, ImportError) as error:
                asset_errors[key] = str(error)
    return {"contract_version": "0.3", "structurally_compatible": not issues,
        "interface_compatible": not report.issues, "execution_compatible": not execution_issues,
        "issues": issues, "warnings": list(report.warnings), "readiness": readiness,
        "resolved_profile": to_data(resolve(space, cap, config.control_mode, config.temporal)) if not issues else None,
        "temporal": temporal_report,
        "execution": {"plan": execution_plan, "issues": execution_issues,
                      "runtime_recheck_required": True, "application_verified": False},
        "asset_checks_requested": verify_assets, "verified_assets": verified, "asset_errors": asset_errors,
        "assets_verified": verify_assets and not asset_errors, "real_validation": "not_performed_by_preflight",
        "synthetic": model.synthetic or benchmark.synthetic, "capability": to_data(cap), "space": to_data(space)}
