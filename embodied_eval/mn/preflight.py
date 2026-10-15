"""GPU-free structural inspection and independently verified asset readiness."""
from ..codec import to_data
from ..errors import ContractError
from .binding import check_compatibility
from .registry import default_registry
from .validation import validate_space


def check(config, registry=None, verify_assets=False):
    registry = registry or default_registry()
    m, b = registry.model(config.model.adapter), registry.benchmark(config.benchmark.adapter)
    m.validate_options(config.model.options)
    b.validate_options(config.benchmark.options)
    cap, space = m.describe(config.model.options), b.describe(config.benchmark.options)
    validate_space(space)
    report = check_compatibility(space, cap, config.binding)
    issues = [to_data(x) for x in report.issues]
    if space.timing_mode == "measured_native" and any(ep.budgets.sim_time_s is not None for ep in config.episodes):
        issues.append({"code": "unsupported_native_budget", "path": "budgets.sim_time_s", "message": "native calls cannot enforce a hard simulation-time cap"})
    readiness = {"model": m.readiness(config.model.options), "benchmark": b.readiness(config.benchmark.options)}
    verified, asset_errors = {}, {}
    if verify_assets:
        for name, entry, options in (("model", m, config.model.options), ("benchmark", b, config.benchmark.options)):
            try:
                verified[name] = entry.verify_assets(options)
            except (ContractError, OSError, ValueError, ImportError) as error:
                asset_errors[name] = str(error)
    return {"contract_version": "0.2", "structurally_compatible": not issues, "issues": issues,
            "warnings": list(report.warnings), "readiness": readiness,
            "asset_checks_requested": verify_assets, "verified_assets": verified, "asset_errors": asset_errors,
            "assets_verified": verify_assets and not asset_errors,
            "real_validation": "not_performed_by_preflight", "synthetic": m.synthetic or b.synthetic,
            "capability": to_data(cap), "space": to_data(space)}
