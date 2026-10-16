"""Explicit OFFLINE config migration. Never changes weights/control/score.

Legacy pairing names are parsed only as historical metadata. Runtime adapters
are still independently selected; no pair factory or simulator is constructed.
"""
from ..codec import decode, to_data
from .._legacy_validation import require
from ..mn.migration import MODELS, BENCHMARKS
from .config import RunConfig, validate_config
from .registry import default_registry
from .matching import resolve
from .models import profile_paths
from .benchmarks import CAMERA_MAPS, state_names


def migrate(raw):
    require(isinstance(raw, dict), "invalid_config", "$", "configuration must be an object")
    version = raw.get("contract_version", "0.1")
    if version == "0.3":
        return validate_config(decode(RunConfig, raw)), {"from_version": version, "changed": False}
    require(version in ("0.1", "0.2"), "unsupported_config_version", "contract_version", "unknown source version")
    if version == "0.1":
        require(set(raw) <= {"contract_version", "run_id", "model", "benchmark", "profile", "episodes"},
                "unknown_field", "$", "unknown source fields cannot be discarded")
        model, bench = raw.get("model", {}), raw.get("benchmark", {})
        require(model.get("adapter") in MODELS and bench.get("adapter") in BENCHMARKS,
                "manual_migration_required", "adapters", "retired mock/tool/discrete configurations have no claimed equivalent; use current synthetic examples")
        family, profile, repo_key = MODELS[model["adapter"]]
        env_family, protocol, env_repo_key = BENCHMARKS[bench["adapter"]]
        model_defaults = dict(scenario="chunk", chunk_length=8, observation_update="every_env_step",
            delta_reference="step_observed", delta_composition="pre_multiply", fault="none", fault_episode_id=None,
            report_forward_count=True, openwam_repo=None, openpi_repo=None, lingbot_repo=None)
        benchmark_defaults = dict(control_period_s=.05, tool_ticks=3, intermediate_observations_supported=True,
            success_at_step=None, fault="none", fault_episode_id=None, fault_after_step=1,
            libero_repo=None, vlabench_repo=None, robotwin_repo=None)
        for name, item, used, defaults in (
            ("model", model, {"adapter", repo_key, "checkpoint_dir", "host", "port", "request_timeout_s"}, model_defaults),
            ("benchmark", bench, {"adapter", env_repo_key, "task_name", "task_config", "instruction_type", "max_seed_candidates", "track", "trial_index"}, benchmark_defaults)):
            require(set(item) <= used | set(defaults), "unknown_field", name, "unknown native configuration fields cannot be discarded")
            require(all(k in used or value == defaults[k] for k, value in item.items()),
                    "manual_migration_required", name, "nondefault retired behavior needs explicit review; do not silently erase it")
        require(set(raw.get("profile", {})) <= {"chunk_mode", "execution_horizon_steps"}, "unknown_field", "profile", "unknown behavior cannot be discarded")
        require(raw.get("profile", {}).get("chunk_mode") == "adapter_owned"
                and raw.get("profile", {}).get("execution_horizon_steps") == 1,
                "manual_migration_required", "profile", "do not silently change runner-owned execution semantics")
        new = {"contract_version": "0.3", "run_id": raw["run_id"], "episodes": raw["episodes"],
            "model": {"adapter": family, "options": {"io_profile": profile, "repo": model[repo_key],
                "checkpoint_dir": model["checkpoint_dir"], **{key: model[key] for key in ("host", "port", "request_timeout_s") if key in model}}},
            "benchmark": {"adapter": env_family, "options": {"repo": bench[env_repo_key], "protocol": protocol,
                **{key: bench[key] for key in ("task_name", "task_config", "instruction_type", "max_seed_candidates", "track", "trial_index") if key in bench}}}}
    else:
        require(set(raw) <= {"contract_version", "run_id", "model", "benchmark", "binding", "episodes"},
                "unknown_field", "$", "do not discard unknown legacy fields")
        new = {key: value for key, value in raw.items() if key != "binding"}
        new["contract_version"] = "0.3"
        model, bench = new["model"], new["benchmark"]
        # Compare the OLD explicit input/output choices to the independent new
        # profile. Nonstandard experiments are not silently reset to defaults.
        require(model["adapter"] not in ("synthetic_policy_a", "synthetic_policy_b"),
                "manual_migration_required", "model", "synthetic fixture identifiers/semantics changed; use the current fixture examples")
        from ..mn.registry import default_registry as original_registry
        original = original_registry()
        old_cap = original.model(model["adapter"]).describe(model["options"])
        old_space = original.benchmark(bench["adapter"]).describe(bench["options"])
        canonical = profile_paths(model["options"]["io_profile"], old_cap)
        reverse_cameras = {value: key for key, value in CAMERA_MAPS[bench["adapter"]].items()}
        reverse_states = {value: key for key, value in state_names(bench["adapter"], old_space).items()}
        expected = {}
        for req in old_cap.inputs:
            if req.key == "drive_targets":
                expected[req.key] = "joint_drive_targets"
            elif req.kind == "task":
                expected[req.key] = "instruction"
            else:
                key = canonical[req.key].split(".", 1)[1]
                table = reverse_cameras if req.kind == "sensor" else reverse_states
                require(key in table, "manual_migration_required", req.key, "canonical field is absent in the selected environment")
                expected[req.key] = table[key]
        binding = raw.get("binding", {})
        require(set(binding) <= {"inputs", "targets", "action_spec_ref", "frame_conversions"},
                "unknown_field", "binding", "unknown binding semantics cannot be discarded")
        require(binding.get("inputs") == expected and not binding.get("frame_conversions"),
                "manual_migration_required", "binding", "custom/missing input or coordinate choices cannot be silently discarded")
        targets = {key: "robot0/" + ("base" if key == "dual" else key) for key in old_cap.output.target_slots}
        require(binding.get("targets") == targets, "manual_migration_required", "binding.targets", "custom action targets need explicit physical review")
        new["control_mode"] = binding["action_spec_ref"]
    config = validate_config(decode(RunConfig, new))
    registry = default_registry()
    profile = resolve(registry.benchmark(config.benchmark.adapter).describe(config.benchmark.options),
                      registry.model(config.model.adapter).describe(config.model.options), config.control_mode)
    return config, {"from_version": version, "to_version": "0.3", "changed": True,
        "weights_normalization_environment_protocol_seed_and_budgets_preserved": True,
        "resolved_action_spec": profile.action_spec_ref, "real_validation": "not_performed"}
