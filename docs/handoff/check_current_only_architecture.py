"""Offline current-only architecture, provenance and independent-profile ledger.

No GPU, simulator, download, large-weight hashing or historical-file rewrite.
Frozen test references are used only to compare the former metadata/calculation
statements, never as a current runtime factory or compatibility fallback.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = "67ca6ad"


def before(path):
    return subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT).decode()


def node(source, name):
    value = ast.parse(source)
    for part in name.split("."):
        value = next(n for n in value.body if getattr(n, "name", None) == part)
    return deepcopy(value)


def computation(value):
    class StripImports(ast.NodeTransformer):
        def visit_Import(self, value):
            return None
        def visit_ImportFrom(self, value):
            return None
    value = StripImports().visit(deepcopy(value))
    if value.body and isinstance(value.body[0], ast.Expr) and isinstance(value.body[0].value, ast.Constant):
        value.body.pop(0)
    return ast.dump(value, include_attributes=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    from embodied_eval.codec import decode, to_data
    from embodied_eval.core import types as leaf
    from embodied_eval.core.validation import require
    from embodied_eval.native import contracts as private
    from embodied_eval.native.model_catalog import PROFILES, capability
    from embodied_eval.native.additional_catalog import POLICIES
    from embodied_eval.native.benchmark_catalog import benchmark_options, space, state_spec
    from embodied_eval.native import protocols
    from embodied_eval.native.joint_native import RAW_GRIPPER_PROTOCOL, RAW_GRIPPER_SIGNATURE
    from embodied_eval.v03.config import RunConfig
    from embodied_eval.v03.preflight import check
    from embodied_eval.v03.registry import default_registry
    from tests.legacy_support.frozen.adapters import (openwam_libero_spec as ow,
        openpi_libero_spec as pi, vlabench_spec as vla, openpi_vlabench_spec as pivla, robotwin_spec as rw)
    receipt = {"kind": "current_only_architecture_and_metadata_not_real_evaluation", "passed": False,
        "at_utc": datetime.now(timezone.utc).isoformat(), "comparison_checkpoint": BASE,
        "public_version": "0.3", "new_GPU_inference": False, "new_simulation": False,
        "large_weight_hashes_rechecked": False, "server_identity_verified": False,
        "formal_stage4_acceptance": False, "contract_frozen": False}
    try:
        runtime = sorted((ROOT / "embodied_eval").rglob("*.py"))
        receipt["runtime_python_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in runtime}
        forbidden = ("embodied_eval.mn", "embodied_eval._legacy", "embodied_eval.stage4_pairs",
                     "embodied_eval.robotwin_pairs", "tests.", "docs.")
        for path in runtime:
            source = path.read_text()
            for item in ast.walk(ast.parse(source)):
                if isinstance(item, ast.ImportFrom) and item.module:
                    require(not item.module.startswith(forbidden), "historical_runtime_dependency", str(path), item.module)
                if isinstance(item, ast.Constant) and item.value in ("0.1", "0.2"):
                    raise ValueError("old version literal remains in runtime: " + str(path))
        receipt["runtime_has_no_old_version_literals_or_historical_dependencies"] = True
        catalogue = before("embodied_eval/native/native_catalog.py")
        scope = dict(ow=ow, pi=pi, vla=vla, pivla=pivla, rw=rw)
        assignment = next(n for n in ast.parse(catalogue).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "PROFILES" for t in n.targets))
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "frozen-model-metadata", "exec"), scope)
        assert PROFILES == scope["PROFILES"]
        receipt["original_weight_profile_identities_unchanged"] = len(PROFILES)
        previous_policies = next(n.value for n in ast.parse(before("embodied_eval/native/additional_catalog.py")).body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "POLICIES" for t in n.targets))
        assert POLICIES == ast.literal_eval(previous_policies)
        receipt["additional_weight_profile_identities_unchanged"] = len(POLICIES)
        # Execute former *metadata* space construction with the same physical
        # record definitions; no former runner or model factory is constructed.
        env_scope = dict(scope, c=private, leaf=leaf, benchmark_options=benchmark_options,
            state_spec=state_spec, RAW_GRIPPER_PROTOCOL=RAW_GRIPPER_PROTOCOL,
            RAW_GRIPPER_SIGNATURE=RAW_GRIPPER_SIGNATURE,
            **{k: getattr(protocols, k) for k in vars(protocols) if k.isupper()})
        exec(compile(ast.Module(body=[node(catalogue, "space")], type_ignores=[]), "frozen-env-metadata", "exec"), env_scope)
        examples = []
        for pattern in ("stage*.json", "v03_*.json", "mn_openwam_dual_wxyz_encoded_absolute.json"):
            examples.extend(sorted((ROOT / "examples").glob(pattern)))
        rows, model_profiles, environments = [], {}, {}
        for path in examples:
            cfg = decode(RunConfig, json.loads(path.read_text()))
            checked = check(cfg)
            assert checked["structurally_compatible"], (path.name, checked["issues"])
            if cfg.benchmark.adapter in ("libero", "vlabench", "robotwin"):
                assert to_data(env_scope["space"](cfg.benchmark.adapter, cfg.benchmark.options)) == to_data(space(cfg.benchmark.adapter, cfg.benchmark.options))
            model_profiles[(cfg.model.adapter, cfg.model.options["io_profile"])] = cfg.model
            environments[(cfg.benchmark.adapter, cfg.benchmark.options["protocol"])] = cfg.benchmark
            rows.append({"configuration": path.name, "model": cfg.model.adapter,
                "io_profile": cfg.model.options["io_profile"], "benchmark": cfg.benchmark.adapter,
                "protocol": cfg.benchmark.options["protocol"], "interface_compatible": checked["interface_compatible"],
                "execution_plan": checked["execution"]["plan"], "model_execution": checked["capability"]["execution"],
                "model_readiness": checked["readiness"]["model"], "benchmark_readiness": checked["readiness"]["benchmark"],
                "assets_rehashed": False, "new_real_validation": False})
        assert (len(rows), len(model_profiles), len(environments)) == (13, 11, 7)
        receipt.update(configurations=rows, unique_weight_io_profiles=11, unique_environment_protocols=7)
        registry = default_registry()
        matrix = []
        for model in model_profiles.values():
            for benchmark in environments.values():
                cfg = decode(RunConfig, json.loads(examples[0].read_text()))
                from dataclasses import replace
                episodes = tuple(replace(ep, budgets=leaf.Budgets(env_steps=4, model_calls=4)) for ep in cfg.episodes)
                checked = check(replace(cfg, model=model, benchmark=benchmark, control_mode=None, episodes=episodes))
                matrix.append({"model": model.adapter, "io_profile": model.options["io_profile"],
                    "benchmark": benchmark.adapter, "protocol": benchmark.options["protocol"],
                    "interface_compatible": checked["interface_compatible"], "preflight_accepted": checked["structurally_compatible"],
                    "issue_codes": sorted({i["code"] for i in checked["issues"]}),
                    "real_validation": False})
        receipt["synthetic_structure_matrix"] = matrix
        receipt["structure_matrix_cases"] = len(matrix)
        receipt["structure_matrix_interface_matches"] = sum(r["interface_compatible"] for r in matrix)
        unchanged = [
            ("native/native_models.py", "NativeActionModel.native_input"),
            ("native/native_models.py", "NativeActionModel.decode_action"),
            ("native/additional_models.py", "JointPolicyModel.native_input"),
            ("native/additional_models.py", "JointPolicyModel.predict"),
            ("native/openwam_arx.py", "OpenWAMArxModel.native_input"),
            ("native/openwam_arx.py", "OpenWAMArxModel.native_rows"),
            ("native/openwam_arx.py", "OpenWAMArxModel.decode_action"),
            ("native/openwam_arx.py", "OpenWAMArxModel.predict"),
            ("native/native_benchmarks.py", "NativeSimulator._project"),
            ("native/native_benchmarks.py", "NativeSimulator._advance_robotwin"),
            ("adapters/native_robotwin_executor.py", "MeasuredRobotwinExecutor.execute"),
            ("adapters/libero_spatial_benchmark.py", "LiberoSpatialBenchmark.advance"),
            ("adapters/libero_spatial_benchmark.py", "LiberoSpatialBenchmark.score"),
            ("adapters/vlabench_benchmark.py", "VLABenchBenchmark.advance"),
            ("adapters/vlabench_benchmark.py", "VLABenchBenchmark.score"),
            ("adapters/robotwin_benchmark.py", "RobotwinBenchmark.score"),
        ]
        receipt["unchanged_calculation_methods"] = []
        for relative, name in unchanged:
            relative = "embodied_eval/" + relative
            assert computation(node(before(relative), name)) == computation(node((ROOT / relative).read_text(), name)), (relative, name)
            receipt["unchanged_calculation_methods"].append({"path": relative, "method": name,
                "comparison": "parsed calculation statements equal; imports/docstrings excluded"})
        # Verify effective former override against today's sole environment
        # implementation, rather than comparing the unused old pair base.
        old_instruction = node(before("embodied_eval/native/native_benchmarks.py"), "_IndependentRobotwinRuntime._choose_instruction")
        for item in ast.walk(old_instruction):
            if isinstance(item, ast.Attribute) and item.attr == "_instruction_protocol":
                item.attr = "variant"
        current_instruction = node((ROOT / "embodied_eval/adapters/robotwin_benchmark.py").read_text(), "RobotwinBenchmark._choose_instruction")
        assert computation(old_instruction) == computation(current_instruction)
        receipt["effective_robotwin_instruction_calculations_preserved"] = True
        assets = sorted((ROOT / "embodied_eval/native/asset_manifests").glob("*.json")) + sorted((ROOT / "embodied_eval/adapters").glob("*.json"))
        for path in assets:
            assert path.read_bytes() == before(str(path.relative_to(ROOT))).encode()
        receipt["pinned_small_asset_manifests_unchanged"] = len(assets)
        baseline_files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE, "--", "embodied_eval"], cwd=ROOT, text=True).splitlines()
        receipt["removed_runtime_files_recoverable_in_git"] = [p for p in baseline_files if not (ROOT / p).is_file()]
        receipt["historical_test_reference_python_files_not_distributed"] = len(list((ROOT / "tests/legacy_support/frozen").rglob("*.py")))
        assert all(digest(ROOT / p) == sha for p, sha in receipt["runtime_python_sha256"].items())
        receipt["passed"] = True
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("passed", "structure_matrix_cases", "structure_matrix_interface_matches")}))


if __name__ == "__main__":
    main()
