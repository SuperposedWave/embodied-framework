"""Snapshot current metadata for a public-field PROPOSAL, not a new protocol.

Only reads tracked source, builds metadata and writes a fresh evidence directory.
No model factories, simulation, weight loading, downloads or upstream edits.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCES = (
    "AGENTS.md", "docs/stage-state.json", "docs/contract-v0.2.md",
    "embodied_eval/contracts.py", "embodied_eval/mn/contracts.py",
    "embodied_eval/mn/interfaces.py", "embodied_eval/mn/config.py",
    "embodied_eval/mn/binding.py", "embodied_eval/mn/transforms.py",
    "embodied_eval/mn/native_catalog.py", "embodied_eval/mn/native_benchmarks.py",
    "embodied_eval/mn/native_models.py", "embodied_eval/mn/additional_catalog.py",
    "embodied_eval/mn/additional_models.py", "embodied_eval/mn/droid_simulator.py",
    "embodied_eval/mn/robodojo_simulator.py", "embodied_eval/mn/openwam_dual_pose.py",
    "embodied_eval/adapters/openwam_libero_spec.py",
    "embodied_eval/adapters/vlabench_spec.py", "embodied_eval/adapters/robotwin_spec.py",
    "examples/mn_openwam_dual_wxyz_encoded_absolute.json",
    "docs/handoff/prepare_mn_extension.py",
    "outputs/onboarding-coverage-baseline-20261015-run1/baseline-check.json",
)


def dump_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    target.relative_to(ROOT / "outputs")
    target.mkdir(parents=True, exist_ok=False)
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    records = {}
    for name in ("embodied_eval/contracts.py", "embodied_eval/mn/contracts.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        classes = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                fields = [{"name": field.target.id, "annotation": ast.unparse(field.annotation),
                           "default": ast.unparse(field.value) if field.value else None,
                           "line": field.lineno}
                          for field in node.body if isinstance(field, ast.AnnAssign)]
                classes[node.name] = {"line": node.lineno, "fields": fields}
        records[name] = classes
    from embodied_eval.codec import to_data
    from embodied_eval.mn.native_catalog import PROFILES, capability, space
    from embodied_eval.mn.additional_catalog import POLICIES, policy_capability
    capabilities = {}
    for profile, item in PROFILES.items():
        options = {"io_profile": profile, "repo": "metadata-only", "checkpoint_dir": "metadata-only"}
        capabilities[profile] = to_data(capability(item[0], options))
    for family, item in POLICIES.items():
        options = {"io_profile": item["profile"], "repo": "metadata-only", "codec_repo": "metadata-only",
                   "checkpoint_dir": "metadata-only", "client_repo": "metadata-only"}
        capabilities[item["profile"]] = to_data(policy_capability(family, options))
    spaces = {}
    for family, protocols in (("libero", ("spatial-30settle", "spatial-10settle")),
                              ("vlabench", ("frozen-track1",)),
                              ("robotwin", ("sealed-instruction", "seeded-instruction"))):
        for protocol in protocols:
            options = {"repo": "metadata-only", "protocol": protocol,
                       "task_name": "select_fruit" if family == "vlabench" else "adjust_bottle"}
            spaces[family + "/" + protocol] = to_data(space(family, options))
    # Pure metadata functions: do not construct either simulator class.
    from embodied_eval.mn.droid_simulator import space as droid_space
    from embodied_eval.mn.robodojo_simulator import space as dojo_space
    spaces["droid_sim/droid-three-scenes-native-v1"] = to_data(droid_space({
        "repo": "metadata-only", "assets_dir": "metadata-only"}))
    spaces["robodojo/stack-bowls-arx-x5-native-v1"] = to_data(dojo_space({"repo": "metadata-only"}))
    assert len(capabilities) == 11 and len(spaces) == 7
    forbidden = ("torch", "jax", "sapien", "mujoco", "isaaclab", "isaacsim")
    loaded = [name for name in forbidden if any(m == name or m.startswith(name + ".") for m in sys.modules)]
    assert not loaded, loaded
    assert before == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    value = {"kind": "current_0_1_and_0_2_metadata_not_0_3_implementation",
             "at_utc": datetime.now(timezone.utc).isoformat(), "base_commit": commit,
             "source_sha256": before, "dataclass_fields": records,
             "model_capabilities": capabilities, "environment_spaces": spaces,
             "metadata_only_placeholder_paths_are_not_runtime_asset_claims": True,
             "model_profile_count": len(capabilities), "environment_protocol_count": len(spaces),
             "GPU_or_simulation_modules_imported": loaded,
             "model_or_environment_factory_called": False,
             "asset_verification_or_real_runtime_performed": False,
             "historical_evidence_rewritten": False}
    dump_new(target / "current-public-space-inventory.json", value)
    print(json.dumps({"output": str(target), "profiles": len(capabilities),
                      "environment_protocols": len(spaces), "source_files": len(before),
                      "GPU_or_simulation_modules_imported": loaded}, ensure_ascii=False))


if __name__ == "__main__":
    main()
