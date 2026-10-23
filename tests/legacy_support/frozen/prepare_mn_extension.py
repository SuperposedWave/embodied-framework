"""New component/configuration evidence only; not a pair registration or score.

Never rewrites historical ledgers, paths or reports. Checks actual published
objects, not partial downloads. No model imports, network calls or GPU work.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
MODELS = WORKSPACE.parent.parent / "Model"
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def configurations():
    """Examples of explicit data wiring, never a runtime compatibility list."""
    import tests.legacy_support.frozen._legacy_contracts as leaf
    import embodied_eval.native.contracts as c
    from tests.legacy_support.frozen.mn.config import RunConfig, AdapterConfig, EpisodeConfig
    from tests.legacy_support.frozen.mn.registry import default_registry
    upstream = WORKSPACE / "upstream"
    model_data = {
        "cosmos3": ("cosmos3", {"io_profile": "nano-policy-droid-joint8", "repo": str(upstream / "cosmos-framework-mn"),
            "codec_repo": str(upstream / "openpi"), "client_repo": str(upstream / "robolab-mn"),
            "checkpoint_dir": str(MODELS / "Cosmos3-Nano-Policy-DROID"), "port": 8940}),
        "dreamzero": ("dreamzero", {"io_profile": "droid-joint8-causal", "repo": str(upstream / "dreamzero-mn"),
            "codec_repo": str(upstream / "openpi"), "checkpoint_dir": str(MODELS / "DreamZero-DROID"), "port": 8941}),
        "pi05_arx": ("openpi", {"io_profile": "arx_joint14", "repo": str(upstream / "xpolicy-reference-mn"),
            "checkpoint_dir": str(MODELS / "pi05_arx_x5_seed0_step59999"), "port": 8942}),
        "openwam_arx": ("openwam", {"io_profile": "eef20_arx_absolute", "repo": str(upstream / "xpolicy-reference-mn"),
            "checkpoint_dir": str(MODELS / "OpenWAM-Alpha-Sim-RoboDojo")}),
    }
    benchmarks = {
        "droid_sim": AdapterConfig(adapter="droid_sim", options={"repo": str(upstream / "sim-evals-mn"),
            "assets_dir": str(MODELS / "DROID-sim-environments"), "protocol": "droid-three-scenes-native-v1", "scene": 1, "device": "cuda:3"}),
        "robodojo": AdapterConfig(adapter="robodojo", options={"repo": str(upstream / "RoboDojo"),
            "protocol": "stack-bowls-arx-x5-native-v1", "layout_index": 0, "device": "cuda:3"}),
        "robotwin": AdapterConfig(adapter="robotwin", options={"repo": str(upstream / "RoboTwin-lingbot"),
            "protocol": "seeded-instruction", "task_name": "adjust_bottle", "task_config": "demo_clean", "instruction_type": "unseen"}),
    }
    rows = {}
    # This list only emits example JSON. Runner/registries do NOT consult it.
    examples = (("cosmos3", "droid_sim"), ("dreamzero", "droid_sim"),
        ("pi05_arx", "robodojo"), ("openwam_arx", "robodojo"),
        ("pi05_arx", "robotwin"), ("openwam_arx", "robotwin"))
    for name, bench in examples:
        family, opts = model_data[name]
        cap = default_registry().model(family).describe(opts)
        if bench == "droid_sim":
            inputs = {"exterior_right": "external_cam_2", "exterior_left": "external_cam", "wrist": "wrist_cam",
                "joints": "arm_joint_positions", "grip": "gripper_open_fraction", "instruction": "instruction"}
            if name == "dreamzero":
                # Fixed DreamZero run_sim_eval.py uses the *legacy variable*
                # right_image for external_cam, despite its +Y/left placement.
                # Wiring is model configuration data, not a simulator branch.
                inputs.update(exterior_right="external_cam", exterior_left="external_cam_2")
            action = "joint7-binary-gripper@1"
        else:
            rw = bench == "robotwin"
            inputs = {"head": "head_camera" if rw else "cam_head", "left_wrist": "left_camera" if rw else "cam_left_wrist",
                "right_wrist": "right_camera" if rw else "cam_right_wrist", "instruction": "instruction"}
            for side in ("left", "right"):
                if name == "pi05_arx":
                    inputs[side + "_joints"] = side + "_joint_positions"
                else:
                    inputs[side + "_position"] = side + "_eef_position"
                    inputs[side + "_quaternion"] = side + "_eef_quaternion"
                inputs[side + "_grip"] = side + ("_gripper_open_fraction" if rw else "_gripper_target")
            action = ("joint-absolute-native-gripper@1" if rw else "dual-joint-raw-gripper@1") if name == "pi05_arx" else (
                "dual-cartesian-absolute@1" if rw else "dual-pose-absolute@1")
        key = name + "_" + bench
        rows[key] = RunConfig(run_id="mn-extension-" + key.replace("_", "-"),
            model=AdapterConfig(adapter=family, options=opts), benchmark=benchmarks[bench],
            binding=c.BindingConfig(inputs=inputs, targets={slot: "robot0/" + slot for slot in cap.output.target_slots}, action_spec_ref=action),
            episodes=(EpisodeConfig(episode_id="ep-0", seed=0,
                budgets=leaf.Budgets(env_steps=800 if bench == "robodojo" else 600, model_calls=800, wall_time_s=3600.)),))
    return rows


def asset_inventory():
    from embodied_eval.native.assets_common import _revision
    locations = {"cosmos3": MODELS / "Cosmos3-Nano-Policy-DROID", "dreamzero": MODELS / "DreamZero-DROID",
        "pi05_arx": MODELS / "pi05_arx_x5_seed0_step59999", "openwam_arx": MODELS / "OpenWAM-Alpha-Sim-RoboDojo",
        "droid_sim": MODELS / "DROID-sim-environments", "robodojo": WORKSPACE / "upstream/RoboDojo"}
    records = {}
    for component, directory in locations.items():
        manifest_path = ROOT / "embodied_eval/mn/asset_manifests" / (component + ".json")
        manifest = read(manifest_path)
        files = []
        for item in manifest["objects"]:
            path = directory / item["relative"]
            if not path.is_file():
                files.append({"file": item["relative"], "status": "missing", "expected_bytes": item["size"]})
                continue
            if path.stat().st_size != item["size"]:
                files.append({"file": item["relative"], "status": "wrong_size", "expected_bytes": item["size"]})
                continue
            actual = digest(path)
            if "sha256" in item:
                passed = actual == item["sha256"]
            else:
                blob = hashlib.sha1(f"blob {item['size']}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        blob.update(chunk)
                passed = blob.hexdigest() == item["git_blob"]
            files.append({"file": item["relative"], "status": "verified" if passed else "hash_mismatch",
                "expected_bytes": item["size"], "sha256": actual})
        complete = all(row["status"] == "verified" for row in files)
        records[component] = {"directory": str(directory), "repository": manifest["repository"], "revision": manifest["revision"],
            "manifest_sha256": digest(manifest_path), "complete_pinned_file_set": complete,
            "verified_files": sum(row["status"] == "verified" for row in files), "required_files": len(files),
            "verified_bytes": sum(row["expected_bytes"] for row in files if row["status"] == "verified"),
            "required_bytes": sum(row["expected_bytes"] for row in files), "files": files,
            "dependencies_verified": False, "runtime_assets_ready": False, "endpoint_identity_verified": False,
            "usd_reference_closure_verified": False if component in ("droid_sim", "robodojo") else None,
            "additional_backbone_assets_verified": False if component in ("cosmos3", "dreamzero", "openwam_arx") else None,
            "real_validation": "not_performed_in_this_extension"}
    sources = {
        "cosmos-framework-mn": "cf5d68c00d97ccd2480a2320ed652b92dec63102",
        "dreamzero-mn": "ab790c198fbce33503358efbbd4187ce9a89adf3", "robolab-mn": "ad45d4f974725d020f82c2b0d77d78533aeba2b3",
        "sim-evals-mn": "3a6b0e8177f696947f49c082728f10ef5133ede1", "openpi": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "xpolicy-reference-mn": "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2", "RoboDojo": "266130a3ec41ba9b20b0e5648d2f3542f219c06c"}
    for name, expected in sources.items():
        _revision(WORKSPACE / "upstream" / name, expected)
    return {"kind": "pinned_component_asset_inventory_not_runtime_readiness", "at_utc": datetime.now(timezone.utc).isoformat(),
        "components": records, "clean_source_revisions": sources, "global_proxy_modified": False,
        "models_loaded": False, "new_real_trials": 0, "official_scores": None}


def coverage(configs, checks):
    old = read(ROOT / "outputs/mn-delivery-20261007-final/coverage-ledger.json")
    rows = []
    for original in old["rows"]:
        row = dict(original)
        model, bench = row["original_model"], row["benchmark"]
        row["previous_delivery_real_trial_completed"] = row.pop("new_real_trial_completed")
        row["new_real_trial_completed"] = False
        case = None
        if model == "LingBot-VA 2.0":
            row.update(scope_status="user_deferred", structurally_composable="not_claimed",
                plain_language_reasons=["用户本轮明确暂不处理LingBot-VA；已实现的LingBot-VLA是另一模型，仍保留，不抵扣VA格。"])
        elif model == "Cosmos 3":
            row["plain_language_reasons"] = ["已新增确切的Cosmos3-Nano-Policy-DROID动作型号；不以同名Reasoner/视频生成型号替代。"]
            if bench in ("libero", "vlabench"):
                row.update(scope_status="model_present_action_structure_mismatch", structurally_composable=False)
                row["plain_language_reasons"].append("这份权重输出7个单臂关节目标，而该具名环境协议只提供末端控制；现有状态/执行接口不满足，不能把关节数组伪装成末端动作。新DROID接线单列，不冒充此格完成。")
            else:
                row.update(scope_status="protocol_identity_still_undefined", structurally_composable="not_checked_undefined_protocol")
                row["plain_language_reasons"].append("模型组件已写好，但原表此环境编号/被测角色仍未明确；新增DROID仿真不自动替代原表R021/R022。")
        elif (model == "π0.5" and bench in ("robodojo", "robotwin")) or (model == "OpenWAM-α" and bench == "robodojo"):
            case = ("pi05_arx" if model == "π0.5" else "openwam_arx") + "_" + bench
            row.update(scope_status="new_structure_ready_runtime_unverified", structurally_composable=checks[case]["structurally_compatible"],
                plain_language_reasons=["模型输入输出配置和环境执行边界已独立实现；显式相机、双臂状态与完整动作接线通过。运行依赖/资产与真实闭环仍未全部核验，不计真实完成。"])
            if bench == "robotwin":
                row["plain_language_reasons"].append("使用ARX训练权重作结构接入示例，并非宣称已在RoboTwin机械臂训练或已验证迁移效果。")
        elif row["scope_status"] == "simulator_protocol_unimplemented":
            row["scope_status"] = "protocol_identity_still_undefined"
            row["plain_language_reasons"] = ["此原表编号/环境/被测角色没有足够定义，不能猜测协议；DROID具名仿真只作新增组件，不偷换原表含义。"]
        elif row["scope_status"] == "independent_components_implemented":
            row["scope_status"] = "historically_implemented_components"
        row["new_configuration_example"] = case
        rows.append(row)
    assert len(rows) == 55 and sum(row["was_one_of_original_49_gaps"] for row in rows) == 49
    return {"kind": "mn_extension_scope_ledger_not_acceptance", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_source_sha256": old["source_sha256"], "original_cells": 55, "original_49_gap_rows": 49,
        "scope_counts_all_55": dict(Counter(row["scope_status"] for row in rows)),
        "scope_counts_original_49_only": dict(Counter(row["scope_status"] for row in rows if row["was_one_of_original_49_gaps"])),
        "native_model_families": 5, "native_io_profiles": 10, "native_benchmark_families": 5, "native_environment_protocols": 7,
        "new_configuration_examples": len(configs), "new_examples_structurally_compatible": sum(c["structurally_compatible"] for c in checks.values()),
        "new_real_trials": 0, "historical_real_trials": 6, "no_placeholder_adapters": True, "user_deferred_model": "LingBot-VA 2.0",
        "retained_distinct_model": "LingBot-VLA 2.0", "unknown_protocols_renamed": False,
        "stage4_formally_accepted": False, "contract_frozen": False, "rows": rows,
        "additional_examples_not_original_cell_claims": list(configs)}


def historical_integrity():
    records = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if name in ("docs/stage2-acceptance.md", "docs/stage2-evidence.json") or name.startswith("outputs/stage2-20261005T063033Z-final/"):
            path = ROOT / name
            records.append({"path": name, "unchanged": path.is_file() and digest(path) == expected})
    assert len(records) == 35 and all(row["unchanged"] for row in records)
    report = ROOT / "outputs/mn-delivery-20261007-final/M+N-architecture-and-evidence.html"
    old_report = {"path": str(report.relative_to(ROOT)), "sha256": digest(report),
        "unchanged": digest(report) == "c913eaf16076b9b9cdc35d49066546326ecb4dc3c092a7dc721411bcdfab2559"}
    assert old_report["unchanged"]
    return {"historical_stage2_sha256_files": records, "historical_mn_report": old_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from embodied_eval.codec import to_data
    from tests.legacy_support.frozen.mn.preflight import check
    from tests.legacy_support.frozen.mn.config import validate_config
    cfgs = configurations()
    checks = {}
    for key, cfg in cfgs.items():
        validate_config(cfg)
        save(args.output / (key + ".json"), to_data(cfg))
        checks[key] = check(cfg)
        save(args.output / (key + "-preflight.json"), checks[key])
        if not checks[key]["structurally_compatible"]:
            raise ValueError(f"explicit example {key} does not structurally match: {checks[key]['issues']}")
        print(f"{key}: structure compatible, real validation not performed", flush=True)
    save(args.output / "coverage-ledger.json", coverage(cfgs, checks))
    save(args.output / "assets.json", asset_inventory())
    baselines = ["mn-extension-baseline-20261007-final4", "mn-extension-baseline-libero-20261007-final4",
        "mn-extension-baseline-libero-pi-20261007-final4", "mn-extension-baseline-vlabench-20261007-final4", "mn-extension-baseline-robotwin-20261007-final4"]
    baselines += ["mn-extension-baseline-" + component + "-20261007-native-diagnostic"
        for component in ("cosmos3", "dreamzero", "openwam_arx")]
    baseline_checks = []
    for name in baselines:
        report = read(ROOT / "outputs" / name / "baseline-check.json")
        log = (ROOT / "outputs" / name / "unittest.txt").read_text()
        cli = [row["actual_exit_code"] for row in report["checks"] if "actual_exit_code" in row]
        passed = report["passed"] and report["checks"][0]["tests_run"] == 306 and cli == [0, 0, 0, 1, 2] and "skipped=" not in log
        assert passed
        baseline_checks.append({"receipt": "outputs/" + name + "/baseline-check.json", "passed": passed,
            "tests": 306, "skips": 0, "cli_exit_codes": cli, "python": report["python"],
            "native_dependency_environment_diagnostic_not_runtime_certification": "native-diagnostic" in name})
    mapping = read(ROOT / "outputs/mn-extension-native-mappings-20261007-final2.json")
    assert mapping["passed"] and len(mapping["results"]) == 9 and not mapping["model_loaded"]
    integrity = historical_integrity()
    save(args.output / "receipt.json", {"kind": "mn_extension_code_and_preparation_not_real_acceptance",
        "at_utc": datetime.now(timezone.utc).isoformat(), "structural_examples_passed": len(checks),
        "baselines": baseline_checks, "native_function_checks": 9, "native_function_check_input": "synthetic",
        "native_function_receipt": "outputs/mn-extension-native-mappings-20261007-final2.json", "new_real_trials": 0,
        "stage4_formally_accepted": False, "contract_frozen": False, **integrity})
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
