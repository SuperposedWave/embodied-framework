"""Audit ledger, NOT a runtime compatibility whitelist or automatic wiring.

The comparison grid uses explicit candidate camera/state wiring data. Missing
inputs and atomic action targets stay failures; no robot or frame is guessed.
Training coverage never gates structural composition. All original cells remain.
"""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CASES = ("openwam_libero", "openpi_libero", "openwam_vlabench", "openpi_vlabench", "openwam_robotwin", "lingbot_robotwin")
ACTION_CANDIDATES = {"Cosmos 3", "LingBot-VA 2.0", "DreamZero-DROID", "π0.5", "OpenWAM-α"}
CAMERA_DATA = {"libero": {"head": "agentview_image", "left_wrist": "robot0_eye_in_hand_image", "right_wrist": "MISSING:right_wrist"},
               "vlabench": {"head": "forward_rgb", "left_wrist": "franka_wrist_rgb", "right_wrist": "right_rgb"},
               "robotwin": {"head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}}
STATE_DATA = {"position": "eef_position", "quaternion": "eef_quaternion", "fingers": "gripper_positions",
              "grip": "gripper_open_fraction", "base": "robot_base_position", "drive_targets": "joint_drive_targets"}
for side in ("left", "right"):
    STATE_DATA.update({side + "_position": side + "_eef_position", side + "_quaternion": side + "_eef_quaternion",
                       side + "_grip": side + "_gripper_open_fraction"})


def read(path):
    return json.loads(path.read_text())


def build(configs, trials, replays):
    from embodied_eval.codec import to_data
    from embodied_eval.config import load_config
    from embodied_eval.mn.preflight import check
    from embodied_eval.mn.registry import default_registry
    source_path = ROOT / "docs/handoff/source-materials/compatibility-draft.json"
    source = read(source_path)
    old = read(ROOT / "outputs/stage4-implementation-coverage-20261006T1640Z.json")
    old_by_cell = {row["source_cell"]: row for row in old["rows"]}
    known = {("OpenWAM-α", "libero"): "openwam_libero", ("OpenWAM-α", "vlabench"): "openwam_vlabench",
             ("OpenWAM-α", "robotwin"): "openwam_robotwin", ("π0.5", "libero"): "openpi_libero",
             ("π0.5", "vlabench"): "openpi_vlabench"}
    rows = []
    for cell in source["combinations"]:
        model, bench = cell["model_name"], cell["benchmark_id"]
        case = known.get((model, bench))
        reasons = []
        if model not in ACTION_CANDIDATES:
            status = "outside_action_model_scope"
            reasons.append("当前具名模型没有本轮纳入的直接机器人动作接口；文本/图片/视频或推理结果转动作需要另建规划系统，本轮不做。")
        elif case:
            status = "independent_components_implemented"
            reasons.append("模型和仿真组件已独立注册；仅本次具名试点有实际证据，不等于原表全部编号或全suite已验证。")
        elif model == "LingBot-VA 2.0":
            status = "action_model_identity_unimplemented"
            reasons.append("原表是LingBot-VA 2.0；本次迁移的是另一模型LingBot-VLA 2.0，不能冒充同一项。需要落实VA 2.0的准确代码、动作接口及可部署资产。")
        elif model == "Cosmos 3":
            status = "action_variant_unselected"
            reasons.append("家族同时有推理、生成和策略型号；只把真正输出动作的Policy型号列为候选，不能拿已有Nano-Reasoner当动作模型。")
        elif bench in ("libero", "vlabench", "robotwin"):
            status = "model_io_profile_missing"
            reasons.append("仿真侧已有独立组件；尚没有该具名模型的可执行输入/动作配置。是否训练过该机器人只作提示，不据此拒绝接入。")
        else:
            status = "simulator_protocol_unimplemented"
            reasons.append("动作模型可列入本轮候选；此项环境/任务/初始化/控制/评分协议尚未实现，不能直接使用项目或论文名称代替仿真接口。")
        if bench not in ("libero", "vlabench", "robotwin"):
            reasons.append({"vidman": "R009/R010没有明确环境定义；CALVIN只是候选，未自动选定。",
                            "dreamzero_protocol": "R021/R022没有明确仿真协议；DROID sim_evals只是候选，不自动换成真实RoboArena或AgiBot。",
                            "robodojo": "需把RoboDojo仿真加载/机器人控制/评分从既有外部依赖中独立出来，尚无0.2注册。",
                            "wiw": "ImageNav/Manipulation及被测角色未选定；本轮不新增预测画面转动作的完整规划系统。"}.get(bench, "环境协议未定义。"))
        if case:
            cfg = load_config(configs / (case + ".json"))
            preflight = check(cfg)
            summary_path = trials / case / "framework/summary.json"
            actual = read(summary_path) if summary_path.is_file() else None
        else:
            preflight, actual = None, None
        rows.append({"source_cell": cell["source_cell"], "original_model": model, "benchmark": bench,
                     "old_status": old_by_cell[cell["source_cell"]]["implementation_status"],
                     "was_one_of_original_49_gaps": old_by_cell[cell["source_cell"]]["implementation_status"] == "blocked_protocol_or_matching_asset",
                     "scope_status": status, "plain_language_reasons": reasons,
                     "structurally_composable": preflight["structurally_compatible"] if preflight else "not_checked_no_model_or_environment_component",
                     "new_real_trial_completed": bool(actual and actual["errors"] == 0), "trial_case": case,
                     "not_a_runtime_pair_whitelist": True})
    parsed = {name: load_config(configs / (name + ".json")) for name in CASES}
    environments = {}
    for config in parsed.values():
        environments.setdefault((config.benchmark.adapter, config.benchmark.options["protocol"]), config.benchmark)
    matrix = []
    for name, original in parsed.items():
        cap = default_registry().model(original.model.adapter).describe(original.model.options)
        for (family, protocol), benchmark in environments.items():
            inputs = {req.key: "instruction" if req.kind == "task" else CAMERA_DATA[family][req.key]
                      if req.kind == "sensor" else STATE_DATA[req.key] for req in cap.inputs}
            binding = replace(original.binding, inputs=inputs)
            candidate = replace(original, benchmark=benchmark, binding=binding)
            checked = check(candidate)
            original_trial = family == original.benchmark.adapter and protocol == original.benchmark.options["protocol"]
            summary_path = trials / name / "framework/summary.json"
            actual = read(summary_path) if original_trial and summary_path.is_file() else None
            native = replays.get(name)
            matrix.append({"model_family": original.model.adapter, "io_profile": original.model.options["io_profile"],
                           "benchmark_family": family, "protocol": protocol, "binding": to_data(binding),
                           "structurally_composable": checked["structurally_compatible"], "issues": checked["issues"],
                           "training_warnings": checked["warnings"], "components_local_paths_available": all(
                               value["local_paths_available"] for value in checked["readiness"].values()),
                           "combination_dependencies_verified": bool(actual),
                           "real_trial_completed": bool(actual and actual["errors"] == 0),
                           "independent_recorded_action_replay_passed": native.get("passed") if original_trial and native else None,
                           "fresh_native_policy_comparison": "not_performed_in_this_batch",
                           "case": name if original_trial else None})
    return {"kind": "mn_scope_and_component_coverage_not_acceptance", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "original_cells": len(rows),
            "original_49_gap_rows": sum(r["was_one_of_original_49_gaps"] for r in rows),
            "scope_counts_all_55": dict(Counter(r["scope_status"] for r in rows)),
            "scope_counts_original_49_only": dict(Counter(r["scope_status"] for r in rows if r["was_one_of_original_49_gaps"])),
            "identity_correction": "AE4原表LingBot-VA 2.0仍未实现；历史LingBot-VLA 2.0试点单列为附加组件证据，不抵扣VA缺口。",
            "native_model_families": 3, "native_io_profiles": 6, "native_benchmark_families": 3, "native_environment_protocols": 5,
            "structural_grid_cells": len(matrix), "structural_grid_compatible": sum(r["structurally_composable"] for r in matrix),
            "structural_grid_note": "这里只检查显式候选接线，不搜索或猜测其他接线；拒绝原因不等于该家族永久不可适配。",
            "no_placeholder_adapters": True, "stage4_formally_accepted": False, "contract_frozen": False,
            "rows": rows, "additional_implemented_model": {"name": "LingBot-VLA 2.0", "case": "lingbot_robotwin"},
            "component_grid": matrix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="append", default=[], help="CASE=receipt.json")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("new output required")
    replays = {value.split("=", 1)[0]: read(Path(value.split("=", 1)[1])) for value in args.replay}
    result = build(args.configs, args.trials, replays)
    assert result["original_cells"] == 55 and result["original_49_gap_rows"] == 49
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "component_grid")}))


if __name__ == "__main__":
    main()
