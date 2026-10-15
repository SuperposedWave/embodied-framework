"""Build a new immutable, source-backed acceptance-preparation HTML snapshot.

No historical rewrite, inference, downloads, public hosting, contract freeze or
stage acceptance. Pending jobs remain pending; failure receipts are preserved.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote
from zoneinfo import ZoneInfo
from prepare_mn_extension import ROOT, WORKSPACE, read, save, digest, historical_integrity, configurations


class EvidenceLink(str):
    """Only escaped, helper-generated local links may enter table cells as HTML."""


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(str(value)) + '</th>' for value in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + (str(value) if isinstance(value, EvidenceLink) else escape(str(value))) + '</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def resource_snapshot():
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:"))}
    def query(fields):
        result = subprocess.run(["nvidia-smi", "--query-" + fields, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True)
        return [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines()]
    return {"kind": "current_read_only_instance_resources_not_personal_quota_or_evaluation_work",
        "at_utc": datetime.now(timezone.utc).isoformat(), "logical_cpus": os.cpu_count(), "memory": memory,
        "shared_filesystem_free_bytes": shutil.disk_usage(ROOT).free, "personal_disk_quota_verified": False,
        "gpus": query("gpu=index,uuid,name,memory.total,driver_version"),
        "gpu_processes": query("compute-apps=gpu_uuid,pid,used_memory"),
        "authorized_occupancy_release_receipts": ["outputs/mn-acceptance-occupancy-release-20261008-run1.json",
            "outputs/mn-acceptance-occupancy-release-20261008-run2.json"],
        "released_occupancy_gpus": [0, 1, 2, 4, 6], "occupancy_is_not_evaluation_work": True,
        "current_gpu_uuids_differ_from_earlier_instance": True, "same_physical_gpu_across_continuation_claimed": False,
        "global_proxy_or_shared_driver_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-run", default="run12")
    parser.add_argument("--review-ready", action="store_true", help="Evidence ready for USER review with preserved failures and deferred DreamZero; not stage pass")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    if args.baseline_run not in ("run7", "run8", "run9", "run10", "run11", "run12"):
        raise ValueError("reviewed_baseline_receipt_set_required")
    baseline_paths = [ROOT / f"outputs/mn-acceptance-baseline-runtime-{index}-20261008-{args.baseline_run}/baseline-check.json" for index in range(5)]
    baseline_names = ["轻量Linux", "DROID", "RoboDojo", "OpenWAM ARX", "RoboTwin"]
    if args.baseline_run in ("run9", "run10", "run11", "run12"):
        dream_run, cosmos_run = {"run9": ("run1", "run2"), "run10": ("run2", "run3"),
            "run11": ("run3", "run4"), "run12": ("run4", "run5")}[args.baseline_run]
        baseline_paths.extend([ROOT / f"outputs/mn-acceptance-baseline-dreamzero-20261008-{dream_run}/baseline-check.json",
            ROOT / f"outputs/mn-acceptance-baseline-cosmos3-20261008-{cosmos_run}/baseline-check.json"])
        baseline_names.extend(["DreamZero", "Cosmos3"])
    baselines = [read(path) for path in baseline_paths]
    counts = {value["checks"][0]["tests_run"] for value in baselines}
    if len(counts) != 1 or not all(value["passed"] for value in baselines):
        raise ValueError("five_completed_passing_consistent_baselines_required")
    count = counts.pop()
    if count != {"run7": 360, "run8": 363, "run9": 367, "run10": 371, "run11": 372, "run12": 376}[args.baseline_run]:
        raise ValueError("actual_baseline_count_mismatch")
    if any([case["actual_exit_code"] for case in value["checks"] if "actual_exit_code" in case] != [0, 0, 0, 1, 2] for value in baselines):
        raise ValueError("legacy_CLI_regression")
    if any(re.search(r"\bskipped=[1-9]\d*", (path.parent / "unittest.txt").read_text()) for path in baseline_paths):
        raise ValueError("unittest_skips_are_not_acceptance_evidence")
    output.mkdir(parents=True, exist_ok=False)
    original = read(ROOT / "outputs/mn-extension-20261007-final2/coverage-ledger.json")
    gap_rows = [row for row in original["rows"] if row["was_one_of_original_49_gaps"]]
    scope = dict(Counter(row["scope_status"] for row in gap_rows))
    if len(gap_rows) != 49 or sorted(scope.values()) != [2, 3, 4, 9, 31]:
        raise ValueError("original_49_scope_inventory_changed")
    history = historical_integrity()
    core = {}
    for line in (ROOT / "outputs/mn-extension-integrity-20261007-final/SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        if filename in ("docs/contract-v0.2.md", "embodied_eval/mn/contracts.py", "embodied_eval/mn/interfaces.py", "embodied_eval/mn/runner.py"):
            actual = digest(ROOT / filename)
            if expected != actual:
                raise ValueError("sealed_public_core_or_runner_changed")
            core[filename] = {"sha256": actual, "unchanged": True}
    if len(core) != 4:
        raise ValueError("four_public_core_seals_required")
    save(output / "historical-integrity.json", dict(history, public_v02_core_and_runner_unchanged=core))
    now = datetime.now(timezone.utc)
    state = read(ROOT / "docs/stage-state.json")
    save(output / "stage-state-snapshot.json", state)
    resources = resource_snapshot()
    save(output / "resources.json", resources)
    synthetic_path = ROOT / "outputs/mn-acceptance-synthetic-architecture-20261008-run1/receipt.json"
    synthetic = read(synthetic_path)
    if not synthetic["passed"] or not synthetic["synthetic"] or synthetic["four_cross_combinations"] != 4:
        raise ValueError("current_synthetic_architecture_evidence_required")
    trial_index_path = ROOT / "outputs/mn-delivery-20261007-final/trial-index.json"
    trials = read(trial_index_path)
    if len(trials) != 6 or any(row["synthetic"] for row in trials):
        raise ValueError("six_original_real_trials_required")
    for trial in trials:
        path = Path(trial["summary"])
        if digest(path) != trial["summary_sha256"] or digest(path.parent / "manifest.json") != trial["manifest_sha256"]:
            raise ValueError("original_trial_evidence_changed")
    windows = []
    for path in sorted((ROOT / "outputs").glob("mn-acceptance-gpu-20261008-run*.json"),
            key=lambda item: int(item.stem.rsplit("run", 1)[1])):
        value = read(path)
        mean = sum(sum(item["utilization_percent"] for item in row["devices"]) / len(row["devices"]) for row in value["rows"]) / len(value["rows"])
        if len(value["rows"]) != value["samples"] or abs(mean - value["aggregate_mean_percent"]) > 1e-10:
            raise ValueError("GPU_window_calculation_mismatch")
        windows.append({"path": str(path.relative_to(ROOT)), "samples": value["samples"], "mean_percent": mean,
            "first_at_utc": value["rows"][0]["at_utc"], "last_at_utc": value["rows"][-1]["at_utc"], "target_met": mean >= 20})
    jobs = {}
    for name, relative in {
        "OpenWAM_RoboDojo_full_limit": "mn-acceptance-openwam-robodojo-native-limit-20261008-run1",
        "pi05_RoboDojo_full_limit": "mn-acceptance-pi05-robodojo-native-limit-20261008-run2",
        "DreamZero_native_two_gpu_request": "mn-acceptance-dreamzero-request-20261008-run5",
        "DreamZero_max_split_allocator_oom": "mn-acceptance-dreamzero-request-20261008-run4",
        "DreamZero_native_default_inference_oom": "mn-acceptance-dreamzero-request-20261008-run3",
        "DROID_independent_recorded_replay": "mn-acceptance-cosmos3-droid-native-replay-20261008-run1",
        "RoboDojo_native_repeat": "mn-acceptance-pi05-robodojo-native-replay-20261008-run8",
        "pi05_RoboTwin_full_native_limit": "mn-acceptance-pi05-robotwin-native-limit-20261008-run2",
        "pi05_RoboTwin_first_invalid_failure_reply": "mn-acceptance-pi05-robotwin-native-limit-20261008-run1",
        "DROID_same_execution_projection": "mn-acceptance-droid-same-execution-observation-20261008-run1",
        "RoboDojo_same_execution_projection": "mn-acceptance-robodojo-same-execution-observation-20261008-run2",
    }.items():
        path = ROOT / "outputs" / relative / "receipt.json"
        jobs[name] = {"directory": "outputs/" + relative, "terminal_receipt_available": path.is_file(),
            "receipt": read(path) if path.is_file() else None,
            "no_terminal_receipt_is_not_success": True}
    snapshot = {"kind": "acceptance_preparation_static_snapshot_not_formal_acceptance", "at_utc": now.isoformat(),
        "at_Asia_Shanghai": now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(), "tests_in_each_of_five_environments": count,
        "tests_in_each_current_completed_environment": count, "completed_baseline_environments": len(baselines),
        "synthetic_tests_not_robot_task_results": True, "CLI_exit_codes": [0, 0, 0, 1, 2], "jobs": jobs,
        "completed_gpu_windows": windows, "original_49_scope_counts": scope,
        "acceptance_ready_claimed": args.review_ready, "readiness_means_materials_for_user_review_not_unconditional_all_gates_pass": True,
        "stage4_accepted": False, "contract_frozen": False,
        "pending_execution_and_differences_not_redefined_as_complete": True}
    review_proof_paths = [
        "outputs/mn-acceptance-openwam-robodojo-accounting-20261008-run1.json",
        "outputs/mn-acceptance-pi05-robodojo-accounting-20261008-run1.json",
        "outputs/mn-acceptance-pi05-robotwin-accounting-20261008-run1.json",
        "outputs/mn-acceptance-pi05-robodojo-native-exchange-comparison-20261008-run3.json",
        "outputs/mn-acceptance-pi05-robotwin-native-exchange-comparison-20261008-run1.json",
        "outputs/mn-acceptance-droid-observation-array-verification-20261008-run1.json",
        "outputs/mn-acceptance-robodojo-observation-array-verification-20261008-run2.json"]
    if args.review_ready:
        if (any(not jobs[name]["terminal_receipt_available"] for name in ("OpenWAM_RoboDojo_full_limit",
                "pi05_RoboDojo_full_limit", "pi05_RoboTwin_full_native_limit"))
                or any(not (ROOT / path).is_file() or not read(ROOT / path)["passed"] for path in review_proof_paths)
                or not state["mn_acceptance_preparation_20261008"]["dreamzero"]["user_deferred_large_memory_gpu_validation"]):
            raise ValueError("terminal_non_deferred_trials_and_mapping_accounting_proofs_required_before_review_ready")
    snapshot["ready_for_user_formal_review_with_known_failures_and_deferred_DreamZero"] = args.review_ready
    save(output / "snapshot.json", snapshot)
    ledger = json.loads(json.dumps(original))
    ledger["acceptance_preparation_snapshot"] = {"at_utc": now.isoformat(), "current_jobs": jobs,
        "original_49_rows_preserved": True, "limited_steps_are_not_full_tasks": True,
        "new_DROID_does_not_replace_original_R021_R022": True, "no_placeholder_adapter_completion": True,
        "legacy_row_fields_describe_20261007_snapshot_only": True,
        "current_validation_field_is_authoritative_for_20261008_runtime_status": True}
    for row in ledger["rows"]:
        if row["scope_status"] != "new_structure_ready_runtime_unverified":
            continue
        job_name = ("pi05_RoboTwin_full_native_limit" if row["benchmark"] == "robotwin" else
            "pi05_RoboDojo_full_limit" if row["original_model"] == "π0.5" else "OpenWAM_RoboDojo_full_limit")
        job = jobs[job_name]
        receipt = job["receipt"]
        row["current_validation"] = {"historical_scope_status_not_rewritten": row["scope_status"],
            "structurally_composable": True, "terminal_real_receipt": job["directory"] + "/receipt.json" if receipt else None,
            "runtime_completed": receipt is not None, "framework_trial_passed": receipt.get("passed") if receipt else None,
            "actual_statistics": receipt.get("execution_statistics") if receipt else None,
            "metrics": receipt.get("metrics") if receipt else None,
            "full_suite_or_generalization_verified": False, "formal_acceptance": False}
    save(output / "coverage-ledger.json", ledger)
    gap_rows = [row for row in ledger["rows"] if row["was_one_of_original_49_gaps"]]
    from embodied_eval.codec import to_data
    examples = configurations()
    example_dir = output / "configuration-examples"
    example_dir.mkdir()
    for name, config in examples.items():
        save(example_dir / (name + ".json"), to_data(config))
    linked = []
    def link(path, label):
        path = Path(path).resolve()
        path.relative_to(ROOT)
        if not path.is_file():
            raise ValueError("required_report_evidence_missing")
        linked.append(path)
        return EvidenceLink('<a href="' + quote(os.path.relpath(path, output), safe="/") + '">' + escape(label) + '</a>')
    def proof(relative, label):
        return link(ROOT / relative, label)
    sections = []
    def section(identifier, title, content):
        sections.append(f'<section aria-labelledby="{identifier}"><h2 id="{identifier}">{escape(title)}</h2>{content}</section>')
    conclusion = "本轮非后置组件的运行、映射及计数证据已经收齐，可提交用户正式验收；已知失败与后置验证必须同时审阅。" if args.review_ready else "架构和软件回归已完成，剩余真实运行与复现差异仍在收尾。"
    section("conclusion", "当前结论", f'''<div class="callout">{conclusion}
        本页是验收准备快照，不是“所有组合都能成功”、阶段4已通过或契约已冻结。</div>
        <p>{len(baselines)}套环境各{count}项软件测试通过，五组旧CLI退出码仍为0/0/0/1/2。CLI指命令行入口；这些通过项是软件与合成数据测试，不是{count}个真实机器人任务。</p>
        <p>已独立注册5个模型家族、10份模型输入输出配置、5个仿真环境家族、7份环境协议。LingBot-VA按用户要求暂缓；已实现的LingBot-VLA是另一个模型，继续保留。</p>
        <p>DreamZero主权重与额外基础组件均完整校验，两卡已实际加载，但默认设置与两种分配策略都在原生图像编码时显存不足，最后一次碎片计数为0。
        用户已明确同意安排H100资源但留待后续对接；当前保留为后置真实验证，不标通过、不自动租卡或修改模型算法。
        RoboDojo和DROID的原生重放已完成，状态/计数一致、图像不同；在同一当前GPU上重复纯原生运行也有像素差异。RoboDojo新长程运行结果与映射证据分别列出，不能用任务成功代替转换正确。</p>''')
    checklist = {"kind": "user_acceptance_checklist_not_self_approval", "at_utc": now.isoformat(),
        "ready_for_user_review_with_known_findings": args.review_ready, "stage4_accepted": False, "contract_frozen": False,
        "gates": [
            {"item": "旧85项、202项和CLI无回归", "result": "通过", "synthetic": True,
                "tests_each_environment": count, "environments": len(baselines), "CLI_exit_codes": [0, 0, 0, 1, 2],
                "proofs": [str(path.relative_to(ROOT)) for path in baseline_paths]},
            {"item": "两个模型两个环境交叉、新增第三插件不改其他组件", "result": "通过", "synthetic": True,
                "proofs": [str(synthetic_path.relative_to(ROOT))]},
            {"item": "相机、完整单双臂、坐标、状态、缓存、变时长和失败计数", "result": "软件检查通过，真实映射及计数另有证据",
                "proofs": [path for path in review_proof_paths if (ROOT / path).is_file()]},
            {"item": "六组原有入口逐项原生对照", "result": "对照完成，严格复现仍有失败，不能当作全部通过",
                "strict_saved_action_replays_passed": 5, "strict_saved_action_replays_failed": 1,
                "proofs": [str(trial_index_path.relative_to(ROOT)), "outputs/mn-delivery-20261007-final/verification.json",
                    "outputs/mn-acceptance-robotwin-planning-comparison-20261008-run2.json",
                    "outputs/mn-acceptance-pi05-controlled-comparison-20261008-run1.json"]},
            {"item": "结构可组合、资产就绪、真实验证分开记账", "result": "已分列，不承诺全环境任务成功", "original_gap_count": 49},
            {"item": "GPU已采窗口总体平均至少20%", "result": "通过" if windows and all(row["target_met"] for row in windows) else "未通过",
                "occupancy_separately_recorded_not_evaluation_work": True, "unsampled_whole_session_mean_guaranteed": False,
                "proofs": [row["path"] for row in windows]},
            {"item": "DreamZero真实动作与DROID闭环", "result": "用户明确安排后置，当前显存不足，无动作输出",
                "user_resource_followup": "两张每卡80GB的H100，后续资源对接时提醒，不保证换卡必然通过"}],
        "remaining_user_decisions": ["是否接受本轮架构交付及明确列出的已知复现限制；不由助手代替验收",
            "DreamZero后续资源与真实验证另行对接；不会把准备好的入口算作已通过"]}
    save(output / "acceptance-checklist.json", checklist)
    section("acceptance-checklist", "验收清单与通过边界", table(["验收项", "当前结论"], [
        [row["item"], row["result"]] for row in checklist["gates"]]) + '<p>' + link(output / "acceptance-checklist.json", "完整验收清单与证据索引") +
        '。待用户确认的是阶段4交付及明确限制，不是重新确认已授权的技术细节。后续H100验证不自动算通过；阶段5未开始，公共契约未冻结。</p>')
    section("architecture", "独立适配架构", '''<p>M指模型数量，N指仿真环境数量。M+N表示各方只维护自己的适配器，而不是为每一对组合写新Python实现。0.2是新公共数据协议版本；0.1旧通路保留。</p>
        <pre>环境原始观测 → 环境适配器 → 公共观测 → 模型适配器
             → 原生模型推理 → 公共动作 → 环境适配器 → 仿真执行</pre>
        <p>箭头表示数据流向。模型适配器不拿环境对象、不按环境名称选实现；环境适配器不知道当前模型。双方独立选择配置后，仅用接线数据指定相机、手臂目标和明确坐标转换。</p>''' + table(["位置", "负责什么", "不负责什么"], [
            ["Model Adapter", "输入排列、图像处理、实测状态转换、归一化、原生模型调用、动作解码、原生动作段缓存", "环境初始化、机器人控制器、官方评分"],
            ["Benchmark Adapter", "环境初始化、观测采集、物理动作转换、控制器、逆运动学、仿真推进、官方评分", "根据模型名称选择算法或权重"],
            ["接线配置", "哪个相机接哪个输入、哪个输出控制哪只手、使用哪条已定义坐标变换", "每个组合再写一套Python适配器"],
            ["Runner", "通用调用顺序、协议校验、已批准预算、计数、证据保存", "图像缩放、模型归一化、机器人控制细节、未建设的硬超时或恢复"]]) + '''
        <p>逆运动学指根据末端位置求关节姿态，属于环境执行侧。缓存指模型一次产生多条动作后依次取用，不能把每次取行算成一次真实模型推理。''' +
        proof("docs/contract-v0.2.md", "公共契约") + '；' + link(output / "historical-integrity.json", "公共核心与历史证据核验") + '</p>')
    section("public-format", "公共格式与明确拒绝", '''<p>公共观测明确图像形状、RGB颜色顺序、关节、末端姿态、夹爪、坐标系、采集时间和公开指令；实际测量与控制目标分开。
        公共动作明确关节目标、绝对位姿或相对位姿，不能只塞一个没含义的浮点数组。双臂必要目标必须完整，不允许丢掉一只手后继续运行。</p>
        <p>模型声明自己需要什么、能输出什么，环境声明能提供什么、接受什么。结构接得上与权重在哪种机器人上训练过分开；训练覆盖给提示，缺相机、缺状态、维数或坐标含义不明确仍拒绝。</p>
        <p>原生扩展保留明确协议名：例如原生夹爪请求可暂时超出合法开度，由原生环境按自己的截断与插值顺序处理，公共合法开度仍是0到1。
        不知道物理含义的数组只能交给声明相同协议的执行端，不能因此声称能跨环境通用。</p>
        <p>可变时长动作分别记原生动作次数、底层物理步数和实际仿真秒数。没有新增硬仿真时间上限；预算允许在一次已开始的原生动作完成后再检查，因此不是强行中断仿真。</p>''')
    section("configurations", "接线与配置示例", '''<p>不同权重选择具名模型配置，不同环境版本选择具名环境协议，分别核验来源和资产。下面只是可检查的示例，不是运行时组合白名单，也不是任务成功承诺。</p>
        <p>Cosmos与DreamZero原生客户端的左右相机变量命名不同，示例显式接线；环境侧没有根据模型切换相机。缺少的相机不会用黑图补齐。</p><ul>''' +
        ''.join('<li>' + link(example_dir / (name + ".json"), name.replace("_", " / ")) + '</li>' for name in examples) + '</ul>')
    section("regression", "软件与合成架构验证", table(["环境", "实际测试数", "五组CLI退出码", "证据"], [
        [name, count, "0 / 0 / 0 / 1 / 2", link(path, "回归回执")] for name, path in zip(baseline_names, baseline_paths)
        ]) + '<p>两个测试模型与两个测试环境的四种交叉组合全部通过，增加第三个模型及第三个环境均无需修改Runner或已有适配器。仍是合成架构测试，没有真实模型或机器人任务成绩：' + link(synthetic_path, "四种交叉与新增插件证据") + '''。</p>
        <p>回归覆盖相机重命名、单/双臂完整映射、显式坐标转换、动作类型不匹配、状态缺失、缓存、执行失联以及变时长计数。原85项、202项和旧退出码均保留。</p>''')
    section("original-six", "原有六组真实入口", table(["组合", "真实动作数", "物理步数", "官方成功值"], [
        [row["case"], row["statistics"]["env_steps"], row["statistics"]["physics_steps"] if row["statistics"]["physics_steps"] is not None else "旧固定周期汇总未提供", row["official_success"]["value"]] for row in trials
        ]) + '<p>六组各是一条已保存真实试验，不代表全suite、多个seed统计或泛化能力。suite是任务集合；seed是初始化的随机种子。' + link(trial_index_path, "六组来源、配置与结果索引") + '''。</p>
        <p>原生客户端输入/输出转换逐项比较过；使用固定合成输出的源函数检查仍单独标合成，不冒充真实重新推理。
        六组记录动作重放中五组严格通过，OpenWAM/RoboTwin整段严格复现未通过，失败保留；任务成功不抵扣映射差异。</p>''' +
        '<p>' + proof("outputs/mn-native-mapping-20261007-final-direct-source/receipt.json", "固定源函数检查") + '；' +
        proof("outputs/mn-delivery-20261007-final/M+N-architecture-and-evidence.html", "历史完整执行对照") + '</p>')
    section("differences", "复现差异的人话说明", '''<h3 id="pi05-rng">π0.5第一条新动作不同</h3>
        <p>把同一份真实起始观测送入模型，在同一个进程内恢复相同随机数状态，整段动作完全一致；换一份随机数状态，动作会改变，说明推理随机性确实影响结果。
        但两个独立进程即使输入、随机数状态、源码、包版本与设备记录相同，首动作仍有最大0.0016450205078125668差异。因此不能把历史差异全归咎于随机性；具体跨进程计算原因仍未定位。</p><p>''' +
        proof("outputs/mn-acceptance-pi05-controlled-comparison-20261008-run1.json", "九次真实推理的受控调查") + '''。</p>
        <h3 id="robotwin-planner">RoboTwin执行同一动作，原生规划本身也会变</h3>
        <p>两次不经过适配器的原生执行，使用同一保存动作、相同起步关节位置和速度，左臂规划轨迹仍不同，右臂本次相同。
        适配器路径与其中一次原生执行的两臂轨迹、全部288条控制操作精确相同。差异定位到原生规划边界，但具体内核、缓存或内部随机状态尚未确定。</p>
        <p>这是一条动作的调查，不是113条历史动作全部复现。没有放宽容差、改规划器或用任务成功掩盖失败。''' +
        proof("outputs/mn-acceptance-robotwin-planning-comparison-20261008-run2.json", "原生重复与适配器对照") + '''。</p>
        <h3 id="robodojo-images">RoboDojo的机器人状态一致，画面还不一致</h3>
        <p>独立原生工厂执行4条保存动作后，五次状态快照、40个物理步和原生评分一致，但三路相机像素不同，严格对照仍未通过。
        在同一当前物理GPU上，两次纯原生运行的相机参数、已记录渲染设置、仿真配置、状态、计数和评分相同，图像仍不同。两轮都在启动之前核验412个所选资产，资产指纹相同。
        说明原生画面本身不能保证逐像素重复；不能直接归咎于适配器，也没有唯一解释全部历史差异或查明具体渲染内核。没有修改渲染开关或像素容差。</p><p>''' +
        proof("outputs/mn-acceptance-pi05-robodojo-native-replay-20261008-run4/native-result.json", "完整四动作原生对照") + '；' +
        proof("outputs/mn-acceptance-robodojo-native-render-comparison-20261008-run2.json", "同一卡两次纯原生画面比较") + '''。</p>
        <h3 id="droid-images">DROID也出现纯原生画面变化</h3>
        <p>独立原生双重重置、4条保存动作、32物理步的状态、时间、指令与结束标志一致，但三路图像严格比较失败。
        同一当前GPU7的两次纯原生重放也有像素差异，相机内外参数、已记录仿真和渲染设置、状态与计数一致。第一张画面各通道平均绝对差约0.210、0.187、0.643；像素每通道取值是0到255。
        这是原生重复实验，不是放宽验收门槛；仍不保证全部内部渲染状态相同，也没有把旧严格失败改成通过。</p><p>''' +
        proof("outputs/mn-acceptance-cosmos3-droid-native-replay-20261008-run1/native-result.json", "DROID独立执行重放") + '；' +
        proof("outputs/mn-acceptance-droid-native-render-comparison-20261008-run1.json", "同一卡DROID纯原生重复") + '''。</p>
        <p>官方IsaacLab2.2的复现文档讨论固定硬件下的物理仿真及浮点执行顺序，不是对本轮所有渲染像素的保证。
        原生物理状态与图像是两种不同的检查，不能互相抵扣：<a href="https://isaac-sim.github.io/IsaacLab/v2.2.0/source/features/reproducibility.html">官方固定版本复现说明</a>。</p>''')
    def actual_trial(name):
        value = jobs[name]["receipt"]
        if value is None:
            return "尚无最终回执，不预记通过"
        stats = value.get("execution_statistics", {})
        success = value.get("metrics", {}).get("success", {})
        score = value.get("metrics", {}).get("score", {})
        success_text = "成功" if success.get("value") is True else "未成功" if success.get("value") is False else "无法取得成功值"
        return (f'实际{stats.get("env_steps")}动作、{stats.get("physics_steps")}物理步、'
            f'{stats.get("sim_time_s")}仿真秒；框架错误{value.get("framework_errors")}、'
            f'退出码{value.get("client_exit_code")}；官方判断{success_text}（{"有效" if success.get("valid") else "无效"}），分数{score.get("value")}；'
            f'结束原因{value.get("termination_reason", "参看回执")}')
    section("new-runtime", "新增组件实际到了哪里", table(["组件/链路", "已经证明", "未完成或限制"], [
        ["OpenWAM ARX / RoboDojo", actual_trial("OpenWAM_RoboDojo_full_limit"), "只有一个固定任务/场景/seed；另一次真实32行映射通过，本次未捕获全程原生生成段，不能宣称319行全部作过原生输出对照"],
        ["π0.5 ARX / RoboDojo", actual_trial("pi05_RoboDojo_full_limit"), "只有一个固定任务/场景/seed；旧165动作被现实时间预算截断仍保留；800是上限不是实做量"],
        ["Cosmos3 / DROID", "真实四动作、32物理步、0.2667仿真秒、正常关闭；保存真实观测的一次32行生成/转换/缓存精确对照；独立原生状态与计数精确一致", "图像严格重放未通过，纯原生重复本身也变化；此来源没有官方任务成功评分，不能造成功率"],
        ["DreamZero-DROID", "主权重15份及额外基础组件15份全文件校验；原生所需导入依赖修复、私有编译器检查通过；两张卡实际加载完成", "默认和两种分配策略均显存不足，尚无真实动作或闭环通过；按用户明确安排后置H100复核"],
        ["π0.5 ARX / RoboTwin", actual_trial("pi05_RoboTwin_full_native_limit"), "首轮在第95动作左臂规划失败，还暴露错误回执编号；修复后新试验单列，不改历史失败或规划器"]]) +
        '<p>' + proof("outputs/mn-acceptance-openwam-real-input-mapping-20261008-run2/receipt.json", "OpenWAM真实输入32行") + '；' +
        proof("outputs/mn-acceptance-cosmos-real-input-mapping-20261008-run1/receipt.json", "Cosmos真实输入32行") + '；' +
        proof("outputs/mn-acceptance-pi05-robodojo-native-exchange-comparison-20261008-run1.json", "π0.5原生请求与缓存") + '；' +
        proof("outputs/mn-acceptance-dreamzero-backbone-verified-20261008-run1/receipt.json", "DreamZero完整基础组件") + '</p>')
    current_runtime_links = []
    for name in ("OpenWAM_RoboDojo_full_limit", "pi05_RoboDojo_full_limit", "pi05_RoboTwin_full_native_limit",
            "pi05_RoboTwin_first_invalid_failure_reply"):
        job = jobs[name]
        if job["terminal_receipt_available"]:
            current_runtime_links.append('<li>' + proof(job["directory"] + "/receipt.json", name + "实际终局回执") + '</li>')
    comparison = ROOT / "outputs/mn-acceptance-pi05-robodojo-native-exchange-comparison-20261008-run3.json"
    if comparison.is_file():
        value = read(comparison)
        if not value["passed"] or value["new_policy_inference"] or value["new_simulator_episode"]:
            raise ValueError("completed_read_only_real_exchange_proof_required")
        current_runtime_links.append('<li>' + link(comparison, "π0.5新长程输入、全生成段与实际缓存顺序") +
            f'：{value["native_requests"]}次请求、生成{value["generated_rows_compared"]}行、执行{value["executed_cache_rows_compared"]}行，数值和类型精确一致。</li>')
    robotwin_comparison = ROOT / "outputs/mn-acceptance-pi05-robotwin-native-exchange-comparison-20261008-run1.json"
    if robotwin_comparison.is_file():
        value = read(robotwin_comparison)
        if not value["passed"] or value["new_policy_inference"] or value["new_simulator_episode"]:
            raise ValueError("read_only_RoboTwin_exchange_mapping_required")
        current_runtime_links.append('<li>' + link(robotwin_comparison, "π0.5/RoboTwin原生输入与关节转换、缓存顺序") +
            f'：{value["native_requests"]}次请求、生成{value["generated_rows_compared"]}行、送入执行边界{value["executed_cache_rows_compared"]}行，数值和类型精确一致。</li>')
    section("current-runtime-proofs", "新运行终局与映射证据", '<ul>' + ''.join(current_runtime_links) + '''</ul>
        <p>长程RoboDojo任务在错误编号修复前加载了Adapter，不能说它们验证了新失败分支。另有修复后同次观测实测与合成失败回执检查。
        任务正常结束指原生环境声明结束，不是全任务集合验证，也不保证换seed、机械臂或环境之后仍能成功。</p>''')
    accounting_rows = []
    for label, relative in (("OpenWAM/RoboDojo", "outputs/mn-acceptance-openwam-robodojo-accounting-20261008-run1.json"),
            ("π0.5/RoboDojo", "outputs/mn-acceptance-pi05-robodojo-accounting-20261008-run1.json"),
            ("π0.5/RoboTwin", "outputs/mn-acceptance-pi05-robotwin-accounting-20261008-run1.json")):
        path = ROOT / relative
        if path.is_file():
            value = read(path)
            if not value["passed"] or value["synthetic"] or value["new_inference_or_simulation"]:
                raise ValueError("read_only_real_accounting_evidence_required")
            counts = value["actual_saved_reply_accounting"]
            accounting_rows.append([label, counts["actual_env_steps"], counts["actual_physics_steps"],
                len(counts["failed_replies"]), proof(relative, "逐条回执与实际汇总复核")])
    section("actual-accounting", "实际执行量逐条核账", table(["运行", "动作数", "物理步数", "失败动作回执数", "证据"], accounting_rows) +
        '''<p>只读检查保存的每条预测与动作回复，核对唯一编号、所有已成功动作的完整目标、实际物理步和时间；失败只确认确实执行的目标，不能漏掉一只手后说整条成功。
        再与Runner结果、原生环境诊断和评分记录核对。本项是证据内部一致性检查，不是新仿真或重新独立测量底层物理，也不把已正确记录的失败变成任务成功。
        只对累积时间的浮点舍入按原协议精度核对；图像、状态和动作数值的严格对照没有放宽。</p>''')
    section("deployment", "部署问题和修复边界", '''<p>DreamZero固定Transformers4.51.3与自动选到的DeepSpeed0.19.7出现循环导入。
        源码对DeepSpeed未锁版本，仅在本项目独立环境中改用官方0.17.6，发布包SHA256校验和原生导入检查通过，Torch及Transformers固定版本不变。
        CUDA12.9编译器来自NVIDIA固定发布的三个完整校验包，只放在本轮产物目录，不升级共享驱动或改变全局环境。</p>
        <p>RoboDojo恢复固定来源的CPU物理仿真，渲染与原生规划仍使用GPU；曾漏掉原生任务条件登记导致空条件误判成功，该失败诊断保留、不作为成绩。
        当前按原生动作前登记任务和过程评分，取最终分数时不重新登记清空进度。原生独立重放缺ffmpeg时暴露项目内已安装、包清单校验过的FFmpeg7.0.2，不关闭原生录像。</p>
        <p>环境应用与模型可各自使用独立Python进程，以隔离不兼容依赖及原生插件收尾问题。环境关闭回复与退出码都须核验；动作执行失联仍是状态未知，无自动重试或恢复。
        独立诊断的进程监督不是Runner硬超时或并发功能。</p><p>''' +
        proof("outputs/mn-acceptance-dreamzero-request-20261008-run1/receipt.json", "双卡首轮导入失败") + '；' +
        proof("outputs/mn-acceptance-dreamzero-import-compatibility-20261008-run1/receipt.json", "缺编译器的安装失败") + '；' +
        proof("outputs/mn-acceptance-dreamzero-import-compatibility-20261008-run2/receipt.json", "DeepSpeed修复与版本核验") + '；' +
        proof("outputs/mn-acceptance-dreamzero-request-20261008-run3/rank-0-result.json", "DreamZero第一卡加载及显存失败") + '；' +
        proof("outputs/mn-acceptance-dreamzero-request-20261008-run3/rank-1-result.json", "DreamZero第二卡加载及显存失败") + '；' +
        proof("outputs/mn-acceptance-cuda129-private-toolchain-20261008-run1/receipt.json", "私有工具链检查") + '；' +
        proof("outputs/mn-acceptance-native-video-runtime-20261008-run1/receipt.json", "私有录像依赖") + '''。</p>
        <p>官方来源：<a href="https://github.com/deepspeedai/DeepSpeed/blob/v0.17.6/deepspeed/runtime/hybrid_engine.py">DeepSpeed固定发布源码</a>；
        <a href="https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json">NVIDIA固定工具链清单</a>。
        大文件下载每条命令先临时直连，失败回原有代理；未输出或保存API密钥。</p>
        <p>当前共享驱动570.124.06低于Isaac Sim5.1建议的580.65.06，实测成功不等于该部署获得官方认证；未升级共享驱动。
        RoboDojo安装脚本最终starlette0.45.3与固定IsaacLab声明0.49.1不一致；Cosmos环境原生安装的NumPy2.2.6超出openpi-client声明范围；
        历史包检查还记录jupyter-compare-view要求Python低于3.13。这些偏差公开保留，软件回归通过不能抵扣完整依赖一致性或未测试的直播/交互功能。</p>
        <p>DreamZero原生路径两卡各自放完整模型，双卡分担的是带指令与不带指令两条计算，不是把权重切成八份。
        固定源码只接受一或两卡。不能把八张卡的总显存相加，就断言这个入口能用；当前每卡约48GiB，GiB是1024的三次方字节。
        显存分配诊断仅设置当前子进程的PyTorch分配策略，不减少图像、精度、采样次数或改分块编码，不算原生任务通过。</p>
        <p>两种碎片处理均失败：限制大块拆分时两卡各已有49,530,590,720字节活跃张量；允许内存块增长时，碎片计数降为0，仍显存不足。
        “活跃张量”指计算仍在使用、不能直接释放的显存内容；“碎片”指未使用但难复用的小块。两张卡分别存完整模型，不共享一个总显存池。</p>
        <p>用户明确回复：“可以安排，但是这个要留到后面进行，我会和你对接，你到时候要提醒我”。
        因此后续资源待办是两张每卡80GB的H100，等其余验收材料收齐或用户对接时提醒；这不是本轮已运行、已租用、定时提醒已创建或保证H100必然通过。
        <a href="https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/README.md">固定DreamZero官方硬件说明</a>；
        <a href="https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management">PyTorch2.8显存管理说明</a>。</p><p>''' +
        proof("outputs/mn-acceptance-dreamzero-request-20261008-run4/rank-0-result.json", "大块不拆分的显存失败") + '；' +
        proof("outputs/mn-acceptance-dreamzero-request-20261008-run5/rank-0-result.json", "增长式分配器第一卡失败") + '；' +
        proof("outputs/mn-acceptance-dreamzero-request-20261008-run5/rank-1-result.json", "增长式分配器第二卡失败") + '''。</p>
        <p>RoboTwin长程首轮第95次动作左臂规划失败，原生执行器确认95动作/2891物理步，Runner因错误回执编号只能校验到94动作/2841物理步。
        不能挑较大的数字当正常总量；该轮状态未知、success为空且无效，旧失败保留。已把两个Benchmark的错误分支改用公共协议原有planning_failed编号，
        意为规划失败；保留execution_failed（执行失败）状态、实际计数和仅确认的命令ID。不扩展协议枚举、不改Runner或规划器。</p>''')
    sections[-1] = sections[-1].replace('</section>', '''<p>修复的是失败时怎么报告，不是把失败轨迹改成成功。新RoboTwin运行重新进行了推理与原生执行，不能假设它和第一轮逐动作相同。
        第二轮没有触发同一规划失败，并不证明原生规划失败已消除；也不把400动作跑通等同于模型完成任务。</p></section>''')
    projection_rows = []
    for name, relative in (("DROID", "outputs/mn-acceptance-droid-observation-array-verification-20261008-run1.json"),
            ("RoboDojo（错误编号修复前）", "outputs/mn-acceptance-robodojo-observation-array-verification-20261008-run1.json"),
            ("RoboDojo（错误编号修复后）", "outputs/mn-acceptance-robodojo-observation-array-verification-20261008-run2.json")):
        path = ROOT / relative
        if path.is_file():
            value = read(path)
            if not value["passed"] or value["synthetic"]:
                raise ValueError("saved_real_projection_evidence_required")
            projection_rows.append([name, "三路相机、全部状态及来源和时间均精确相同", value["snapshots"], proof(relative, "保存数组逐项复核")])
    section("same-execution-mapping", "同一次仿真，映射有没有改坏数据", '''<p>独立重开环境会有渲染差异，因此另加同一次执行的检查：
        观察Adapter正常消费的那份原生观测，不额外拍照、渲染或调用模型；逐项比较转换后的公共图像、关节、位姿和夹爪。
        原始数组和公共数组都保存，再只读复核文件指纹、数值和类型。各完成4条保存动作，得到重置加4次动作共5份快照，正常关闭并退出0。</p>''' +
        table(["环境", "同次映射结果", "快照数", "证据"], projection_rows) +
        '<p>这证明这些执行的边界映射正确，不宣称所有历史像素轨迹完全重现，也不撤销独立原生重放的严格失败。</p>')
    section("resources", "资源和GPU约束", table(["项目", "当前查询值"], [
        ["GPU", "；".join(f"编号{row[0]}：{row[2]}，{row[3]}MiB，驱动{row[4]}" for row in resources["gpus"])],
        ["CPU逻辑核", resources["logical_cpus"]], ["主机内存字节", resources["memory"]["MemTotal"]],
        ["共享文件系统剩余字节", resources["shared_filesystem_free_bytes"]],
        ["个人磁盘配额", "未核实，不能用共享空闲量替代"],
        ["已释放占卡GPU编号", "0、1、2、4、6；控制器与其余3、5、7号占卡保留"]]) +
        '<p>GPU编号只表示当前实例顺序，不等于始终同一张物理卡。MiB是1024×1024字节。SHA256是文件内容指纹，用来核对数据是否变化。' +
        link(output / "resources.json", "当前硬件标识与进程快照") + '</p>' + table(["采样窗口", "样本数", "八卡总体均值", "证据"], [
            [str(index + 1), row["samples"], f'{row["mean_percent"]:.6f}%', proof(row["path"], "原始采样")] for index, row in enumerate(windows)]) +
        '''<p>目标是总体平均至少20%，不是每张卡必须20%。统计是每个样本内八卡平均，再对已采样窗口求平均。
        包含占卡负载，不作为模型评测工作量、效率或成绩；不能宣称未采样时段也保证达标，不能把重叠窗口当成唯一的全程时间平均。</p>''')
    labels = {"outside_action_model_scope": "本轮范围外", "user_deferred": "用户暂缓VA", "protocol_identity_still_undefined": "环境编号或被测角色未定",
        "model_present_action_structure_mismatch": "实际动作/状态结构不满足", "new_structure_ready_runtime_unverified": "架构已接通，当前真实验证结果另列（保留旧分类来源）"}
    section("gap-summary", "原49组缺口重新分类", table(["类别", "格数", "人话解释"], [
        ["纯视频/图像/文字等范围外", 31, "本轮不新增预测画面转机器人动作的规划系统，不能给不能直接出动作的模型挂空适配器"],
        ["LingBot-VA暂缓", 4, "原49缺口中4格；全部55格中VA为5格。已实现VLA不能冒充VA"],
        ["协议/角色未定义", 9, "原表编号、版本、输入输出或被测角色不足，不能猜，也不把新DROID替换R021/R022"],
        ["Cosmos结构不满足", 2, "这份模型出单臂关节动作，而对应环境协议只接受末端控制；没明确接口不能硬接"],
        ["新增结构已接上", 3, "π0.5/RoboTwin、π0.5/RoboDojo、OpenWAM/RoboDojo；最新运行状态逐格补充，两条RoboDojo已有原生任务终局，不抵扣全suite或训练覆盖"]]) +
        '<p>结构可组合、运行资产就绪、已真实验证分别记账，理论支持不等于跨机器人都能完成任务。' + link(output / "coverage-ledger.json", "55格覆盖账本与原49格索引") + '</p>')
    gap_body = []
    for index, row in enumerate(gap_rows, 1):
        gap_body.append(f'<h3 id="gap-{index:02d}">{escape(row["original_model"])} / {escape(row["benchmark"])}</h3>'
            f'<p>原表单元格 {escape(row["source_cell"])}，结论：{escape(labels[row["scope_status"]])}。</p>'
            + ('<p>下列说明保留10月7日分类时的来源状态，不代表当前下载或运行仍未完成。</p>' if "current_validation" in row else '')
            + ''.join('<p>' + escape(reason) + '</p>' for reason in row["plain_language_reasons"]))
        if "current_validation" in row:
            value = row["current_validation"]
            if value["runtime_completed"]:
                stats = value["actual_statistics"]
                success = value["metrics"].get("success", {})
                score = value["metrics"].get("score", {})
                success_text = "成功" if success.get("value") is True else "未成功" if success.get("value") is False else "无法取得有效成功值"
                gap_body.append(f'<p>当前实际终局：{stats["env_steps"]}次动作，{stats["physics_steps"]}个物理步，{stats["sim_time_s"]}仿真秒。'
                    + escape(f'官方判断{success_text}，分数{score.get("value")}；运行证据回执为{"通过" if value["framework_trial_passed"] else "失败"}') +
                    '。一份真实运行不代表全任务或泛化完成。' + proof(value["terminal_real_receipt"], "该格当前真实回执") + '</p>')
            else:
                gap_body.append('<p>当前新长程运行尚无终局回执，已有有限步接入证据；不预记新试验通过。</p>')
    section("gaps-each", "原49组逐项说明", ''.join(gap_body))
    remaining = '''<ol>
        <li>DreamZero默认及两种分配策略均显存不足，按用户明确安排后置H100验证。其余材料收齐或用户对接时，提醒两张每卡80GB资源；不预认动作/闭环通过，新仿真入口只是已准备未执行。</li>
        <li>两条RoboDojo长程试验已有正常原生终局，实做319/280动作；继续复核逐条回执与汇总计数，不把800上限当实际执行量，不宣称全suite验证。</li>
        <li>π0.5/RoboTwin长程试验的错误回执已修为现有允许编号，补做新实际运行；原生规划失败仍是失败，不能强迫执行或把未完成400步称为完整任务成功。</li>
        <li>DROID与RoboDojo独立重放和同GPU纯原生重复已完成，状态/计数一致、像素不一致；同次原始与公共数组映射精确相同，两种不同检查分别留证。</li>
        <li>对全部要求逐条核对当前证据，封存新源码与报告摘要、检查网页和全部本地证据链接，再提交用户正式验收。</li>
        </ol>'''
    if args.review_ready:
        remaining = '''<ol><li>DreamZero按用户明确安排后置。后续用户资源对接时提醒两张每卡80GB的H100；当前仍无真实动作/仿真闭环，不保证换卡必然通过。</li>
            <li>两条RoboDojo原生任务结束、计数已核对，仅代表当前一个固定任务和场景，不代表全suite、多个seed统计或跨机器人泛化。</li>
            <li>π0.5/RoboTwin新试验已有终局回执，映射、缓存与计数证据已收齐；任务成绩与框架错误按实测分别报告，不强迫失败动作执行，不新增自动重试。</li>
            <li>严格复现仍有未通过项：OpenWAM/RoboTwin历史整段轨迹，以及RoboDojo/DROID独立重开环境的像素。原生重复与同次映射调查已完成，但具体内部非确定性根因没有唯一定位，旧失败不会改成通过。</li>
            <li>用户正式确认是否接受本轮架构交付和明确限制。网页与文件完整性核验、源码摘要封存另有回执；它们不等于阶段4已通过。阶段5未开始，公共契约未冻结。</li></ol>'''
    section("remaining", "验收与后续边界" if args.review_ready else "正式验收前仍需收尾", remaining + '''<p>旧0.1入口、历史Mac绝对路径、旧报告和35份阶段2证据保持原样。没有Git提交或remote的框架工作区不伪造提交。
        DataRef载荷读取、Runner硬超时/并发/恢复、硬仿真时间预算以及画面转动作规划器仍未建设，也不列成本轮已完成项。</p>
        <p>网页仅在实例本机127.0.0.1受限提供报告和明确链接的文档。实例的localhost不是用户电脑；用户浏览需实际平台端口转发或SSH隧道，不发布公网。</p><p>''' +
        link(output / "snapshot.json", "当前验收准备状态") + '；' + link(output / "stage-state-snapshot.json", "阶段记录快照") + '</p>')
    title = "M+N 架构与正式验收提交报告" if args.review_ready else "M+N 架构与正式验收准备报告"
    template = Path("/root/.codex/skills/html-report/assets/report-template.html").read_text()
    shell = template[:template.index("<body>")].replace('lang="en"', 'lang="zh-CN"').replace("Report title", title)
    report = output / "M+N-acceptance-preparation-and-evidence.html"
    report.write_text(shell + '<body><div class="report-shell"><main class="report-content"><h1 id="report-title">' + title +
        '</h1><p class="lead">快照时间（北京时间）：' + escape(snapshot["at_Asia_Shanghai"]) + '。进行中结果不预记通过，验收权仍由用户保留。</p>' +
        ''.join(sections) + '</main></div></body></html>', encoding="utf-8")
    save(output / "evidence-sha256.json", {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(linked + [report]))})
    print(str(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
