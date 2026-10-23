"""New static runtime evidence snapshot; never overwrite historical reports."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote
from build_mn_report import resources, table, digest
from prepare_mn_extension import read, save, historical_integrity
from download_stage4_assets import verify
from prepare_dreamzero_backbone import MODEL_ROOT

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("/root/.codex/skills/html-report/assets/report-template.html"))
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    baseline_names = ("baseline", "baseline-droid", "baseline-openwam", "baseline-cosmos3")
    baseline_paths = [ROOT / f"outputs/mn-runtime-{name}-20261008-final2/baseline-check.json" for name in baseline_names]
    baselines = [read(path) for path in baseline_paths]
    for record in baselines:
        assert record["passed"] and record["checks"][0]["tests_run"] == 320
        assert [row["actual_exit_code"] for row in record["checks"] if "actual_exit_code" in row] == [0, 0, 0, 1, 2]
    asset_path = ROOT / "outputs/mn-runtime-preparation-20261008-run1/assets.json"
    assets = read(asset_path)
    assert all(row["complete_pinned_file_set"] for row in assets["components"].values())
    file_count = sum(row["verified_files"] for row in assets["components"].values())
    file_bytes = sum(row["verified_bytes"] for row in assets["components"].values())
    ow_path = ROOT / "outputs/mn-runtime-openwam-arx-request-20261008-run4/component-result.json"
    cosmos_path = ROOT / "outputs/mn-runtime-cosmos3-request-20261008-run2/component-result.json"
    hold_path = ROOT / "outputs/mn-runtime-droid-hold-20261008-run3/component-result.json"
    ow, cosmos, hold = [read(path) for path in (ow_path, cosmos_path, hold_path)]
    assert all(record["passed"] for record in (ow, cosmos, hold))
    trial_path = args.trial.resolve() / "receipt.json"
    trial = read(trial_path)
    result_path = args.trial.resolve() / "framework/episodes/ep-0/result.json"
    diag_path = args.trial.resolve() / "framework/episodes/ep-0/diagnostics.internal.json"
    summary_path = args.trial.resolve() / "framework/summary.json"
    result = read(result_path) if result_path.is_file() else None
    diag = read(diag_path) if diag_path.is_file() else None
    stage2 = historical_integrity()
    sealed = {}
    for line in (ROOT / "outputs/mn-extension-integrity-20261007-final/SHA256SUMS").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        if filename in ("docs/contract-v0.2.md", "embodied_eval/native/contracts.py", "embodied_eval/native/interfaces.py", "embodied_eval/native/runner.py"):
            actual = digest(ROOT / filename)
            assert actual == expected
            sealed[filename] = {"expected_sha256": expected, "actual_sha256": actual, "unchanged": True}
    assert len(sealed) == 4
    stage2["public_v02_core_and_runner_unchanged"] = sealed
    save(target / "historical-integrity.json", stage2)
    resource = resources()
    save(target / "resources.json", resource)
    # Final-file checks and partial-cache bytes are deliberately separate.
    dream_manifest_path = ROOT / "outputs/mn-runtime-dreamzero-backbone-20261008-run2/manifest.json"
    manifest = read(dream_manifest_path)
    objects = []
    for item in manifest["objects"]:
        path = MODEL_ROOT / item["relative"]
        row = {"file": item["relative"], "required_bytes": item["size"], "verified": False}
        if path.is_file():
            row.update(verified=True, sha256=verify(path, item))
        objects.append(row)
    chunks = [p for p in (MODEL_ROOT / ".persistent-http-ranges").glob("*/*.chunk") if p.with_suffix(".json").is_file()]
    dream = {"repository": manifest["repository"], "fixed_revision": manifest["revision"], "objects": objects,
        "selected_bytes": sum(row["size"] for row in manifest["objects"]),
        "verified_files": sum(row["verified"] for row in objects),
        "verified_bytes": sum(row["required_bytes"] for row in objects if row["verified"]),
        "old_range_cache_bytes_not_checkpoint_not_rehashed": sum(path.stat().st_size for path in chunks),
        "complete_native_auxiliary_assets": all(row["verified"] for row in objects), "model_loaded": False}
    save(target / "dreamzero-auxiliary-snapshot.json", dream)
    gpu_paths = [ROOT / f"outputs/mn-runtime-gpu-20261008-run{i}.json" for i in (1, 2, 3)]
    gpu_paths = [path for path in gpu_paths if path.is_file()]
    windows = [read(path) for path in gpu_paths]
    for window in windows:
        actual = sum(sum(d["utilization_percent"] for d in r["devices"]) / 8 for r in window["rows"]) / window["samples"]
        assert abs(actual-window["aggregate_mean_percent"]) < 1e-10
    overall = sum(row["aggregate_mean_percent"]*row["samples"] for row in windows)/sum(row["samples"] for row in windows)
    snapshot = {"kind": "runtime_progress_not_stage4_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
        "pinned_asset_files": file_count, "pinned_asset_bytes": file_bytes,
        "new_tests": 14, "total_tests": 320, "baseline_environments_this_wave": len(baselines),
        "cli_exit_codes": [0,0,0,1,2], "openwam_real_inference_and_conversion": ow["passed"],
        "cosmos_real_inference_synthetic_observation": cosmos["passed"], "droid_isolated_hold": hold["passed"],
        "cosmos_droid_trial": trial, "gpu_completed_window_weighted_mean_percent": overall,
        "occupancy_is_not_evaluation_work": True, "contract_frozen": False, "stage4_accepted": False}
    save(target / "snapshot.json", snapshot)
    ledger = read(ROOT / "outputs/mn-runtime-preparation-20261008-run1/coverage-ledger.json")
    ledger["runtime_20261008"] = {"cosmos_droid_receipt": str(trial_path.relative_to(ROOT)),
        "original_R021_R022_reclassified": False, "openwam_arx_inference_and_conversion": str(ow_path.relative_to(ROOT)),
        "dreamzero_native_auxiliary_assets_complete": dream["complete_native_auxiliary_assets"],
        "no_placeholder_adapter_completion": True, "formal_acceptance": False}
    save(target / "coverage-ledger.json", ledger)
    report_path = target / "M+N-runtime-progress-and-evidence.html"
    linked = []
    def link(path, label):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("report evidence missing")
        linked.append(path)
        return '<a href="' + quote(os.path.relpath(path, target), safe="/") + '">' + escape(label) + '</a>'
    sections = []
    def section(identifier, title, body):
        sections.append(f'<section aria-labelledby="{identifier}"><h2 id="{identifier}">{title}</h2>{body}</section>')
    section("outcome", "本轮结果", f'''<div class="callout">本轮补上真实加载、动作转换和DROID执行证据，不是“所有组合已完成验收”。
        M是模型数量，N是仿真环境数量。现有独立注册仍为5个模型家族、10份输入输出配置、5个仿真家族、7份环境协议。
        LingBot-VA暂缓，原有LingBot-VLA保留；阶段4未正式验收，契约未冻结。</div>
        <p>原有306项测试无回归，加14项诊断安全/进程边界测试后共320项。四套本轮环境均通过320项和CLI退出码0/0/0/1/2。
        原先八套环境的306项回执保留，不把旧回执改写成320项。</p>
        <p>六类固定文件共{file_count}个、{file_bytes:,}字节均已核验。这里的“齐了”只指选定文件清单；
        不包括模型原生加载器另需的全部基础组件，也不代表RoboDojo完整任务和素材闭包已经核验。</p>''')
    section("components", "模型和环境到了哪一步", table(["组件", "这轮确认", "还不能声称"], [
        ["OpenWAM ARX", "真实权重生成32条20维原生动作；输入及32条转换数值/类型与固定源码一致", "尚无本轮真实环境执行；未比较两次独立重新推理"],
        ["Cosmos3-Nano-Policy-DROID", "真实权重+额外固定VAE完成一次GPU推理，公共关节动作合法", "单次推理用合成观测，不是任务成功证明"],
        ["DROID sim_evals场景1", "真实三相机、实测关节；隔离后两条保持动作、16物理步、0.1333秒，正常退出", "未覆盖场景2/3；没有源码定义的官方成功评分"],
        ["DreamZero-DROID", f"主权重15/15齐；原生策略导入/GPU矩阵运算通过。额外基础组件{dream['verified_files']}/15已核验", "DreamZero权重加载和真实推理仍未完成"],
        ["RoboDojo stack_bowls", "具名执行边界与选定412个素材文件已核验", "IsaacSim5.1/IsaacLab2.3/CuRobo及全部引用尚未验证；不升级共享驱动"],
        ["π0.5 ARX / RoboTwin", "此前两轮各50动作真实运行证据保留", "两轮第一动作差异原因仍未证明；有限步不等于全任务验收"]
    ]) + '<p>' + link(ow_path, "OpenWAM实际推理与转换") + '；' + link(cosmos_path,"Cosmos实际推理") + '；' + link(hold_path,"DROID隔离后执行") + '</p>')
    trial_text = "整轮正常结束" if trial["passed"] else "整轮未通过，不能只看Runner的完成数"
    detail = table(["项目","实际值"], [
        ["整轮结果", trial_text], ["子进程退出码",trial.get("client_exit_code","未取得")],
        ["Runner错误数",trial.get("framework_errors","未取得")],
        ["动作/模型调用",result["execution_statistics"] if result else "未取得"],
        ["环境原生计数",diag["benchmark"] if diag else "未取得"],
        ["官方成功评分",result["metrics"]["success"] if result else "不可取得"]])
    section("real-trial", "Cosmos与DROID有限步链路", '''<p>模型在GPU4的独立环境中运行原生WebSocket服务，仿真在GPU0的DROID环境中运行。
        使用同一个0.2 Runner和双方独立Adapter，配置只声明相机、关节与夹爪接线；动作上限4条，场景1、seed0。
        这是有限步接入诊断，不是完整任务评测。没有官方成功评分时明确记不可取得，不编造成功率。</p>''' + detail +
        '<p>' + link(trial_path,"整轮监督回执") + ('；'+link(summary_path,"Runner汇总") if summary_path.is_file() else '') +
        ('；'+link(result_path,"逐trial结果") if result_path.is_file() else '') + '</p>')
    section("fixes", "问题原因与修复边界", '''<h3 id="cuda-fix">OpenWAM不能用GPU，不是权重坏了</h3>
        <p>没有固定CUDA分支的初次安装选择了CUDA13的PyTorch，现有驱动不兼容。
        仅在本项目新建的OpenWAM环境中改为源码允许的PyTorch2.10.0/CUDA12.8及torchvision0.25.0；依赖检查和实际推理通过。
        没有升级驱动、修改旧环境或改模型算法。</p>
        <h3 id="cosmos-assets">Cosmos加载器还需要额外文件</h3>
        <p>先前导入失败来自原生CUDA库查找位置，给当前进程指定已安装库路径后解决。
        加载器随后还要读取Wan2.2 VAE；按固定源码指定的发布版本下载2,818,839,170字节并校验SHA256，
        通过源码支持的配置覆盖只改本地资产路径。没有改采样器、归一化或控制规则。</p>
        <h3 id="droid-close">DROID执行成功不代表进程退出成功</h3>
        <p>先补上私有GLU/OpenGL库，解决启动缺库。原生快速关闭会提前终止Python，导致来不及保存汇总；
        关闭快速模式后又观察到Python收尾阶段段错误。失败回执保留。</p>
        <p>当前DROID环境Adapter使用自己的独立Python子进程：初始化、控制器、观测和物理步进都仍执行原生代码。
        环境与应用关闭返回之后，该子进程直接退出，避免再次运行不安全的插件收尾；父进程校验关闭回复和退出码再向Runner返回。
        Runner没有模型/机器人分支，没有新增并行trial、重试、恢复或硬超时协议；动作执行中失联仍标状态未知，不自动重试。</p>
        <h3 id="dream-assets">DreamZero主权重齐了，仍不能直接启动</h3>
        <p>固定原生构造器先加载Wan2.1基础组件再加载DreamZero权重，额外清单共82,243,108,389字节。
        本轮不改原生构造顺序，不跳过组件冒充原生复现。旧分块保留，换用同一固定版本的整文件直连优先下载；
        每个文件完整摘要匹配后才记就绪。缓存分块字节不是完整权重。</p>''' +
        '<p>' + link(ROOT/"outputs/mn-runtime-openwam-arx-cuda128-env-20261008-run1/receipt.json","CUDA环境修复") + '；' +
        link(ROOT/"outputs/mn-runtime-cosmos3-native-vae-20261008-run2/receipt.json","额外VAE摘要") + '；' +
        link(ROOT/"outputs/mn-runtime-droid-hold-20261008-run2/receipt.json","旧关闭段错误") + '；' +
        link(ROOT/"outputs/mn-runtime-cosmos3-droid-bounded-20261008-run1/receipt.json","首次链路退出失败") + '；' +
        link(target/"dreamzero-auxiliary-snapshot.json","DreamZero文件快照") + '</p>')
    section("resources", "资源与GPU约束", '<p>资源是实例当时查询值；共享磁盘剩余量不是个人配额。</p>' +
        table(["资源","查询值"], [["CPU逻辑核",resource["logical_cpus"]], ["主机内存字节",resource["host_memory_bytes"]],
        ["共享盘剩余字节",resource["shared_filesystem_free_bytes"]], ["GPU",resource["gpu_query"]],
        ["完成采样窗口总体平均",f"{overall:.2f}%（按8卡和样本数加权）"]]) +
        '''<p>保持总体平均至少20%的目标。本次没有关闭占卡程序。占卡进程PID及显存单独盘点，
        平均利用率包含占卡负载，不是模型推理用量、任务成绩或效率测量；不能从卡级平均值推算评测单独贡献。</p><p>''' +
        link(target/"resources.json","资源快照") + '；' + '；'.join(link(path,"GPU窗口"+str(i+1)) for i,path in enumerate(gpu_paths)) + '</p>')
    section("gaps", "原49组缺口怎么记", '''<p>原49组仍按本轮范围分为：31组不在直接动作模型范围，4组LingBot-VA由用户暂缓，
        9组环境编号或被测角色未定义，2组Cosmos与原表环境的实际动作/状态结构不满足，3组新增接线仍缺完整运行验证。
        DROID新链路不能自动替代原表含义未定的R021/R022，因此不以本轮运行抵扣它们。</p>
        <p>“结构可组合”指必要相机、状态、动作和坐标含义接得上；“运行资产就绪”指匹配文件与依赖已核实；
        “已真实验证”还要注明是否只有一次推理、有限步链路、全任务以及原生对照。三者不能合并成一个“完成”标签。
        训练覆盖仍是提示，不是固定组合白名单，也不能保证跨机器人任务表现。</p>''' +
        '<p>' + link(target/"coverage-ledger.json","本轮覆盖账本") + '；' +
        link(ROOT/"outputs/mn-extension-20261007-final2/M+N-adapter-extension-and-gaps.html","保留的详细49组报告") + '</p>')
    section("next", "下一轮工作与验收边界", '''<ol><li>DreamZero额外基础组件完整核验后，按固定原生路径做双GPU权重加载、推理及输入/动作对照。</li>
        <li>准备RoboDojo的组件私有运行环境，核验原生控制器、素材引用和实际驱动兼容性；不擅自升级共享驱动。</li>
        <li>对结构满足的组合补足原生输入、动作转换、执行记录对照，再扩展有限步诊断至具名完整任务。</li>
        <li>继续调查π0.5同起点重新推理第一动作差异；不把随机性当作已经证实的原因，也不以任务成功代替映射正确。</li>
        <li>提交阶段4验收证据；未经用户另行确认，不冻结契约或进入下一阶段。</li></ol>
        <p>本轮没有需要重复确认的模型选择或公共核心变更。未来若必须改变公共契约、未定义协议或共享驱动，仍先说明具体影响。</p>''')
    section("evidence", "证据入口", '''<p>报告是生成时的静态快照；下载完成后的新结果需要另出新回执，不回写本页或历史Mac路径。
        SHA256是文件内容指纹，用于核验内容是否变化。</p><ul>''' + ''.join('<li>'+link(path,"软件回归："+name)+'</li>'
        for name,path in zip(baseline_names,baseline_paths)) + '<li>'+link(asset_path,"494个固定文件清单")+'</li><li>' +
        link(target/"historical-integrity.json","历史交接摘要复核")+'</li><li>'+link(target/"snapshot.json","结构化本轮摘要")+'</li></ul>')
    template = args.template.read_text()
    title = "M+N 运行推进与证据报告"
    shell = template[:template.index("<body>")].replace('lang="en"','lang="zh-CN"').replace("Report title",title)
    report_path.write_text(shell+'<body><div class="report-shell"><main class="report-content"><h1 id="report-title">'+title+
        '</h1><p class="lead">核验时间：'+escape(snapshot["at_utc"])+ '；UTC为协调世界时。只记录实际结果，不自动认定阶段验收。</p>'+
        ''.join(sections)+'</main></div></body></html>',encoding="utf-8")
    evidence = {str(path.relative_to(ROOT)):digest(path) for path in sorted(set(linked+[report_path]))}
    save(target/"evidence-sha256.json", evidence)
    print(report_path)


if __name__ == "__main__":
    main()
