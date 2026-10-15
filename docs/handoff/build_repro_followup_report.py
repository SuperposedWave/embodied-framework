"""Seal this bounded execution investigation and render its Chinese report.

Reads actual receipts; never invents strict passes, fresh model inference or
stage acceptance. All new output files are exclusive. No raw process arguments,
environment values, upstream logs or credentials are collected.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess

from probe_mn_components import ROOT, write

PREFIX = "outputs/repro-followup-20261015-"
RUNS = ("native3", "native4", "adapter1", "fixed-native1", "fixed-adapter1", "native-off1", "native-off2")


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def collect():
    baseline_path = "outputs/repro-followup-baseline-20261015-final/baseline-check.json"
    baseline = read(baseline_path)
    cli = [item["actual_exit_code"] for item in baseline["checks"] if "actual_exit_code" in item]
    if not baseline["passed"] or baseline["checks"][0]["tests_run"] != 441 or cli != [0, 0, 0, 1, 2]:
        raise ValueError("full_441_tests_and_five_CLI_gates_required")
    runs = {}
    for name in RUNS:
        path = PREFIX + name
        lifecycle, component = read(path + "/receipt.json"), read(path + "/component-result.json")
        if not lifecycle["completed"] or not component["completed"] or not component["diagnostic_source_unchanged_during_run"]:
            raise ValueError("completed_source_stable_diagnostic_required")
        comparison = component["comparison_with_historical_framework"]
        steps = [len(action["physics"]) for action in component["actions"]] if component["observer_enabled"] else None
        # Off controls report an independent total; do not manufacture missing
        # per-action physical counts from a difference-to-reference field.
        cameras = [{key: value["native_pixel_sha256"] for key, value in row["fields"].items()
                    if "native_pixel_sha256" in value} for row in comparison["rows"]]
        runs[name] = {"component": path + "/component-result.json", "receipt": path + "/receipt.json",
            "mode": component["mode"], "planning_mode": component["planning_mode"],
            "observer_enabled": component["observer_enabled"], "physics_steps_by_action": steps,
            "physics_steps_total": comparison["native_physx_steps"],
            "initial_public_observation_matches_saved_framework": all(value["exact_equal"] for value in comparison["rows"][0]["fields"].values()),
            "first_divergence_from_saved_framework": comparison["first_divergence"],
            "camera_sequence_sha256": cameras,
            "diagnostic_source_unchanged_during_run": component["diagnostic_source_unchanged_during_run"],
            "diagnostic_source_sha256": component["diagnostic_source_sha256_at_start"]}
    comparisons = {name: read(PREFIX + name + ".json") for name in ("native-repeat", "adapter-native", "fixed-comparison")}
    native_repeat = comparisons["native-repeat"]["comparison"]
    fixed = comparisons["fixed-comparison"]["comparison"]
    if not native_repeat["native_planner_output_variation_observed_without_Adapter"]:
        raise ValueError("claimed_native_variation_requires_recorded_witness")
    if (not fixed["controlled_saved_plan_replay_not_fresh_planning"] or not fixed["all_returned_plan_arrays_exact"]
            or not fixed["all_control_operations_exact"] or not fixed["all_recorded_articulation_physics_states_exact"]):
        raise ValueError("fixed_plan_mapping_physics_claim_not_supported")
    fixed_camera_equal = runs["fixed-native1"]["camera_sequence_sha256"] == runs["fixed-adapter1"]["camera_sequence_sha256"]
    off = {"both_have_new_planner_and_physics_observers_disabled": True,
        "fresh_model_inference": False,
        "physics_totals": [runs[name]["physics_steps_total"] for name in ("native-off1", "native-off2")],
        "camera_sequences_equal": runs["native-off1"]["camera_sequence_sha256"] == runs["native-off2"]["camera_sequence_sha256"],
        "observer_off_does_not_mean_uninstrumented_official_evaluator": True}
    history = read("outputs/openwam-dual-pose-mapping-20261015-run2/receipt.json")
    protected = {path: sha(path) == expected for path, expected in history["source_and_history_sha256_before"].items()}
    old_hashes = read("outputs/native-comparison-delivery-20261009-final/evidence-sha256.json")
    old_evidence = {path: sha(ROOT / path.lstrip("/")) == expected for path, expected in old_hashes.items()}
    production = git("ls-files", "embodied_eval", "docs/contract-v0.1.md", "docs/contract-v0.2.md", "SHA256SUMS", "docs/openwam-dual-pose-20261015.md").splitlines()
    unchanged = {}
    for path in production:
        old = subprocess.run(["git", "show", "baseline-20261015:" + path], cwd=ROOT, capture_output=True, check=True).stdout
        unchanged[path] = hashlib.sha256(old).hexdigest() == sha(ROOT / path)
    if not all(protected.values()) or not all(old_evidence.values()) or not all(unchanged.values()):
        raise ValueError("historical_sources_evidence_or_production_components_changed")
    gpu_paths = ("outputs/repro-followup-gpu-20261015-run1.json", "outputs/repro-followup-gpu-20261015-run2.json")
    gpu = [read(path) for path in gpu_paths]
    total_samples = sum(item["samples"] for item in gpu)
    gpu_summary = {"receipts": list(gpu_paths), "samples": total_samples,
        "sample_weighted_eight_gpu_mean_percent": sum(item["samples"] * item["aggregate_mean_percent"] for item in gpu) / total_samples,
        "target_percent": 20, "includes_existing_occupancy": True,
        "occupancy_is_not_evaluation_work_or_score": True, "full_wall_clock_lifetime_coverage": False}
    evidence = [baseline_path, *gpu_paths, *[run[key] for run in runs.values() for key in ("component", "receipt")],
        *[PREFIX + name + ".json" for name in comparisons],
        "outputs/openwam-dual-pose-delivery-20261015-final/receipt.json",
        "outputs/openwam-dual-pose-mapping-20261015-run2/receipt.json",
        "outputs/openwam-dual-pose-videos-20261015-run1/verification.json"]
    return {"kind": "git_baseline_and_bounded_openwam_execution_followup_not_stage_acceptance",
        "at_utc": datetime.now(timezone.utc).isoformat(), "delivery_ready": True,
        "source_commit_at_build": git("rev-parse", "HEAD"), "baseline_commit": git("rev-parse", "baseline-20261015^{commit}"),
        "branch": git("branch", "--show-current"), "git_remote_names": git("remote").splitlines(),
        "historical_commit_history_reconstructed": False, "external_upload_performed": False,
        "software": {"receipt": baseline_path, "tests": 441, "new_synthetic_tests": 25, "five_CLI_exit_codes": cli},
        "runs": runs, "comparisons": comparisons, "observer_off_control": off,
        "fixed_plan_camera_sequences_equal": fixed_camera_equal,
        "fixed_plan_replay_is_diagnostic_not_new_policy_evaluation": True,
        "protected_historical_source_and_inputs": protected, "historical_comparison_evidence_hash_checks": old_evidence,
        "all_production_Adapter_Runner_and_contract_files_unchanged_from_Git_baseline": unchanged,
        "gpu": gpu_summary, "evidence_sha256": {path: sha(ROOT / path) for path in evidence},
        "prototype_attempts_preserved_not_final_source_version": [PREFIX + "native1", PREFIX + "native2"],
        "new_model_inference_or_new_weight_download": False,
        "strict_original_fresh_rollout_trajectory_gate_passed": False,
        "planner_internal_rng_or_kernel_unique_root_identified": False,
        "task_success_from_fixed_plan_replay_used_as_acceptance": False,
        "server_identity_verified": False, "stage4_formally_accepted": False, "contract_frozen": False}


CSS = '''
    :root { color-scheme: light dark; --canvas:#fafbff; --paper:#fff; --text:#242630; --muted:#666b7a; --accent:#354381; --tint:#eef0f8; --line:#dfe2ec; --code:#20232d; }
    * { box-sizing:border-box; }
    html { scroll-behavior:smooth; }
    body { margin:0; background:var(--canvas); color:var(--text); font:16px/1.76 Inter,"PingFang SC","Microsoft YaHei",system-ui,-apple-system,sans-serif; }
    .report-shell { display:block; width:min(100% - 40px,820px); margin:32px auto 64px; }
    main.report-content { min-width:0; padding:42px 48px; background:var(--paper); border:1px solid var(--line); border-radius:8px; counter-reset:section; }
    h1 { margin:0 0 .7em; color:var(--accent); font-size:1.85em; line-height:1.25; scroll-margin-top:24px; }
    h2 { margin:2em 0 .8em; padding-bottom:.35em; border-bottom:1px solid var(--line); font-size:1.42em; counter-increment:section; counter-reset:subsection; scroll-margin-top:24px; }
    h2::before { content:counter(section, cjk-ideographic) "、"; color:var(--accent); }
    h3 { margin:1.5em 0 .6em; font-size:1.18em; counter-increment:subsection; scroll-margin-top:24px; }
    h3::before { content:counter(section) "." counter(subsection) " "; color:var(--accent); }
    h1:target, h2:target, h3:target { border-radius:4px; background:var(--tint); box-shadow:0 0 0 6px var(--tint); }
    p,ul,ol { margin:.8em 0; }
    .lead { color:var(--muted); font-size:1.05em; }
    .callout { margin:1.25em 0; padding:14px 16px; background:var(--tint); border-left:4px solid var(--accent); }
    .table-wrap { overflow-x:auto; }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:10px 12px; border:1px solid var(--line); text-align:left; vertical-align:top; }
    th { background:var(--tint); }
    code { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
    pre { overflow-x:auto; padding:16px; color:#f5f6fa; background:var(--code); border-radius:8px; }
    a { color:var(--accent); text-underline-offset:2px; overflow-wrap:anywhere; }
    @media (max-width:760px) { body { font-size:15px; } .report-shell { display:block; width:100%; margin:0; } main.report-content { width:100%; margin:0; padding:58px 18px 48px; border:0; border-radius:0; } }
    @media (prefers-color-scheme:dark) { :root { --canvas:#14161d; --paper:#1b1e27; --text:#e8eaf0; --muted:#aeb3c2; --accent:#aeb9ff; --tint:#282d42; --line:#383d4d; } }
    @media print { :root { --canvas:#fff; --paper:#fff; --text:#111; --accent:#222; --tint:#f2f2f2; --line:#bbb; } body,.report-shell,main.report-content { background:#fff; } .report-shell { display:block; width:100%; margin:0; } main.report-content { width:100%; padding:0; border:0; } a { color:inherit; } }
'''


def render(data, output):
    def link(path, text):
        target = os.path.relpath(ROOT / path, output)
        return f'<a href="{escape(target, quote=True)}">{escape(text)}</a>'
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">'+escape(v)+'</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'
    runs = data["runs"]
    native = data["comparisons"]["native-repeat"]["comparison"]
    fixed = data["comparisons"]["fixed-comparison"]["comparison"]
    witness = native["same_recorded_MotionGen_input_different_output_witnesses"]
    first = native["rows"][0]
    gpu = data["gpu"]
    ledger = [
        ("项目没有 Git 历史", "已解决", "先建立可追溯基线，再分别提交修复和报告。没有伪造早期提交，也没有上传外网。"),
        ("新公共位姿不能用旧脚本独立回放", "已解决", "补了位姿分支，明确把公共 xyzw 排列转回原生 wxyz；旧关节和旧原生数组分支保留。"),
        ("保存记录的双臂完整性和错误输入检查不足", "已补齐", "拒绝重复目标、缺一只手、错误坐标系、错误夹爪协议、非有限数和有损数值转换。"),
        ("关节实测状态与旧图像混用采集时间", "检查工具已修正", "旧图像可保留上次时间；刚读取的实测关节使用当前仿真时间。没有改环境采样或历史记录。"),
        ("362 与 357 个物理步的差异只定位到执行侧", "定位进一步推进", "现已记录到原生规划器输入相同而输出轨迹不同；不是仅凭最后成功率推测。"),
        ("规划变化以后，执行端是否又引入差异", "完成受控诊断", "给两侧固定同一组原生规划结果，逐条控制命令与已记录的关节物理状态完全一致。固定轨迹不是新推理评测。"),
        ("独立重新规划能否严格逐位重现", "仍未解决", "保留严格不一致结论。没有改种子策略、求解器、控制器或验收容差来让它显示通过。"),
        ("完整任务、多次随机起点与正式验收", "本轮未推进", "本轮复用四条已有真实动作，只做执行诊断，不新增完整任务成功率，也不自动批准阶段。")]
    changes = [
        (".gitignore / README.md", "增加版本管理说明；排除虚拟环境、运行产物、密钥、常见权重和视频。", "Git 管理源码，不等于已经备份模型和实验结果。"),
        ("replay_mn_native_actions.py", "加入公共位姿读取、双臂完整性校验、实测关节读取和独立采集时间。", "让检查工具真正认识新格式，而不是把位姿误当关节角。"),
        ("compare_native_robotwin_closedloop.py", "加强独立证据读取器的协议、维数和重复目标检查。", "不接受看上去像数组但含义不清的记录。"),
        ("robotwin_execution_observer.py", "新增只读规划／控制／物理记录器；异常时恢复临时绑定。", "在边界上记下收到什么、规划出什么、每一步实际变成什么。"),
        ("diagnose_openwam_pose_execution.py", "新增有监督的四动作原生／框架对照，支持不加新增记录器的控制组及固定规划诊断。", "只停止自己创建的子进程，不把原生日志或进程环境写入证据。"),
        ("compare_robotwin_execution_audits.py", "逐项比较实际规划输入、输出、控制指令和物理状态。", "报告第一个分叉点，不用任务成功掩盖映射错误。"),
        ("test_robotwin_execution_audit.py", "新增25项合成测试。", "它们证明检查逻辑和边界行为，不冒充真实模型成绩。"),
        ("build_repro_followup_report.py / stage-state.json", "封存证据摘要、生成本报告并登记最新进展。", "新报告新目录，旧报告和历史路径不回写。")]
    run_rows = []
    labels = {"native3":"原生重新规划 A", "native4":"原生重新规划 B", "adapter1":"框架重新规划", "fixed-native1":"原生固定规划", "fixed-adapter1":"框架固定规划", "native-off1":"原生控制组 A（关闭新增记录器）", "native-off2":"原生控制组 B（关闭新增记录器）"}
    for name, run in runs.items():
        run_rows.append((labels[name], " / ".join(map(str, run["physics_steps_by_action"])) if run["physics_steps_by_action"] else "仅保存独立总数，不补猜逐动作数", str(run["physics_steps_total"]), link(run["component"], "实际记录")))
    evidence_rows = [("软件回归", link(data["software"]["receipt"], "441项与五组CLI回执")),
        ("原生重复规划", link(PREFIX+"native-repeat.json", "规划与物理分叉对照")),
        ("框架／原生重新规划", link(PREFIX+"adapter-native.json", "未强求相同轨迹的实际对照")),
        ("固定规划的两侧对照", link(PREFIX+"fixed-comparison.json", "控制命令与逐步关节状态对照")),
        ("历史公共空间适配交付", link("outputs/openwam-dual-pose-delivery-20261015-final/receipt.json", "上一轮交付回执（保留原样）")),
        ("上轮可读说明", link("docs/openwam-dual-pose-20261015.md", "训练编码、控制器和四元数来源")),
        ("本轮总回执", link(str(output.relative_to(ROOT)/"receipt.json"), "本轮事实与证据摘要")),
        ("本轮源码摘要", link(str(output.relative_to(ROOT)/"source-sha256.json"), "Git已纳入源码的SHA256"))]
    old_video = "outputs/openwam-dual-pose-videos-20261015-run1/"
    video_links = '；'.join(link(old_video+name+".mp4", label) for name,label in (("comparison","上轮两侧对比视频"),("framework","上轮框架视频"),("native","上轮原生视频")))
    t = table
    body = f'''
<h1 id="report-title">EmbodiedEval 本轮修改与复现差异报告</h1>
<p class="lead">2026-10-15｜先建立 Git，再推进 OpenWAM／RoboTwin 剩余执行差异。下列数字由保存的回执生成，不是全任务榜单。</p>
<section aria-labelledby="summary"><h2 id="summary">先说结论</h2>
<div class="callout"><strong>本轮解决了版本管理和回放检查工具的实际缺口，并把差异定位推进到原生规划器内部；没有把“仍不能严格复现”改写成“全部已修好”。</strong></div>
<p>源码现有 Git 基线 <code>{data['baseline_commit'][:7]}</code>，修复与证据工具分批提交。软件回归为 <strong>441 项通过</strong>，包括原有416项和新增25项合成测试；五组命令行退出码仍为 <code>0 / 0 / 0 / 1 / 2</code>。</p>
<p>使用相同的四条已保存真实模型动作，两个不经过 Adapter 的原生进程，第一条动作的左右臂规划输入、起步关节位置与速度都一致，但返回的规划轨迹不同。反过来，把同一组已保存规划结果分别交给原生执行和框架执行，两侧控制指令与记录的每个物理步关节位置、速度完全一致。</p>
<p>人话就是：<strong>“怎么把动作翻译给环境”目前能对齐；“同样的目标每次规划成完全相同的一条路”目前不能保证。</strong>第二件事不能靠把四元数换个顺序来解决，也不能把固定轨迹的诊断通过，算作真实重新规划已经通过。</p>
<p>本轮没有新模型推理，没有下载权重，没有改生产 Adapter、Runner、公共契约、控制器或评分规则；阶段4仍未正式验收，契约仍未冻结。</p></section>
<section aria-labelledby="scope"><h2 id="scope">本轮范围与问题账本</h2>
<p>这次继续处理上一轮公共位姿适配后的残留问题，不扩大到其他模型或全身机器人。Adapter 是模型／环境格式的翻译层；Runner 是只负责调用顺序、计数和留证的通用执行器。两者没有因为调试而混入对方职责。</p>
{t(['问题','本轮状态','说人话的解释'], ledger)}
</section>
<section aria-labelledby="git"><h2 id="git">Git 如何管理这些版本</h2>
<p>Git 仓库建在 <code>{escape(str(ROOT))}</code>，不是含上游源码和虚拟环境的父目录。初始提交纳入297个既有源文件、配置、测试和文档，标签为 <code>baseline-20261015</code>。此前没有提交，所以没有伪造“过去每一天”的历史版本。</p>
<p>当前工作分支为 <code>{escape(data['branch'])}</code>；生成报告时的代码提交为 <code>{data['source_commit_at_build']}</code>。最终交付索引另记录报告封存后的提交与仓库快照，避免把生成报告时尚不存在的提交号写进去。</p>
{t(['进入Git','不进入Git','原因'], [('源码、测试、配置、项目文档','权重、虚拟环境、outputs运行产物、缓存、密钥、视频','代码可以回看和比较；大资产与历史结果通过文件路径及摘要单独管理。')])}
<p>本地没有配置远端，没有推送或公开上传。交付会附 Git bundle：它是一份可复制的仓库历史快照，可以在另一目录恢复源码，但<strong>不包含模型权重、环境和被排除的实验结果</strong>。本地提交和 bundle 都还在本实例磁盘时，不能叫异地备份。</p>
<pre><code>git status --short
git log --oneline --decorate
git diff baseline-20261015
git bundle verify /实际保存位置/EmbodiedEval-repro-followup-20261015.bundle</code></pre>
<p>上面的“实际保存位置”要替换成最终交付中的 bundle 路径。前两条分别查看未提交改动和版本历史，第三条查看相对基线的改动；都不会撤销文件。不建议用强制重置来回看版本，以免误删未提交工作。</p>
</section>
<section aria-labelledby="changes"><h2 id="changes">逐项修改说明</h2>
{t(['文件／组件','改了什么','为什么要改'], changes)}
<h3 id="pose-reader">公共位姿回放为什么要专门补一条分支</h3>
<p>旧工具只认识“环境专属16个数”或“双臂关节14个数”。新的公共动作是两只手各自的明确位姿和夹爪请求，不能再靠数组长度猜。现在读取器先核实两只手和夹爪都在、坐标系是 world（仿真世界坐标），再把每只手的位置与朝向装回环境原生动作。</p>
<p><code>xyzw</code> 指方向四元数的 x、y、z 分量在前，实数分量 w 在后；<code>wxyz</code> 指 w 在前。比如不旋转在公共空间是 <code>[0,0,0,1]</code>，环境原生输入是 <code>[1,0,0,0]</code>。这里只改变表示顺序，不改变物体的实际方向、坐标系或工具参考点。两只手各自的位置、朝向、夹爪及原始数值类型都完整保留。</p>
<p>旧 OpenWAM 权重的特殊训练旋转编码仍由上一轮新增的具名模型配置处理。本轮没有重新训练、换归一化统计、改模型内部55／80维排列，也没有用另一个 ARX 配置的成功替代本链路证据。</p>
<h3 id="timestamps">为什么实测关节和图像的时间可能不同</h3>
<p>最后一步动作执行后，可以继续读取关节位置，却按原本预算与取图节奏不再拍新图。此时关节是“刚刚测到的”，图像是“上次拍到的”。新的检查工具分别比较这两种采集时间，避免把正确的关节时间误判成图像时间不一致。没有为了对齐而多拍图、多走物理步，或把驱动目标冒充实测关节。</p>
</section>
<section aria-labelledby="diagnosis"><h2 id="diagnosis">复现差异到底出在哪里</h2>
<h3 id="starting-point">原来已经知道什么</h3>
<p>上一轮四动作新推理对照中，两侧四条原始模型回复完全一样，公共动作转回环境的字节和数值类型也一样；但框架走了362个物理步，原生参考走了357个物理步，第一次观测分叉发生在第二条动作之后。这只能说明问题已在执行链路，不能进一步断言是规划器还是物理引擎。</p>
<h3 id="native-repeat">本轮记录到了更早的分叉点</h3>
<p>这次不重新请求模型，只使用保存的四条动作，独立重建环境与相同种子搜索历史，记录规划器实际收到的关节和目标、返回轨迹、发给机器人的控制命令，以及每个已完成物理步的关节位置和速度。记录器不抽随机数，不额外调用规划、取图、渲染或物理步；但读取 CUDA 结果会同步计算，可能影响耗时，因此观测开启与关闭控制组分别记录。</p>
<p>固定版本的 CuRobo MotionGen 是环境原生运动规划器：它把“手要到哪里”变成“每个物理步各关节走到哪里”。两次原生重复运行在第一条动作上，共记录到 {len(witness)} 个“已记录输入相同、输出不同”的臂侧证据。实际起步的完整关节位置和速度也相同。</p>
{t(['第一动作的比较项','两次原生运行的实际结果'], [('起始关节、目标位置、目标朝向','已记录数值和类型完全相同'),('Python／NumPy／Torch全局随机状态摘要','完全相同；并未穷尽规划器自有采样器和内部状态'),('左臂规划位置数组最大差值',f"{first['plans'][0]['position'].get('maximum_absolute_difference')} 弧度"),('右臂规划数组长度','60个关节路点与50个关节路点'),('第一物理步之前的控制指令','已经不同，不是先出现物理漂移才改变控制'),('是否确定某个内部随机采样器或GPU内核是唯一原因','没有；边界证据到规划器输入／输出，内部根因仍需单独调查')])}
<p>这里“弧度”是关节角度单位。右臂路径长度不同，并不一定让这一条总物理步数不同，因为两臂按各自路径进度交错执行，较长的一侧可能决定总时长；后续动作又会从已经不同的关节状态出发，差异继续传下去。</p>
<p>不能把这里写成“π或OpenWAM推理随机性造成”：本轮根本没有调用模型。也不能写成“只要设置同一个随机种子就必定一致”：已记录的全局随机状态相同，原生规划输出仍不同。可能涉及规划器内部采样状态、并行数值或求解过程，但这些只是待验证方向，不是已经证明的唯一根因。</p>
<p>{link(PREFIX+'native-repeat.json','查看原生重复的逐项比较')}；{link(PREFIX+'adapter-native.json','查看框架与原生重新规划的逐项比较')}。</p>
</section>
<section aria-labelledby="controlled"><h2 id="controlled">固定规划的诊断意味着什么</h2>
<p>为进一步把“规划差异”和“执行差异”分开，取原生运行 A 已返回的实际规划数组，分别交给原生 <code>take_action</code> 和框架 Benchmark 执行路径。此处明确跳过这四次动作的新规划求解，原生控制循环、夹爪插值、物理步和评分函数仍保留。它是受控实验，不是生产配置，不是新的策略评测。</p>
{t(['检查项','实际结论'], [('两侧返回给控制循环的规划数组','完全一致'),('控制命令与调用顺序','完全一致'),('每个已记录物理步的完整关节位置和速度','完全一致'),('两侧消耗的物理步数',str(runs['fixed-native1']['physics_steps_total'])+' 与 '+str(runs['fixed-adapter1']['physics_steps_total'])),('五个公开观测时点的原始相机字节摘要','一致' if data['fixed_plan_camera_sequences_equal'] else '仍有差异，见回执'),('是否证明重新规划严格复现通过','没有；不能用固定规划替代真实求解'),('是否捕获了物理引擎所有隐藏状态','没有；本轮比较的是明确记录的关节位置、速度、控制和相机')])}
<p>人话就是：同一张“每一步怎么走”的路线表交给两条执行路径，记录到的机器人运动能逐步对齐。这给“框架执行端没有额外改坏这组动作”提供了实证，但没有替原生规划器保证每次都画出同一张路线表。</p>
<p>{link(PREFIX+'fixed-comparison.json','固定规划完整对照证据')}。报告中的“完全一致”指对应数组数值、数值类型、命令顺序或字节摘要的严格比较，没有新增容差。</p>
</section>
<section aria-labelledby="runs"><h2 id="runs">实际运行与测试清单</h2>
<p>下面每行均为真实仿真中的四条保存动作。不是七次新的模型推理，也不是七个完整任务成功率样本。“物理步”是仿真引擎实际完成的一小步；“动作”是外部提交的一条目标，一条动作可能包含很多物理步。</p>
{t(['运行','四条动作各自的物理步数','物理步总数','证据'], run_rows)}
<p>关闭新增记录器的两个原生控制组，物理步总数为 {data['observer_off_control']['physics_totals'][0]} 和 {data['observer_off_control']['physics_totals'][1]}，整段相机字节摘要{'相同' if data['observer_off_control']['camera_sequences_equal'] else '不同'}。仍保留原来的独立 PhysX 计数器，因此不是“未经任何修改的官方整套评测程序”。</p>
<p>早期两次诊断 <code>native1</code> 和 <code>native2</code> 也保留，没有覆盖或删除；它们位于检查工具修订期间，不作为最终同版本对照。最终表内运行均记录了诊断源码摘要，并检查执行过程中源码未发生变化。</p>
<h3 id="regression">轻量回归没有退步</h3>
<p>修改前416项通过；修改后441项通过，无跳过。新增25项均为合成测试，覆盖公共位姿／旧原生／关节分支、双臂缺失和重复、错误坐标与协议、精度损失、非有限值、记录器不增加物理步或消耗随机数、异常恢复绑定、固定轨迹标记、错误证据拒绝，以及图像／关节采集时间分离。</p>
<p>五组 CLI（命令行程序）分别为 chunk（连续动作段）、tool（工具调用）、cache（缓存）、score-error（评分出错）、rejected（预检查拒绝），退出码为 <code>0 / 0 / 0 / 1 / 2</code>。退出码0表示正常完成，1表示预期运行错误，2表示预检查拒绝；后两组不是“测试没修好”。</p>
<p>{link(data['software']['receipt'],'查看实际441项及五组CLI回执')}。没有声称所有真实模型运行环境都各自重跑了441项。</p>
</section>
<section aria-labelledby="integrity"><h2 id="integrity">资产、历史和GPU</h2>
<p>沿用 OpenWAM 源码 <code>48bd67b89d489b14d03b8d92bc66e65d306df32e</code>、RoboTwin 源码 <code>0aeea2d669c0f8516f4d5785f0aa33ba812c14b4</code> 的既有资产。本轮每个仿真诊断重新核验环境来源与配置，不重新加载或下载24.8GB模型权重；权重及 normalization（匹配权重的标准化统计）的既有摘要仍见上轮交付，不冒充本轮重新推理。</p>
<p>本轮与 Git 基线比较的生产 Adapter、Runner、公共契约等 {len(data['all_production_Adapter_Runner_and_contract_files_unchanged_from_Git_baseline'])} 个文件全部不变；上轮登记的 {len(data['protected_historical_source_and_inputs'])} 项只读来源／历史输入摘要全部一致；10月9日比较材料中 {len(data['historical_comparison_evidence_hash_checks'])} 个既有文件重新核验摘要一致。SHA256 是文件内容摘要，用来发现文件是否被改动，不等于模型服务能证明自己加载了哪份权重。</p>
<p>机器仍是既有8张 GPU，驱动查询显示 GeForce RTX 4090、每卡49140MiB显存；本轮真实诊断使用原来的模拟器环境与 GPU 0，不改变系统驱动。集中采样 {gpu['samples']} 次，全八卡采样加权平均利用率为 <strong>{gpu['sample_weighted_eight_gpu_mean_percent']:.2f}%</strong>，高于20%目标。包含既有占卡负载，没有把占卡算成评测工作或成绩；采样窗口之间仍有间隔，不声称覆盖本轮每一秒。</p>
<p>没有改全局代理，没有安装新环境或下载大文件，没有保存 API 密钥、完整环境变量、外部进程参数或原生日志。原生异常仅保存允许的错误类型与安全位置。</p>
</section>
<section aria-labelledby="remaining"><h2 id="remaining">剩余缺口与下一步</h2>
{t(['事项','现在能说什么','推进边界'], [('公共位姿映射','上一轮保存113条观测、111条动作的映射证据保留；本轮回放工具补齐。','不能拿结构接通承诺跨机械臂任务成功。'),('原生规划严格可重复','确认存在相同已记录输入得到不同规划轨迹的现象。','如果要改求解器确定性设置、重置内部种子、固定路径或更换规划器，须先说明协议影响并确认。'),('验收如何看待自然的原生差异','仍维持原严格失败，不自动豁免。','可以提交“映射严格一致＋原生重复变动单列”的验收建议，但不能本轮擅自降低门槛。'),('完整任务与多起点统计','本轮是四动作执行诊断，不是完整成功率评测。','仍需按固定任务／预算／种子，独立做真实模型与原生参考运行。'),('模型服务的权重身份','仍没有可信的服务器加载身份回执。','本地文件摘要不能替代服务端身份验证。'),('DreamZero与LingBot-VA','保持后置，不计为本轮完成。','DreamZero的双H100 80GB待你后续对接；LingBot-VA继续暂缓，不用VLA抵扣。'),('公共契约与正式阶段验收','本轮无公共字段或协议变化。','阶段4未自动通过、契约未冻结。')])}
<p>本轮没有要求你为已确认范围内的检查再做细碎决定。真正需要你确认的是以后是否改变规划确定性／验收协议，或安排后置资源；在此之前，现有严格门槛与历史失败结论都保留。</p>
</section>
<section aria-labelledby="evidence"><h2 id="evidence">证据与报告访问</h2>
{t(['材料','入口'], evidence_rows)}
<p>以下是<strong>上一轮</strong>有限步新推理保存的视频，供理解历史分叉，未重新配音或改写成这次固定规划实验的视频：{video_links}。本轮新增的是规划数组、控制命令和逐物理步记录，没有为拍视频增加采样或渲染。</p>
<p>报告预览服务只监听实例本机 <code>127.0.0.1</code>，只提供本报告明确链接的证据和旧视频，不开放工作区目录。HTTP 检查另存回执；远程桌面要通过平台端口转发或 SSH 隧道才能访问。没有实际桌面转发证据前，不声称你的浏览器已打开。</p>
</section>'''
    return '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EmbodiedEval 本轮修改与复现差异报告</title><style>'+CSS+'</style></head><body><div class="report-shell"><main class="report-content">'+body+'</main></div></body></html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    data = collect()
    write(output / "receipt.json", data)
    source_files = git("ls-files").splitlines()
    write(output / "source-sha256.json", {path: sha(ROOT / path) for path in source_files if path != "docs/stage-state.json"})
    with (output / "EmbodiedEval-repro-followup-20261015.html").open("x", encoding="utf-8") as stream:
        stream.write(render(data, output))
    print(str(output / "EmbodiedEval-repro-followup-20261015.html"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
