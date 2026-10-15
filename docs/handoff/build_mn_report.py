"""Build a source-backed, static M+N architecture/evidence delivery.

Never updates stage state or old reports. Failed comparisons remain failed;
delivery integrity and formal evaluation acceptance are separate fields.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from build_mn_coverage import build, CASES


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def resources():
    disk = shutil.disk_usage(ROOT)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True).stdout
    occupancy = Path("/proc/88718/comm")
    return {"at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "host_memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            "shared_filesystem_total_bytes": disk.total, "shared_filesystem_free_bytes": disk.free,
            "shared_free_space_is_not_personal_quota": True, "gpu_query": gpu,
            "gpu_compute_processes_pid_name_uuid_memory_only": processes,
            "existing_occupancy": {"pid": 88718, "process_name": occupancy.read_text().strip() if occupancy.exists() else None,
                "changed_by_this_work": False, "counted_as_evaluation_work_or_score": False,
                "process_utilization_separately_measured": False},
            "credentials_printed_or_saved": False, "global_proxy_modified": False,
            "new_large_weights_downloaded_in_this_architecture_batch": False,
            "mac_venv_used": False, "framework_git_available": (ROOT / ".git").exists() or (ROOT.parent / ".git").exists()}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.paths = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if tag == "a" and key == "href" and value and not urlsplit(value).scheme and not value.startswith("#"):
                self.paths.append(unquote(urlsplit(value).path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--fresh-differences", type=Path, required=True)
    parser.add_argument("--meter", type=Path, required=True, help="Separate simultaneous meter diagnostic receipt")
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=Path, action="append", required=True)
    parser.add_argument("--replay", action="append", required=True, help="CASE=receipt.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("/root/.codex/skills/html-report/assets/report-template.html"))
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    replay_paths = {v.split("=", 1)[0]: Path(v.split("=", 1)[1]).resolve() for v in args.replay}
    replays = {key: read(path) for key, path in replay_paths.items()}
    if set(replays) != set(CASES):
        raise ValueError("six attempted replay receipts required, failures must not be omitted")
    ledger = build(args.configs, args.trials, replays)
    mapping, architecture, fresh = read(args.mapping), read(args.architecture), read(args.fresh_differences)
    meter = read(args.meter)
    baselines = [read(path) for path in args.baseline]
    gpu = [read(path) for path in args.gpu]
    resource = resources()
    trials = []
    for case in CASES:
        folder = args.trials / case / "framework"
        manifest = read(folder / "manifest.json")
        summary = read(folder / "summary.json")
        episode = manifest["tasks"][0]["episode_id"]
        result = read(folder / "episodes" / episode / "result.json")
        diagnostic = read(folder / "episodes" / episode / "diagnostics.internal.json")
        trials.append({"case": case, "summary": str((folder / "summary.json").resolve()), "summary_sha256": digest(folder / "summary.json"),
            "manifest_sha256": digest(folder / "manifest.json"), "run_status": result["run_status"],
            "official_success": result["metrics"]["success"], "statistics": result["execution_statistics"],
            "native_policy_requests": diagnostic["model"].get("native_requests"),
            "synthetic": summary["synthetic"], "model": manifest["config"]["model"], "benchmark": manifest["config"]["benchmark"]})
    def link(path, label):
        return '<a href="' + escape(os.path.relpath(Path(path).resolve(), target), quote=True) + '">' + escape(label) + '</a>'
    names = {"openwam_libero": "OpenWAM / LIBERO", "openpi_libero": "π0.5 / LIBERO", "openwam_vlabench": "OpenWAM / VLABench",
             "openpi_vlabench": "π0.5 / VLABench", "openwam_robotwin": "OpenWAM / RoboTwin", "lingbot_robotwin": "LingBot-VLA 2.0 / RoboTwin"}
    status_names = {"outside_action_model_scope": "本轮范围外", "action_variant_unselected": "动作型号未确定",
        "action_model_identity_unimplemented": "确切动作模型未实现", "simulator_protocol_unimplemented": "仿真协议未实现",
        "independent_components_implemented": "独立组件已实现", "model_io_profile_missing": "模型输入输出配置缺失"}
    sections = []
    def section(ident, title, body):
        sections.append(f'<section aria-labelledby="{ident}"><h2 id="{ident}">{title}</h2>{body}</section>')
    section("conclusion", "交付结论", f'''<div class="callout"><strong>独立架构已实现，验收结论仍分开提交。</strong>
      新0.2通路把模型与仿真注册拆开，新增一侧不要求逐个修改另一侧。原0.1和历史证据保留。
      结构检查、资产检查、真实任务结果、原生逐步对照不是同一层通过。</div>
      <p>M指模型数量，N指仿真Benchmark数量。M+N表示各方只实现自己的边界，组合通过接线数据完成；不是承诺任意权重都能控制任意机器人。
      本轮只处理直接输出机器人动作的模型与仿真环境，不加“预测画面变成动作”的规划系统。</p>
      <p>原202项加61项新检查共263项，四套环境的测试及旧五组CLI分别保留证据。
      已有六组入口迁移到3个模型家族、6个具名输入输出配置，3个仿真家族、5个环境协议。
      六组转换对照使用历史真实观测和<strong>合成输出夹具</strong>；真实单次和原生重放另列，不能互相替代。</p>
      <p>完整公共契约：{link(ROOT / 'docs/contract-v0.2.md', 'contract-v0.2.md')}。
      覆盖账本：{link(target / 'coverage-ledger.json', '独立组件与原表逐格账本')}。阶段4没有自动通过，契约没有冻结。</p>''')
    section("architecture", "架构怎么接起来", '''<pre>环境原始观测 → 环境Adapter → 公共观测
公共观测 → 显式接线 → 模型Adapter → 模型推理
模型结果 → 模型Adapter → 公共动作
公共动作 → 环境Adapter → 仿真执行与官方评分</pre>''' + table(["部件", "负责", "不承担"], [
        ("Model Adapter", "输入排列、图像处理、状态转换、标准化、模型调用、动作解码及动作缓存", "不接收环境对象，不按当前Benchmark名称选择代码"),
        ("Benchmark Adapter", "初始化、公开状态、控制器、逆运动学、仿真推进、官方评分", "不接收模型配置、权重、端点或模型仓库"),
        ("接线配置", "相机槽、控制目标、源/目标坐标系", "不含组合专用Python函数、评分或控制器代码"),
        ("Runner", "通用调用顺序、校验、预算、计数、证据与错误收尾", "不缩放图像，不解码模型动作，不执行逆运动学")]) + '''
      <p>逆运动学（IK）具体指根据末端位置求关节目标，留在环境执行端。新注册表没有模型×环境白名单；资产、来源和选项各自验证。
      0.1仍走原路径，只有明确版本为0.2的配置才走新路径。新通路复用了环境自身加载/评分组件，未调用旧组合工厂。</p>''')
    section("semantics", "公共格式不是一串数字", table(["公开字段/类型", "人话含义", "明确拒绝的情况"], [
        ("SensorSample / StateSample", "图像或状态，同时写明类型、采集时间和来源", "必要样本缺失、格式不符、超时龄或只有尚不可读取的DataRef"),
        ("measurement / controller_target", "实际测量的状态 / 控制器想达到的目标", "把RoboTwin关节/夹爪驱动目标写成实测位置"),
        ("JointTarget", "哪个关节，用弧度位置、速度或力矩中的哪种模式", "环境只实现位置控制，却收到速度/力矩；漏关节"),
        ("PoseTarget / PoseDelta", "绝对末端目标 / 有明示参考的相对目标", "坐标不明、相对参考错误或执行端未声明支持"),
        ("GripperCommand", "哪个夹爪，米宽度还是0到1开度或夹持力", "把归一化开度当物理米宽度；漏一只夹爪"),
        ("ControllerCommand", "双方明确签名相同的具名原生指令", "原生数组含义或签名不同却当通用动作使用")]) + '''
      <p>四元数xyzw表示x、y、z、w的排列；原生wxyz必须显式转换。公共米/弧度与归一化控制量不能混用。
      坐标转换必须声明且有可用变换；它不会猜不同工具参考点或机械臂关节对应关系。双臂原子执行要求左右两边全部目标，不能默默把另一只手丢掉。</p>
      <p>源欧拉角提示保存模型原生弧度表示，防止四元数往返转换增加舍入；与公共方向严格校验等价后才使用，转换坐标后移除旧提示。
      OpenWAM旧双臂四元数约定冲突仍隔离在具名原生协议，未伪装成跨环境物理位姿。</p>''')
    sections[-1] = sections[-1].replace('</section>', '''<p>LingBot的原生夹爪请求可能略超0到1。RoboTwin先把原始请求与当前目标插值，再逐物理步截断；模型端提前截断会改变执行轨迹。
      因而这两个请求使用带相同声明签名的原生扩展，12个关节仍是公共关节位置。公共夹爪开度依然严格限制0到1，并未放宽单位或范围。
      RoboTwin公开夹爪字段也是控制目标，不是实测开度，已经明确标注；需要真实开度的模型不能假装这个输入满足要求。</p></section>''')
    section("components", "现有独立组件", table(["模型家族", "具名输入输出配置", "动作域"], [
        ("OpenWAM-alpha", "eef10_osc / eef10_absolute / eef20_dual_legacy", "LIBERO归一化控制器 / 物理绝对位姿 / 明确隔离的旧双臂原生数组"),
        ("π0.5 / openpi", "state8_osc / state7_absolute", "归一化控制器 / 物理绝对位姿；各自保留5行原生缓存"),
        ("LingBot-VLA 2.0", "joint14", "12个公共关节位置和2个具名原生夹爪请求，保留完整原生动作段缓存")]) + table(
        ["仿真家族", "环境协议", "初始化/控制边界"], [
        ("LIBERO", "spatial-30settle / spatial-10settle", "分别保留固定源码与30或10次原生静置动作，不由当前模型名决定"),
        ("VLABench", "frozen-track1", "固定select_fruit场景、原生第二次reset、世界位姿→原生逆运动学→关节执行与评分"),
        ("RoboTwin", "sealed-instruction / seeded-instruction", "两种具名源码/指令协议，控制目标与实测末端分开，按实际物理步计时")]) + '''
      <p>输入输出配置是权重自己的规则，环境协议是仿真自己的规则，两者分别选择。训练覆盖未核实只提示，不作接入门槛。
      有公共动作类型不等于所有执行端都实现了它：现有机器人仿真未声明的相对位姿、速度、力矩等仍拒绝，不以理论可支持冒充实际实现。</p>''')
    software_rows = []
    for path, value in zip(args.baseline, baselines):
        cases = {row["case"]: row for row in value["checks"]}
        codes = '/'.join(str(cases[k]["actual_exit_code"]) for k in ("chunk", "tool", "cache", "score-error", "rejected"))
        software_rows.append((Path(value["interpreter"]).parent.parent.name, value["python"], cases["unittest"]["tests_run"], codes, "通过" if value["passed"] else "失败"))
    section("software", "软件与合成架构证据", table(["环境", "Python", "测试项数", "旧CLI退出码", "回执"], software_rows) + f'''
      <p>四种交叉组合使用不同的两个策略类和两个仿真类；仿真类分别用50毫秒与100毫秒控制周期，第二策略输出变化目标。
      再注册第三策略、第三环境，不修改Runner或既有Adapter即可运行。
      这些都是<strong>synthetic=true的合成架构测试</strong>，不是模型能力或Benchmark成绩。
      {link(args.architecture, '四组交叉及两组扩展的保存证据')}。</p>
      <p>新检查覆盖相机与机器人重命名、完整双臂目标、坐标变换、输入缺失、单位/类型拒绝、缓存清理、动态指令、采集时间、变时长计数与错误收尾。
      新增原生执行端合成探针另检查公共双臂位姿、12关节与2夹爪、左右手显式交换、原生类型保留、速度/宽度模式拒绝和完整执行回执；探针没有真实仿真或评分。
      零执行拒绝会停止而非无限重试；评分失败写null/invalid，状态未知时不评分。</p>
      <p>{'；'.join(link(path, '软件回放 ' + str(i + 1)) for i, path in enumerate(args.baseline))}。</p>''')
    trial_rows = []
    for row in trials:
        stats = row["statistics"]
        trial_rows.append((names[row["case"]], "完成" if row["run_status"]=="completed" else row["run_status"],
                           "成功（有效）" if row["official_success"]["value"] and row["official_success"]["valid"] else row["official_success"], stats["env_steps"],
                           stats["physics_steps"] if stats["physics_steps"] is not None else "未单列", f'{stats["sim_time_s"]:.6f}', row["native_policy_requests"]))
    section("real-trials", "六组新真实试验", table(["入口", "运行状态", "官方成功", "动作次数", "物理步", "仿真秒", "策略请求"], trial_rows) + '''
      <p>动作次数、物理步数、仿真秒数分别计账。固定周期路线的物理步未单列时显示None，不捏造计数；表中None就是没有该数值。
      策略请求是客户端请求数，不是服务器内部前向次数；后者无法核实则为null。
      仿真秒数在表中只显示六位小数，原始精度仍保存在结果文件，比较标准没有因此改变。
      本表都是单个固定试点，不是完整suite、泛化成功率或正式阶段4通过。</p>
      <p>真实标记和本地资产摘要不等于服务器提供了可信权重身份回执：所有新真实manifest仍保留server_identity_verified=false。
      每次运行的manifest保存源码SHA256摘要与独立组件配置；源码未冻结时不同试验的源码摘要可能不同，不能假定它们来自同一个Git提交。</p>
      <p>''' + '；'.join(link(row['summary'], names[row['case']]) for row in trials) + '。</p>')
    replay_rows = [(names[case], "通过" if replays[case]["passed"] else "不通过", replays[case].get("replayed_actions"),
                    replays[case].get("compared_snapshots"), replays[case].get("first_divergence"),
                    replays[case].get("error_type", "无启动错误")) for case in CASES]
    section("native-checks", "转换与原生执行逐项对照", f'''<p>六组历史公开观测配合固定合成输出，输入数值、NumPy类型和原生执行参数全部精确一致。
      {link(args.mapping, '6/6转换对照回执')}。这是转换检查，没有调用新模型，也没有产生官方任务成绩。</p>''' + table(
        ["入口", "记录动作原生重放", "重放动作", "比较观测", "首差异动作", "启动错误"], replay_rows) + '''
      <p>记录动作重放不重新推理，直接调用原生环境初始化、动作执行与官方评分，不调用Benchmark Adapter的advance。
      RoboTwin用独立PhysX观察器计数；PhysX是实际物理仿真系统，数值与生产计数器分开核对。
      逐状态、逐像素、采样时刻、终止状态和指标均保留差异。通过不能说明重新推理一致，失败也不能用“最后成功了”覆盖。</p>
      <p>''' + '；'.join(link(replay_paths[case], names[case] + ' 重放') for case in CASES) + '。</p>')
    sections[-1] = sections[-1].replace('</section>', f'''<p>专项检查在同一轮新的OpenWAM/RoboTwin执行中，同时观察生产计数器和底层PhysX。
      该轮执行{meter.get('executed_actions')}次动作、生产与独立物理计数分别为{meter.get('production_physics_steps')}与{meter.get('independent_physx_steps')}，专项结论为{'通过' if meter['passed'] else '不通过'}。
      它只说明该轮计数是否正确；不把这个数替换成历史{meter.get('historical_physics_steps')}，也不撤销跨进程重放的轨迹差异。
      {link(args.meter, '同轮双计数器专项')}。</p>
      <p>转换夹具直接执行固定原生客户端函数或未改写的纯源码片段：Python语法树选择只避开原生CLI的仿真启动，不改公式。
      原生源码文件及摘要随夹具保存；它仍只有固定输入/输出案例，不是全量数值证明。重新推理专项还会使用被检查的投影工具，所以不能独立替代转换夹具与原生执行重放。</p></section>''')
    fresh_rows = [(names[r["case"]], r["first_native_input_values_and_dtypes_exact"], ', '.join(r["different_first_input_keys"]) or '无',
                   r["first_decoded_execution_parameters_exact"], str(r["historical_actions"]) + ' → ' + str(r["new_actions"])) for r in fresh["results"]]
    section("differences", "重新推理的差异怎么理解", table(["入口", "第一份输入相同", "不同输入字段", "首动作参数相同", "旧→新动作数"], fresh_rows) + f'''
      <p>这份诊断没有再发模型请求，只检查实际新旧试验第一份原生输入和解码执行参数。
      若输入不同，先调查相机/状态/指令，不先归因于随机性。若输入相同但首动作不同，差异位于推理或服务状态这一侧；可能涉及采样、数值后端或状态，但没有受控实验就不能确定唯一原因。
      第一条相同也不能证明全程一致。{link(args.fresh_differences, '重新推理差异专项')}。</p>
      <p>本次先有一次LIBERO配置初始化顺序错误，已修复；其后GPU3的π0.5服务在首推理发生CUDA非法访存，尚未执行动作。
      小型Torch运算在GPU3与GPU2均通过，不能据此判定硬件损坏或排除特定JAX工作负载问题。
      换到GPU2模型服务继续验证，失败日志保留，未更改官方评分、随机性开关、占卡或全局代理。</p>''')
    sections[-1] = sections[-1].replace('</section>', '''<p>初次LingBot新通路试验因越界夹爪请求在第一条动作执行前被公共开度校验拒绝，执行动作数为0。
      根据固定源码改为上述具名原生请求后另建目录重跑；失败不删除，也不能称作实际任务失败率。
      两组RoboTwin还因夹爪来源标注修正而另开目录重测，之前的运行保留，不把控制目标当成实测状态。</p></section>''')
    sections[-1] = sections[-1].replace('</section>', '<p>' +
        link(ROOT/'outputs/mn-real-trials-20261007T0957Z-pi05-libero/progress.json','初始化失败记录') + '；' +
        link(ROOT/'outputs/mn-real-trials-20261007T1005Z-retry/openpi_libero/framework/summary.json','GPU3首推理失败') + '；' +
        link(ROOT/'outputs/mn-real-lingbot-robotwin-20261007T1043Z/lingbot_robotwin/framework/summary.json','旧夹爪语义拒绝记录') + '；' +
        link(ROOT/'outputs/mn-diagnostic-case-boundary-20261007T1043Z.json','四组完成后仅停止自有监督进程的记录') + '。</p></section>')
    count_rows = [(status_names[k], v) for k, v in ledger["scope_counts_original_49_only"].items()]
    section("coverage", "原49格缺口重新分类", table(["本轮分类", "原49格中的数量"], count_rows) + f'''
      <p>31格因为没有直接动作接口而在本轮范围外，18格仍需明确动作型号、模型输入输出配置或仿真协议。
      “范围外”不是完成适配，也不是证明模型永远不能参与具身系统。
      原AE4格指LingBot-VA 2.0，已实现的是另一个模型LingBot-VLA 2.0，所以原表55格严格身份下只对应5格已有组件，VLA试点另列。
      AE4这个记号具体指原Excel的AE列第4行；所有原表格子都保留，不回写原表。</p>
      <p>实际6个模型配置与5个环境协议形成30格显式候选接线检查，其中{ledger['structural_grid_compatible']}格结构通过。
      未真实运行的交叉配置仍标为未验证；结构失败列出缺相机、状态、目标或动作域，不靠训练覆盖拒绝，也不自动补另一只手。
      {link(target / 'coverage-ledger.json', '30格结构检查及55格来源账本')}。</p>''')
    section("gap-details", "49格逐条说明", '<p>下表保留原格号、模型名和环境名；不写占位Adapter，也不把缺权重训练覆盖当作结构拒绝理由。</p>' + table(
        ["原格", "模型", "环境", "本轮结论", "人话说明"], [(r["source_cell"], r["original_model"], r["benchmark"], status_names[r["scope_status"]],
        ' '.join(r["plain_language_reasons"])) for r in ledger["rows"] if r["was_one_of_original_49_gaps"]]))
    gpu_mean = sum(value["aggregate_mean_percent"] * value["samples"] for value in gpu) / sum(value["samples"] for value in gpu)
    section("resources", "资源与GPU目标", f'''<p>当前Linux x86_64，{resource['logical_cpus']}个逻辑CPU；主机总内存约{resource['host_memory_bytes'] / 2**30:.1f} GiB。
      GiB表示2的30次方字节。显卡名称和显存以实例nvidia-smi报告为准；共享文件系统可用空间不是个人配额。
      使用Linux轻量环境与已有独立模型/仿真环境，没有沿用Mac虚拟环境，也没有新增大权重下载。
      {link(target / 'resources.json', '实际资源与GPU进程快照')}。</p>
      <p>三个已完成窗口均超过20%目标，最长1800次窗口的八卡采样均值为{gpu[-1]['aggregate_mean_percent']:.2f}%。所有窗口样本合并的算术均值为{gpu_mean:.2f}%。
      部分窗口时间重叠，合并均值不是整轮唯一时间加权平均；每个窗口开始/结束以原始采样时间为准，不把未采样时段当作已测量。
      这个数值包含既有占卡程序；占卡PID、显存和是否改动单独记录，<strong>占卡不计为评测工作量或成绩</strong>。
      没有单独测量其进程利用率，因此不声称去掉占卡后评测本身仍达到20%。</p>
      <p>{'；'.join(link(path, 'GPU采样窗口 ' + str(i + 1)) for i, path in enumerate(args.gpu))}。</p>''')
    section("usage", "怎么使用与下一道门禁", '''<pre>.venv/bin/python -m embodied_eval check --config examples/mn_synthetic_run.json
.venv/bin/python -m embodied_eval run --config examples/mn_synthetic_run.json --output outputs/a-new-mn-run</pre>
      <p>以上只运行合成示例。真实配置预检查可加--verify-assets核验本地摘要；真实run还需相应独立客户端环境和匹配的受监督模型服务。
      新模型或新仿真按各自接口写一个组件并注册，声明实际能力，再写接线数据，不修改Runner或已有另一侧Adapter。</p>
      <p>本轮提交的是新增架构与分层证据。仍待单独确认的是正式阶段4验收和契约冻结；尚未解决的严格复现差异继续保留。
      未实现DataRef读取、硬超时、并发Runner、自动恢复、未知编号解释、额外训练或视频规划器，也不从单次成功推断跨机械臂能力。</p>''')
    title = "EmbodiedEval：M+N独立架构与验证证据"
    body = '<h1 id="report-title">' + title + '</h1><p class="lead">' + escape(datetime.now(timezone.utc).isoformat()) + ' · 仅本轮动作模型与仿真范围 · 未正式验收/冻结</p>' + ''.join(sections)
    template = args.template.read_text()
    report_html = template.replace('<html lang="en">', '<html lang="zh-CN">').replace('<title>Report title</title>', '<title>' + title + '</title>')
    start, end = report_html.index('<main class="report-content">') + len('<main class="report-content">'), report_html.index('</main>')
    report_html = report_html[:start] + body + report_html[end:]
    report_path = target / "M+N-architecture-and-evidence.html"
    report_path.write_text(report_html, encoding="utf-8")
    (target / "coverage-ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
    (target / "resources.json").write_text(json.dumps(resource, ensure_ascii=False, indent=2) + "\n")
    (target / "trial-index.json").write_text(json.dumps(trials, ensure_ascii=False, indent=2) + "\n")
    validation = subprocess.run([sys.executable, "/root/.codex/skills/html-report/scripts/validate_report.py", str(report_path)], capture_output=True, text=True)
    links = Links(); links.feed(report_html)
    missing = [path for path in links.paths if not (target / path).exists()]
    historical = []
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if relative.startswith("outputs/stage2-20261005T063033Z-final/") or relative in ("docs/stage2-acceptance.md", "docs/stage2-evidence.json"):
            historical.append({"path": relative, "unchanged": digest(ROOT / relative) == expected})
    checks = {"report_html": validation.returncode == 0, "report_local_links": not missing,
        "historical_stage2_evidence": len(historical) == 35 and all(row["unchanged"] for row in historical),
        "software_baselines": all(v["passed"] and v["checks"][0]["tests_run"] == 263 for v in baselines),
        "synthetic_architecture": architecture["passed"], "native_mapping_fixtures": mapping["passed"],
        "six_real_trial_results_present": len(trials) == 6, "six_attempted_native_replays_present": len(replays) == 6,
        "simultaneous_meter_diagnostic_present": args.meter.is_file(),
        "original_inventory_preserved": ledger["original_cells"] == 55 and ledger["original_49_gap_rows"] == 49}
    receipt = {"kind": "mn_delivery_integrity_not_formal_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_integrity_passed": all(checks.values()), "checks": checks, "missing_local_links": missing,
        "html_validation_output": validation.stdout + validation.stderr, "visual_browser_check_performed": False,
        "historical_stage2_sha256_files": historical, "stage4_formally_accepted": False, "contract_frozen": False,
        "real_trials_completed": sum(row["run_status"] == "completed" for row in trials),
        "recorded_action_replays_passed": sum(row["passed"] for row in replays.values()),
        "fresh_native_policy_bitwise_acceptance": False, "gpu_sample_weighted_mean_percent": gpu_mean,
        "simultaneous_meter_crosscheck_passed": meter["passed"],
        "report_sha256": digest(report_path)}
    (target / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "historical_stage2_sha256_files"}, ensure_ascii=False))
    return 0 if receipt["delivery_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
