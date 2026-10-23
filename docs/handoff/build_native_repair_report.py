"""Evidence-backed Chinese report; runtime completion and strict equality differ."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"openwam_robotwin": "OpenWAM／RoboTwin：原生旧编码",
          "openwam_dual_physical": "OpenWAM／RoboTwin：公共标准位姿",
          "openwam_libero": "OpenWAM／LIBERO", "openwam_vlabench": "OpenWAM／VLABench",
          "openpi_libero": "π0.5／LIBERO", "openpi_vlabench": "π0.5／VLABench",
          "lingbot_robotwin": "LingBot-VLA／RoboTwin"}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", required=True, help="Explicit ID=case-directory; never auto-select a passing attempt")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    def link(path, label):
        path = Path(path).absolute()
        path.resolve().relative_to(ROOT)
        return '<a href="' + quote(os.path.relpath(path, args.output), safe="/") + '">' + escape(label) + '</a>'
    evidence = ROOT / "outputs/native-repair-evidence-gate-20261023-final1/receipt.json"
    repair = read(ROOT / "outputs/native-env-repair-20261023-run1/receipt.json")
    readiness_path = ROOT / "outputs/native-runtime-readiness-20261023-run2/receipt.json"
    readiness = read(readiness_path)
    gate = read(evidence)
    assert repair["passed"] and readiness["passed"] and gate["passed"]
    rows = []
    for item in args.case:
        name, directory = item.split("=", 1)
        if name not in LABELS or any(row["id"] == name for row in rows):
            raise ValueError("explicit_unique_reviewed_case_required")
        directory = Path(directory).absolute()
        directory.resolve().relative_to(ROOT / "outputs")
        framework = directory / "framework"
        manifest = read(framework / "manifest.json")
        summary = read(framework / "summary.json")
        native = read(directory / "native/native-result.json")
        result = None
        if manifest:
            episode = manifest["config"]["episodes"][0]["episode_id"]
            result = read(framework / "episodes" / episode / "result.json")
        completed = bool(summary and summary["completed"] == 1 and summary["errors"] == 0 and not summary["synthetic"])
        rows.append({"id": name, "directory": str(directory.relative_to(ROOT)), "framework_completed": completed,
            "framework_statistics": result["execution_statistics"] if result else None,
            "native_completed": bool(native and native["completed"]), "strict_equal": bool(native and native["strict_equal"]),
            "initial_observation_exact": native.get("initial_observation_exact") if native else None,
            "first_divergence": native.get("first_divergence") if native else None,
            "native_actions": native.get("native_actions") if native else None,
            "native_physics_steps": native.get("native_physics_steps") if native else None,
            "native_sim_time_s": native.get("native_sim_time_s") if native else None})
    completed_pairs = sum(row["framework_completed"] and row["native_completed"] for row in rows)
    strict = sum(row["strict_equal"] for row in rows)
    action_count = sum((row["framework_statistics"] or {}).get("env_steps", 0) for row in rows) + sum(row["native_actions"] or 0 for row in rows)
    supplemental_path = ROOT / 'outputs/current-wire-diagnostics-20261023/openwam_robotwin-repeat-actual-controller.json'
    supplemental = read(supplemental_path)
    supplementary_actions = 2 * len(supplemental['rows']) if supplemental else 0
    gpu_path = ROOT / 'outputs/native-env-repair-20261023-run1/gpu-utilization.json'
    gpu = read(gpu_path)
    out = ['<h1 id="report-title">原生环境修复与真实对照报告</h1>',
           '<p class="lead">2026年10月23日。本轮保持公共0.3协议、权重、归一化、原生控制器和评分规则；DreamZero推理继续等待后续硬件。</p>',
           '<section><h2 id="conclusion">本轮结论</h2>',
           f'<p>9个失效的Python链接已修复；18套环境现在都能找到解释器。14套选定环境完成依赖检查。下表明确列出的{len(rows)}条链路中，{completed_pairs}条完成框架与独立原生两侧运行，{strict}条满足本轮记录字段的一致性检查，主对照实际执行{action_count}条策略动作。另做一次双侧控制器边界补充对照，执行{supplementary_actions}条；两类合计{action_count + supplementary_actions}条，不重复计数。</p>',
           '<div class="callout">“两边能跑完”和“逐项一致”是两回事；4步诊断也不等于完整任务成功率。阶段4未自动通过，公共契约未冻结。</div></section>',
           '<section><h2 id="environments">环境到底修了什么</h2>',
           '<p>原来的链接指向已经消失的 /root 下Python安装位置。现在指向共享目录中的对应版本；原配置和链接目标均已备份，可回查。3.11.17和3.12.15的Python二进制与10月19日归档指纹一致，已安装依赖的版本元数据没有被升级。SHA256指纹是用于核对文件内容是否一致的一串校验值。</p>',
           '<div class="table-wrap"><table><thead><tr><th>环境</th><th>解释器</th><th>原依赖数量</th><th>本轮边界</th></tr></thead><tbody>']
    for row in gate["repairs"]:
        note = "只修解释器；不启动DreamZero推理" if row["environment"].startswith("dreamzero") else "原安装未完成的空训练脚手架；不当作推理环境" if row["environment"] == "pi05-arx-mn" else "解释器恢复；依赖结果另见检查回执"
        out.append(f'<tr><td>{escape(row["environment"])}</td><td>{row["interpreter"]}</td><td>{row["unchanged_installed_distributions"]}</td><td>{note}</td></tr>')
    out += ['</tbody></table></div>',
            '<p>LIBERO使用CPU版PyTorch读取初始化状态是正常分工；模型推理和仿真渲染用各自的GPU组件，不能因为它的PyTorch不带CUDA就判定环境坏了。DROID／RoboDojo的条款接受仅设置在本项目进程中。首次180秒冷导入超时、诊断要求不合适及未传条款变量的记录均保留；后续检查修正诊断条件，不靠升级依赖抹掉失败。</p>',
            '<p>' + link(evidence, '修复、恢复备份与依赖未升级证据') + '；' + link(readiness_path, '14套依赖检查明细') + '。</p></section>',
            '<section><h2 id="live-comparisons">真实运行与原生对照</h2>',
            '<p>独立原生侧直接调用固定来源的模型客户端、转换函数和仿真组件，不走我们的Adapter执行方法。这不是未经修改的完整官方评测脚本；保存的动作只用于比对，没有拿旧动作替代新推理。四元数、关节、夹爪、相机和计数仍依照各自原生规则处理。</p>',
            '<div class="table-wrap"><table><thead><tr><th>链路</th><th>框架运行</th><th>原生运行</th><th>字段一致</th><th>最早差异</th><th>证据</th></tr></thead><tbody>']
    for row in rows:
        directory = ROOT / row["directory"]
        links = []
        for path, label in ((directory / "framework/summary.json", "框架摘要"), (directory / "native/native-result.json", "原生比较")):
            if path.is_file():
                links.append(link(path, label))
        divergence = "未发现" if row["strict_equal"] else "未完成／未知" if not row["native_completed"] else "起始观测" if row["first_divergence"] == 0 else "动作 " + str(row["first_divergence"])
        out.append(f'<tr><td>{LABELS[row["id"]]}</td><td>{"完成" if row["framework_completed"] else "未完成"}</td><td>{"完成" if row["native_completed"] else "未完成"}</td><td>{"满足本轮检查" if row["strict_equal"] else "不满足／未核实"}</td><td>{divergence}</td><td>{"；".join(links)}</td></tr>')
    out += ['</tbody></table></div><p>每条链路两侧各执行4条动作；起始观测是否一致、后续动作／状态／相机是否一致分别核对。同一对照内固定模型配置、权重、环境版本、任务和随机起点；不同链路不保证采用同一个环境版本，不能拿此表横向比较模型强弱。单臂参考中仿真时钟沿用原检查允许的一亿分之一秒舍入误差；没有为了本轮通过而放宽动作、状态或像素比较。</p>',
            '<p>RoboTwin的一条策略动作会触发原生控制器执行多次物理计算，所以“4条动作”和“363次物理步”不是矛盾。下表的步数是实际物理步数，不是推理次数，也不是把模型一次输出的整个动作段都执行完。</p>',
            '<div class="table-wrap"><table><thead><tr><th>RoboTwin链路</th><th>框架物理步</th><th>原生物理步</th><th>框架／原生仿真秒</th></tr></thead><tbody>']
    for row in rows:
        stats = row['framework_statistics'] or {}
        if stats.get('physics_steps') is not None:
            out.append(f'<tr><td>{LABELS[row["id"]]}</td><td>{stats["physics_steps"]}</td><td>{row["native_physics_steps"]}</td><td>{stats["sim_time_s"]:.9f}／{row["native_sim_time_s"]:.9f}</td></tr>')
    out += ['</tbody></table></div></section>',
            '<section><h2 id="differences">差异要怎么理解</h2>',
            '<h3 id="same-results">本轮一致的链路</h3>',
            '<p>OpenWAM／LIBERO、OpenWAM／VLABench以及标准公共位姿的OpenWAM／RoboTwin，要分别以各自行内结果为准。达到“一致”的含义是：选定起始观测、4条动作及执行后的状态、相机原图和计数满足检查，不是完整任务成功。标准位姿配置仍在模型边界保留权重需要的特殊编码，公共朝向使用明确的xyzw排列，即向量分量x、y、z在前，实数分量w在后；本轮没有再改转换算法。</p>',
            '<h3 id="openpi-differences">π0.5：输入对得上，新动作仍不同</h3>',
            '<p>初始化专项检查中，改变建环境前的随机种子没有改变本次LIBERO起始图像或选定原生数组，所以没有据此修改初始化协议，也不能用它解释历史差异的唯一原因。</p>',
            '<p>本轮π0.5／LIBERO与VLABench的起始图像、状态相同；从保存的公共输入重建模型编码，图像、状态、文字、数据类型逐项相同，但重新推理后的动作仍不同，随后轨迹分叉。该重建检查不冒充此前未保存的实际网络请求。差异已缩小到独立重新推理相关环节，不能凭任务成功率把它关闭，也不能未经验证就归因于随机种子。</p>']
    wire = ROOT / 'outputs/current-wire-diagnostics-20261023/openpi-vlabench-first-input.json'
    if wire.is_file():
        out.append('<p>' + link(wire, 'π0.5／VLABench首个模型输入编码检查') + '。</p>')
    wire_libero = ROOT / 'outputs/current-wire-diagnostics-20261023/openpi-libero-first-input.json'
    if wire_libero.is_file():
        out.append('<p>' + link(wire_libero, 'π0.5／LIBERO首个模型输入编码检查') + '。</p>')
    out += ['<p>π0.5／LIBERO第一条原生控制动作的最大分量差约0.000823；这是控制器原生单位中的分量差，不统一叫米。一步后末端位置相差约0.00000464米。VLABench第一条动作的位置分量差约0.0000826米。这些量级说明差异很早出现，不证明任务最终会失败；也不证明唯一原因就是采样随机性。需进一步做同一请求的多次原生推理与编译／数值执行专项检查。</p>',
            '<h3 id="controller-differences">OpenWAM旧编码：动作相同，物理执行仍不同</h3>',
            '<p>“原生旧编码”是当前0.3下保留的具名权重编码，不是恢复旧0.1／0.2运行通路。主对照的4个实际模型返回动作全部相同，但物理步数是370／367；后续请求随两侧状态分叉而不同。不能假定两个排列错误抵消，也不能用标准位姿那一组的一致结果替它关闭问题。</p>',
            '<p>另做一次被动捕获：不是只计算“应该送什么”，而是记录控制器实际收到的数组。公共动作转实际执行数组逐条一致，两侧实际数组也逐条一致，类型均为32位浮点数；物理步数仍为367／365。证据把问题缩小到同动作下的原生执行重复性，但还不能确定是规划器、仿真、初始化隐藏状态中的哪一项。未擅自换控制器、改容差或使用旧动作代替新推理。</p>',
            '<p>' + link(supplemental_path, '补充对照：实际控制器参数与物理计数') + '。</p>',
            '<h3 id="lingbot-differences">LingBot：实际请求一样，返回的动作段不一样</h3>',
            '<p>这次保存了双方真正发送的首次请求，图像、状态和指令一致；返回的50行、每行14个数的动作段不同，本轮只各执行其中前4条，没有偷偷全部执行。公共动作转实际控制器数组的4条检查全部通过；264／265物理步的差异不能直接当成转换错误。LingBot源码已设置随机种子42，并非简单的“忘记固定种子”；编译、设备数值和采样路径的唯一原因仍未核实。</p>']
    for name in ('openwam_robotwin', 'openwam_dual_physical', 'openwam_libero', 'openwam_vlabench', 'lingbot_robotwin'):
        wire_path = ROOT / f'outputs/current-wire-diagnostics-20261023/{name}-actual-wire.json'
        if wire_path.is_file():
            out.append('<p>' + link(wire_path, LABELS[name] + '：实际模型请求与返回逐项比较') + '。</p>')
    out += ['<p>' + link(ROOT / 'outputs/current-wire-diagnostics-20261023/lingbot_robotwin-actual-controller.json', 'LingBot：公共动作到实际控制器检查') + '。</p></section>',
            '<section><h2 id="videos">对比视频</h2>',
            '<p>视频只利用已有观测被动生成，没有额外调用相机、推理或物理步。画面按真实仿真时间保持上一张已采集观测；视频的30帧每秒是播放编码速度，不是相机采样率。精度检查使用无损原图和数值，不凭压缩视频看起来相似判断。</p>']
    for row in rows:
        video_root = ROOT / 'outputs/current-native-videos-20261023' / row['id']
        if (video_root / 'comparison.mp4').is_file():
            out.append('<p>' + escape(LABELS[row['id']]) + '：' + link(video_root / 'comparison.mp4', '上下对照视频') + '；' + link(video_root / 'framework.mp4', '框架单侧') + '；' + link(video_root / 'native.mp4', '原生单侧') + '；' + link(video_root / 'receipt.json', '视频来源与指纹') + '。</p>')
    out += ['</section><section><h2 id="deployment">部署修复与资源</h2>',
            '<p>OpenWAM和LingBot共享盘冷启动先后达到900秒诊断上限。另建本机磁盘副本后验证：OpenWAM的3份检查点共21个文件、74,505,679,385字节，LingBot一份17个文件、25,518,316,625字节，合计约100GB。逐文件源／副本SHA256一致，包含相同归一化统计和配置。没有下载新大权重，没有改权重、精度、图像尺寸、采样次数或控制器。本机缓存只适用于当前实例，不能拿临时路径直接提交离线平台任务。</p>',
            '<p>' + link(ROOT / 'outputs/native-local-checkpoint-cache-20261023-run1/receipt.json', 'OpenWAM副本文件指纹') + '；' + link(ROOT / 'outputs/native-local-lingbot-cache-20261023-run1/receipt.json', 'LingBot副本文件指纹') + '。启动栈采样不读取局部变量或认证信息，不把单次栈采样当唯一根因证明。</p>',
            '<p>LingBot本机副本首次运行仍在第一个动作请求的300秒等待上限失败；下一次独立诊断把请求等待改为1200秒、协作式墙钟预算改为1800秒，动作上限仍为4，双方成功。成功轮首次请求到响应的文件记录间隔约169秒，小于原300秒；缓存预热也可能影响结果，不能声称“只加超时”就是唯一修复原因。这是诊断配置，不是给Runner新加硬超时、重试或动作重发。</p>',
            '<p>OpenWAM／LIBERO原来指向的上游目录已有未提交兼容改动，导致固定来源核验拒绝。本轮不回滚它，只把当前示例指向已经存在、相同固定revision且工作树干净的副本。原脏目录、历史配置和结果保持；库版本、初始状态、任务、控制器和评分未更换。π0.5的原生服务改为仅监听本机，仍用原固定源码的同一个策略工厂。</p>',
            '<p>启智CLI登录已过期，服务端返回invalid_grant。本轮没有新建平台任务；需要用户执行一次 qz login 才能恢复平台排队，不能通过更换项目或索取Token绕过。本机实际有4张约48GB显存的4090，没有停掉既有占卡程序。</p>',
            '<p>本机SAPIEN渲染设备检查通过，不代表此前H100平台节点缺图形驱动入口的问题已修复。运行日志中的libtpu.so属于JAX探测TPU后端的提示，不是本轮CUDA运行失败；启动时的裸TCP探测也会留下握手断开日志。是否成功按实际动作、结果文件和退出码判断，不按出现warning字样判断。</p>',
            '<p>GPU均值按本轮实际采样窗口报告，目标20%；包含既有占卡负载，不能当作纯评测工作量或模型成绩，也不是整个项目生命周期均值。</p>']
    if gpu:
        out.append(f'<p>实际窗口：{escape(gpu["samples"][0]["at_utc"])} 至 {escape(gpu["samples"][-1]["at_utc"])}，{len(gpu["samples"])}次采样，四卡算术平均{gpu["aggregate_mean_percent"]:.2f}%，{"达到" if gpu["target_met"] else "未达到"}20%目标。' + link(gpu_path, '完整GPU采样') + '。</p>')
    out += ['</section>',
            '<section><h2 id="failed-attempts">失败尝试如何处理</h2>',
            '<p>失败目录全部保留，不把后一次成功写回前一次失败；新轮是人工启动的独立诊断，不是运行器自动重试。run4中π0.5／LIBERO原生检查在动作前因诊断读取版本字段的位置错误退出，修正为observation.context.contract_version并加测试后，用新的原生侧目录对照已完成的框架侧，框架文件指纹保持。run5的OpenWAM与run7的LingBot是共享盘冷启动超时，run8是LingBot首次请求超时，run6前两组有效、后续LIBERO来源检查失败，不把整个run6汇总改成成功。</p>',
            '<p>' + link(ROOT / 'outputs/current-native-live-20261023-run5/receipt.json', 'OpenWAM早期启动失败') + '；' + link(ROOT / 'outputs/current-native-live-20261023-run7/receipt.json', 'LingBot早期启动失败') + '；' + link(ROOT / 'outputs/current-native-live-20261023-run8/receipt.json', 'LingBot请求等待失败') + '。模型原始日志未保存或公开，错误类别与必要源码位置另行留证，避免把认证内容写进报告。</p></section>',
            '<section><h2 id="regression">回归与未完成项</h2>',
            '<p>公共运行仍只有0.3。模型、环境的独立注册与公共空间没有回退；本轮修复核对时75个生产Python文件与上次交付一致，安装包和修复后的原生子进程构造、传输、关闭另有检查。新增诊断代码不等于新公共协议。</p>',
            '<p>' + link(ROOT / 'outputs/native-repair-baseline-20261023-final3/baseline-check.json', 'Python3.13回归') + '；' + link(ROOT / 'outputs/native-repair-baseline-py310-20261023-final3/baseline-check.json', 'Python3.10回归') + '。两套各688项且零跳过，原676项保留，另加12项诊断边界测试；五组CLI退出码保持0/0/0/1/2。桥接34项零跳过另见修复证据门禁；本轮没有新增MassWorld桥接真实运行。</p>',
            '<ul><li>DreamZero真实动作与闭环等待用户后续安排双卡、每卡80GB的H100；目前没有启动或更改量化、采样方案。</li><li>权重训练动作的真实物理时长、部分Benchmark对动作段消费参数的官方权限仍缺完整来源；不从视频帧率或日志猜成已知，不自动重采样。</li><li>有限步对照不代表全任务、多随机起点成功率评测。ARX、Cosmos等未列入本轮新实跑表的链路，不用依赖检查或旧结果冒充新闭环。</li><li>历史差异与本轮结果分别保留；不能因新的一次一致就宣称所有历史差异唯一根因已解决。</li><li>阶段4正式验收和契约冻结仍需用户确认。</li></ul></section>']
    # Repository shell already derives from the html-report skill template;
    # future operators need not install a skill in their own home directory.
    template = (ROOT / 'docs/handoff/report-shell-v03.html').read_text()
    html = template.replace('{{BODY}}', '\n'.join(out)).replace('<title>公共空间 0.3 迁移与验证报告</title>', '<title>原生环境修复与真实对照报告</title>')
    report_path = args.output / 'native-environment-repair-and-comparison.html'
    report_path.write_text(html, encoding='utf-8')
    receipt = {"kind": "native_environment_repair_delivery_report_not_stage_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
               "report": str(report_path.relative_to(ROOT)), "cases": rows, "completed_pairs": completed_pairs,
               "strict_pairs": strict, "primary_actual_policy_actions": action_count,
               "supplementary_actual_policy_actions": supplementary_actions,
               "actual_policy_actions": action_count + supplementary_actions, "formal_acceptance": False,
               "DreamZero_deferred": True}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({"report": str(report_path), "completed_pairs": completed_pairs, "strict_pairs": strict}))


if __name__ == '__main__':
    main()
