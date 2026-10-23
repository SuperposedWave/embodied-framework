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
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    def link(path, label):
        path = Path(path).absolute()
        path.resolve().relative_to(ROOT)
        return '<a href="' + quote(os.path.relpath(path, args.output), safe="/") + '">' + escape(label) + '</a>'
    evidence = ROOT / "outputs/native-repair-evidence-gate-20261023-run1/receipt.json"
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
            "native_actions": native.get("native_actions") if native else None})
    completed_pairs = sum(row["framework_completed"] and row["native_completed"] for row in rows)
    strict = sum(row["strict_equal"] for row in rows)
    action_count = sum((row["framework_statistics"] or {}).get("env_steps", 0) for row in rows) + sum(row["native_actions"] or 0 for row in rows)
    out = ['<h1 id="report-title">原生环境修复与真实对照报告</h1>',
           '<p class="lead">2026年10月23日。本轮保持公共0.3协议、权重、归一化、原生控制器和评分规则；DreamZero推理继续等待后续硬件。</p>',
           '<section><h2 id="conclusion">本轮结论</h2>',
           f'<p>9个失效的Python链接已修复；18套环境现在都能找到解释器。14套选定环境完成依赖检查。下表明确列出的{len(rows)}条链路中，{completed_pairs}条完成框架与独立原生两侧运行，{strict}条满足本轮记录字段的一致性检查，合计实际执行{action_count}条策略动作。</p>',
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
    out += ['</tbody></table></div><p>单臂参考中仿真时钟沿用原检查允许的一亿分之一秒舍入误差；没有为了本轮通过而放宽动作、状态或像素比较。RoboTwin的动作次数、真实物理步数与仿真秒数分别检查，动作相同不自动代表物理执行一样。</p></section>',
            '<section><h2 id="differences">差异要怎么理解</h2>',
            '<p>初始化专项检查中，改变建环境前的随机种子没有改变本次LIBERO起始图像或选定原生数组，所以没有据此修改初始化协议，也不能用它解释历史差异的唯一原因。</p>',
            '<p>本轮π0.5／LIBERO与VLABench的起始图像、状态相同；从保存的公共输入重建模型编码，图像、状态、文字、数据类型逐项相同，但重新推理后的动作仍不同，随后轨迹分叉。该重建检查不冒充此前未保存的实际网络请求。差异已缩小到独立重新推理相关环节，不能凭任务成功率把它关闭，也不能未经验证就归因于随机种子。</p>']
    wire = ROOT / 'outputs/current-wire-diagnostics-20261023/openpi-vlabench-first-input.json'
    if wire.is_file():
        out.append('<p>' + link(wire, 'π0.5／VLABench首个模型输入编码检查') + '。</p>')
    wire_libero = ROOT / 'outputs/current-wire-diagnostics-20261023/openpi-libero-first-input.json'
    if wire_libero.is_file():
        out.append('<p>' + link(wire_libero, 'π0.5／LIBERO首个模型输入编码检查') + '。</p>')
    out += ['<p>新启动的原生服务与实际请求、返回动作尽量分别留证。LingBot源码已设置随机种子42，并非简单的“忘记固定种子”；若动作仍不同，还需要核对真实请求、返回段、编译与GPU数值执行，不能先猜结论。</p></section>',
            '<section><h2 id="videos">对比视频</h2>',
            '<p>视频只利用已有观测被动生成，没有额外调用相机、推理或物理步。画面按真实仿真时间保持上一张已采集观测；视频的30帧每秒是播放编码速度，不是相机采样率。精度检查使用无损原图和数值，不凭压缩视频看起来相似判断。</p>']
    for row in rows:
        video_root = ROOT / 'outputs/current-native-videos-20261023' / row['id']
        if (video_root / 'comparison.mp4').is_file():
            out.append('<p>' + escape(LABELS[row['id']]) + '：' + link(video_root / 'comparison.mp4', '上下对照视频') + '；' + link(video_root / 'receipt.json', '视频来源与指纹') + '。</p>')
    out += ['</section><section><h2 id="deployment">部署修复与资源</h2>',
            '<p>OpenWAM共享盘启动曾在参数复制阶段达到原有900秒诊断上限。另建本机磁盘副本后重新验证；3份检查点共21个文件、约74.5GB，逐文件核对源和副本SHA256一致，包含相同归一化统计和配置。没有改权重、精度、图像尺寸、采样次数或控制器。本机缓存位于临时目录，只适用于当前实例，不能拿这些临时路径直接提交离线平台任务。</p>',
            '<p>' + link(ROOT / 'outputs/native-local-checkpoint-cache-20261023-run1/receipt.json', '本机副本完整文件指纹') + '。启动栈采样不读取局部变量或认证信息，不把单次栈采样当唯一根因证明。</p>',
            '<p>启智CLI登录已过期，服务端返回invalid_grant。本轮没有新建平台任务；需要用户执行一次 qz login 才能恢复平台排队，不能通过更换项目或索取Token绕过。本机实际有4张约48GB显存的4090，没有停掉既有占卡程序。</p>',
            '<p>本机SAPIEN渲染设备检查通过，不代表此前H100平台节点缺图形驱动入口的问题已修复。运行日志中的libtpu.so属于JAX探测TPU后端的提示，不是本轮CUDA运行失败；启动时的裸TCP探测也会留下握手断开日志。是否成功按实际动作、结果文件和退出码判断，不按出现warning字样判断。</p>',
            '<p>GPU均值按本轮实际采样窗口报告，目标20%；包含既有占卡负载，不能当作纯评测工作量或模型成绩，也不是整个项目生命周期均值。</p></section>',
            '<section><h2 id="regression">回归与未完成项</h2>',
            '<p>公共运行仍只有0.3。模型、环境的独立注册与公共空间没有回退；本轮修复核对时75个生产Python文件与上次交付一致，安装包和修复后的原生子进程构造、传输、关闭另有检查。新增诊断代码不等于新公共协议。</p>',
            '<p>' + link(ROOT / 'outputs/native-repair-baseline-20261023-final2/baseline-check.json', 'Python3.13回归') + '；' + link(ROOT / 'outputs/native-repair-baseline-py310-20261023-final2/baseline-check.json', 'Python3.10回归') + '。两套各688项，原676项保留，另加12项诊断边界测试；五组CLI退出码保持0/0/0/1/2。桥接34项另见修复证据门禁。</p>',
            '<ul><li>DreamZero真实动作与闭环等待用户后续安排双卡、每卡80GB的H100；目前没有启动或更改量化、采样方案。</li><li>权重训练动作的真实物理时长、部分Benchmark对动作段消费参数的官方权限仍缺完整来源；不从视频帧率或日志猜成已知，不自动重采样。</li><li>有限步对照不代表全任务、多随机起点成功率评测。ARX、Cosmos等未列入本轮新实跑表的链路，不用依赖检查或旧结果冒充新闭环。</li><li>历史差异与本轮结果分别保留；不能因新的一次一致就宣称所有历史差异唯一根因已解决。</li><li>阶段4正式验收和契约冻结仍需用户确认。</li></ul></section>']
    # Repository shell already derives from the html-report skill template;
    # future operators need not install a skill in their own home directory.
    template = (ROOT / 'docs/handoff/report-shell-v03.html').read_text()
    html = template.replace('{{BODY}}', '\n'.join(out)).replace('<title>公共空间 0.3 迁移与验证报告</title>', '<title>原生环境修复与真实对照报告</title>')
    report_path = args.output / 'native-environment-repair-and-comparison.html'
    report_path.write_text(html, encoding='utf-8')
    receipt = {"kind": "native_environment_repair_delivery_report_not_stage_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
               "report": str(report_path.relative_to(ROOT)), "cases": rows, "completed_pairs": completed_pairs,
               "strict_pairs": strict, "actual_policy_actions": action_count, "formal_acceptance": False,
               "DreamZero_deferred": True}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({"report": str(report_path), "completed_pairs": completed_pairs, "strict_pairs": strict}))


if __name__ == '__main__':
    main()
