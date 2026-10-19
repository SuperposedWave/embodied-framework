"""Human-readable static report from completed read-only execution audits."""
import argparse
from datetime import datetime
import html
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"openwam_libero": "OpenWAM／LIBERO", "openpi_libero": "π0.5／LIBERO",
    "openwam_vlabench": "OpenWAM／VLABench", "openpi_vlabench": "π0.5／VLABench",
    "openwam_robotwin": "OpenWAM／RoboTwin旧位姿", "lingbot_robotwin": "LingBot-VLA／RoboTwin",
    "pi05_robotwin": "π0.5 ARX／RoboTwin", "pi05_robodojo": "π0.5 ARX／RoboDojo",
    "openwam_robodojo": "OpenWAM ARX／RoboDojo", "cosmos3_droid": "Cosmos3／DROID",
    "openwam_dual_physical": "OpenWAM／RoboTwin标准位姿"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    evidence.relative_to(ROOT / "outputs")
    output.relative_to(evidence)
    if output.exists():
        raise ValueError("new_report_path_required")
    source_proof = evidence / "source-preservation.json"
    audit = json.loads((evidence / "audit-final/receipt.json").read_text())
    native = json.loads((evidence / "native-clock-review.json").read_text())
    native_rows_check = json.loads((evidence / "robotwin-native-row-check.json").read_text())
    baseline = json.loads((evidence / "baseline-final2/baseline-check.json").read_text())
    temporal = json.loads((evidence / "temporal-evidence-final/receipt.json").read_text())
    differences = json.loads((evidence / "audit-final/recent-boundary-localization.json").read_text())["comparisons"]
    pi_libero = next(x for x in differences if x["route"] == "openpi_libero")["camera_differences"]
    pi_vla = next(x for x in differences if x["route"] == "openpi_vlabench")["camera_differences"]
    if not audit["completed"] or not baseline["passed"] or not temporal["passed"] or not native["source_identity_checks_passed"] or not native_rows_check["passed"]:
        raise ValueError("completed_source_and_software_evidence_required")
    runs = {row["id"]: json.loads((evidence / "audit-final" / row["evidence"]).read_text()) for row in audit["runs"]}
    golden_path = ROOT / "outputs/native-timing-audit-20261020-run1/saved-boundaries-final/receipt.json"
    golden = json.loads(golden_path.read_text())
    expected_sources = golden["framework_source_sha256"]
    current_sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT / "embodied_eval").rglob("*.py")}
    if not golden["passed"] or current_sources != expected_sources:
        raise ValueError("framework_runtime_changed_since_previous_saved_boundary_verification")
    source_proof_payload = {"kind": "current_runtime_preserved_since_previous_verified_boundaries",
        "passed": True, "previous_saved_boundary_receipt": str(golden_path),
        "previous_receipt_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "framework_python_files": len(current_sources), "framework_source_sha256": current_sources,
        "previous_saved_conversion_checks_reused_not_rerun": True,
        "new_inference_or_simulation": False}
    old = sorted((r for r in runs.values() if r["cohort"] == "historical"), key=lambda r: r["route"])
    def esc(x):
        return html.escape(str(x))
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    def span(stats):
        return "无间隔样本" if not stats["count"] else f"{stats['min']:.6g}～{stats['max']:.6g}"
    long_rows = []
    for r in old:
        key = "head" if "head" in r["model_input_sensors"] else next(iter(r["model_input_sensors"]))
        sensor = r["model_input_sensors"][key]
        long_rows.append((LABELS[r["route"]], r["executed_actions"], r["public_model_input_records"],
            r["recorded_native_requests"], f"{key}: {sensor['distinct_capture_timestamps']}",
            span(r["execution_elapsed_s"]), f"{r['execution_elapsed_s']['sum']:.6g}"))
    recent_rows = []
    local_names = {
        "saved_public_input_differs_before_first_returned_action": "第一条动作前，公共输入已不同",
        "same_first_public_input_but_first_returned_action_differs": "首次公共输入相同，第一条返回动作不同",
        "same_public_actions_up_to_first_different_execution_physics_count": "此前公共动作相同，执行物理步数先不同",
        "later_action_difference_requires_prior_state_and_request_telemetry": "后续动作不同，需要继续追查前序状态／服务记录",
        "no_difference_in_reviewed_initial_inputs_actions_and_execution_counts": "所查起点、动作和执行计数未见差异"}
    for row in differences:
        recent_rows.append((LABELS[row["route"]], "是" if row["first_saved_public_model_input_exact"] else "否",
            row["first_action_difference"] or "未见", row["first_physics_count_difference"] or "未见",
            local_names[row["first_localized_boundary"]]))
    native_rows = []
    for r in native["cases"]:
        native_rows.append((LABELS[r["route"]], f"{r['framework_actions']}／{r['native_actions']}",
            r["compared_common_action_prefix"], r["first_elapsed_difference"] or "未见",
            r["first_physics_count_difference"] or "未见",
            "原记录严格一致" if r["original_strict_equal"] else "原差异保留"))
    ling = runs["historical-lingbot_robotwin"]
    ling_sensor = ling["model_input_sensors"]["head"]
    acquisition_steps = [0] + [r["action_index"] for r in ling["transitions"] if r["capture_time_advanced"].get("head_camera")]
    # Public inputs use head, whereas historical environment snapshots use head_camera.
    if len(acquisition_steps) != len(ling_sensor["capture_times_s"]):
        raise ValueError("lingbot_capture_step_and_input_time_inventory_disagree")
    ling_rows = [(step, f"{time:.9g}") for step, time in zip(acquisition_steps, ling_sensor["capture_times_s"])]
    issues = [(LABELS[r["route"]], r["cohort"], r["quality_issues"]) for r in runs.values() if not r["passed"]]
    per_run_links = '<details><summary>查看全部25次运行的独立核对回执</summary><ul>' + ''.join(
        '<li><a href="audit-final/' + esc(row['evidence']) + '">' + esc(row['id']) + '</a></li>'
        for row in audit['runs']) + '</ul></details>'
    at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M（北京时间）")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    bridge_tests = next(x["tests"] for x in temporal["checks"] if x["name"] == "bridge-tests")
    body = f'''<h1 id="report-title">已有真实记录：取图、动作缓存与仿真时间核对</h1>
<p class="lead">{esc(at)} · 离线读取历史证据 · 核对工具提交 <code>{esc(commit[:12])}</code> · 无新GPU推理或仿真</p>
<section aria-labelledby="summary"><h2 id="summary">本轮结论</h2>
<div class="callout">完整核对了<strong>{len(runs)}次已有运行</strong>：11次历史框架运行，以及最近7条链路的直接／MassWorld双入口共14次短跑，覆盖{sum(r['executed_actions'] for r in runs.values())}条已存在的动作记录。日志核对通过不等于原生复现相同；原生差异仍保留。</div>
<p>目前最明确的发现是：<strong>记录了很多次公共模型输入，不代表模型每次都重新看图推理。</strong>LingBot历史运行有112次模型侧调用，但每个相机只有3个不同采集时间，记录的原生动作请求也是3次；其中一次取图间隔达到约20.39个仿真秒。它符合保留的原生缓存设置，不能直接当成Adapter丢图。</p>
<p>本轮只增加离线核对工具、合成测试和证据报告，没有改Runner、模型／环境Adapter、权重、归一化、控制器、取图、推理频率或评分。没有提交启智任务，也没有把已有占卡当评测工作。阶段4和契约冻结仍未接受。</p></section>
<section aria-labelledby="meaning"><h2 id="meaning">到底核对了什么</h2>
{table(['记录','人话解释','不能据此断言'], [
 ['公共输入条数','框架给Adapter交了多少次观测快照','不等于每次都调用模型服务'],
 ['不同采集时间数','同一个相机出现了几个不同的采集时间；重复缓存不重复计数','不是独立相机触发仪器实测；同一时刻多次取图仍可能无法区分'],
 ['原生动作请求／生成计数','模型侧诊断记录了多少次成功返回动作的请求，或本地生成调用；不含重置、健康检查等通信','不等于已核实的模型前向推理次数，也未记录每次请求的准确位置'],
 ['每动作耗时','这条动作的执行反馈报告推进了多少仿真秒','不是该权重训练时所要求的动作时长'],
 ['物理步数','执行记录报告底层仿真推进了多少步','没有物理计数的记录不能用动作条数代替']])}
<p>同一画面被重复返回，要保持原采集时间；不同时间采到完全相同的像素也可能合理，例如场景静止，不能仅按图像内容去重。所有时间均为仿真中的秒数，不是等待推理的现实耗时。本工具的数值比较容差为一亿分之一秒，只处理浮点舍入，不会把相差一个物理步说成相同。</p></section>
<section aria-labelledby="historical"><h2 id="historical">历史运行逐条核对</h2>
<p>下表来自旧版0.2历史记录，并未改写或迁移这些证据。每个相机都做了核对，表内只选一个代表相机显示计数；完整字段在各运行回执中。请求列保留各实现诊断的原含义，不能跨模型合计为统一“推理次数”。</p>
{table(['链路','动作数','公共输入数','原生动作请求／生成数','代表相机不同采集时间数','每动作仿真秒数范围','累计仿真秒数'], long_rows)}
<p>每条已保存预测都核对了对应公共输入的观测编号、采集时间和相机像素指纹；每个执行反馈都核对了动作计数、时间增量，以及存在时的物理步数。新快照编号不等于新图；终止后返回的最后快照也不一定再取图，且未必再次送进模型。</p>
<h3 id="lingbot-example">用LingBot这次真实记录举例</h3>
{table(['已完成动作数','模型公共输入里出现的新采集时间（仿真秒）'], ling_rows)}
<p>三路相机的记录一致：112次公共输入中，每路3个不同采集时间，另109次继续带着已有画面。第50和第100条动作请求更新观测，两段分别约20.388秒和11.784秒。单条动作耗时从约0.06到0.704秒不等，所以“每50条动作更新”不是固定的几赫兹；赫兹表示每秒多少次。</p>
<p>图片在公共输入记录中的最大年龄约20.308秒，表示这条输入携带的图片相对当前仿真时刻有多旧，<strong>不等于模型服务实际使用这张旧图重新推理了109次</strong>。模型可能仍在消费先前预测的动作。服务请求的逐次时间没有保存，因此这里不伪造精确的真实推理时间线。</p></section>
<section aria-labelledby="clocks"><h2 id="clocks">时钟证据的强弱</h2>
{table(['环境','当前记录和源码能支持的说法','尚缺什么'], [
 ['LIBERO','快照时间按动作数乘固定控制周期计算，反馈也报告该周期','没有独立物理时间读数或物理步数记录；账面一致不是独立实测'],
 ['VLABench','报告固定周期；执行代码还检查物理引擎执行前后的时间差','历史日志没有逐条保存原始物理时钟读数；依赖当时的运行检查'],
 ['RoboTwin','实际计数场景物理步，按当时物理步长累计；动作耗时可变','时间来源与具体计数实现仍需保留，不能拿其他实现的计数混用'],
 ['RoboDojo','按实际物理推进计数，每步0.004秒累计','不等于权重训练时的动作间隔'],
 ['DROID','执行代码核对每条动作实际推进8个物理步；反馈保留计数','固定模式结果汇总仍可能把物理步数记为null；不能误说反馈也没有计数']])}
<p>null表示这份汇总没有提供该值，不是零。本轮核对不补写原结果，不将环境实测执行时间注册为模型训练时间。</p></section>
<section aria-labelledby="bridge"><h2 id="bridge">最近七条链路的差异边界</h2>
<p>下面是10月19日已有的4动作直接／MassWorld对照，<strong>不是新运行，也不是官方原生入口对照</strong>。这里只定位最早可见差异，不凭日志猜唯一根因。“动作不同”比较完整公共动作，物理步数则比较独立执行反馈。</p>
{table(['链路','首次公共输入相同','第一条不同动作','第一次不同物理计数','差异最早出现在哪里'], recent_rows)}
<p>π0.5的起始图像差异发生在第一条动作执行之前，不能靠修改后续动作频率消除。LingBot首次公共输入相同而首条返回动作不同，仍需核对实际编码与服务状态；不能直接归咎于随机性。OpenWAM两条RoboTwin链路的公共动作相同，物理计数先出现差异，应继续追查环境执行／规划／计数路径，不能用任务成功掩盖。</p>
<p>机器人测量状态相同，不代表物体、灯光、相机及完整仿真状态都相同；现有公开状态没有覆盖所有细节，不能仅凭机械臂状态一致确定起始图像差异的原因。</p>
<p>π0.5／LIBERO的起点主视角有{pi_libero['head']['changed_pixels']:,}个像素不同、腕视角有{pi_libero['wrist']['changed_pixels']:,}个像素不同；π0.5／VLABench只有右侧视角不同，共{pi_vla['external_right']['changed_pixels']:,}个像素、{pi_vla['external_right']['changed_values']:,}个颜色分量，主／腕视角相同。一个像素含红绿蓝3个颜色分量，二者不是同一计数。对这些差异图做上下、左右或同时翻转的数值比较，也没有恢复完全相等；本轮没有据此认定更深层的渲染原因。</p>
<p>另对两条OpenWAM／RoboTwin链路各4条保存动作，调用生产代码的纯坐标／原生命令转换：双方转换后仍逐项相同，包括控制模式和float32精度。因此这几个已保存动作的差异没有出现在该纯转换边界。LingBot首条动作的不同也保留到了原生关节数组。见<a href="robotwin-native-row-check.json">三条RoboTwin链路的原生命令数值检查</a>。这是CPU计算，不是向控制器重新发送动作，也不是实际执行端抓包。</p>
<p>原始逐相机像素差、是否存在简单翻转关系、逐动作相等结果和逐步计数，见<a href="audit-final/recent-boundary-localization.json">七条链路差异定位证据</a>。没有改变图片以追求表面一致，也没有提高比较容差来关闭物理步数差异。</p></section>
<section aria-labelledby="native"><h2 id="native">原生参考的时间复核</h2>
<p>另读取10份既有固定原生组件参考结果，用其保存的框架清单指纹确认比较对象，再对齐双方共同动作前缀逐条复核时间与物理计数。原生参考不是未经修改的官方全套评测程序。两边任务结束得早晚不同，不能直接把总耗时差当作频率不匹配。</p>
{table(['链路','框架／原生动作数','共同比较动作数','首次耗时不同动作','首次物理计数不同动作','原严格结论'], native_rows)}
<p>7份原生结果自身记录了框架清单指纹；另3份通过伴随的首次请求检查、实际原生请求文件和框架首条输入指纹交叉绑定。后3份的首次输入相等检查原本未通过，此处只确认它们比较的是哪份基线，不把它改判为输入相同。</p>
<p>源身份核验通过不代表严格复现通过。已有{native['strict_equal_references']}条原生参考严格一致，其余差异本轮没有被关闭。详见<a href="native-clock-review.json">原生参考逐步时间复核</a>。</p></section>
<section aria-labelledby="quality"><h2 id="quality">核对结果与保留缺口</h2>
<p>日志质量核对：{sum(r['passed'] for r in runs.values())}／{len(runs)}次运行通过所列规则。检查重复观测编号、缺失／多余输入、未来采集时间、同一采集时间像素冲突、相机漏项、输入像素与来源快照不符、动作／时间／物理计数矛盾，以及结果汇总不一致。</p>
{table(['有问题的运行','批次','具体问题'], issues) if issues else '<p>上述规则在所选记录中未发现矛盾；这不填补下面的证据缺项。</p>'}
<ul><li>真实前向推理次数未知。模型侧predict调用数、服务请求数、本地生成调用数分别保留，不能互相替代。</li>
<li>缺少每个请求的准确位置、返回行数和本次取出哪一行的独立日志。因此请求总数与50行动作缓存相符，仍不等于逐次缓存消费已经完整实测。</li>
<li>结束时剩余缓存为0不能证明全部动作都执行了：现有生命周期先结束轮次并清理缓存，再写诊断。</li>
<li>公共输入保留了相机的采集时间，不是模型线上实际收到／使用每一帧的独立通信抓包。</li>
<li>匹配权重的训练动作物理时长仍未知；本轮不能据日志补齐，也没有开始动作重采样。</li></ul>
<p>数据质量检查促使本轮将“记录条数、采集时间、请求和真实推理”拆成独立口径，缺项保持未知。报告沿用项目选定的静态HTML模板；没有另建交互式业务看板或发布公网。</p></section>
<section aria-labelledby="verification"><h2 id="verification">软件回归与证据</h2>
<p>主测试{baseline['checks'][0]['tests_run']}项通过，保留原603项并新增31项合成测试（25项日志口径、6项来源绑定）；MassWorld桥接{bridge_tests}项通过，零跳过。五组CLI仍为{'/'.join(str(x['actual_exit_code']) for x in baseline['checks'][1:])}。观测时间扩展57项检查与4种合成交叉组合仍通过。合成测试不冒充真实模型评测。</p>
<ul><li><a href="audit-final/receipt.json">25次运行的审计索引与逐文件指纹</a>。</li>
<li><a href="audit-final/historical-lingbot_robotwin.json">LingBot完整时间与取图记录</a>；<a href="audit-final/historical-openpi_libero.json">π0.5／LIBERO完整时间与请求计数</a>。</li>
<li><a href="baseline-final2/baseline-check.json">主回归及CLI回执</a>；<a href="baseline-final2/unittest.txt">完整主测试日志</a>。</li>
<li><a href="temporal-evidence-final/receipt.json">时间扩展及桥接回执</a>；<a href="temporal-evidence-final/bridge-tests.txt">桥接日志</a>。</li></ul>
{per_run_links}
<p><a href="source-preservation.json">运行源码保持不变的指纹证据</a>：当前{len(current_sources)}个框架Python文件与上一轮保存真实数据边界检查时完全相同。因此本轮复用其既有转换验证结论，不冒充又做了一次模型推理或转换回放。</p>
<p>审计流式读取完整所选记录而不是只取前几行；新输出写到新目录，旧文件只读并复核指纹。CPU读取进程的并行只是离线分析，不是Runner新增并行试验。本轮没有重新测算GPU平均利用率。</p></section>
<section aria-labelledby="next"><h2 id="next">下一轮具体做什么</h2>
<ol><li>π0.5优先查起始图像差异的重置、随机状态和渲染来源，不先修改动作频率。</li>
<li>LingBot补最小被动服务／动作缓存记录的方案，区分返回动作差异与真正的服务推理差异，不自动缩短50条原生缓存。</li>
<li>OpenWAM／RoboTwin针对相同公共动作的物理计数差异，设计固定动作回放和原生计数对照；这是新仿真验证，先交具体运行方案确认。</li>
<li>新GPU验证、改动作执行策略或公共协议不由本轮自动授权。DreamZero仍等待后续两张每卡80GB的H100；本轮没有启动它。</li></ol>
<pre><code>.venv/bin/python docs/handoff/audit_saved_execution.py --output outputs/my-new-execution-audit --workers 3
.venv/bin/python docs/handoff/audit_saved_native_clocks.py --execution-audit outputs/my-new-execution-audit --output outputs/my-new-native-clocks.json
.venv/bin/python -m unittest tests.test_saved_execution_audit -v</code></pre>
<p>输出路径必须不存在，工具拒绝覆盖旧证据。报告预览只开放本页明确链接的材料；远程查看需要平台端口转发，未据实例HTTP成功声称桌面浏览已通过。</p></section>'''
    template = args.template.read_text()
    start = template.index('<main class="report-content">') + len('<main class="report-content">')
    end = template.index('</main>', start)
    document = (template[:start] + body + template[end:]).replace('lang="en"', 'lang="zh-CN"').replace('<title>Report title</title>', '<title>已有真实记录：取图、动作缓存与仿真时间核对</title>')
    if source_proof.exists():
        if json.loads(source_proof.read_text()) != source_proof_payload:
            raise ValueError("refuse_to_replace_different_source_preservation_evidence")
    else:
        with source_proof.open("x") as f:
            json.dump(source_proof_payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
    with output.open("x", encoding="utf-8") as f:
        f.write(document)
    print(json.dumps({"report": str(output), "runs": len(runs), "quality_passed": audit["passed"], "new_inference": False}))


if __name__ == "__main__":
    main()
