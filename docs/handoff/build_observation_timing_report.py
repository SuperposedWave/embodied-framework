"""Build a static human-readable report from completed observation-time evidence."""
import argparse
from datetime import datetime
import html
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    evidence.relative_to(ROOT / "outputs")
    output.relative_to(evidence)
    if output.exists() or (evidence / "alignment-examples.json").exists():
        raise ValueError("refuse_to_overwrite_report_or_examples")
    baseline = json.loads((evidence / "baseline-final/baseline-check.json").read_text())
    temporal = json.loads((evidence / "temporal-evidence-final/receipt.json").read_text())
    saved = json.loads((evidence / "saved-boundaries-final/receipt.json").read_text())
    assert baseline["passed"] and temporal["passed"] and saved["passed"]
    tests = baseline["checks"][0]["tests_run"]
    focused = next(c["tests"] for c in temporal["checks"] if c["name"] == "temporal-tests")
    bridge = next(c["tests"] for c in temporal["checks"] if c["name"] == "bridge-tests")
    def esc(value):
        return html.escape(str(value))
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + \
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    examples = {}
    for case, index in (("exact", 10), ("approximate", 5)):
        path = evidence / f"temporal-evidence-final/cli-{case}/episodes/ep-0/observation_alignment.internal.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        examples[case] = rows[index]
    bridge_outputs = list((evidence / "bridge-history-exact").glob("backend-*/backend"))
    assert len(bridge_outputs) == 1, "one_explicit_synthetic_bridge_run_required"
    compared = []
    for relative in ("episodes/ep-0/model_input.public.jsonl", "episodes/ep-0/trace.internal.jsonl",
                     "episodes/ep-0/observation_alignment.internal.jsonl", "episodes/ep-0/result.json", "temporal-plan.json"):
        direct = evidence / "temporal-evidence-final/cli-exact" / relative
        bridged = bridge_outputs[0] / relative
        a, b = direct.read_bytes(), bridged.read_bytes()
        assert a == b, (relative, "direct_bridge_bytes_differ")
        compared.append({"relative": relative, "direct": str(direct), "bridge": str(bridged),
                         "sha256": hashlib.sha256(a).hexdigest(), "bytes_equal": True})
    bridge_comparison = {"kind": "synthetic_history_direct_vs_massworld_bytes", "passed": True,
        "new_real_inference": False, "compared": compared}
    time_rows = []
    for case, row in examples.items():
        for sample in row["selections"]["sensors.head"]:
            time_rows.append(("精确" if case == "exact" else "授权近似", round(row["decision_sim_time_s"], 6),
                round(sample["requested_at_s"], 6), round(sample["captured_at_s"], 6), round(sample["error_s"], 6)))
    native_rows = []
    for item in temporal["native_declarations"]:
        first = next(iter(item["sample_timing"].values()), {"kind": "unknown", "period_s": None})
        native_rows.append((item["model"], item["profile"], item["benchmark"], first["kind"],
            "未核实" if item["model_timing"]["native_action_period_s"] is None else item["model_timing"]["native_action_period_s"]))
    at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M（北京时间）")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    body = f'''<h1 id="report-title">图像时间对齐首轮修改与证据</h1>
<p class="lead">{esc(at)} · EmbodiedEval 0.3可选扩展 · 实现提交 <code>{esc(commit[:12])}</code></p>
<section aria-labelledby="summary"><h2 id="summary">结论与范围</h2>
<div class="callout"><strong>已实现：</strong>模型声明“需要什么时间的图像／状态”，公共组件按真实采集时间选择历史，缺数据或不满足时间条件就明确拒绝。<strong>没有实现：</strong>任意模型动作跨频率转换、统一接管模型服务内部缓存、实时异步执行。</div>
<p>旧配置继续保留原生行为，默认不会新增历史缓存、取图或模型推理。本轮是通用机制和软件验证，不代表所有真实权重的采样时间已经核实，更不代表阶段4验收或契约冻结。</p></section>
<section aria-labelledby="changes"><h2 id="changes">这次具体改了什么</h2>
<ul><li>模型按公共路径声明需要0.2秒前、0.1秒前和现在等真实样本；多相机、状态可以分别声明。</li>
<li>环境独立声明等间隔、事件触发或尚未核实的采集能力。相机名义帧率、物理步长和实际公开图片交付频率不再混为一谈。</li>
<li>公共历史缓冲按时间保留数据。同一张采集被重复返回只记一次；不能改时间戳或重复一张图凑出多帧。</li>
<li>默认只接受精确时间匹配；选择附近帧必须显式授权，并记录实际误差。不能使用本次决策时尚未可用的图片。</li>
<li>启动缺历史默认提前拒绝。只有模型配置明确支持“起步先用较短历史”，才提供真实存在的少量帧；中途丢帧不能借此忽略。</li>
<li>历史状态转换使用对应采集时刻的坐标变换，缺少就拒绝，不套用最新机器人姿态。</li>
<li>Runner只增加通用缓冲调用与证据保存；模型编码、原生动作缓存、环境控制器、逆运动学和评分归属不变。</li></ul>
<p>所有时间均为仿真中的秒数；“首次可用时间”是该样本第一次进入公共观测时的仿真时间，不是已实现实时模式或通信延迟补偿。</p></section>
<section aria-labelledby="examples"><h2 id="examples">按时间选图是什么样子</h2>
<p>下面直接读取本轮合成运行日志，不是示意数字。精确例子从每0.02秒返回一张图的环境选取每0.1秒一张的历史。近似例子每0.04秒返回一张图，目标0.1秒没有对应图片，显式授权后选0.08秒，仍保留真实时间。</p>
{table(['方式','决策时间（秒）','希望图片时间（秒）','实际采集时间（秒）','实际减目标（秒）'], time_rows)}
<p><a href="alignment-examples.json">查看上述原始选帧回执</a>。最近帧等距时选择较早的一张；选附近帧不是生成中间画面，也不是提高模型的真实决策频率。</p></section>
<section aria-labelledby="verification"><h2 id="verification">测试与已有行为保护</h2>
{table(['检查','实际结果','证据边界'], [
    ['主测试集', f'{tests}项通过，原515项＋新增{focused}项，零跳过', 'CPU软件／合成测试，不是新真实成绩'],
    ['MassWorld桥接', f'{bridge}项通过，零跳过', '已有桥接行为无回归'],
    ['历史输入穿过MassWorld', '实际CLI合成运行完成，5份关键证据字节一致', '没有新增逐组合桥接逻辑'],
    ['原五组CLI', '/'.join(str(c['actual_exit_code']) for c in baseline['checks'][1:]), '保留0/0/0/1/2退出语义'],
    ['新增历史CLI', '精确0／授权近似0／网格拒绝2／启动拒绝2', '拒绝时不创建运行产物目录'],
    ['独立组合', f"{len(temporal['crosses'])}种交叉组合通过", '两个合成模型×两个合成环境，不是新增逐组合Adapter'],
    ['保存真实数据边界', f"{len(saved['cases'])}条链路，每条{saved['cases'][0]['transitions']}次已保存转换", '无新模型推理、无新环境运行，不关闭原生复现差异']])}
<p>新增测试覆盖时钟网格、实际间隔、因果性、近似授权、历史不足、重复采集、多视角完整性、状态精度、历史坐标变换、缓存频次、容量、跨轮次隔离、冲突声明和极端数值。另保留旧输入张量、采集时间、原生动作数值、精度及取图标记的只读回放检查。</p>
<ul><li><a href="baseline-final/baseline-check.json">主回归与五组CLI回执</a>；<a href="baseline-final/unittest.txt">完整主测试日志</a>。</li>
<li><a href="temporal-evidence-final/receipt.json">时间扩展、桥接、四种组合与原生声明审计</a>；<a href="temporal-evidence-final/temporal-tests.txt">新增测试日志</a>；<a href="temporal-evidence-final/bridge-tests.txt">桥接测试日志</a>。</li>
<li><a href="saved-boundaries-final/receipt.json">保存真实数据的边界与历史文件SHA256保护回执</a>。</li></ul>
<p><a href="bridge-history-comparison.json">同一历史输入配置在直接入口／MassWorld入口的逐文件比较</a>：公共模型输入、执行记录、历史选帧记录、单轮结果和时间方案字节一致。该项仍是合成数据，不冒充真实模型验证。</p>
<p>本轮没有提交启智任务、下载权重或新增GPU推理，没有改变占卡程序，也没有重新测算整个项目GPU平均利用率。</p></section>
<section aria-labelledby="native"><h2 id="native">现有真实配置的时间声明</h2>
<p>下表来自只读注册信息审计，共{len(native_rows)}份已有配置，不是本轮新增真实运行。unknown表示未核实，event表示事件触发；不能把它们当作固定取图频率。所有现有原生模型继续使用原来的当前画面编码，不擅自增加历史。</p>
{table(['模型','具名输入输出配置','环境','静态取图声明','权重原生动作间隔'], native_rows)}
<p>VLABench加载环境后读取真实控制周期，才会声明运行时等间隔采集；静态默认0.002秒没有被拿来当相机周期。RoboTwin仍按原生动作请求取图。DROID／RoboDojo没有在本轮补充核实公开采样周期。</p>
<p>动作间隔未核实仍允许已有原生模式保留历史行为，但不能拿去声称跨频率动作等价。OpenWAM的缓存仍在服务内；新的通用调度器没有接管它。</p></section>
<section aria-labelledby="use"><h2 id="use">如何接入与复现</h2>
<p>模型开发者声明历史偏移、匹配容差、启动规则和来源，并从公共历史编码自己的输入。环境开发者独立声明可提供的真实样本及时间。用户选择是否开启观测对齐和是否授权近似，不填写逐组合相机接线。</p>
<pre><code>.venv/bin/python -m embodied_eval check --config examples/synthetic_history_exact_run.json
.venv/bin/python -m embodied_eval run --config examples/synthetic_history_exact_run.json --output outputs/my-history-fixture
.venv/bin/python docs/handoff/check_observation_timing.py --output outputs/my-history-evidence</code></pre>
<p>这些示例明确为synthetic，即合成测试。输出目录必须是新目录，已有结果不会覆盖。</p>
<p><a href="../../docs/contract-v0.3.md">完整公共契约及新增字段解释</a>；<a href="../../docs/adapter-onboarding.md">Adapter接入指引</a>。</p></section>
<section aria-labelledby="remaining"><h2 id="remaining">剩余缺口与下一步</h2>
<ol><li>继续按固定源码、匹配权重核实真实输入的历史要求及动作时间，不能把数据集帧率、生成视频抽帧率直接当作在线控制频率。</li>
<li>需要完整历史但启动无历史、事件取图却要求固定间隔历史的情况，当前明确拒绝；需要有来源的新初始化／采集协议，不能自动补帧或额外推进机器人。</li>
<li>现有原生编码器不接受新历史后悄悄丢弃；若某个权重确实需要历史，须另建具名配置、实现对应模型编码并做有限步原生对照。</li>
<li>动作时间重采样、统一重新规划、实时异步、硬仿真时间上限和自动恢复均未实现。本轮没有借机修改控制器、评分、训练机器人接入门槛或已有复现结论。</li></ol>
<div class="callout">当前交付的是“公共观测时间机制可用且原生边界受保护”，不是“所有模型的全部时间差异已经解决”。阶段4仍待用户正式验收，契约尚未冻结。</div></section>'''
    template = args.template.read_text(encoding="utf-8")
    start = template.index('<main class="report-content">') + len('<main class="report-content">')
    end = template.index('</main>', start)
    document = (template[:start] + body + template[end:]).replace('lang="en"', 'lang="zh-CN"').replace('<title>Report title</title>', '<title>图像时间对齐首轮修改与证据</title>')
    (evidence / "alignment-examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (evidence / "bridge-history-comparison.json").write_text(json.dumps(bridge_comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.write_text(document, encoding="utf-8")
    print(json.dumps({"report": str(output), "main_tests": tests, "new_tests": focused, "bridge_tests": bridge,
                      "saved_routes": len(saved["cases"]), "new_real_inference": False}))


if __name__ == "__main__":
    main()
