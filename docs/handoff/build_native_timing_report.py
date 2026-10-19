"""Render source-audit and regression evidence using the HTML report template."""
import argparse
from datetime import datetime
import html
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
    if output.exists():
        raise ValueError("refuse_to_overwrite_report")
    audit = json.loads((evidence / "source-audit-final/receipt.json").read_text())
    baseline = json.loads((evidence / "baseline-final/baseline-check.json").read_text())
    saved = json.loads((evidence / "saved-boundaries-final/receipt.json").read_text())
    temporal = json.loads((evidence / "temporal-evidence-final/receipt.json").read_text())
    replay = json.loads((evidence / "source-audit-final/dreamzero-native-assembly-synthetic.json").read_text())
    assert all(x["passed"] for x in (audit, baseline, saved, temporal, replay))
    def esc(x):
        return html.escape(str(x))
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    profiles = {}
    for case in audit["cases"]:
        profiles.setdefault((case["family"], case["profile"]), case)
    rows = []
    for (family, profile), case in profiles.items():
        timing = case["timing_declaration"]
        fact = timing["provenance"]["source_audit"]
        if family == "openwam":
            training = "33行训练窗口；32行动作；未来视频每4行取一帧，动作不因此抽稀"
            cache = "本地Adapter管缓存" if timing["replanning_owner"] == "adapter" else "原生服务管缓存；每次请求取一条不等于每次重新推理"
        elif family == "openpi":
            training = "ARX缺匹配训练采样清单" if profile == "arx_joint14" else "示例转换器10帧/秒；尚未证明为匹配权重的完整训练清单"
            cache = f"Adapter每次取{fact['execution_rows_per_request']}条执行，再请求"
        elif family == "lingbot_vla":
            training = "50行动作；未来视频教师默认1帧/秒，可被数据批次覆盖；不是在线1帧/秒"
            cache = "当前具名服务返回50条；Adapter执行缓存"
        elif family == "cosmos3":
            training = "固定配方15帧/秒；服务默认生成条件15、32行动作；权重checkpoint.json为空"
            cache = f"Adapter默认取{fact['execution_rows_per_request']}条；可用具名运行配置缩短"
        else:
            training = "训练视频索引0到24、动作索引0到23；评估视频索引仅0"
            cache = f"Adapter默认取{fact['execution_rows_per_request']}条；服务另管4帧和因果隐状态"
        rows.append((family, profile, "每个相机当前1张", training, cache, "尚未核实"))
    source_count = len({(x["repo"], x["revision"], x["relative"]) for c in audit["cases"] for x in c["sources"]})
    metadata_count = len({x["path"] for c in audit["cases"] for x in c["small_weight_metadata"]})
    tests = baseline["checks"][0]["tests_run"]
    bridge = next(x["tests"] for x in temporal["checks"] if x["name"] == "bridge-tests")
    pad_rows = []
    for call in replay["calls"]:
        ids = call["views"]["video.exterior_image_1_left"]["pixel_ids"]
        pad_rows.append((call["RPC_call"], ", ".join(str(x) for x in ids), len(ids), call["distinct_input_calls_in_external_0"]))
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M（北京时间）")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    body = f'''<h1 id="report-title">模型时间规则：来源审计与修改证据</h1>
<p class="lead">{esc(now)} · EmbodiedEval 0.3 · 实现提交 <code>{esc(commit[:12])}</code> · 无新GPU推理</p>
<section aria-labelledby="summary"><h2 id="summary">本轮结论</h2>
<div class="callout">已把<strong>13份现有配置、11种权重输入输出规格</strong>的时间规则说明接到预检查中，并修正OpenWAM ARX的缓存归属声明。现有在线接口接收当前画面；训练未来视频、动作缓存和服务内部历史另记，不再混成一个“采样率”。<strong>尚不能据此声称任意模型的动作已跨频率对齐。</strong></div>
<p>FPS表示每秒帧数；“动作行”指模型一次返回的动作数组中的一条命令；“缓存”指这批动作尚未执行完前继续取出下一条，而非重新运行模型。本轮的来源审计标记<code>native-timing-source-audit@1</code>只是第一版审计说明，不是新公共协议或执行模式。</p>
<p>本轮没有增加公共历史输入、没有改模型权重、归一化、原生动作数值、取图节奏、控制器或官方评分，也没有新建启智任务。阶段4和契约冻结仍未接受。</p></section>
<section aria-labelledby="implementation"><h2 id="implementation">具体修改</h2>
<ol><li>新增模型侧时间审计说明，按具名输入输出规格选择，不按Benchmark名称分支。预检查会同时展示当前画面接口、训练监督、服务历史、执行缓存和未核实的动作时长。</li>
<li>修正OpenWAM ARX的说明：该配置由本地Adapter调用生成引擎并保存动作，而不是由网络服务保存动作。普通OpenWAM网络入口仍声明服务管理缓存。这里只修说明，没有迁移缓存或改变调用。</li>
<li>新增离线来源核验：对实际源码与固定Git版本逐文件比较，核查相关小型权重配置，对已有资产清单逐项核对元数据锁。来源漂移、标记缺失或元数据不符会失败。</li>
<li>用原生DreamZero图像组装函数做CPU合成回放，实际验证其内部补帧行为；不导入原生模型，不加载权重。</li>
<li>补接入指引及31项回归测试。不让未知配置自动继承已审计声明，不把“当前图像接口”写成“模型没有任何内部历史”。</li></ol>
<p>涉及的代码为<code>native/timing_profiles.py</code>、<code>v03/models.py</code>及离线审计工具；Runner、Benchmark Adapter和公共物理字段未改。详细接入规则见<a href="../../docs/adapter-onboarding.md">Adapter接入指引</a>。</p></section>
<section aria-labelledby="profiles"><h2 id="profiles">逐配置时间规则</h2>
<p>下表合并相同规格，保留11种不同输入输出规格。13份配置中，ARX两个模型分别用于两个环境，因此配置数更多。表中训练规则来自本地固定来源；“当前1张”表示外部送入接口的单张画面，不否认服务内部另有缓存。</p>
{table(['模型','具名规格','外部送图','训练／生成时间数字','动作缓存','每条动作秒数'], rows)}
<p>完整13份配置各自的固定revision、权重引用、源码SHA256、相关行号和小配置锁结果见<a href="source-audit-final/receipt.json">原始来源审计回执</a>。revision是固定源码提交编号；SHA256是核对文件内容是否变化的指纹。</p></section>
<section aria-labelledby="interpretation"><h2 id="interpretation">几个数字为什么不能直接用</h2>
<h3 id="openwam-stride">OpenWAM：每4帧抽图，不是4倍动作间隔</h3>
<p>匹配目录中的配置使用33行训练窗口，视频每4行选一帧；读取器保留32行动作，并不跟着每4行只留一条。在线推理则用当前合成视图和当前状态生成未来动作。仅有“4”无法推出相邻动作间隔几秒，照它拉长动作时长会改变控制。</p>
<h3 id="openpi-converter">π0.5：示例10帧每秒，不等于当前权重必须10赫兹执行</h3>
<p>LIBERO转换器明确是最小示例，逐行搬运已过滤数据，并写入每秒10帧的元数据；它没有证明发布权重用的正是这份转换过程。VLABench示例转换器的数据集名称是预训练集，匹配训练配置用的却是微调集，不能把前者的频率直接搬给后者。训练读取器按数据集FPS选未来动作行，但本机还缺对应训练数据的完整采样清单。赫兹在这里指每秒执行多少次；它与视频文件编码帧率也不是一回事。</p>
<h3 id="lingbot-teacher">LingBot：1帧每秒是训练教师，不是每秒只看一次环境</h3>
<p>匹配权重配置中的1.0用于未来视频监督；读取器给教师提供当前图及未来图，并可按真实数据集频率和50行动作跨度重算该值。在线接口仍收三个相机的当前图，不能因为有三张图就把它们当成三个历史时刻。</p>
<h3 id="cosmos-conditioning">Cosmos：15是生成条件，尚不是动作标定证书</h3>
<p>固定配方按每秒15个采样点取数据，服务默认也把15送进生成条件；但匹配目录中的<code>checkpoint.json</code>为空，不能证明当前发布权重完整沿用了这份训练配置。服务的默认<code>history_length=1</code>还会被当前状态行占用，不代表额外一张过去图。原来的运行默认值保留，不擅自把1/15秒登记为已核实的物理动作时长。</p></section>
<section aria-labelledby="dreamzero"><h2 id="dreamzero">DreamZero内部补帧实测</h2>
<p>原生图像组装函数首轮使用1帧，后续选最近4帧，不足时向前重复最早图像。下表用10、20、30等像素编号代表依次送入的不同<strong>合成</strong>图片；它们不是时间戳。RPC表示一次向模型服务发起的请求，不等于一次模型动作。</p>
{table(['请求序号','原生函数拼出的外部相机0图片编号','输出帧数','来自多少次不同输入'], pad_rows)}
<p>第二次请求已经输出4帧，但其中只有2次不同输入。三路相机均检查了原生结果、顺序和uint8数值类型。来源见<a href="source-audit-final/dreamzero-native-assembly-synthetic.json">原生纯函数合成回放</a>，不是DreamZero推理通过证据。</p>
<div class="callout">当前保留这段原生补帧代码，不能对外声称它是公共空间按真实时间选出的4帧。若将来要改成4张真实等间隔历史，必须另建具名模型配置、核对原生缓存与因果状态、完成来源和对照，不能直接把公共缓冲塞进去替换。</div>
<p>DreamZero真实动作验证仍等待用户后续安排两张每卡80GB的H100；本轮没有再次启动推理，也没有替换量化、分辨率或采样策略。</p></section>
<section aria-labelledby="evidence"><h2 id="evidence">验证结果与证据边界</h2>
{table(['验证','本轮结果','能证明什么'], [
    ['全量主测试', f'{tests}项通过，零跳过；572项旧覆盖＋31项新增', '元数据变更及既有软件行为无回归，不是真实任务评分'],
    ['五组原有CLI', '/'.join(str(x['actual_exit_code']) for x in baseline['checks'][1:]), '退出语义保持0/0/0/1/2'],
    ['MassWorld桥接', f'{bridge}项通过，零跳过', '桥接与观测历史机制仍能工作'],
    ['保存真实数据边界', f"{len(saved['cases'])}条路线，各1次已保存转换", '输入张量、采集时间、原生动作数值／精度／取图标记保持，不是新闭环'],
    ['固定来源审计', f'{source_count}个不同源码文件，{metadata_count}份不同小型权重元数据', '引用源码字节匹配固定Git版本；元数据按各自锁状态核查'],
    ['DreamZero图像组装', '5次纯函数调用、三路合成图像', '确认内部不足4帧会重复；不是权重推理或时间标定']])}
<ul><li><a href="baseline-final/baseline-check.json">全量与CLI回执</a>；<a href="baseline-final/unittest.txt">完整603项测试日志</a>。</li>
<li><a href="temporal-evidence-final/receipt.json">观测时间及桥接检查回执</a>；<a href="temporal-evidence-final/bridge-tests.txt">桥接日志</a>。</li>
<li><a href="saved-boundaries-final/receipt.json">保存真实数据映射与历史文件指纹保护回执</a>。</li></ul>
<p>来源核验只读被引用的源码和小型配置，没有重新读取全部大权重。本轮OpenWAM LIBERO的配置仅新记SHA256，未发现预先锁定的配置指纹，回执明确区别于已有资产锁匹配，不将它包装成已核实的完整训练来源。没有保存整份训练YAML或认证信息。</p></section>
<section aria-labelledby="next"><h2 id="next">剩余缺口和下一步</h2>
<ol><li><strong>物理动作时长：</strong>补齐发布权重对应的数据集元信息、动作抽样、过滤规则和时间参照；目前所有配置仍明确未知，不自动取环境周期或FPS倒数。</li>
<li><strong>原生服务历史：</strong>DreamZero内部积帧没有携带公共采集时间；本轮只把事实核实并暴露出来，不接管或美化它。其他模型也不因声明了当前画面接口就声称已完整标定内部状态。</li>
<li><strong>新真实对照：</strong>本轮没有新GPU推理；旧π0.5图像差异、LingBot动作差异、RoboTwin物理步数差异等严格复现缺口仍保留，保存样本转换通过不能关闭它们。</li>
<li><strong>动作跨频率执行：</strong>动作插值、统一重规划、实时异步和硬仿真时间预算不在已批准首轮范围。若继续做，须先明确哪些动作可重采样、相对哪个时刻生效、是否改变原生评测协议，再提交确认。</li></ol>
<p>目前可以继续用原生模式进行已有协议的运行，也可用已交付的合成例子验证公共观测历史机制；不能承诺模型任意换环境后具有相同物理时间含义，更不能承诺任务成功率等同。</p>
<pre><code>.venv/bin/python docs/handoff/audit_native_timing.py --output outputs/my-new-native-timing-audit
.venv/bin/python -m unittest tests.test_native_timing_profiles -v</code></pre>
<p>命令中的输出路径应换成不存在的新目录，工具拒绝覆盖旧证据。报告服务只开放本页明确链接的证据，不开放整个工作区；远程浏览需要平台端口转发。</p></section>'''
    template = args.template.read_text()
    start = template.index('<main class="report-content">') + len('<main class="report-content">')
    end = template.index('</main>', start)
    document = (template[:start] + body + template[end:]).replace('lang="en"', 'lang="zh-CN"').replace('<title>Report title</title>', '<title>模型时间规则：来源审计与修改证据</title>')
    output.write_text(document, encoding="utf-8")
    print(json.dumps({"report": str(output), "main_tests": tests, "source_files": source_count,
                      "weight_metadata": metadata_count, "new_real_inference": False}))


if __name__ == "__main__":
    main()
