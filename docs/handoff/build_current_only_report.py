"""Build a static human-readable delivery report from completed receipts."""
import argparse
from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    'baseline': 'outputs/current-only-baseline-20261023-final4/baseline-check.json',
    'python310': 'outputs/current-only-baseline-py310-20261023-final4/baseline-check.json',
    'bridge': 'outputs/current-only-bridge-20261023-final/receipt.json',
    'architecture': 'outputs/current-only-architecture-audit-20261023-final2/receipt.json',
    'saved': 'outputs/current-only-saved-boundaries-20261023-final3/receipt.json',
    'package': 'outputs/current-only-package-check-20261023-final2/receipt.json',
    'inventory': 'outputs/current-only-runtime-inventory-20261023-run1/receipt.json',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(path, label):
    assert (ROOT / path).is_file(), path
    return '<a href="' + quote('/' + path, safe='/') + '">' + escape(label) + '</a>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(str(c)) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    data = {key: json.loads((ROOT / path).read_text()) for key, path in PATHS.items()}
    for key, record in data.items():
        if key != 'inventory':
            assert record['passed'], key
    architecture, saved = data['architecture'], data['saved']
    for record in (architecture, saved):
        hashes = record.get('runtime_python_sha256', record.get('framework_source_sha256'))
        assert all(sha(ROOT / path) == digest for path, digest in hashes.items()), 'runtime changed after receipt'
    # The completed boundary gate already rehashed its immutable large traces.
    # Bind its receipt, rather than claiming another full historical audit here.
    assert saved['historical_hashes_unchanged'] is True
    tests = data['baseline']['checks'][0]['tests_run']
    assert tests == data['python310']['checks'][0]['tests_run'] == 676
    assert [r['actual_exit_code'] for r in data['baseline']['checks'][1:]] == [0, 0, 0, 1, 2]
    removed = len(architecture['removed_runtime_files_recoverable_in_git'])
    runtime_count = len(architecture['runtime_python_sha256'])
    notes = {
        'eef10_osc': 'OpenWAM单臂归一化控制器编码；模型服务持有原生动作缓存，不能从一次返回一条推断只生成一条。',
        'state8_osc': 'π0.5单臂8维输入编码；原生每次取动作段前5条，剩余尾段不被Runner接管。',
        'eef10_absolute': 'OpenWAM单臂绝对末端动作；底座偏移、旋转编码、图像处理都留在模型侧。',
        'state7_absolute': 'π0.5单臂7维输入／动作规则；保留原生位置与夹爪转换，原生消费前5条。',
        'eef20_dual_legacy': '具名原样参考扩展：输出不是公共物理位姿，只能进入声明相同原生协议的执行端。它仍属于0.3，不是旧运行协议回退。',
        'eef20_dual_wxyz_encoded_absolute': '同一旧双臂权重的标准化入口：特殊训练编码只在模型内部，公共朝向为xyzw；不是拿ARX配置替代它。',
        'joint14': 'LingBot使用双臂控制目标和三张当前图；14维排列只在模型编码器重建。服务默认50不等于已确认任意服务实收50，按实际返回整段消费。',
        'arx_joint14': 'π0.5的ARX双臂关节配置；同一模型Adapter对接RoboTwin或RoboDojo，固定原生50条参考。',
        'eef20_arx_absolute': 'OpenWAM的ARX绝对双臂位姿配置；模型侧持有缓存与权重校准，条数沿用部署配置，不能猜为32。',
        'nano-policy-droid-joint8': 'Cosmos3的7关节加夹爪公共动作；模型Adapter持有原生32条消费偏好，调整仍要先有环境许可。',
        'droid-joint8-causal': 'DreamZero原生8条参考；服务有因果历史，不因数组可以截短就声明任意改频已支持。真实推理仍等待后续硬件。',
    }
    body = ['<h1 id="title">单一0.3运行版本与独立Adapter：本轮修改报告</h1>',
        '<p class="lead">2026-10-23（北京时间）｜代码、静态检查、保存真实记录转换与新推理分开报告。实施提交1905ba7；清理前检查点67ca6ad。</p>',
        '<h2 id="conclusion">先说结论</h2>',
        '<div class="callout"><strong>公开运行版本已统一为0.3，模型与环境分别注册；执行规则按Benchmark优先解析，不新增逐组合Adapter。</strong>旧入口、旧导入别名和组合工厂已移出安装包。但这不代表所有组合、所有原生环境和所有任务都已真实验收。</div>',
        f'<p>两套Python各{tests}项测试、零跳过；MassWorld桥接34项；两套入口的五组CLI均为0/0/0/1/2。11条保存真实链路各检查1个已有转换，合计22份观测和11条动作；没有新GPU推理、仿真、下载或平台任务。GPU总体平均20%的要求保留给后续集中真实验证，本轮没有重新测量或以占卡负载充作评测工作量。</p>',
        f'<p><strong>仍需直说的限制：</strong>所有现有权重的动作物理时长仍未取得完整训练来源；真实Benchmark的动作消费权限没有一概登记为“自由可调”；{len(data["inventory"]["missing_interpreters"])}套旧原生环境的解释器链接失效。DreamZero双H10080GB、旧严格复现差异和正式阶段验收仍未关闭。</p>',
        '<h2 id="scope">本轮授权与没有擅自改变的内容</h2>',
        '<p>按已批准方案实现执行策略解析，并清理旧运行版本与历史兼容依赖。没有更换权重、归一化统计、原生控制器、逆运动学或评分规则；没有修改只读上游、历史报告、旧结果和Mac绝对路径。没有把视频生成模型扩展成机器人规划器，也没有新增全身机器人、动作重采样、自动补帧、硬超时、并发评测、失败重试或恢复。</p>',
        '<p>“一个版本”指唯一公开运行／配置／消息协议为0.3，不是要求所有权重使用同一种神经网络编码，也不是所有第三方依赖必须塞进同一个Python环境。历史测试参考留作对照，不随安装包发布；它们不是生产兼容入口。</p>',
        '<h2 id="architecture">现在的架构与边界</h2>',
        '<p>Adapter指各自负责格式转换和生命周期的适配代码；Benchmark是仿真评测环境，Runner是通用执行循环。公共位置用米、角度用弧度、时间用秒。xyzw表示四元数的向量分量x、y、z在前、实数分量w在后；wxyz则把w放在第一位。整体取反仍是同一朝向。统一排列并没有补出缺失的坐标原点或工具点偏置。</p>',
        '<pre>环境原生观测 → Benchmark Adapter → 公共观测\n                                      ↓\n                  Model Adapter → 模型推理／唯一动作缓存\n                                      ↓\n+                公共动作 → Benchmark Adapter → 仿真与官方评分\n\n通用Runner：调用顺序、预算、校验、计数、证据\n通用执行解析器：根据双方独立声明，决定设置是否满足规则</pre>'.replace('\n+', '\n'),
        table(['位置', '现在只负责什么', '本轮变化'], [
            ('core/', '米、弧度、秒、xyzw位姿、类型化动作、坐标路径和注册容器', '几何校验不再依赖旧协议校验器'),
            ('v03/', '唯一公共协议、公共字段匹配、观测历史、执行策略、通用Runner', '增加纯解析器与计划／应用回执；无模型或环境名称分支'),
            ('native/model_catalog.py、model_assets.py', '模型配置与源码／权重／统计身份', '不再从环境空间定义取模型身份'),
            ('native/benchmark_catalog.py、benchmark_assets.py、environment_specs/', '环境版本、机器人、控制、场景与评分资产', '不再从模型目录取环境规则'),
            ('adapters/', '仍使用的环境加载器、原生通信、物理计数等工具', '删除旧模型包装器和旧组合工厂'),
            ('integrations/massworld/', '把整次机器人评测委托给EmbodiedEval', '链接execution-plan.json及其SHA256，不重复解析或缓存'),
            ('tests/legacy_support/frozen/', '历史回归、旧证据的显式只读解析', '不发布、不作为run/check回退'),
        ]),
        '<p>M+N中的M是模型数量，N是环境数量。每一方只实现自己的Adapter和声明。添加新组件只登记新增组件，不修改Runner或其他Adapter；结构缺项则拒绝，而不是再写一份逐组合Python胶水。</p>',
        '<h2 id="cleanup">哪些历史兼容已清理</h2>',
        table(['原有问题', '这次处理', '为什么这样处理'], [
            ('mn与_legacy旧导入路径', '生产包删除；旧pickle类路径也不再承诺可加载', '避免旧协议继续作为隐形API'),
            ('0.1／0.2旧Runner、配置、逐组合工厂', '移到不发布的历史测试参考', '保留原测试的判定能力，不继续生产运行'),
            ('CLI migrate兼容入口', '退役，正常CLI只有run与check', '当前配置只保留0.3；历史证据解析不等于旧配置执行'),
            ('12个使用旧Runner／旧配置的独立脚本入口', '明确退出并提示当前0.3工具；纯参考函数仅供历史测试／只读核对', '避免从正常CLI之外又启动旧协议评测'),
            ('内部0.1／0.2消息封装', '私有槽／原生空间记录无版本；控制消息共享当前0.3定义', '不是仅把旧版本字符串换成新名字'),
            ('模型／环境混合目录及资产核验', '拆成独立模型目录与独立环境目录', '新增模型不必触碰环境元数据'),
            ('RoboTwin底层装载器残留model_repo与模型名分支', '移除；按sealed／seeded具名环境协议选指令', '旧有效指令算法保持，环境不再加载模型仓库初始化代码'),
            ('退役配置仍给当前测试供例', '当前13份例子直接读0.3文件，历史格式只用于历史测试', '防止新测试依靠旧迁移路径才能构造当前配置'),
        ]),
        f'<p>安装包从114个运行时Python文件收敛为{runtime_count}个；{removed}个旧运行／重复数据路径移出生产目录。历史参考含{architecture["historical_test_reference_python_files_not_distributed"]}个Python文件，打包检查确认没有被带进wheel。wheel是Python可分发安装包。退役源码可从Git检查点67ca6ad恢复；清理了80个旧mn编译缓存，它们可由Python重建。历史报告与原始运行数据未删除。</p>',
        '<h2 id="execution-policy">Benchmark优先到底怎么执行</h2>',
        '<p>不是给整个Benchmark贴一个“固定／自由”的标签，而是逐个参数判断：例如物理周期可以固定，动作消费条数却可能允许模型决定。环境规则不明确时，不把某个模型示例的默认值升级成官方规定。</p>',
        table(['规则', '人话', '程序行为'], [
            ('fixed', '环境明文规定只能用这个值', '模型真正支持才运行；用户覆盖冲突就拒绝'),
            ('range', '环境明确允许这些值／这个范围', '先用模型原生偏好；允许范围内的显式实验；不偷偷裁剪到边界'),
            ('policy_owned', '环境明确把该参数交给模型', '用模型自身规则与可支持设置'),
            ('reference_default', '只找到一份参考实现的默认值', '保留匹配原生入口，不把默认值当硬规定'),
            ('unknown', '规则或权重时间含义还没核实', '保留明确标注的原生参考；改动／严格同协议比较不能放行'),
        ]),
        '<p>例如：假设环境一条动作推进0.05秒，模型一次预测50条，但执行5条后重新看图，那么原生重推理间隔为5×0.05＝0.25仿真秒；不是一定执行完50条。这个例子是算术说明，不是给所有真实权重补出的时间标定。变时长动作不能这样乘：必须分别累计原生动作次数、物理步数和实际秒数。</p>',
        '<p>模型声明预测总长度、原生消费偏好、原生参考值、可调范围、唯一缓存归属及动作时长来源；环境声明自己的约束和适用版本／任务／机器人／控制器。纯解析器没有图像处理、动作解码或仿真代码。Runner在环境加载后复查实际周期，再要求模型给出应用回执；模型仍持有唯一缓存。</p>',
        '<p><code>execution.consume_actions</code>表示消费几条再重新推理；<code>require_verified</code>表示是否必须所有必要执行来源都核实。默认false允许标注未校准的原生参考；true遇到未知就拒绝。模型选项里的旧open_loop_horizon也不能绕过环境权限。时间声明和执行声明对缓存归属或动作时长相互矛盾时，预检查会拒绝。</p>',
        '<p><code>execution-plan.json</code>保存预检查计划、实际环境复查计划、应用回执、双方原始声明及逐轮种子／初始化。protocol_verified只表示声明层参数来源已解出，不等于服务身份、真实推理或阶段验收通过。predict调用、原生请求、实收动作条数、入缓存条数与真实前向次数分别记录；真实前向次数不明继续为null，即未知。</p>',
        '<h2 id="profiles">现有13条入口逐条说明</h2>',
        '<p>下表“消费”是当前模型声明，不是已给所有Benchmark批准的自由调参。缓存指把一次返回的后续动作暂存起来；“未确认”不能填成1、32或50来凑数。</p>']
    rows = []
    for row in architecture['configurations']:
        decl = row['model_execution']
        rows.append((row['configuration'], row['model'] + ' / ' + row['io_profile'],
            row['benchmark'] + ' / ' + row['protocol'],
            ('未确认原生条数' if decl['preferred_consume_actions'] is None else str(decl['preferred_consume_actions']) + '条')
            + '；缓存：' + ('模型服务' if decl['cache_owner'] == 'server' else '模型Adapter'), notes[row['io_profile']]))
    body += [table(['当前配置', '模型配置', '环境协议', '消费／缓存', '具体说明'], rows),
        '<p>OpenWAM旧双臂原样参考扩展仍是当前0.3的一种具名原生编码，不是0.1／0.2兼容入口；它没有被包装成公共物理位姿，也不宣称可跨环境。另一个标准化配置已经提供公共xyzw位姿。不同编码配置与“只保留一个公开协议版本”并不冲突，但原样扩展的使用范围必须继续受协议签名限制。</p>',
        '<p>π0.5普通单臂前缀消费可声明1至50条；Cosmos3可声明1至32条；实际修改仍须环境明确许可。π0.5 ARX只声明原生50条参考。OpenWAM远端缓存、ARX部署条数和LingBot服务返回段没有被假装远程改好；不满足固定要求时拒绝。DreamZero只声明原生8条，不宣称截短后因果历史仍正确。</p>',
        '<h2 id="coverage">覆盖账本怎么读</h2>',
        f'<p>当前是5类动作模型、5类仿真环境、11种具名权重输入输出配置、7种环境协议、13份现有配置。静态矩阵做了{architecture["structure_matrix_cases"]}次声明匹配，{architecture["structure_matrix_interface_matches"]}次接口结构满足，57次明确拒绝。没有为这77次另写77个Adapter。</p>',
        '<p>这不是20组真实任务已通过。相机／状态缺失、单双臂不完整、关节或动作类型不符、控制器原生协议不符等仍不能启动。原49组缺口与这里的77次统计口径不同：前者按旧模型／Benchmark表格记格子，后者按当前具名权重与环境协议计数，不能用20或57覆盖旧49记录，更不能用占位Adapter标完成。</p>',
        '<p>环境协议仍有明确试点边界：LIBERO只含已有具名空间任务，VLABench只含select_fruit固定轨迹，RoboTwin只含adjust_bottle／demo_clean，RoboDojo是已有stack-bowls协议；DROID有已有场景接口但没有新增官方成功评分。覆盖接口不等于全Benchmark所有任务、所有机器人、所有种子都完成了。</p>',
        '<h2 id="evidence">实际验证证据与能证明什么</h2>',
        table(['检查', '本轮实际结果', '不能推出什么'], [
            ('主测试，Python 3.13.5', f'{tests}项，零跳过；原634项判定保留，新增42项执行／隔离检查', '不是新模型成绩'),
            ('主测试，Python 3.10.21', f'{tests}项，零跳过；五组CLI为0/0/0/1/2', '不是全部第三方GPU依赖都兼容'),
            ('MassWorld桥接', '34项，零跳过；五组CLI同为0/0/0/1/2；执行计划文件及摘要可回查', '不是重跑MassWorld所有世界模型后端'),
            ('保存真实记录边界', '11条链路，每条1个已有动作转换；22份观测、11条预测，图像、状态、精度、采集时间与执行参数保持', '不是新推理、全轨迹重现或任务成功'),
            ('代码计算语句审计', '16处转换／执行／评分计算及有效RoboTwin选指令算法核对通过', '不是声称所有代码一字不改'),
            ('资产身份', '11种权重配置身份、7种环境元数据与12份小型资产清单保持', '不是重新哈希全部大权重，也不是在线服务加载身份验证'),
            ('安装包', f'实际打包{runtime_count}个当前Python文件，旧入口／历史测试未打包；两Python中合成运行通过', '不是完整原生GPU环境验收'),
            ('原生子进程探针', '健康Python 3.11环境中构造、消息收发、关闭确认与退出码通过；未begin、未加载权重、未推理', '不是原失效OpenWAM运行环境已修复或完整推理通过'),
        ]),
        '<p>保存记录用显式离线0.2解析器读取，再只在内存转换消息外壳以检查当前公共边界。原文件、原路径和数值不改；没有给当前Runner增加旧版本回退。已有严格原生一致仍只按先前确认的两条处理，其余重新推理差异不因本轮转换通过而关闭。</p>',
        '<h2 id="cannot-flatten">哪些东西不能硬塞成完全一样</h2>',
        table(['对象', '为什么不能抹掉', '现在的处理与仍缺什么'], [
            ('权重的归一化、状态／旋转排列、机器人训练参考', '它们定义权重实际学过的数据，改错了就不是同一模型输入', '留在具名Model Adapter配置；不交给Runner，也不按当前Benchmark名称分支'),
            ('环境控制器、逆运动学、评分与变时长轨迹', '它们定义这台仿真机器人怎样动、怎样判成功', '留在Benchmark Adapter；不能为了模型方便偷偷改环境规则'),
            ('未知基座外参和工具点偏置', '只有统一xyzw排列，仍不知道两坐标原点／工具点之间差多少', '缺变换就拒绝；有完整标定才能转换，不能猜单位变换'),
            ('远端服务缓存与因果历史', '外层截数组不代表服务端缓存、采样、隐藏状态同步改变', '没有设置／确认能力时只保留原生参考；固定值不满足就拒绝，后续需服务协议和验证证据'),
            ('具名原生控制器扩展', '归一化控制量或特殊插值规则没有自动变成通用物理位姿', '仅同协议与结构签名匹配端可执行，不按模型＋环境名单硬编码'),
            ('权重训练动作物理时长与Benchmark权限来源', '运行日志能证明推进了多久，不能证明权重训练时每个动作标签对应几秒，也不能替官方规定', '继续明确未知；严格门禁不放行，不通过视频FPS倒推'),
        ]),
        '<p>这些不是要求每个组合再写一个Adapter。前两项已经实现软件职责分离；后几项是互操作所需信息或服务能力不足。框架可以让问题被清楚地拒绝／记录，却不能凭软件命名把缺失信息创造出来。</p>',
        '<h2 id="runtime-gaps">运行准备与尚未关闭的问题</h2>',
        '<p>本轮只读盘点18套独立环境，9套python链接指向当前实例不存在的旧/root管理路径。它们依次是：<code>' + escape('、'.join(data['inventory']['missing_interpreters'])) + '</code>。这说明旧“目录在、曾经跑过”不能当作今天可运行。没有修改这些共享环境，也没有通过全局/root别名隐藏问题；包的通用子进程探针使用现有健康环境，仅证明协议传输和关闭流程。</p>',
        '<ol><li>需要按原有锁定版本重建或重定位这些原生运行环境，并逐一验证依赖；这是可修复的运行准备问题，不是理论上无法解耦的Adapter限制。</li><li>需要补齐权重动作时间来源和各Benchmark具名协议对动作消费条数的正式许可；当前13份参考配置不能冒充已完成严格时间标定。</li><li>需要在环境就绪后做当前版本的有限步新推理／仿真和同条件原生对照，再扩到完整任务与多种子。当前保存记录回放不替代它。</li><li>π0.5两组输入差异、LingBot重新推理动作差异、RoboTwin物理步数差异保留；不能用任务成功掩盖转换或原生复现问题。</li><li>DreamZero仍需你后续对接两张每卡80GB的H100再实测；官方测试硬件不是必过保证。本轮没有租卡或改量化、图像尺寸、采样。</li></ol>',
        '<p>因此，本轮是代码架构与软件证据交付，不是“全部残留问题已清零”，也不是阶段4正式通过或公共契约冻结。</p>',
        '<h2 id="onboarding">新增模型或环境的调试流程</h2>',
        '<ol><li>先固定自己这一侧的源码、权重／资产、归一化、机器人与原生输入输出示例。</li><li>模型把公共head等字段编码成自身输入并把动作解码回来；环境把自己的观测映射成公共字段并实现控制／评分。不要新增组合名单。</li><li>声明实际输入、输出、坐标、参考部位、完整双臂／关节目标、观测时间要求，以及独立执行能力／环境约束与来源。</li><li>实现独立注册、选项校验、资产校验和生命周期；需要改消费设置时实现真正的应用回执。不能控制服务端就声明不支持。</li><li>不启动GPU做预检查；缺相机、状态、动作类型、坐标关系或规则时先定位到具体字段。</li><li>用保存原生观测和动作做边界对照，检查图像、左右臂、夹爪、精度、四元数整体反号等价性、缓存及采集时间。</li><li>分别验证模型能推理、环境能初始化／执行，再做有限步闭环、同条件原生对照及视频，最后才做全任务、多种子统计。</li></ol>',
        '<p>常规“相机字段改名、模型图像排列、已知坐标变换”只改新增方的Adapter；真正耗时的是首次厘清权重动作含义、原生依赖、服务历史和控制器，而不是为每个组合复制代码。公共格式无法表达的新物理含义仍先提交契约缺口与影响，不静默扩协议。</p>',
        '<h2 id="files">证据与版本位置</h2>',
        '<p>SHA256是文件字节的固定摘要，用来核对文件有没有变，不是模型评测分数。下列链接给出实际回执，不把预检查、资产存在或组件导入当作真实任务成功。</p>',
        '<p>源码检查点：67ca6ad；实施提交：1905ba7；交付标签：<code>current-v03-decoupled-20261023</code>。分支仍为feat/massworld-integration。旧文件恢复请在独立检出目录查看检查点，不把旧代码混回当前运行包。</p>',
        '<ul>' + ''.join('<li>' + link(path, key + '：' + path) + '</li>' for key, path in PATHS.items()) + '</ul>',
        '<p>' + link('docs/adapter-onboarding.md', '完整接入与调试指引') + ' · ' + link('docs/contract-v0.3.md', '当前0.3字段与执行契约') + ' · ' + link('docs/handoff/current-only-tools-20261023.json', '当前与退役工具清单') + '</p>',
        '<p>报告由html-report阅读模板生成，为无脚本、无外部资源的单文件网页。本机服务只监听127.0.0.1并按显式链接白名单供文件；实例地址不是你电脑地址，远程浏览需要真实端口转发。HTTP字节／权限检查与浏览器视觉检查分开记录。</p>',
    ]
    template = (ROOT / 'docs/handoff/report-shell-v03.html').read_text()
    html = template.replace('公共空间 0.3 迁移与验证报告', '单一0.3运行版本与独立Adapter：本轮修改报告').replace('{{BODY}}', '\n'.join(body))
    path = args.output / 'current-only-decoupling-report.html'
    path.write_text(html, encoding='utf-8')
    receipt = {'kind': 'current_only_delivery_report_not_stage_acceptance', 'at_utc': datetime.now(timezone.utc).isoformat(),
        'report': str(path.relative_to(ROOT)), 'report_sha256': sha(path),
        'implementation_commit': subprocess.check_output(['git', 'rev-parse', '1905ba7'], cwd=ROOT, text=True).strip(),
        'evidence': {k: {'path': p, 'sha256': sha(ROOT / p)} for k, p in PATHS.items()},
        'new_inference': False, 'new_simulation': False, 'formal_acceptance': False, 'contract_frozen': False}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(path), 'bytes': path.stat().st_size}))


if __name__ == '__main__':
    main()
