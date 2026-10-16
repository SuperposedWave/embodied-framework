"""Evidence-backed 0.3 migration report; no new inference, claims or rewrites."""
import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def link(path, label):
    path = path.resolve()
    path.relative_to(ROOT)
    assert path.is_file(), path
    return '<a href="' + quote('/' + path.relative_to(ROOT).as_posix(), safe='/') + '">' + escape(label) + '</a>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + cell + '</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def section(key, title, body):
    return '<section aria-labelledby="' + key + '"><h2 id="' + key + '">' + title + '</h2>' + body + '</section>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--secondary-baseline', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--camera-audit', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    base, mapping, inv, real = [read(path) for path in (args.baseline, args.mapping, args.inventory, args.real)]
    assert base['passed'] and mapping['passed'] and inv['passed']
    secondary = read(args.secondary_baseline)
    assert secondary['passed'] and secondary['checks'][0]['tests_run'] == base['checks'][0]['tests_run']
    count = base['checks'][0]['tests_run']
    codes = [r['actual_exit_code'] for r in base['checks'] if 'actual_exit_code' in r]
    assert codes == [0, 0, 0, 1, 2]
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M 北京时间')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    pieces = ['<h1 id="report-title">公共空间 0.3 迁移与验证报告</h1><p class="lead">EmbodiedEval · ' + now + ' · 当前入口迁移、实际证据与仍未完成事项</p>']
    passed = sum(r['passed'] for r in real['cases'])
    pieces.append(section('conclusion', '先说结论',
        '<div class="callout"><strong>当前使用入口和既有Adapter公共边界已迁移到0.3。</strong>环境先输出统一字段，模型直接从统一字段取数据；普通配置不再逐对填写接线。不是把旧字段换个外壳后继续让用户手工对应。</div>'
        + '<p>实际完成：' + str(count) + '项软件／合成检查、零跳过；五组当前CLI（命令行入口）退出码' + '/'.join(map(str,codes))
        + '。保存的11条真实链路共55份观测、44条动作通过边界值对照。新四步真实短跑完成' + str(passed) + '/' + str(len(real['cases'])) + '条；逐条结果见后文，失败不隐藏。</p>'
        + '<p>这不意味着所有模型在所有环境中都能成功。也不代表阶段4正式验收、契约冻结、全部任务覆盖或DreamZero真实动作已通过。</p>'))
    pieces.append(section('meaning', '用人话解释现在的数据流',
        '<pre>环境自己的相机／状态名称\n  → 环境Adapter：识别真实含义，输出head、wrist等公共字段\n  → 公共观测：统一字段＋单位＋坐标＋采集时间＋来源\n  → 模型Adapter：取公共字段，转换成匹配权重需要的输入\n  → 推理／原生动作缓存\n  → 模型Adapter：解码成公共关节、位姿、夹爪或具名原生扩展\n  → 环境Adapter：转换成环境原生控制、推进仿真、调用官方评分</pre>'
        + '<p>Adapter就是某一侧的适配器。M+N表示M个模型和N个环境分别实现自己的Adapter，不写逐对组合工厂。Runner只负责顺序、预算、计数、校验和保存证据，不做图像缩放、归一化、控制器或逆运动学。逆运动学是根据末端目标求关节位置。</p>'
        + '<p>模型的InputSpec（输入需求）声明公共路径，例如sensors.head；OutputSpec（输出说明）声明公共目标，例如robot0/left_eef，即机器人0的左侧末端。预检查自动生成ResolvedProfile（匹配回执），用于记录实际路径，不再要求用户手填binding（逐对字段对应配置）。</p>'))
    camera_rows = []
    for key, s in inv['benchmarks'].items():
        if any(key.split('/')[0] == row[0] for row in camera_rows):
            continue
        family = key.split('/')[0]
        camera_rows.append((family, '<br>'.join('<code>' + escape(v['native_source']['field']) + ' → ' + escape(k) + '</code>' for k,v in s['sensors'].items())))
    pieces.append(section('fields', '公共字段到底统一了什么',
        table(['环境家族', '原生来源 → 实际公共键'], [[escape(k), v] for k,v in camera_rows])
        + '<p>head表示工作区总览的逻辑角色，不保证物理安装在头部；wrist是单臂腕部，left_wrist/right_wrist是双臂各自腕部。DROID外部相机暂用external_0/1稳定身份，不凭模型的旧变量名猜物理左右。Cosmos和DreamZero的原生相机顺序不同，由各自模型配置独立处理。</p>'
        + '<p>公共状态分开记录末端位置／朝向、实测关节、关节控制目标、实测夹爪和夹爪控制目标。双臂分别加left_和right_前缀。RoboTwin的14维混合驱动目标拆成左6关节、左夹爪、右6关节、右夹爪；只有LingBot的模型私有输入中重新拼回14维，不能把目标冒充实测位置。</p>'
        + '<p>每份样本保留采集时间、数值类型、有效性和来源；每个状态声明单位、物理关节顺序及参考部位。完整逐字段说明见' + link(ROOT/'docs/contract-v0.3.md','0.3契约与字段目录') + '。</p>'
        + '<p>公共模型输入里的键仍然叫head等公共名称。native_source（原生来源）只是追溯资料，不是模型据以取数据的环境字段。</p>'))
    pieces.append(section('camera-basis', '相机方位的核实边界',
        '<p>VLABench外部右相机并非只凭right这个变量名判断：固定场景XML给出相机位置(0.775,-0.856,1.209)米，机器人基座位置(0,-0.4,0.78)米、绕竖直轴转角1.57弧度。换到基座坐标后，相机侧向坐标约为-0.775米，位于右侧，因此使用external_right。XML是原生场景结构文件。</p>'
        + '<p>这一静态侧向核实不等于已发布完整的运行期相机或基座标定；没有据此补入猜测外参。DROID仍保持external_0/1，不沿用模型变量名作物理左右结论。' + link(args.camera_audit,'固定来源、位置与计算回执') + '</p>'))
    pieces.append(section('geometry', '坐标统一不是猜测标定',
        '<p>位置用米，时间用秒，转角用弧度。四元数统一为xyzw：向量分量x、y、z在前，实数w在后。四元数整体取反仍表示同一朝向。排列变换、坐标系变换和工具参考点变换是三件不同的事。</p>'
        + '<p>公共world保留各环境协议核实的世界原点；不同环境的world不是同一个物理位置。未知基座外参时，robot0/base保持不连通，不补假单位变换。位姿转换只沿已声明的真实旋转和平移路径执行。</p>'
        + '<p>参考部位由reference_sites目录明确说明；它记录测量位置、测量朝向或控制目标究竟对应哪个部位。LIBERO现有位置来自grip_site、朝向来自right_hand，仍保留两个部位，不强行拼成同一个工具点。无法确定偏置时拒绝，坐标转换不能代替工具标定。</p>'
        + '<p>旧OpenWAM双臂含糊输出仍保留具名原生扩展；另建的eef20_dual_wxyz_encoded_absolute配置继续按匹配权重的特殊训练编码读写，再进公共物理位姿。没有为“看起来标准”更改权重输入、归一化统计或控制器。</p>'
        + '<p>测试覆盖无旋转、三轴90度与180度、一般旋转及四元数整体反号。公共单位四元数沿用平方长度距1的绝对容差0.000001，不放宽为任务成功；合成旋转往返的位置误差要求小于0.000000000001米。</p>'))
    pieces.append(section('scope', '迁移覆盖与结构边界',
        '<p>当前5个模型家族：OpenWAM-alpha、π0.5/openpi、LingBot-VLA、Cosmos3 Nano Policy、DreamZero；5个仿真家族：LIBERO、VLABench、RoboTwin、DROID sim_evals、RoboDojo。它们共11份模型输入输出配置、7份具名环境协议。LingBot-VLA和暂缓的LingBot-VA不是同一个模型。</p>'
        + table(['模型具名配置', '读取公共数据／输出类型'], [[escape(k), '<code>' + escape(', '.join(r['public_path'] for r in v['inputs'])) + '</code><br>输出：' + escape(', '.join(v['output']['command_kinds']))] for k,v in inv['models'].items()])
        + '<p>结构网格检查77格，其中20格结构满足。其余格的拒绝原因保留在' + link(args.inventory,'逐格结构与历史完整性清单') + '；这不是运行时白名单，也不是77格都已经运行。新增模型／环境只注册自己，不修改Runner或另一方Adapter。</p>'
        + '<p>缺必要相机、状态、关节顺序、明确坐标、完整双臂或可执行动作类型时仍拒绝。权重是否在某机械臂上训练过只作提示，不作为架构名单门槛。结构满足不保证这份权重有能力完成任务。</p>'))
    pieces.append(section('changes', '文件与配置改了什么',
        '<pre>embodied_eval/\n├── contracts.py / interfaces.py / config.py / validation.py / runner.py\n│   └── 当前0.3公共入口\n├── v03/\n│   ├── contracts.py / config.py：公共类型与严格配置\n│   ├── matching.py / preflight.py：自动匹配和无GPU预检查\n│   ├── models.py：公共键到权重私有编码，保留原生缓存\n│   ├── benchmarks.py：环境原生字段到公共字段及执行边界\n│   ├── transforms.py：有依据的坐标变换工具\n│   ├── registry.py / interfaces.py：双方独立登记和生命周期\n│   ├── validation.py / runner.py：消息校验和串行执行\n│   ├── migration.py：显式离线迁移，不静默吞掉旧设置\n│   └── synthetic.py：明确标注合成的架构测试组件\n├── mn/、adapters/：仍复用的独立原生组件与编码\n└── _legacy_*：共享叶类型／旧检查，不是正常配置回退\ntests/\n├── fixtures/legacy-configs/：13份原配置的完整历史副本\n├── legacy_support/：只在测试中运行的旧配置和Runner参考\n└── test_v03_public_space.py：52项新增专项检查</pre>'
        + '<p>六组原入口和OpenWAM物理双臂示例已迁成0.3，另补齐六份ARX／DROID当前配置示例。五份旧mock运行示例移出活动examples目录，可在测试夹具和pre-v03-20261016标签恢复；没有删除历史证据。旧合同消息样本仍是历史测试资料，不是当前运行配置。</p>'
        + '<p>正常run/check拒绝0.1、0.2、缺版本和未知版本，不隐式回退。离线migrate命令只转换可证明等价的配置，不覆盖文件；自定义旧接线、未知字段、未核实坐标转换或退役工具行为需要明确审查，不猜测抹掉。</p>'))
    cli_rows = [[escape(r['case']), str(r['actual_exit_code']), escape(json.dumps(r.get('execution_statistics',{}), ensure_ascii=False))] for r in base['checks'] if 'actual_exit_code' in r]
    pieces.append(section('software', '软件回归和命令行证据',
        '<p>' + str(count) + '项、零跳过：保留既有443项检查覆盖，新增52项0.3专项。覆盖2个模型×2个环境的四种交叉组合，以及仅新增注册就接入第三组件。此项明确是合成架构测试，不是真实机器人能力。</p>'
        + '<p>专项还覆盖公共命名、必要主视角、相机重命名、双臂完整性、实测／目标来源、关节顺序、参考部位、坐标路径、缓存与跨轮次清空、动态指令、变时长计数、错误版本／重复JSON键／非有限数／无效样本／未知字段及预检查不加载GPU库。</p>'
        + table(['当前合成用例', '实际退出码', '实际计数'], cli_rows)
        + '<p>退出码0是运行完成，1是运行错误，2是配置／结构拒绝。它们保持0/0/0/1/2的门禁形式，但旧工具调用与Runner持有动作段已经退出当前协议，不冒称是完全相同的五组旧语义。物理步和原生动作数始终分开记录。</p>'
        + '<p>轻量Python 3.13.5与原生Python 3.10.21环境分别执行相同的495项检查及五组CLI，均通过；不是七套原生依赖环境全部重跑。</p>'
        + '<p>' + link(args.baseline,'Python 3.13软件／CLI回执') + ' · ' + link(args.baseline.parent/'unittest.txt','Python 3.13完整测试日志') + ' · ' + link(args.secondary_baseline,'Python 3.10软件／CLI回执') + ' · ' + link(args.secondary_baseline.parent/'unittest.txt','Python 3.10完整测试日志') + '</p>'))
    pieces.append(section('saved', '保存真实数据的转换对照',
        '<p>读取既有真实运行的前四步：11条链路，每条5份观测和4条动作，合计55份观测、44条动作。逐项核对模型私有输入的传感器／状态值、数值类型、采集时间、任务指令，以及公共动作重新进入环境后的执行值、精度和取图标记。</p>'
        + table(['既有链路', '观测／动作', '结果'], [[escape(r['case']), str(r['observations'])+' / '+str(r['predictions']), '输入与动作边界数值完全一致'] for r in mapping['cases']])
        + '<p>上述是保存真实数据的映射检查，没有新推理或新物理执行。' + str(len(mapping['protected_sha256'])) + '份被读历史文件在前后核验摘要不变；当前源码摘要与提交也单独绑定。原生函数本身没有重新实现，仍复用固定来源组件。这个检查不能证明独立重新推理产生相同动作，更不能证明整段任务轨迹一致。</p>'
        + '<p>' + link(args.mapping,'逐条检查、源码与历史摘要回执') + '</p>'))
    real_rows = []
    for row in real['cases']:
        target = Path(row.get('output_directory', args.real.parent/row['case']))
        summary_path = target/'framework/summary.json'
        stats, score = '未取得有效执行统计', '未取得评分'
        if summary_path.is_file():
            summary = read(summary_path)
            cfg = read(target/'config.json')
            result_path = target/'framework/episodes'/cfg['episodes'][0]['episode_id']/'result.json'
            value = read(result_path)
            st = value['execution_statistics']
            stats = escape(json.dumps(st,ensure_ascii=False))
            metric = value['metrics'].get('success',{})
            score = '官方未成功（仅四步，不能评估完整任务能力）' if metric.get('valid') and metric.get('value') is False else '官方成功' if metric.get('valid') and metric.get('value') is True else '评分不可用'
            stats += '<br>' + link(summary_path,'运行摘要') + ' · ' + link(result_path,'结果与计数')
        real_rows.append([escape(row['case']), '完成' if row['passed'] else '未通过，保留失败回执', stats, score])
    gpu = read(args.real.parent/'gpu-utilization.json')
    mean = gpu['aggregate_mean_percent']
    from datetime import datetime as dt
    times = [dt.fromisoformat(r['at_utc']) for r in gpu['samples']]
    max_gap = max((b-a).total_seconds() for a,b in zip(times,times[1:]))
    pieces.append(section('runtime', '本轮新真实短跑',
        '<p>每条最多4动作，使用既有匹配权重、归一化、原生控制器和任务初始化；四步是迁移诊断预算，不是更换完整任务评分协议。与保存数据映射检查分开记录，不用任务成功替代转换正确。</p>'
        + table(['链路', '完成情况', '实际统计', '官方任务结果'], real_rows)
        + '<p>本批监督期间记录两个不重叠GPU采样窗口，共' + str(len(times)) + '次，全卡样本平均为' + (format(mean,'.2f')+'%' if mean is not None else '未知') + '，目标20%；最大相邻采样间隔约' + format(max_gap,'.2f') + '秒。这里是样本平均，不是无间断每秒均值或整个项目生命周期的保证。包含既有占卡负载，不能记作评测工作量或成绩。没有改全局代理、驱动或占卡程序。</p>'
        + '<p>' + link(args.real,'本批真实运行监督回执') + ' · ' + link(args.real.parent/'gpu-utilization.json','GPU实际采样') + '</p>'))
    pieces.append(section('attempts', '启动工具的问题也保留记录',
        '<p>首个顺序监督器的OpenWAM/LIBERO四步已完成；准备下一条时，端口重绑预检查收到“地址占用”，监督器以失败结束，原回执保留。后续各条用独立端口，不把这个工具失败当作模型／环境执行失败，也不将原失败覆盖成通过。</p>'
        + '<p>预检查补了端口回收选项，并以自己创建的本地TCP连接复现关闭后的普通重绑拒绝与可回收重绑成功。TCP是本地网络连接协议。该复现不倒推证明原事件的唯一内部原因；当前Adapter、权重、控制器与评分没有因此改变。</p>'
        + '<p>两条已完成的π0.5启动日志命中“共享库”关键词分类，但客户端四步和退出均成功。该分类只是保留的关键词提示，不等同于运行错误；未保存原始日志，也不据此唯一认定某个库缺失或擅自更换依赖。</p>'))
    pieces.append(section('use', '后续接入与运行怎么做',
        '<ol><li>固定新模型自己的源码、权重、归一化和输入输出示例；或固定新环境自己的版本、任务、控制器与评分。</li>'
        + '<li>在自己的Adapter中写清原生格式与公共字段的转换，声明输入路径、实际目标、单位、坐标和参考部位。已有公共键不能重新改成环境私有名。</li>'
        + '<li>只登记自己的ModelRegistration或BenchmarkRegistration，即模型／环境注册记录，不修改另一方Adapter或Runner。</li>'
        + '<li>先做不加载GPU的check，再核验资产、保存数据转换、有限步执行和独立原生对照；完整任务和多种子统计另做。</li></ol>'
        + '<pre>.venv/bin/python -m embodied_eval check --config examples/mn_synthetic_run.json\n.venv/bin/python -m embodied_eval run --config examples/mn_synthetic_run.json --output outputs/my-v03-run\n.venv/bin/python -m embodied_eval migrate --config /path/to/old.json --output outputs/migrated-v03.json</pre>'
        + '<p>自定义权重需要不同输入时，新增具名模型配置，不写Benchmark名称分支；环境控制差异使用自己的具名环境协议。详细步骤见' + link(ROOT/'docs/adapter-onboarding.md','更新后的接入指引') + '。</p>'))
    pieces.append(section('remaining', '仍然存在的缺口',
        '<ul><li>DreamZero仍无真实动作通过，等待后续对接两张每卡80GB的H100；届时提醒，不擅自改量化、图像或采样。此次只完成0.3结构与编码通路。</li>'
        + '<li>本轮没有宣布五套Benchmark全部任务、全部初始化和多种子成功率覆盖；仍受已批准具名任务协议限制。</li>'
        + '<li>π0.5独立重新推理差异、RoboTwin原生重新规划差异不会因统一字段而自动消失；原严格差异继续保留。</li>'
        + '<li>LIBERO工具参考点、DROID完整外参等没有核实的几何关系仍不能猜测补齐；旧含糊原生数组仍限制在同签名协议内。</li>'
        + '<li>没有新增DataRef外部载荷读取、硬超时、并发Runner、恢复、全身机器人、视频转动作规划器或MassWorld整合；LingBot-VA继续暂缓。</li>'
        + '<li>阶段4未正式接受，0.3契约未冻结；本报告是提交证据，不是替用户完成验收。</li></ul>'))
    pieces.append(section('version', '版本、证据和可恢复性',
        '<p>迁移前标签pre-v03-20261016指向57f566b。当前分支feat/public-v03；核心实施提交cdd1ddb；报告构建时源码提交为<code>' + escape(source) + '</code>。之后的文档／回执提交独立记录。</p>'
        + '<p>源码、配置、测试和文档进入本地Git；未设置远端、未上传。历史outputs、权重、环境、缓存不进入Git，旧路径与报告不回写。被移出的5份旧mock配置可从标签和测试夹具恢复；这是退出当前入口，不是销毁历史。</p>'
        + '<p>历史0.1／0.2脚本如需重跑，使用迁移前标签的独立检出目录。当前三类证据脚本是replay_v03_baseline.py（软件）、check_v03_saved_boundaries.py（保存数据）、run_v03_bounded.py（有限步真实运行）。报告服务仅监听实例本机回环地址，不发布公网；远程桌面仍需实际端口转发。</p>'))
    shell = (ROOT/'docs/handoff/report-shell-v03.html').read_text()
    path = args.output/'public-v03-migration-report.html'
    with path.open('x',encoding='utf-8') as stream:
        stream.write(shell.replace('{{BODY}}','\n'.join(pieces)))
    receipt = dict(kind='v03_migration_report_index_not_formal_acceptance', report=str(path.relative_to(ROOT)),
        source_commit_at_build=source, software_tests=count, CLI=codes, saved_real_observations=55, saved_real_actions=44,
        bounded_real_passed=passed, bounded_real_total=len(real['cases']), metadata_cells=77, structurally_compatible_cells=20,
        baseline=str(args.baseline), mapping=str(args.mapping), inventory=str(args.inventory), real=str(args.real),
        GPU_mean_percent=mean, stage4_accepted=False, contract_frozen=False, new_external_upload=False)
    with (args.output/'receipt.json').open('x') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    main()
