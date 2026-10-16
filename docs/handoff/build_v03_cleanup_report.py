"""Build a static, source-backed internal-cleanup report; no historical rewrites."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.resolve().relative_to(ROOT / 'outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    index = json.loads(args.index.read_text())
    def evidence(key):
        return json.loads((ROOT/index[key]).read_text())
    bases = [evidence(k) for k in ('baseline', 'second_environment_baseline')]
    mapping, audit, package = (evidence(k) for k in ('saved_boundary', 'source_audit', 'package_check'))
    assert all(b['passed'] and b['checks'][0]['tests_run'] == 515 for b in bases)
    assert all([r['actual_exit_code'] for r in b['checks'][1:]] == [0, 0, 0, 1, 2] for b in bases)
    assert mapping['passed'] and len(mapping['cases']) == 11
    assert sum(r['observations'] for r in mapping['cases']) == 55
    assert sum(r['predictions'] for r in mapping['cases']) == 44
    assert audit['passed'] and package['passed']
    assert subprocess.run(['git', 'diff', '--quiet', index['source_commit'], '--', 'embodied_eval', 'tests', 'pyproject.toml'], cwd=ROOT).returncode == 0
    def link(path, title):
        return '<a href="' + escape(os.path.relpath(ROOT/path, args.output), quote=True) + '">' + escape(title) + '</a>'
    def section(key, title, body):
        return '<section aria-labelledby="'+key+'"><h2 id="'+key+'">'+title+'</h2>'+body+'</section>'
    def table(headers, rows):
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    parts = ['<h1 id="report-title">0.3 内部去旧化修改报告</h1>',
        '<p class="lead">2026-10-16 · 已实现内部依赖分层；公共协议、权重、数值处理、控制器及评分语义保持不变。本报告不代表阶段4验收。</p>']
    parts.append(section('conclusion', '先说结论',
        '<p>这次整理的是“代码归谁管”，不是重新实现一套机器人控制。此前公共类型夹在旧契约里、现用原生组件放在旧架构目录里；现在把它们分成清楚的公共基础层和原生组件层，0.3正常运行不再需要导入旧架构模块。</p>'
        '<div class="callout">两套Python各515项测试、零跳过；五组CLI退出码均为0/0/0/1/2。11条保存真实链路的55份观测、44条动作转换检查通过。这些是软件与保存数据验证，不是新增真实推理或任务成绩。</div>'
        '<p>CLI指命令行入口。515项由原495项加本轮20项组成，没有删除旧检查凑通过。新组件可以直接实现0.3接口，不必先写一份旧版Adapter。</p>'))
    parts.append(section('ownership', '目录与职责现在如何分开',
        '<pre>embodied_eval/\n  v03/          公共字段、双方公共边界、匹配、调度\n  core/         公共物理类型、校验、坐标数学、注册容器\n  native/       原生模型编码、缓存、执行封装和资产核验\n  adapters/     仍复用的环境加载器、控制工具及历史参考\n  mn/           旧架构参考与迁出组件的导入别名\n  _legacy_*     历史解码/测试兼容，不是当前运行入口</pre>'
        '<p>Adapter就是负责某一侧格式转换的适配器；Runner就是只管调用顺序、预算、计数和保存证据的调度器。公共层负责统一含义，原生层负责按固定来源完成具体处理。</p>'
        + table(['改动', '做了什么', '为什么'], [
            ['公共基础层', '31种原有物理数据类型迁到core/types.py，另将两边相同的Action合成一份定义。', '位姿、数组、关节命令不是旧版本专属，不应从历史契约取得。'],
            ['共同校验与数学', '抽出11个原有校验辅助函数、空间结构校验及坐标路径计算。', '直接校验当前结构，不为校验再造一份旧版空间。'],
            ['原生组件层', '15份已有原生组件代码迁入native/，另设原生独立注册表。', '已经验证过的模型处理、控制和评分有明确归属，不再挂在旧架构下面。'],
            ['历史兼容', '17个mn旧模块路径变成薄导入别名；_legacy_contracts同样转为别名。', '历史测试、旧对象解码和旧导入路径继续有依据，不维护第二份计算实现。'],
            ['接入指引', '明确新组件应使用v03接口和core基础类型。', '新接入不必沿着历史实现再包一层。']
        ])))
    parts.append(section('bridges', '具体去掉了哪些过渡转换',
        '<ol><li><p>模型输出动作时：以前把原生处理代码返回的动作字段逐项拷进另一种同形状对象。现在公共边界和原生处理共享同一个动作定义，直接保留动作对象、精度标记、夹爪命令和取图标记。</p></li>'
        '<li><p>环境接收动作时：以前做完坐标处理后，还要再重建一份私有动作对象。现在把已经处理好的动作直接交给执行器，不再重复装箱。需要的真实坐标转换仍然执行。</p></li>'
        '<li><p>校验公共空间时：以前为调用旧校验器，把整个0.3空间临时投影成0.1空间。现在共同的物理结构校验直接检查0.3空间，再检查公共命名、测量来源、关节顺序和参考部位。</p></li></ol>'
        '<p>以主视角为例：环境原生相机名 → 环境Adapter输出公共head → 模型Adapter读取head → 模型自己的图像处理。head指工作区总览角色，不保证实际安装在头上。用户仍不用为每个模型与环境组合填写字段对应关系。</p>'
        '<p>本轮没有把坐标变换、图像处理或机器人执行塞入Runner，也没有引入模型与环境的组合白名单。</p>'))
    audit_rows = [
        ['Python 3.13.5', '515项，零跳过；五组退出码0/0/0/1/2'],
        ['Python 3.10.21（robotwin-stage4既有环境）', '515项，零跳过；五组退出码0/0/0/1/2'],
        ['公共数据定义', f'{audit["public_v03_classes_fields_defaults_annotations_unchanged"]}种0.3公共记录的字段、默认值和类型声明保持一致'],
        ['原生计算与执行', '15份原生组件、9份加载器/规格及Runner、坐标数学逐项比较语法结构'],
        ['固定资产清单', '6份清单与整理前Git版本逐字节一致；没有更换权重或归一化'],
        ['历史证据', str(len(audit['protected_historical_files']))+'份文件按既有公开摘要重新核验，未变化'],
        ['安装包', str(package['packaged_python_files_exactly_match_current_source'])+'份Python源码与当前源码一致，6份原生资产清单齐全；解包后两套Python均能独立导入和运行合成用例'],
        ['原生子进程启动', '已有OpenWAM原生解释器实际启动新模块、传递私有上下文、返回诊断并确认关闭，退出码0；没有开始任务、加载权重或生成动作']
    ]
    parts.append(section('verification', '验证结果与证据边界', table(['检查', '实际结果'], audit_rows)
        + '<p>语法结构比较会忽略import导入语句和模块说明；子进程启动模块路径由mn改为native是明确允许的文字变化。AST指把Python源码解析后的语法结构；这种比较说明计算语句未被重写，但不能独自证明导入对象正确，因此同时做了完整回归、禁止旧模块导入检查、安装包隔离检查和实际子进程启动检查。</p>'
        '<p>新测试覆盖：当前入口不加载旧协议模块、旧Runner或组合工厂；共享动作身份与序列化；历史导入/对象路径兼容；错误坐标与四元数拒绝；原生清单完整性；公共校验不再创建旧空间。xyzw指朝向四元数的x、y、z在前、实数分量w在后。</p>'))
    parts.append(section('saved-data', '保存真实数据的逐链路检查',
        '<p>下面使用已有真实运行保存下来的数据，重新通过当前公共命名、交给模型原生编码器的私有输入样本、交给执行器的动作边界。比较样本数组、采集时间、动作数值、精度和取图标记；不重新调用模型或推进仿真。后续原生图像处理与动作解码代码另以源码审计核对，没有据此宣称重算了GPU推理输入或物理轨迹。</p>'
        + table(['已有链路记录', '观测', '动作', '结果'], [
            [escape(row['case']), str(row['observations']), str(row['predictions']), '私有输入样本与执行器入参精确一致'] for row in mapping['cases']])
        + '<p>合计11条链路、55份观测、44条动作。这里证明格式和物理动作映射没有因代码整理改变，不证明重新推理会逐位一致，更不能推出任务成功率。</p>'))
    parts.append(section('diagnostics', '本轮调试中发现并处理的问题',
        '<p>首轮完整回归没有直接通过：环境执行器仍要给旧加载器发送私有上下文与固定周期动作，不能把它们误认为纯物理基础类型。已明确改为引用native/environment_records，保留原字段和版本标记；随后两套完整回归通过。</p>'
        '<p>最初选到的另一套Python 3.10环境缺少jsonschema测试依赖，未擅自安装或修改；改用此前已有依赖的robotwin-stage4环境。资产复制完整性检查也发现过一份清单不完整，最终按整理前Git内容恢复，六份清单均重新通过逐字节核验。失败/中止的诊断目录保留，最终证据使用本报告所列目录。</p>'
        '<p>第一次保存数据检查启动后源码又有修正，因此主动中止，另建目录重跑，避免把不同源码时点的结果拼成一次通过。</p>'))
    parts.append(section('remaining', '还有哪些内容刻意保留',
        '<ul><li><p>native/contracts.py保留原生组件实际使用的冻结私有消息，其0.2标记不代表公开支持0.2配置；native/environment_records.py保留加载器内部的0.1消息。同一公共动作已经共享，其余私有上下文、状态和任务转换暂时保留。</p></li>'
        '<li><p>adapters目录没有清空：当前仍调用其中的环境加载器和原生工具，另外有历史包装供回归参考。迁出的旧模块保留别名，旧资产清单留作冻结参考；当前核验以native中的清单为准。</p></li>'
        '<li><p>原生复现差异、全任务/多种子覆盖、DreamZero双H10080GB真实推理和暂缓的LingBot-VA没有被本次整理解决。阶段4仍未正式接受，公共契约仍未冻结。</p></li></ul>'
        '<p>所以，本轮结论是“当前通路的公共基础与原生实现已分层，重复转换减少”，不是“所有历史代码删除”或“全部模型评测完成”。新Adapter不必依赖这些历史私有消息。</p>'
        '<p>本轮未下载大权重、未更改全局代理或上游源码、未修改占卡程序；没有新增集中GPU验证，因而不新增GPU平均利用率达标声明，也不把既有占卡计为评测工作。</p>'))
    parts.append(section('versioning', '版本管理与复跑方式',
        '<p>整理前标签：<code>pre-v03-internal-cleanup-20261016</code>，对应<code>15c2b2f</code>。本轮实现提交：<code>'+escape(index['source_commit'])+'</code>。旧报告、结果和Mac路径未回写，没有建立remote或上传源码。</p>'
        '<p>下列命令均在项目根目录执行。输出目录必须是新的，不覆盖已有结果：</p>'
        '<pre>.venv/bin/python docs/handoff/replay_v03_baseline.py \\\n  --output outputs/my-cleanup-baseline --expected-tests 515\n\n'
        '../upstream/envs/robotwin-stage4/bin/python docs/handoff/replay_v03_baseline.py \\\n  --output outputs/my-cleanup-py310 --expected-tests 515\n\n'
        '.venv/bin/python docs/handoff/check_v03_saved_boundaries.py \\\n  --output outputs/my-cleanup-saved-boundaries\n\n'
        '.venv/bin/python docs/handoff/check_v03_internal_cleanup.py \\\n  --output outputs/my-cleanup-source-audit</pre>'))
    items = [(index[k], title) for k, title in (
        ('baseline', 'Python 3.13完整回归回执'), ('second_environment_baseline', 'Python 3.10完整回归回执'),
        ('saved_boundary', '11条保存真实链路边界回执'), ('source_audit', '源码、契约字段与历史摘要审计'),
        ('package_check', '安装包隔离检查与原生子进程启动回执'))]
    items += [(str(Path(index['baseline']).parent/'unittest.txt'), 'Python 3.13逐项测试日志'),
        (str(Path(index['second_environment_baseline']).parent/'unittest.txt'), 'Python 3.10逐项测试日志'),
        ('docs/adapter-onboarding.md', '更新后的接入指引'), ('docs/contract-v0.3.md', '0.3契约（仅源码归属文字更新）'),
        ('outputs/v03-delivery-20261016-final4/public-v03-migration-report.html', '此前0.3迁移报告，原件保留')]
    parts.append(section('evidence', '可检查的附件', '<ul>'+''.join('<li>'+link(path, title)+'</li>' for path, title in items)+'</ul>'
        '<p>本报告是独立新文件。预览服务只绑定实例回环地址并显式提供选定附件，不开放工作区目录；用户电脑访问远程实例仍需实际端口转发。</p>'))
    shell = (ROOT/'docs/handoff/report-shell-v03-cleanup.html').read_text()
    head, rest = shell.split('    <main class="report-content">', 1)
    _, tail = rest.split('    </main>', 1)
    html = head.replace('lang="en"', 'lang="zh-CN"').replace('<title>Report title</title>', '<title>EmbodiedEval 0.3内部去旧化修改报告</title>')
    html += '    <main class="report-content">\n'+ '\n'.join(parts) +'\n    </main>'+tail
    target = args.output/'public-v03-internal-cleanup-report.html'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(html)
    receipt = dict(kind='v03_internal_cleanup_report_not_stage_acceptance', passed=True,
        at_utc=datetime.now(timezone.utc).isoformat(), report=str(target.relative_to(ROOT)),
        source_commit=index['source_commit'], public_contract_version='0.3', public_wire_changed=False,
        tests_each_environment=515, test_skips=0, current_CLI=[0, 0, 0, 1, 2],
        saved_real_observations=55, saved_real_actions=44, new_inference=False, new_simulation=False,
        historical_files_unchanged=len(audit['protected_historical_files']), stage4_accepted=False, contract_frozen=False)
    with (args.output/'receipt.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
