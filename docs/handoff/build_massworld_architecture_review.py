"""Build a source-backed, review-only MassWorld/EmbodiedEval comparison.

Reads the user-supplied archive without importing or running its code. Does not
modify either production package, run setup, download assets or start workloads.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from urllib.parse import quote
from urllib.request import ProxyHandler, build_opener
import zipfile

from serve_report import ROOT, catalog

EXPECTED_ARCHIVE_SHA256 = "e64747418982c8787645925b86f78f120024bea3fbe1dc84496e3ac80966034a"
ARCHIVE_ROOT = "MassWorld-main/"
BACKEND_BASE = "deaa6cf"
SELECTIONS = {
    "README.md": [(1, 30)],
    "docs/architecture/unified-evaluation-platform/architecture.md": [(1, 17), (88, 120), (214, 229)],
    "WorldEvalKit/docs/architecture.md": [(1, 112)],
    "WorldEvalKit/docs/phased-evaluation.md": [(1, 18)],
    "WorldEvalKit/docs/robotics-native.md": [(1, 47), (70, 115)],
    "WorldEvalKit/src/worldevalkit/contracts/artifacts.py": [(63, 99), (131, 172)],
    "WorldEvalKit/src/worldevalkit/contracts/cases.py": [(190, 229), (251, 265)],
    "WorldEvalKit/src/worldevalkit/contracts/models.py": [(12, 29), (210, 278)],
    "WorldEvalKit/src/worldevalkit/adapter.py": [(35, 90), (235, 260)],
    "WorldEvalKit/src/worldevalkit/registry.py": [(156, 202)],
    "WorldEvalKit/src/worldevalkit/planner.py": [(80, 127)],
    "WorldEvalKit/src/worldevalkit/episode.py": [(12, 105)],
    "WorldEvalKit/src/worldevalkit/contracts/plan.py": [(10, 38), (147, 164)],
    "WorldEvalKit/src/worldevalkit/benchmarks/_robotics.py": [(23, 35), (66, 114), (141, 146)],
    "WorldEvalKit/src/worldevalkit/models/_native_policy.py": [(12, 40)],
    "WorldEvalKit/src/worldevalkit/benchmarks/robodojo.py": [(68, 96)],
    "WorldEvalKit/src/worldevalkit/benchmarks/robotwin_hard.py": [(59, 80)],
    "WorldEvalKit/src/worldevalkit/benchmarks/droid_sim.py": [(48, 59), (174, 226)],
    "WorldEvalKit/src/worldevalkit/workers/droid_sim.py": [(26, 54), (126, 148)],
    "WorldEvalKit/src/worldevalkit/runtime.py": [(170, 227)],
    "WorldEvalKit/src/worldevalkit/execution.py": [(324, 339)],
    "WorldEvalKit/src/worldevalkit/stores.py": [(148, 177)],
    "WorldEvalKit/src/worldevalkit/resources/robotics/sources.json": [(1, 30)],
    "WorldEvalKit/pyproject.toml": [(1, 42)],
    "LICENSE": [(1, 5)], "WorldEvalKit/LICENSE": [(1, 4)],
}

BODY = r'''
<h1 id="report-title">MassWorld 与 EmbodiedEval 架构整合评审</h1>
<p class="lead">2026-10-16 · 以用户最新上传的 MassWorld 源码压缩包为准 · 只读源码评审，不是合并实现、性能测试或阶段验收。</p>
<section aria-labelledby="verdict">
<h2 id="verdict">结论与推荐方案</h2>
<div class="callout"><strong>可以整合，而且互补性较强；不建议把两个 Runner 和两种公共数据格式直接揉成一个。</strong>推荐由 MassWorld / WorldEvalKit 管整次评测任务，EmbodiedEval 作为机器人闭环执行后端，继续管理相机、机器人状态、动作、控制和逐步证据。后端指一个被上层调用的执行模块，不要求部署成独立网络服务。</div>
<p>这不是说两边现在已经兼容。源码确认了足够的扩展点，所以“分层对接”可行；具体转换正确性和真实运行仍须单独实施、测试。外层统一任务、身份和结果；内层按领域保留严格语义。视频生成、图像评分和机器人动作不用强迫共用一个浮点数组。</p>
<p>建议的最终形态可以放在同一个 Git 仓库，也可以先保留独立包。是否把文件物理搬到一起是管理选择，不是接口兼容的前提。第一步采用可独立安装的插件，避免大规模搬迁后才发现控制含义不一致。</p>
<pre aria-label="推荐的分层整合关系">MassWorld 统一入口与任务编排
├─ WorldEvalKit 现有离线链路
│    数据 → 图像／视频生成 → 官方或具名评分
├─ WorldEvalKit 现有原生任务链路
│    固定服务协议 → 原生整批运行 → 结果导入
└─ EmbodiedEval 机器人闭环后端（新增桥接）
     模型 Adapter ↔ 类型化公共观测／动作 ↔ 环境 Adapter
     仅这一层负责逐步推进仿真

上面三条共用：运行身份、来源、文件索引、覆盖与结果展示
不能共用成一条：任意动作解释、评分规则、自动恢复语义</pre>
<p>Adapter 是“把某一方原生格式翻译成公共格式”的组件。Runner 是调用与计数循环。这里“共用上层”不意味着两套循环同时发动作：<strong>一次机器人运行只能有一个实际推进仿真的循环</strong>。</p>
</section>

<section aria-labelledby="basis">
<h2 id="basis">审查依据与版本边界</h2>
<div class="table-wrap"><table><thead><tr><th>对象</th><th>本次依据</th><th>不能据此声称什么</th></tr></thead><tbody>
<tr><td>MassWorld 最新附件</td><td>@@ARCHIVE_NAME@@；归档注释标识 <code>0425da26bc79b6dcc5debd40d904cc063950aaec</code>；292 个文件，206 个 Python 文件。</td><td>归档注释不是在线提交签名验证；不声称已取到 GitHub 当前分支。旧压缩包不作为本次结论依据。</td></tr>
<tr><td>附件内容指纹</td><td><code>@@ARCHIVE_SHA@@</code></td><td>SHA256 是文件内容指纹，只证明讨论使用同一份压缩包，不是运行结果。</td></tr>
<tr><td>在线仓库</td><td>本轮公开访问返回 404，匿名 Git 读取未成功；后续以用户提供附件完成检查。</td><td>不能据此判断仓库一定删除或一定私有；没有要求提供或保存访问密钥。</td></tr>
<tr><td>EmbodiedEval</td><td>评审基线 <code>deaa6cf</code>；当前运行通路为 0.1／0.2。统一公共字段的 0.3 仍是待用户确认的提案。</td><td>不能把上一份字段方案当作已获准实施或已冻结契约。</td></tr>
<tr><td>本轮检查</td><td>对最新附件做路径、软链接、文件摘要和 Python 语法树检查；206 份 Python 源文件均可解析。比对现有 133 份生产、示例、测试和契约文件未变。</td><td>语法树检查不执行代码。静态发现的 377 个 test_ 方法没有在本轮运行，不能写成“377 项测试通过”。</td></tr>
</tbody></table></div>
<p>源码和文件摘要见<a href="source-index.json">来源索引</a>；关键实现原文及行号见<a href="code-evidence.json">代码证据节选</a>。所有关于“已实现”的判断以最新附件代码为据；架构草案只作为目标设计，不冒充现有功能。</p>
</section>

<section aria-labelledby="comparison">
<h2 id="comparison">两边分别擅长什么</h2>
<div class="table-wrap"><table><thead><tr><th>方面</th><th>MassWorld 最新代码</th><th>我们当前代码</th><th>合并取舍</th></tr></thead><tbody>
<tr><td>总体范围</td><td>WorldEvalKit 已覆盖离线生成／评分、异步生成、部分机器人闭环及原生批任务。</td><td>聚焦能输出机器人动作的模型与仿真环境，强调转换和执行证据。</td><td>让 WorldEvalKit 做广域任务框架，机器人细节继续由专门后端管理。</td></tr>
<tr><td>扩展方式</td><td>ModelAdapter／BenchmarkAdapter 类，自动发现与外部插件 entry point（包声明的扩展入口）。</td><td>模型、环境独立注册；声明、工厂、资产核验、来源分别提供。</td><td>可直接利用外部插件机制建立桥，不为每个模型×环境写包装器。</td></tr>
<tr><td>通用数据</td><td>Artifact 代表有类型名的内联数据或文件；Port 代表具名输入输出口；Case 区分公开 inputs 与私有 targets。</td><td>Observation、StateSample、Action 等显式规定单位、坐标、采集时间、动作目标和精度。</td><td>Artifact 当外层容器，机器人类型化对象当内部载荷；通用容器不代替物理语义。</td></tr>
<tr><td>匹配</td><td>Planner 依 protocol、semantic、类型、单／多项和 constraints 自动匹配；多候选会拒绝猜测。</td><td>0.2 仍依赖显式绑定，同时做机器人结构匹配；0.3 提案拟统一键名并取消普通成对绑定。</td><td>保留其自动语义匹配，补入我们的机器人结构、参考点、完整目标和时间校验。</td></tr>
<tr><td>执行调度</td><td>Stage Engine（阶段执行器）已有阶段依赖图、缓存、重试、环境准备及运行记录。</td><td>串行闭环，原生动作数／物理步数／仿真时间分别记账；不提供通用恢复和并发。</td><td>外层管整次工作；内层管每一步。不能自动继承外层重试／恢复作为机器人新能力。</td></tr>
<tr><td>离线交接</td><td>infer／judge 已实现生成与评分分机、清单及文件校验；明确拒绝自定义阶段图和闭环任务。</td><td>有实际公开输入、执行记录、原生转换对照和比较视频；非通用 infer／judge。</td><td>机器人运行先走 run + 自定义阶段，不强套离线分阶段命令。</td></tr>
<tr><td>运行环境</td><td>已有 core、Conda、venv、Docker 等执行路径和 setup.lock。通用 remote 枚举存在，但 RuntimeExecutor 的 remote 分支会拒绝，远端调用另由模型处理。</td><td>固定资产核验、独立模型服务／部分专用子进程及轻量驱动环境。</td><td>保留隔离，不合成一个巨型 Python／CUDA 环境；不把声明的 remote 当完整远程作业后端。</td></tr>
<tr><td>大平台能力</td><td>qzcli 作业提供者、部署注册与统一控制平面出现在目标架构文档；在本次 src 中未发现对应完整实现。</td><td>没有这套云端控制平面。</td><td>可以对齐设计，但不能把接入 qzcli 当作复制现成功能。qzcli 是文档计划复用的算力命令行工具，本轮未检查其实际能力。</td></tr>
</tbody></table></div>
<p>代码依据：<code>registry.py</code>、<code>planner.py:80–127</code>、<code>contracts/artifacts.py</code>、<code>episode.py</code>、<code>runtime.py:170–227</code>、<code>docs/phased-evaluation.md:1–18</code>；完整路径及原文在<a href="code-evidence.json">节选</a>中。</p>
</section>

<section aria-labelledby="conflicts">
<h2 id="conflicts">不能直接合并的具体原因</h2>
<h3 id="physical-meaning">类型名相同，不保证物理含义相同</h3>
<p>对方的 Artifact.type_id 是开放字符串，例如 robot.joint_position_action。它适合标记“这是什么数据”，但不自动说明 8 个数中哪个是夹爪、关节顺序是什么、夹爪 1 是开还是关、四元数是 xyzw 还是 wxyz、状态是测量还是控制目标。现有 DROID 输入甚至仍含 external_cam 等环境原名；这与我们正在补齐的统一命名是同类缺口。</p>
<p>解决办法不是废掉 Artifact，而是为机器人载荷增加具名、带版本的强约束规范。普通视频继续是 core.video；机器人观测／动作使用专门类型和校验器。模型内部的 55／80 维编码不进入这个规范。跨坐标转换需要真实变换和工具偏移，不能靠 semantic 字符串自动完成。</p>
<p>代码依据：<code>contracts/artifacts.py:131–172</code>、<code>workers/droid_sim.py:26–54</code>。Artifact 可以承载 JSON Schema（数据结构规则），框架也支持 validators（额外校验函数），因此无需为每个机器人类型修改核心枚举；但必须把校验接到实际模型／执行边界，不能只给类型起名字。</p>

<h3 id="native-harness">它的原生整批入口不是通用公共动作接口</h3>
<p>RoboDojo 的原生任务要求 <code>native_harness=xpolicylab</code>；RoboTwin hard 要求 <code>native_harness=starvla-batched</code>。native_harness 在这里是“原生评测程序要求的服务协议”，不是机器人公共动作标准。NativePolicy.predict 还会明确报错，要求由 Benchmark 的任务阶段启动 service_spec 对应服务并运行原生整批程序。</p>
<p>这条路可以诚实保留，用于已有原生复现和大范围任务记录。它没有按模型名分支，边界有约束；但不能直接因此声称任意具身模型都能跨环境组合。要进入统一公共空间，必须给相应模型／环境各自补真正的数据边界转换，而不是在外面再套一个 Artifact。</p>
<p>另外，这里的 X-WAM、LaWAM 不等于我们的 OpenWAM；LingBot-World 视频模型不等于 LingBot-VLA 动作模型；Cosmos 视频生成配置不等于 Cosmos3-Nano-Policy-DROID。共享名称前缀不能作为资产或动作兼容依据。</p>

<h3 id="lifecycle">生命周期与计数不同</h3>
<p>它的通用 EpisodeRunner 主要是 reset → invoke → step，按 max_steps 计数，最后关闭环境。我们的模型接口还有 begin_episode、observe、predict、end_episode，动作带取图节奏，执行后分别返回原生动作数、物理步数、仿真秒数和状态是否已知。</p>
<p>因此不能机械地把 predict 改个函数名就迁过去。首版不替换我们的内循环；后续若要共用一个库，先补生命周期、执行反馈和计数接口。模型动作段缓存由模型持有，不能让上层再切一遍动作段。</p>

<h3 id="static-profile">静态类声明与具名权重配置不同</h3>
<p>对方的 ModelAdapter.spec／BenchmarkAdapter.spec 是类级声明；我们的能力随 io_profile 或环境 protocol 选项变化。例如 OpenWAM 单臂 OSC、双臂绝对位姿和 ARX 权重需要不同输入输出规则。</p>
<p>可用其已有多 Endpoint 机制表示不同具名规则，或从我们的元数据生成“一份具名配置一个插件声明”。Endpoint 指一个明确输入输出接口。生成时仍按模型配置和环境协议分别处理，不产生模型×环境的成对插件；注册信息必须与实际选择的权重一致，不能把未加载的配置也报为可运行。</p>

<h3 id="retry-cache">缓存、重试和恢复不能照搬</h3>
<p>Stage 默认允许缓存，原生任务可设置多次尝试；现有机器人循环没有完整环境、模型缓存和随机状态恢复协议。已经发出去但回执丢失的动作，如果重试，可能实际执行两次。缓存命中也不是“刚刚又跑了一遍模型”。</p>
<p>桥接首版明确设为 <code>Cache(enabled=False)</code>、<code>Retry(attempts=1)</code>：不缓存运行阶段，只尝试一次；不续跑中断回合。外层若终止进程，标记取消／执行状态未知并保留最后已确认计数，不补造结果。评分导入可以后续独立设计可复用缓存，但不能默认为机器人恢复功能。</p>
<p>权重身份也不能降级：其 <code>_robotics.file_identity</code> 对大于 16MiB 文件采用大小和修改时间，而我们的相关真实配置已有固定 revision／SHA 核验。整合时应把已核验摘要带入运行身份，不能用大小和时间替代；这不表示对方所有文件管理都没有内容摘要。</p>

<h3 id="metrics-difference">结果、任务范围与版本不能混为一谈</h3>
<p>它的 EvaluationResult.metrics 直接接受数值表；我们的 Metric 还包含 valid、unavailable_reason、单位和来源，且允许布尔值或 null。没有评分器、评分器失败、任务未成功是三种不同情况。最小桥接可只把有效数值用于展示，完整指标记录另存为产物；不可用项不能转成 0。</p>
<p>两边叫 RoboTwin 的并不一定是同一协议：对方 hard 是 demo_randomized，我们当前主要真实入口是 demo_clean；对方 RoboDojo 固定源码 b8e5eed，我们为既有 ARX 入口固定的是 266130a。迁移要分别命名，不偷换版本、难度、初始化或评分后再拼成绩。</p>
</section>

<section aria-labelledby="design">
<h2 id="design">推荐的整合方案</h2>
<h3 id="bridge-first">先接通整次任务，后统一机器人规范</h3>
<p>先做一个通用的 EmbodiedEval 执行插件，由 WorldEvalKit 的自定义 Stage 调用。Stage 是有明确输入、输出和失败状态的一段工作。它接收分别选定的模型配置、环境协议和回合配置，经我们的预检查后启动现有执行后端，再导入结果、视频和来源。这不是为每一对组合创建新 Adapter。</p>
<p>这一阶段可以继续调用现有 0.2 后端，但仍保留并显式记录旧绑定，不冒充公共键名已统一。随后在用户确认 0.3 字段方案后迁移公共命名，外层任务入口无需再改。</p>
<p>桥接运行必须使用当前已审定的任务、起点和预算。引入对方全量任务表不代表我们的单任务协议已自动扩展。对方原生整批链路可继续保留在独立协议下，与新的公共动作链路分别标记。</p>

<h3 id="shared-data">统一外层容器，保留强类型内部载荷</h3>
<p>下面的 <code>embodied.*</code> 类型名均是<strong>拟议的新插件类型</strong>，不是现有实现。点号表示类型的命名空间，用于区分不同领域；不是把两个包复制到同一个目录。</p>
<div class="table-wrap"><table><thead><tr><th>拟议外层产物</th><th>内部内容</th><th>谁负责正确性</th></tr></thead><tbody>
<tr><td><code>embodied.observation</code></td><td>带版本的公共观测：相机、状态、时间、坐标、测量来源。</td><td>环境 Adapter 产出；机器人校验器验证；模型 Adapter 只取公共字段。</td></tr>
<tr><td><code>embodied.action</code></td><td>关节／绝对位姿／相对位姿／夹爪等显式命令，含目标身份及单位。</td><td>模型 Adapter 解码；环境 Adapter 转成原生控制并执行。</td></tr>
<tr><td><code>embodied.space_spec</code></td><td>字段说明、机器人、坐标关系、允许执行方式、必需目标、原生协议签名。</td><td>环境独立声明；深层匹配委托机器人预检查，不只比 Artifact 类型名。</td></tr>
<tr><td><code>embodied.episode_result</code></td><td>执行状态、真实计数、完整 Metric、错误及真实／合成／有限步标记。</td><td>后端原始结果为事实源；上层只投影显示，不重解释成功条件。</td></tr>
<tr><td><code>core.video</code> 与证据文件</td><td>比较视频、原生对照、轨迹和摘要，带内容指纹与访问级别。</td><td>现有产物存储管理文件；私有初始化和评分信息不能跟随公开模型输入传播。</td></tr>
</tbody></table></div>
<p>运行阶段先使用后端现有内联载荷。把完成后的文件交给 Artifact Store，不等于实现了 DataRef 读取器；不要因此打开尚未定义的远程图片／共享内存读取路径。文件容器与机器人实时输入通道是两层不同能力。</p>

<h3 id="identity-and-score">身份与指标只维护一条可追溯链</h3>
<p>建立外层 run_id 与后端 run_id 的明确映射，再关联 episode_id、observation_id、request_id、command_id 和原生来源。它们依次表示运行、回合、观测、预测请求和具体命令的身份。文件摘要、协议版本、权重／统计身份及已知的校验状态全部保留。</p>
<p>WorldEvalKit 当前 EvidenceEvent 可以在 data 中附加这些关联字段，因此初版不必马上扩核心事件字段或建设数据库；后续再与其目标事件规范对齐。动作执行证据不能抽样丢弃，运维进度显示可以与之分开。</p>
<p>有效的数值指标进入外层 metrics；布尔成功值保留在逐回合原始指标中，汇总成功率时明确分母。无评分器则外层留空并标明原因；执行异常则运行失败或状态未知。框架执行完成、映射正确、任务成功、正式可比较四个结论分开，不生成跨 Benchmark 的统一总分。</p>
</section>

<section aria-labelledby="changes">
<h2 id="changes">双方具体需要改什么</h2>
<div class="table-wrap"><table><thead><tr><th>位置</th><th>建议改动</th><th>首轮是否必须</th></tr></thead><tbody>
<tr><td>我们：独立插件包</td><td>提供一个通用 Stage handler（阶段处理函数）、模型配置声明桥、环境协议声明桥、结果导入器。可经对方 entry point 注册，不改它的全局分派。</td><td>必须。它是对接代码，不是成对 Adapter。</td></tr>
<tr><td>我们：公共字段</td><td>按待确认 0.3 方案规范 head／wrist、实测／目标、xyzw、关节身份和参考点。保留原生编码在模型内部，维持旧版行为。</td><td>完整统一目标必须；最小任务桥可先用 0.2，但必须声明仍有旧绑定。</td></tr>
<tr><td>我们：预检查与资产</td><td>输出可供上层消费的结构检查、资产就绪、服务身份、训练覆盖警告，保留完整固定来源。</td><td>必须对接；不需要把资产核验移入 Runner。</td></tr>
<tr><td>我们：结果与证据</td><td>对外导出完整指标、确认计数、错误、视频索引及严格原生对照状态；私有 trace 单独保存。</td><td>必须；不能只导出成功率。</td></tr>
<tr><td>对方：插件入口与 Stage</td><td>利用已存在 build_plan／handlers／外部插件入口，新增机器人后端桥；明确不调用通用 EpisodeRunner 再套一层仿真。</td><td>需要新增插件，第一版预计无需大改核心，仍须验证真正接通。</td></tr>
<tr><td>对方：机器人协议规范</td><td>增加具名版本化公共观测／动作端口及真实边界校验；不再把 robot.droid_observation 这样的特定环境结构当跨环境标准。</td><td>机器人公共空间目标必须；不影响离线图像／视频模型。</td></tr>
<tr><td>对方：已有原生整批入口</td><td>保留 xpolicylab／starvla-batched，标明原生服务协议限制。若要转为公共动作接口，再分别拆模型和环境；不直接改掉原生数值和评分。</td><td>首轮无需重写；严格隔离即可。</td></tr>
<tr><td>对方：核心 Metric／Episode 类型</td><td>后续可原生支持有效性、来源和分项计数；首轮通过完整结果 Artifact 和 metadata 兼容，暂不破坏旧指标类型。</td><td>首轮可不改；长期共用闭环库时再评审。</td></tr>
<tr><td>双方：版本与依赖</td><td>包名、公共契约版本、环境协议版本分开；各运行环境继续隔离。同名 Benchmark 用命名空间区分协议；大型资产保留摘要，不靠路径／修改时间替代。</td><td>必须。</td></tr>
<tr><td>双方：来源与许可材料</td><td>快照根目录为 MIT，WorldEvalKit 子目录为 Apache-2.0，另有第三方来源和许可说明。文件搬迁／发布前梳理权属及保留材料；本项目当前未发现独立 LICENSE 文件。</td><td>代码搬迁／发布前确认；本轮仅记录快照事实，不给出许可兼容法律结论。</td></tr>
</tbody></table></div>
<p>整个工作仍在独立项目和插件内进行；不把我们的实现挪到相邻 XPolicyLab 工作目录，也不修改只读上游。若未来接入 X-WAM，固定来源的 XPolicyLab 依赖只是模型运行依赖，不应变成框架核心。</p>
</section>

<section aria-labelledby="alternatives">
<h2 id="alternatives">三种路线怎么选</h2>
<div class="table-wrap"><table><thead><tr><th>路线</th><th>优点</th><th>代价／风险</th><th>建议</th></tr></thead><tbody>
<tr><td>分层插件整合</td><td>很快形成一个任务入口；保留现有数据契约、转换和证据。外层复用对方成熟部分。</td><td>要做清楚结果、身份和声明桥，短期有两个包。</td><td><strong>推荐。</strong>两个包不是两套同时执行的控制循环。</td></tr>
<tr><td>把机器人能力逐步并入 WorldEvalKit</td><td>长期可以同仓、同发版，把 EmbodiedEval 作为 robotics 子包或可选依赖。</td><td>要稳定领域规范、生命周期、校验、结果类型；历史兼容和全部转换重验工作更多。</td><td>分层验证通过后再做，不要求一开始就搬源码。</td></tr>
<tr><td>让我们吞下视频生成／通用编排</td><td>保留当前包作为唯一入口。</td><td>会重复建设对方已有的环境准备、异步生成、评分交接、阶段执行和产物管理。</td><td>只有明确不采用 MassWorld 的长期产品方向时再选，当前没有看到足够理由。</td></tr>
</tbody></table></div>
<p>不推荐的第四种做法是直接替换我们的 Runner、把所有命令降成 Artifact.data 原生数组，然后宣布统一。这会丢掉刚刚核实的机器人语义，也会把旧四元数、夹爪和时间问题重新带回来。</p>
</section>

<section aria-labelledby="rollout">
<h2 id="rollout">建议实施顺序与验收</h2>
<ol>
<li><strong>确认架构方向及接口影响。</strong>选择“MassWorld 上层 + 机器人专门后端”，连同待确认公共字段提案一并评审。0.3 尚未实施，不自动冻结。</li>
<li><strong>最小任务桥，只做合成验证。</strong>分别注册两模型和两环境，验证四种交叉组合；新增第三个组件不改另一侧。证明没有成对工厂、重复启动模型或双重推进仿真。</li>
<li><strong>结果与失败先验收。</strong>测试无评分器、评分失败、任务失败、动作状态未知、缓存关闭、重试关闭、私有目标不泄漏、ID 与文件摘要完整。</li>
<li><strong>完成公共字段迁移及离线转换。</strong>逐具名模型配置核验保存的原生输入／输出；保留双臂、夹爪、精度、缓存、采样时间和固定权重。名称统一不是转换已验收。</li>
<li><strong>有限步真实对照。</strong>从资产已就绪且离线门禁通过的链路开始；先比较映射，再比较新推理、规划／执行和视频，最后看任务成功。DreamZero 仍等待用户安排后续硬件。</li>
<li><strong>再扩任务覆盖和物理合仓。</strong>任务范围、起点、难度、预算、聚合与评分分别提交确认；整合后继续保留离线、原生批任务、公共动作链路的不同标记。</li>
</ol>
<p>回归要求保留我们已完成的 443 项测试与五组 CLI 退出码 0／0／0／1／2；对方也要在独立环境跑其实际测试，不能把本轮静态发现的 377 个测试方法当已通过。CLI 是命令行入口；这里的退出码包含正常、评分错误和预检查拒绝用例，不是五个真实 Benchmark 成绩。</p>
<p>本次只做源码评审，没有跑双方真实评测、启动付费 API、安装依赖、下载权重或改占卡负载。对方文档记载的全量复现成绩属于其历史声明，本轮没有取得完整原始证据并重验，不能加入我们的验收账本。</p>
<div class="callout"><strong>最值得确认的不是“要不要把两个文件夹拼起来”，而是：</strong>是否以 MassWorld 为统一任务入口，以我们这套强类型机器人契约和独立 Adapter 为闭环底座，先做无损任务桥，再逐步统一公共字段。确认前，本轮不修改任一方的生产实现。</div>
</section>

<section aria-labelledby="evidence">
<h2 id="evidence">证据与继续阅读</h2>
<ul>
<li><a href="source-index.json">最新快照来源索引</a>：压缩包摘要、归档自述版本、全部 292 份文件摘要、206 份 Python 的静态解析及我方文件保留检查。</li>
<li><a href="code-evidence.json">关键实现证据</a>：包含原始路径、行号、文件摘要和代码节选；没有执行其中代码。</li>
<li><a href="../../docs/public-space-fields-review-20261016.html">待确认的公共字段完整评审稿</a>；<a href="../../docs/contract-v0.2.md">当前 0.2 契约</a>。</li>
<li><a href="../../outputs/public-space-review-20261016-run1/verification.json">上一轮 443 项测试和 CLI 回执索引</a>：历史事实，不冒充本次重跑。</li>
</ul>
<p class="muted">报告按自包含 HTML 保存；预览只绑定实例本机，远程需端口转发。未上传仓库、未保存密钥、未修改历史报告或真实结果。方案中的新协议、类型、插件和共仓结构均为建议，不是已具备能力。</p>
</section>
'''


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def build(archive, output):
    output.resolve().relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    archive_sha = sha(archive.read_bytes())
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("review is pinned to the user's latest supplied archive")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        assert len({x.filename for x in entries}) == len(entries)
        assert sum(x.file_size for x in entries) < 50_000_000
        file_index, parse_errors, syntax_count, test_methods = {}, [], 0, 0
        for entry in entries:
            name = PurePosixPath(entry.filename)
            assert not name.is_absolute() and ".." not in name.parts
            assert not stat.S_ISLNK(entry.external_attr >> 16)
            if entry.is_dir():
                continue
            assert entry.filename.startswith(ARCHIVE_ROOT)
            local = entry.filename.removeprefix(ARCHIVE_ROOT)
            payload = bundle.read(entry)
            file_index[local] = {"sha256": sha(payload), "bytes": len(payload)}
            if local.endswith(".py"):
                try:
                    tree = ast.parse(payload.decode("utf-8"))
                    syntax_count += 1
                    if "/tests/" in local:
                        test_methods += sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and node.name.startswith("test_") for node in ast.walk(tree))
                except SyntaxError as error:
                    parse_errors.append({"path": local, "line": error.lineno, "message": error.msg})
        assert len(file_index) == 292 and syntax_count == 206 and not parse_errors
        assert test_methods == 377
        evidence = []
        for local, ranges in SELECTIONS.items():
            payload = bundle.read(ARCHIVE_ROOT + local)
            lines = payload.decode("utf-8").splitlines()
            for start, end in ranges:
                evidence.append({"path": local, "file_sha256": sha(payload), "line_start": start,
                                 "line_end": min(end, len(lines)),
                                 "text": "\n".join(lines[start - 1:end])})
        revision = bundle.comment.decode("utf-8")
    previous = json.loads((ROOT / "outputs/public-space-review-20261016-run1/verification.json").read_text())
    preserved = previous["preserved_production_example_test_and_contract_files"]
    assert len(preserved) == 133
    assert all(sha((ROOT / name).read_bytes()) == value for name, value in preserved.items())
    our_paths = ("embodied_eval/mn/contracts.py", "embodied_eval/mn/interfaces.py",
                 "embodied_eval/mn/runner.py", "embodied_eval/mn/preflight.py",
                 "embodied_eval/mn/registry.py", "docs/public-space-fields-review-20261016.html",
                 "docs/stage-state.json")
    source = {"kind": "readonly_architecture_review_not_integration_or_runtime_validation",
              "at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(archive),
              "archive_sha256": archive_sha, "archive_self_reported_revision": revision,
              "revision_verified_against_online_git": False,
              "archive_file_count": len(file_index), "archive_files": file_index,
              "python_files_syntax_parsed": syntax_count, "syntax_failures": parse_errors,
              "static_test_method_count_NOT_executed": test_methods,
              "upstream_code_imported_or_executed": False,
              "our_review_base": BACKEND_BASE,
              "our_source_sha256": {name: sha((ROOT / name).read_bytes()) for name in our_paths},
              "our_production_example_test_contract_files_preserved": preserved,
              "our_baseline_actual_tests_in_previous_turn": 443,
              "our_baseline_five_CLI_exit_codes_in_previous_turn": [0, 0, 0, 1, 2],
              "tests_rerun_this_review": False, "new_inference_or_simulation": False,
              "public_contract_or_protocol_or_production_code_changed": False,
              "integration_implementation_started": False,
              "proposed_0_3_approved_or_implemented": False}
    dump(output / "source-index.json", source)
    dump(output / "code-evidence.json", {"latest_archive_sha256": archive_sha, "excerpts": evidence})
    template = (ROOT / "docs/public-space-fields-review-20261016.html").read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", template, flags=re.S).group(1)
    body = BODY.replace("@@ARCHIVE_NAME@@", escape(archive.name)).replace("@@ARCHIVE_SHA@@", archive_sha)
    report = ("<!doctype html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
              "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
              "<title>MassWorld 与 EmbodiedEval 架构整合评审</title><style>" + css +
              "</style></head><body><div class=\"report-shell\"><main class=\"report-content\">" +
              body + "</main></div></body></html>\n")
    with (output / "MassWorld-EmbodiedEval-architecture-review.html").open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(json.dumps({"report": str(output / "MassWorld-EmbodiedEval-architecture-review.html"),
                      "archive_files": len(file_index), "Python_syntax_checked": syntax_count,
                      "production_files_preserved": len(preserved), "runtime_tests_executed": False}, ensure_ascii=False))


def check_http(output, port):
    report = output / "MassWorld-EmbodiedEval-architecture-review.html"
    route, routes, hashes = catalog(report)
    opener = build_opener(ProxyHandler({}))
    checks = []
    for path, (payload, _) in routes.items():
        with opener.open(f"http://127.0.0.1:{port}" + quote(path, safe="/"), timeout=10) as response:
            assert response.status == 200 and response.read() == payload
            checks.append({"route": path, "status": 200, "sha256": hashes[path]})
    with opener.open(f"http://127.0.0.1:{port}/", timeout=10) as response:
        assert response.read() == routes[route][0]
    dump(output / "http-verification.json", {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": sha(report.read_bytes()), "HTTP_checks": checks,
        "proposal_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "listen_address": "127.0.0.1", "port": port, "user_desktop_forwarding_verified": False,
        "review_only_not_integration_acceptance": True})
    print(json.dumps({"HTTP_routes_verified": len(checks), "port": port}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-http-port", type=int)
    args = parser.parse_args()
    if args.verify_http_port:
        check_http(args.output, args.verify_http_port)
    else:
        if not args.archive:
            parser.error("--archive required when building")
        build(args.archive, args.output)


if __name__ == "__main__":
    main()
