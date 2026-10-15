# EmbodiedEval

独立的具身评测框架。阶段 1 契约和阶段 2 CLI / 串行执行循环已实现；阶段 3 的 OpenWAM×LIBERO 单次真实闭环及原生逐步对照已获工程验收。阶段4按“先完成有依据的适配器，再集中真实测试”推进，未定义协议不以占位实现冒充完成。

设计参考 XPolicyLab 的适配思路，运行时不导入 XPolicyLab。mock示例与单元测试是合成/接口检查；真实闭环结果另有明确来源与验收范围，不能混作官方全量成功率。

2026-10-07已按用户确认的方案新增0.2独立通路：M指模型数量，N指仿真Benchmark数量，M+N表示双方分别实现自己的Adapter，以显式接线配置组合，而非逐对新增Python适配器。只纳入直接输出机器人动作的模型和仿真Benchmark；旧0.1不改语义或历史证据。新增通路及集中验证仍须单独验收，契约未冻结。

## 0.2 独立接入

模型注册表与环境注册表分别管理自己的参数、资产和来源；匹配根据相机、状态含义/单位/维数、坐标、动作模式和完整目标，而不依据模型×环境名单或训练机械臂身份。训练覆盖未核实只给提示；缺输入、夹爪单位不符、双臂不完整或具名原生协议不匹配仍拒绝。

~~~sh
.venv/bin/python -m embodied_eval check --config examples/mn_synthetic_run.json
.venv/bin/python -m embodied_eval run --config examples/mn_synthetic_run.json --output outputs/my-mn-run
~~~

`check`是GPU-free预检查（不启动GPU推理或仿真）；`--verify-assets`可额外核验本地资产，不下载模型。运行资产和真实验证状态分开，不从结构通过推定能完成任务。新注册位于`embodied_eval/mn/registry.py`和`native_catalog.py`，观测/动作契约位于`mn/contracts.py`，双方独立接口位于`mn/interfaces.py`。

六组既有入口已拆成3个模型家族、6个具名权重输入输出配置，以及3个仿真家族、5个具名环境协议；不同权重与环境协议独立选择。配置翻译脚本`docs/handoff/check_mn_preparation.py`只生成新配置，运行时不包装旧组合工厂。公共状态明确区分实测值和控制目标，动作保留关节/绝对与相对位姿/夹爪/速度类型；环境只执行自己已声明的类型。无法证明物理含义的旧原生数组仍只在同签名协议内流转。

公共契约细节、当前模式限制和独立插件接入方法见[0.2契约](docs/contract-v0.2.md)。首轮新增61项合成架构/边界检查与原202项合计263项；10月7日扩展时306项，10月8日最新376项。四种交叉组合和第三组件扩展另保存合成证据。旧五组CLI退出码仍为0/0/0/1/2。新真实结果不替换下文的历史阶段4数字。

本轮0.2交付见[独立架构与验证报告](outputs/mn-delivery-20261007-final/M+N-architecture-and-evidence.html)、[新覆盖账本](outputs/mn-delivery-20261007-final/coverage-ledger.json)与[六组实际试验索引](outputs/mn-real-index-20261007-final/receipt.json)。六组固定单次真实试验均成功，5组记录动作原生重放精确通过；OpenWAM/RoboTwin仍有逐步轨迹差异。总体实现与验证门禁分开，不能将单次成功或合成架构检查当作正式验收。

真实0.2配置示例在[本轮独立配置及资产预检查](outputs/mn-preparation-20261007-final-verified/receipt.json)所列目录：例如`openpi_vlabench.json`分别选择模型`openpi/state7_absolute`和环境`vlabench/frozen-track1`，再声明相机、状态和末端/夹爪接线。该目录的端口只是诊断配置，不表示服务仍在运行。具名夹爪原生扩展保留RoboTwin“先插值、后逐步截断”的请求含义；公共开度仍严格限定0到1。

## 当前交付

### 2026-10-08最新：本轮验收提交材料

七套独立环境各376项软件/合成检查通过、零跳过，CLI退出码0/0/0/1/2无回归。新真实运行：OpenWAM/RoboDojo实做319动作、3190物理步，π0.5/RoboDojo280动作、2800物理步，均原生任务结束且官方成功、分数100；各只有一个固定stack_bowls任务与初始场景，不是全suite成功率。π0.5/RoboTwin达到固定400动作上限、12967物理步，框架错误0、官方任务未成功；不能把运行完成说成任务成功。

π0.5两条新长程运行的全部原生请求、整段生成动作、实际缓存顺序，以及三条新运行的逐条回执/计数均已只读核对。OpenWAM另有保存真实观测的32行完整映射证明，本次319动作没有捕获全程原生生成段，不宣称全程模型输出逐行对照。同次DROID/RoboDojo原始观测与公共观测映射精确相同；独立重开环境的像素严格复现失败仍保留。π0.5跨进程重新推理、OpenWAM/RoboTwin历史整段轨迹差异也没有被撤销，具体内部非确定性根因仍未唯一定位。

DreamZero主/辅助资产完整、双卡模型实际加载，但原生推理三次显存不足，仍没有动作或闭环通过。用户明确将两张每卡80GB的H100验证留待后续对接，届时提醒；不自动租卡、不改量化、图像或采样。LingBot-VA仍按用户要求暂缓；LingBot-VLA保留。49个原缺口逐项说明范围外、未定义协议、结构不满足和已验证进展，没有占位完成。

[当前验收报告](outputs/mn-acceptance-delivery-20261008-final/M+N-acceptance-preparation-and-evidence.html)、[验收清单](outputs/mn-acceptance-delivery-20261008-final/acceptance-checklist.json)、[新覆盖账本](outputs/mn-acceptance-delivery-20261008-final/coverage-ledger.json)。网页仅在实例回环地址服务，远程电脑需实际端口转发；文件完整性、网页检查与源码封存有单独回执，均不是用户已通过阶段4。阶段4待正式确认，契约未冻结，阶段5未开始。下方旧章节保留各自历史日期和数字，不作当前进度。

### 2026-10-07追加：暂缓VA，扩展其他具名组件

本轮新增Cosmos3-Nano-Policy-DROID和DreamZero-DROID模型边界、π0.5/OpenWAM的ARX输入输出配置，以及DROID sim_evals/RoboDojo仿真边界。现有5个真实模型家族、10份模型输入输出配置、5个仿真家族、7份具名协议。LingBot-VA按用户要求暂缓；已实现的LingBot-VLA保留。旧三家族/六入口交付是历史范围，不回写。

软件回归增加至306项（原263项＋43项）：五套既有环境及三个新增原生依赖环境中的软件诊断均零跳过，五组旧CLI仍0/0/0/1/2。9项合成输入的固定源函数对照通过，最新接线按各自原生客户端明确区分Cosmos与DreamZero的左右相机命名。π0.5 ARX匹配权重18/18核验，独立兼容环境在GPU2产生完整50条14维动作；这是合成观测上的真实推理，不是Benchmark成绩。另以50动作预算接入真实RoboTwin，第一轮50动作/1486物理步/约5.944秒，重复50动作/1523物理步/约6.092秒，两次有效success=false、框架无错误。首轮原生输入完全相同而首动作不同，原因未定位；有限步运行并非全任务验收或原生策略严格一致证明。

原训练依赖中NumPy低于2与LeRobot 0.4.4间接要求NumPy至少2互相冲突。兼容推理环境明确使用源码支持的旧LeRobot导入分支，固定0cf8648、rerun-sdk 0.21.0和源锁optax/chex；不宣称原训练环境已复现，不修改参考仓库或旧环境。GPU3原生CUDA非法访存失败单列，GPU2通过不能证明具体硬件故障原因。

新配置示例与资产/预检查由`docs/handoff/prepare_mn_extension.py`生成，仍是独立接线数据，不是运行时组合白名单。DROID没有原生任务成功评分；RoboDojo本轮只具名stack_bowls/ARX/layout0，场景引用、依赖和共享驱动兼容未全部核验。其余大权重仍须完整发布后校验。公共核心和旧0.1未改，不把结构通过当作所有组合已完成。

Cosmos、DreamZero与OpenWAM ARX的新Linux依赖环境已安装，DreamZero/OpenWAM原生模块导入通过；Cosmos仍有两处第三方包元数据不兼容及原生导入失败，不能称为运行就绪。DROID锁定为IsaacLab2.2.0/IsaacSim5.0.0.0：原flatdict4.0.1构建缺pkg_resources问题通过固定构建工具并核验原sdist摘要绕过，完整环境安装在直连及原代理路线均超时，仍在有界续装。安装器成功、软件测试通过、原生导入通过、真实模型/仿真运行是不同层次。

[本轮HTML报告与逐格说明](outputs/mn-extension-20261007-final2/M+N-adapter-extension-and-gaps.html)；[新覆盖账本](outputs/mn-extension-20261007-final2/coverage-ledger.json)；[最新说明与证据](docs/stage4-review.md)。报告预览仅监听实例回环地址，远程浏览需端口转发。

| 模块 | 用途 |
| --- | --- |
| embodied_eval/contracts.py | 带类型的数据结构，保留单位、坐标系、机器人身份和时间语义 |
| embodied_eval/interfaces.py | ModelAdapter / BenchmarkAdapter 的生命周期接口 |
| embodied_eval/codec.py | 严格 JSON 编解码；拒绝未知字段、重复键和非有限数值 |
| embodied_eval/compatibility.py | 按双方声明和版本化配置判断兼容性，返回阻塞原因 |
| embodied_eval/visibility.py | 公开任务、观测、反馈和空间定义的白名单副本 |
| embodied_eval/validation.py | 动作、张量、工具参数、执行反馈和结果的语义校验 |
| embodied_eval/geometry.py | 显式单位、位姿、坐标树和 quaternion / rot6d 转换 |
| embodied_eval/runner.py / __main__.py | 串行执行闭环、CLI 与错误收尾 |
| embodied_eval/config.py / artifacts.py | 配置、注册与本地产物 |
| embodied_eval/adapters/ | mock及六组完整具名真实注册入口；模型处理、原生控制与评分留在适配边界 |
| examples/contract_examples.json | 8 组可解码的决策实例及公开视图、工具反馈和结果 |
| embodied_eval/mn/ | 独立0.2契约、注册、接线、预检查、串行Runner、模型/仿真边界；不调用旧组合工厂 |
| tests/ | 当前376项软件/合成检查，保留原85项、202项及旧CLI行为；真实闭环证据另列 |

## 运行契约检查

需要 Python 3.10 或更高版本。轻量迁移基线用Python3.13.5；软件检查与五组旧CLI在轻量环境及三套独立Python3.10仿真侧环境复现，实际计数与跳过状态以新baseline-check.json及unittest.txt为准。基础环境未安装可选真实客户端依赖时可能跳过映射检查，应核对日志。工具参数和控制器参数使用 JSON Schema Draft 2020-12；轻量依赖锁定在 uv.lock，真实原生环境分别锁定，不能共用这份锁宣称完整模型可运行。

~~~sh
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m examples.validate_contracts
~~~

最后一条命令重新生成并校验合成 JSON 示例，不执行 episode 或模型推理。也可直接使用 .venv/bin/python 执行相同命令。

## 运行合成 episode

~~~sh
.venv/bin/python -m embodied_eval run --config examples/mock_chunk_run.json --output outputs/my-chunk-run
.venv/bin/python -m embodied_eval run --config examples/mock_tool_run.json --output outputs/my-tool-run
.venv/bin/python -m embodied_eval run --config examples/mock_cache_run.json --output outputs/my-cache-run
~~~

输出目录必须不存在；省略 --output 会生成唯一目录。CLI 退出码为 0（运行完成）、1（运行/评分异常）、2（配置/兼容性拒绝）。mock结果标记 synthetic=true（合成数据），具名真实配置的来源标记为false；该标记本身不证明服务权重身份或真实评测通过。任务 success 与运行完成分开。完整产物和限制见阶段 2 验收说明。

## 阶段4具名适配器

0.1的六组完整入口已注册：历史 OpenWAM×LIBERO，新增 π0.5×LIBERO、OpenWAM×VLABench、π0.5×VLABench、OpenWAM×RoboTwin、LingBot-VLA 2.0×RoboTwin。配置位于 `examples/stage3-openwam-libero-spatial-pilot.json` 和五份 `examples/stage4-*.json`；0.1仅允许其既有具名组合，新0.2不沿用该组合白名单。原表另外49格没有占位实现。原表LingBot-VA 2.0与此处VLA 2.0是不同模型，新覆盖账本单列纠正，不拿VLA证据抵扣VA缺口。全部具名专用权重、VLABench镜像与RoboTwin场景已完整校验；镜像不能冒充无官方摘要的网盘原包。

五组新增适配的真实固定单次闭环均已完成并成功：π0.5×LIBERO 103动作、OpenWAM×VLABench 139动作、π0.5×VLABench 165动作、OpenWAM×RoboTwin 113动作、LingBot×RoboTwin 111动作。四组独立记录动作重放精确通过，但π0.5×LIBERO和LingBot×RoboTwin的重新推理对照，以及OpenWAM×RoboTwin的原生轨迹重放仍有严格复现差异。任务成功、重放通过和完整策略一致不是同一结论。π0.5×VLABench旧300步未成功试验单列，不与原生浮点修正后的新试验混用；不从一次结果变化推断修正提高模型质量。当前实际结果和复现差异以 [阶段状态](docs/stage-state.json)、[阶段4证据](docs/stage4-review.md) 和 [架构与细节汇总](outputs/stage4-delivery-20261006T1435Z/EmbodiedEval-architecture.html) 为准；不是全suite成绩或正式阶段4验收。

路径针对当前Linux实例；固定模型服务与仿真依赖必须独立准备，不能用基础 `.venv` 或Mac `.venv` 直接宣称可运行。原生服务监督脚本在 `docs/handoff/`，会校验匹配资产、拒绝占用的本地端口，并只清理本次启动的服务器；其外部时间限制不是Runner的通用硬超时或恢复能力。

原55个组合的实现/缺口逐项记录见 [阶段4方案与证据](docs/stage4-review.md)。[已批准的可变时长核心与原始公开映射](docs/variable-transition-review.md)已实施，独立记录原生动作/物理步/实际仿真时间，保留固定模式，拒绝原生模式硬仿真时间预算。RoboTwin完整公开状态和原生取图节奏已实现并真实运行；驱动目标不当实测角度。物理计数证明与跨进程逐像素复现分别设门禁。硬超时、并发、自动恢复及DataRef文件载荷读取均未因此建设。

## 接入的边界

- 模型 Adapter 负责模型原生预处理、输出解析及模型内部状态。
- Benchmark Adapter 负责原生任务和观测转换、机器人控制、预算内执行及官方评分。
- 框架负责兼容性与消息校验、公开视图、生命周期编排和统计；当前串行执行，一个环境与一个模型实例同时只持有一个 episode。
- checkpoint_ref / normalization_ref / source_revision 是双方声明的身份。检查通过仅说明声明一致；真实权重、机器人效果和原生协议仍须在阶段 3 对照验证。
- 中间观测可以表达工具内部的每个环境步；工具执行中再次决策、在线抢占和恢复协议尚未建设。
- DataRef 已校验元数据格式，但文件读取、共享内存、远端传输及实际摘要验证尚未实现。
- 世界模型预测目前是数据封装。候选动作输入保留，当前校验会拒绝使用；规划器及世界模型评分须在阶段 5C 确认。

## 审阅入口

- [GPU 实例交接文档与迁移命令](docs/GPU实例交接文档.md)
- [数据契约与职责](docs/contract-v0.1.md)
- [0.2独立模型/环境契约与接线](docs/contract-v0.2.md)
- [阶段 1 验收说明](docs/stage1-acceptance.md)
- [阶段 2 实施前确认材料](docs/stage2-review.md)
- [阶段 2 验收说明与运行产物](docs/stage2-acceptance.md)
- [阶段 3 真实试点确认材料](docs/stage3-review.md)
- [阶段 4 实现方案和最新证据](docs/stage4-review.md)
- [RoboTwin已批准时间协议、公开映射与实施证据](docs/variable-transition-review.md)
- [项目架构、默认值、55格范围与实际证据汇总](outputs/stage4-delivery-20261006T1435Z/EmbodiedEval-architecture.html)
- [阶段状态](docs/stage-state.json)

按照用户约定：每个阶段先确认细节，再实施；完成后提交证据，经确认后进入下一阶段。当前代码可审阅，契约尚未标记为正式冻结。
