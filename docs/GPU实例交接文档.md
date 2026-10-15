# EmbodiedEval：GPU 实例交接文档

交接基线：2026-10-05（Asia/Shanghai）。本文件交接当前工作区中的源码、设计和验收证据，供 GPU 实例上的工程师或 Codex 继续工作。Linux/GPU 上的结果须重新取得；本文记录的通过结果来自本地 macOS。

## 1. 接手时先掌握的状态

项目是独立的 **EmbodiedEval**，不是对 XPolicyLab 现有工程继续加 Adapter。阶段 1 的数据契约、阶段 2 的 CLI / 串行 Runner / 合成 Adapter 已实现，85 项测试通过。**真实模型、权重和仿真 Benchmark 均未接入，也没有任何真实成功率结果。**

用户要求：**“每个阶段先确认细节，再实施；完成后提交验收证据，经确认后进入下一阶段。”** 当前阶段 2 等待验收，阶段 3 的具体组合和运行配置等待确认。此次迁移交接不代表自动批准阶段 3，也不代表选择了某个模型。

接手后可以先校验交接包、盘点 GPU 环境、重建轻量环境、重跑已有测试与 mock 用例、核实已有资产。这些准备工作完成后，提交阶段 2 验收结果和具名的阶段 3 配置。获得确认再安装该试点所需的重型环境、下载权重、编写真实 Adapter 和执行真实评测。

### 阅读顺序与事实依据

1. 项目根目录 `AGENTS.md`：持续工作与阶段确认规则。
2. `docs/stage-state.json`：当前阶段、授权范围与开放项。
3. 本文件：迁移方式、实现边界、下一步。
4. `docs/stage2-acceptance.md`、`docs/stage2-evidence.json`：现有验收事实。
5. `docs/contract-v0.1.md`、`embodied_eval/contracts.py`、`embodied_eval/interfaces.py`：现行契约。
6. `examples/contract_examples.json`：可解码的实际消息样例。
7. `docs/stage3-review.md`：真实试点确认入口。
8. `docs/handoff/source-materials/`：原工作簿与阶段 0 核实资料的只读副本。

类型定义和实际消息以当前源码为准。前期研究材料中的“尚未创建项目”“尚未实施阶段 1”等文字描述的是当时状态，不能覆盖当前 `stage-state.json`。旧设计草稿也不能替代已实现接口。

## 2. 项目目标、范围与阶段进度

目标是通过两侧 Adapter 和统一语义契约接入不同模型与 Benchmark，保留每个模型、机器人、控制器和官方评测协议的实际含义。新增对象应主要通过 Adapter 和版本化配对配置完成，不能仅因向量维度相同就认定兼容。

首版范围是仿真中的具身操作与导航，后续支持动作策略（VLA/WAM）、工具型 VLM Agent 和世界模型相关协议。使用已有匹配权重；训练、微调、真机部署暂不建设。CLI、单机执行，模型可在后续通过服务调用；当前实现只有单进程串行路径。

| 阶段 | 目标 | 当前状态 | 接下来需要确认什么 |
| --- | --- | --- | --- |
| 0 | 对象、协议、权重、机器人与试点核实 | 已整理 55 个计划组合；真实试点和资源未确认 | 模型/权重、Benchmark 子协议、机器人、GPU、原生入口 |
| 1 | Adapter 职责与标准数据契约 | 已实现；用户允许进入阶段 2；尚未正式冻结契约 | 若真实接入暴露契约缺口，提交影响后确认调整 |
| 2 | 最小 CLI、执行循环和测试替身 | 已实现，85 项测试通过；等待本阶段验收 | GPU 环境复测与现有验收材料 |
| 3 | 首个真实动作策略闭环 | 确认材料已写；未选定组合、未实施 | 具名映射、依赖、时序、预算和原生对照标准 |
| 4 | 第二模型与第二 Benchmark，检验复用 | 未开始 | 分批选择能验证接口边界的兼容对象 |
| 5A / 5B / 5C | Agent / 导航 / 世界模型协议 | 未开始；分别确认实施 | 工具语义 / 导航可见性 / 被测组件及固定辅助系统 |
| 6 | 并发、资源、超时、恢复、汇总与接入模板 | 未开始 | 隔离、重试边界、恢复粒度和 Adapter 验收要求 |

已有授权记录：阶段 1“可以的，执行”；阶段 2“继续推进”，范围为已提交的 CLI、串行 Runner、mock Adapter 方案。阶段内常规修复可以继续完成；跨阶段、改公共契约或改变官方协议时，先提交具体变化再确认。

## 3. 代码位置与迁移注意事项

本地源码：

```text
/Users/boboyuyong/Library/CloudStorage/OneDrive-Personal/WorkSpace/1TempFiles/EmbodiedEval
```

只读参考工程：

```text
/Users/boboyuyong/Library/CloudStorage/OneDrive-Personal/WorkSpace/1TempFiles/XPolicyLab-github
```

**不要在 XPolicyLab-github 目录内实施独立框架。EmbodiedEval 运行时不导入 XPolicyLab。** GPU 上不需要把 XPolicyLab 整个运行环境一并安装。

当前 Git 分支为 `codex/contracts-v0.1`。仓库没有任何提交，没有配置 remote；已有源码只是工作区文件，原有文件为 intent-to-add 状态。没有可用的 HEAD commit，也没有可 clone 的远端框架仓库。仅复制 Git 提交、推送当前分支或 `git archive HEAD` 都不能迁移当前代码。

`.venv/`、缓存、构建目录、`outputs/` 都被忽略。**macOS 的 .venv 不可直接搬到 Linux 使用；验收轨迹不会自动随 Git 迁移。** 本次交接包专门包含最终阶段 2 产物，排除 .git、.venv、缓存和其他临时输出。

### 交接包内容

包名：`EmbodiedEval-GPU-handoff-20261005.tar.gz`，本地位于项目 `dist/`，同时提供同名 `.sha256` 校验文件。解压只有一个顶层目录 `EmbodiedEval/`。

| 目录/文件 | 内容 |
| --- | --- |
| 项目源码、tests、examples、pyproject.toml、uv.lock | 当前可运行实现与依赖定义 |
| docs、AGENTS.md、README.md | 契约、阶段规则、确认材料与证据 |
| docs/handoff/source-materials | 原表副本、阶段 0 兼容性 Markdown / JSON 和候选核实材料 |
| docs/handoff/replay_baseline.py | 重跑现有 85 项测试及五组 CLI 验收的辅助脚本 |
| docs/handoff/交接启动提示词.md | 可复制给 GPU 实例上的 Codex 的任务上下文 |
| outputs/stage2-20261005T063033Z-final | 最终本地 mock 验收轨迹，不含其他 smoke 临时目录 |
| HANDOFF_CONTENTS.json | 打包来源、阶段范围和文件统计 |
| SHA256SUMS | 除其自身外的包内文件摘要 |

包内没有模型权重、API 密钥、模型 SDK、CUDA、仿真器或场景资产，也没有原 Git 元数据。可在 GPU 上按团队规范创建新仓库或连接指定远端，但不能伪称已存在提交历史。

历史 JSON 和 Markdown 中保留了本地绝对路径，作为来源记录。GPU 上按本文件提供的相对目录定位；**不要为迁移批量替换历史证据中的绝对路径**，否则摘要与审计记录会改变。新运行自然产生 GPU 上的新路径。若历史 result_ref 指向 Mac，直接读取同一产物目录内 `episodes/<episode_id>/result.json`。

## 4. GPU 上迁移与恢复的具体命令

以下命令供接手者执行，GPU 主机尚未指定，本次交接没有实际传输。示例目录可改为有写权限的目录；选择一个新的落地点，避免覆盖已有工作。

### 4.1 从本地上传

在本地项目根目录执行，先把主机占位符改为实际地址：

```sh
GPU_DEST='user@your-gpu-host'
ssh "$GPU_DEST" 'mkdir -p ~/handoff'
scp dist/EmbodiedEval-GPU-handoff-20261005.tar.gz dist/EmbodiedEval-GPU-handoff-20261005.tar.gz.sha256 "$GPU_DEST:~/handoff/"
```

也可以通过云实例上传功能传送这两个文件；校验和文件必须一同保留。

### 4.2 在 GPU 主机校验、解压

```sh
cd ~/handoff
sha256sum -c EmbodiedEval-GPU-handoff-20261005.tar.gz.sha256
mkdir -p ~/work/embodied-eval-handoff-20261005
tar -xzf EmbodiedEval-GPU-handoff-20261005.tar.gz -C ~/work/embodied-eval-handoff-20261005
cd ~/work/embodied-eval-handoff-20261005/EmbodiedEval
sha256sum -c SHA256SUMS
```

落地点如已包含 `EmbodiedEval/`，先选另一个目录。校验通过后，先读根目录 AGENTS.md 和当前阶段材料；不要立即修改代码或安装多个 Benchmark。

### 4.3 盘点资源

```sh
pwd
uname -a
python3 --version
nvidia-smi
df -h .
```

记录系统/CPU 架构、Python、GPU 型号/数量/显存、驱动、可用磁盘，以及已有权重/场景路径。`nvidia-smi` 显示的信息不能单独证明某个模型环境可用；选定框架、CUDA runtime、渲染和原生推理后仍需实测。不要输出凭据或包含密钥的环境变量。

### 4.4 重建框架轻量环境

项目声明 Python >=3.10，现有实际验证是 **macOS arm64 / Python 3.13.5 / jsonschema 4.26.0**。尚未验证 Linux、CUDA 或 GPU。真实模型若要求 Python 3.10/3.11 等版本，以已核实的原生依赖为准，并在所选版本上跑回归；不要强行把真实模型装进 3.13。

若实例已提供 uv，优先从锁文件重建：

```sh
uv sync --frozen
.venv/bin/python --version
```

当前 Mac 会话的 uv 不在 PATH，不能假设 GPU 也有 uv；这里没有自动安装 uv。uv 创建的环境可以直接用 `.venv/bin/python` 运行项目，不要求它另外提供 pip。

没有 uv 时使用实例选定的 Python（把 python3 换为相应解释器）：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip freeze
```

pip 方案遵循 pyproject 的版本范围，**不保证与 uv.lock 的全部传递依赖完全相同**。记录实际依赖和测试结果；需要严格同一依赖集时，使用可用的 uv 按锁文件安装。当前框架只有 jsonschema 这一项直接运行依赖，不依赖 torch 或仿真器。

轻量框架环境和真实模型/仿真环境的安装方式在阶段 3 确认，必要时隔离。不要在未选试点时统一安装表格中所有模型的依赖。

### 4.5 复现基线

在项目根目录、已安装项目的解释器中执行：

```sh
.venv/bin/python docs/handoff/replay_baseline.py
```

辅助脚本创建新的 `outputs/handoff-baseline-*` 目录，运行已有 unittest 和五个 CLI 用例，保存命令输出及核验结果。它不调用 GPU、真实模型或 API，也不修改历史验收文件。预期 85 tests OK，退出码组合为 0 / 0 / 0 / 1 / 2。

需要单独查看时：

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m embodied_eval run --config examples/mock_chunk_run.json
.venv/bin/python -m embodied_eval run --config examples/mock_tool_run.json
.venv/bin/python -m embodied_eval run --config examples/mock_cache_run.json
.venv/bin/python -m embodied_eval run --config examples/mock_score_error_run.json
.venv/bin/python -m embodied_eval run --config examples/mock_rejected_run.json
```

最后两个命令预期分别退出 1、2，不是迁移失败。退出 0 表示运行完成，**不保证任务成功**。每次显式 `--output` 必须指定尚不存在的目录；省略它会生成唯一目录。

已有消息样例可用 `examples.validate_contracts.build_examples()` 检查。`python -m examples.validate_contracts` 会重新生成 `examples/contract_examples.json`，因此不属于纯只读检查；如执行，在新副本中运行并核对差异。示例模块不作为独立已安装包发布，命令需要在源码根目录执行。

## 5. 实现结构与三方职责

```text
原生任务/观测 → Benchmark Adapter → TaskSpec / Observation
    → 框架校验、白名单视图 → PublicTask / ModelInput
    → Model Adapter → 模型原生输入 → 推理
    → Model Adapter → Prediction
    → 框架校验与时序 → Benchmark Adapter → 原生控制/工具
    → StepResult → 新观测、公开执行反馈
    → Benchmark 官方 scorer → EpisodeResult
```

| 模块 | 当前作用 | 真实接入时的边界 |
| --- | --- | --- |
| contracts.py / interfaces.py | 类型与 Adapter Protocol | 公共语义；修改先审阅影响 |
| codec.py | 严格类型 JSON 编解码 | 未知字段、重复键、NaN/Infinity、错误类型都拒绝 |
| compatibility.py | 声明与 Profile 兼容检查、必需输入检查 | 通过检查只说明声明匹配，不能证明实际权重匹配 |
| visibility.py | 公开任务/空间/输入与反馈副本 | 不向模型暴露私有评分、初始化信息或隐藏状态 |
| validation.py / geometry.py | 张量、动作、预算、消息关联、单位与几何校验 | 不做隐式缩放、坐标转换或 IK |
| runner.py | 串行生命周期与 run_episode 标准循环 | 不添加模型 resize、去归一化、Benchmark 评分规则 |
| config.py | 严格 CLI 配置和内置 mock 注册 | 尚未提供任意真实 Adapter 的插件配置 |
| artifacts.py / __main__.py | 产物与 CLI | 真实接入需完善真实性标记与原生依赖记录 |
| adapters/mock_spec.py / mock_model.py / mock_benchmark.py | 共享测试规格、合成策略和环境 | 测试替身，不能代表真实机器人动力学 |
| tests/test_contracts.py / test_runner.py | 53 + 32 项现有测试 | 保留回归；按真实映射增加有意义的检查 |

**模型侧 Adapter**：声明源码/权重/归一化/机器人能力；从标准输入选取、排序、resize、tokenize 和归一化；解析原生输出；管理模型历史、KV cache、内部 chunk 缓存及重置。模型收到的 SpaceSpec 也是过滤后的公开版本。

**Benchmark 侧 Adapter**：原生任务枚举与 reset；把原生 RGB/depth/状态转换为有身份、时间、单位、坐标的 Observation；把标准动作/工具变为原生控制，负责增量参考、IK、规划、限位、每 tick 的实际预算检查；保留官方终止和评分定义。

**框架**：兼容性、消息关联和语义校验；信息白名单；调用顺序、chunk 时序和统计；错误分类、评分收尾、结果保存与资源释放。标准格式不会替代模型专属预处理和机器人原生控制语义。

### 已实现 Adapter 接口

```python
class ModelAdapter(Protocol):
    def describe(self) -> CapabilitySpec: ...
    def begin_episode(self, context: Context, task: PublicTask,
                      space: SpaceSpec, profile: AdaptationProfile) -> None: ...
    def observe(self, model_input: ModelInput) -> None: ...
    def predict(self, request_context: Context) -> Prediction: ...
    def end_episode(self, context: ModelEndContext) -> None: ...
    def close(self) -> None: ...

class BenchmarkAdapter(Protocol):
    def describe(self) -> SpaceSpec: ...
    def enumerate_tasks(self, selection: dict[str, object]) -> list[TaskSpec]: ...
    def reset(self, task: TaskSpec, context: Context) -> Observation: ...
    def advance(self, control: ActionStep | DiscreteAction,
                request: Context, remaining_budget: Budgets) -> StepResult: ...
    def execute_tool(self, call: ToolCall, request: Context,
                     remaining_budget: Budgets) -> StepResult: ...
    def score(self, context: Context) -> dict[str, Metric]: ...
    def close(self) -> None: ...
```

### 生命周期

`describe / profile / enumerate_tasks / preflight → reset → begin_episode → observe → predict → advance 或 execute_tool → observe … → score → end_episode → close`。

一个模型和环境实例同时只持有一个 episode。正常连续 episode 可复用实例，但 begin/reset 清空状态；任一 episode 发生异常后关闭双方并重建实例再进入下一条。结束和 close 的第二个错误要保留，不能遮蔽原始异常。score 结果不会交给模型 end_episode，后者只有通用结束原因。

## 6. 数据契约中不能丢失的语义

| 契约 | 必须保留的信息 |
| --- | --- |
| SpaceSpec | Benchmark 协议/版本、机器人、关节/EEF/夹爪、传感器、坐标树、控制周期、动作/控制器/工具规格 |
| CapabilitySpec | 模型版本、checkpoint 和 normalization 引用、角色、支持的 embodiment、必需输入、输出规格、历史更新与 chunk 能力 |
| AdaptationProfile | 配对配置版本、双方身份、命名输入绑定、输出允许范围、frame 转换、horizon、公开视图；unresolved 非空直接阻塞 |
| TaskSpec / PublicTask | 公开指令/目标、episode/协议、初始化和预算；初始化引用、私有评测条件不发送给模型 |
| Observation / ModelInput | Context、observation_id、env_step、sim_time_s、传感器时间/有效性、机器人状态、允许公开的执行反馈 |
| Prediction | 对应 request_id 和源 observation_id、profile、单步/chunk/离散/工具/停止；未来预测与真实传感器分开 |
| StepResult | 实际消耗、执行状态、新观测、中间观测、原生 terminated/truncated、内部评分信息 |
| EpisodeResult | 官方指标来源与有效性、结束原因、执行统计、manifest 和产物引用 |

以下规则已由合成测试验证，真实接入须逐项对照原生代码：

- Context 含 run/env/episode/message 身份，契约版本 `0.1`。拒绝跨 episode、旧请求、旧观测和不匹配 Profile 的输出。
- 多相机、多臂、多机器人通过命名实体表示；不要把标准动作固化为一个 7D 数组。
- 物理位置 m、转动关节 rad、移动关节 m，速度与力各有单位；旋转用 quaternion_xyzw。Transform 是 T_parent_child。rot6d 是旋转矩阵前两列顺次拼接。
- 控制器归一化参数保留 controller_native 域；不能伪装为 m/rad。相同形状不等于相同动作。
- RGB 在标准层是 RGB；depth 为 m。真实模型需要不同颜色/尺寸时在模型 Adapter 中显式说明转换，不做无依据的通道交换。
- Tensor 明确 dtype/shape/layout，inline data 与 DataRef 二选一。DataRef 当前只有格式和长度校验，没有载荷读取或实际摘要校验。
- 缺失、null、invalid 与有效零值分开；缺必需输入不能补零。预算 null 为无上限，0 为无剩余。
- ActionStep 是一个完整逻辑控制 tick，duration 等于控制周期；离散持续时间为整数 tick。双臂同 tick 是一步。
- PoseDelta 明确 chunk_start / step_observed / previous_command 参考和 pre/post multiply；Benchmark 侧解析实际锚点，不偷用 chunk 起点。
- runner_owned 返回整个 chunk，Runner 只消费本次 horizon，丢弃尾部再推理。adapter_owned 自持缓存，每次只暴露一步，horizon=1。episode 边界清缓存。
- every_env_step 与 decision_boundary 分开。长工具/离散动作在 every_env_step 模式下必须返回完整、连续的中间观测。
- 工具是同步原子调用，返回后更新历史；目前不支持工具运行中的再次决策或抢占。工具按实际环境 tick 消耗预算。
- terminated/truncated 可以同时为真；模型主动 StopRequest 不等于成功。success 由官方 scorer 决定。
- 合法任务失败为 `value=false, valid=true`；无法评分为 `value=null, valid=false, unavailable_reason`。运行异常不能默认记任务失败或成功。
- model_calls 与真实 model_forward_count 分开，未知前向次数为 null。mock forward 是合成计数，不是实际网络性能数据。
- 公共视图是白名单深拷贝；评分、隐藏坐标、未允许字段不能进入模型。人工仍需审计被错误标记为公开的信息。
- 工具/控制器参数使用 JSON Schema Draft 2020-12，仅允许本地引用，不联网加载 schema。

## 7. 已有验证证据与产物

`docs/stage2-test-output.txt` 是完整 85 项 unittest 输出；`docs/stage2-evidence.json` 包含版本、源码 SHA-256、CLI 退出码和实际统计。阶段 1 的 `docs/validation-evidence.json` 是历史基线，不能当作阶段 2 当前源码摘要。

| CLI 用例 | 预期退出码与状态 | 实际基线统计 |
| --- | --- | --- |
| chunk 8 / horizon 2 | 0，completed，budget_exhausted | 6 ticks、0.30 s、3 predict、7 observe、3 mock forward |
| 工具每次请求 3 ticks | 0，completed，budget_exhausted | 两次实际 3 / 2 ticks，总 5 ticks、0.25 s、2 predict、2 tool call、6 observe |
| adapter cache，两条 episode | 0，两条 completed | 每条 3 ticks、0.15 s、3 predict、1 mock forward、4 observe；新 episode 缓存/历史从空开始 |
| scorer 故障 | 1，error | success=null、valid=false、unavailable_reason=scorer_failed |
| chunk ownership 不兼容 | 2，preflight 拒绝 | reset 前拒绝，未创建 rejected 结果目录 |

前三种 mock 的 success=false/valid=true 是固定测试规则，不是实际 Benchmark 结论。模拟时间中 0.30000000000000004 等浮点表示按容差比较。

包内保留目录：`outputs/stage2-20261005T063033Z-final/{chunk,tool,cache,score-error}`。每个运行含 `manifest.json`、`summary.json`；每条 episode 含：

```text
episodes/<episode_id>/
  task.internal.json
  trace.internal.jsonl
  model_input.public.jsonl
  diagnostics.internal.json
  result.json
```

manifest 保存配置、身份、seed、源码/锁文件摘要及 synthetic=true。public 输入轨迹和 internal 轨迹分别存储。日志中的 internal 信息不能重新作为模型输入。

## 8. 下一阶段会遇到的实际缺口

这些是当前实现限制，不是已完成能力。接手者不要依赖不存在的功能：

1. **真实 Adapter 注册尚未建设。** config.py 的 adapter Literal、工厂、Profile、fallback metrics、diagnostics 都是 mock 路径。真实接入需明确注册与专属配置，不是只替换一个名字。
2. **产物中的 synthetic=True 仍是 mock 固定标记。** artifacts.py 和 runner summary 在真实接入时需要按来源明确标记，扩充模型、权重、torch/CUDA/仿真器/资产版本记录；不能把固定 True 改成 False 就宣称完成真实评测。
3. **载荷和日志性能未验证。** 真实图像用 inline 嵌套 JSON、深拷贝和逐步写日志可能成本很高。DataRef materialization、传输和视频存储需在试点中确定；若改变公共语义先确认。
4. **没有硬超时、多进程、batch、自动重试或恢复。** wall_time 只在同步调用前后协作检查，不能中断卡住的模型/仿真调用。不能把它当作可靠故障隔离。
5. **执行状态未知只能记录最后确认进度。** termination_reason=execution_state_unknown；summary 的 execution_statistics_complete=false、scope=last_confirmed；指标不可用。整数统计不代表完整实际消耗。真实协议如需部分物理步/未知统计，先讨论契约扩展。
6. **没有远端模型协议实现。** HTTP/RPC 序列化、状态 reset、请求幂等性、断线/迟到响应需核实后适配；不能盲目重试已执行的动作。
7. **控制时钟有约定。** 当前 reset 从 env_step=0 / sim_time=0 开始，按完整控制 tick 计时；原生 warm-up、多个 physics substep 或可变时长必须逐项对齐，不能暗中计入/排除预算。
8. **导航、Agent、世界模型仅有部分基础表达。** 离散动作和工具路径用 mock 验证；没有真实 Agent Runtime、导航指标或世界模型规划/scorer。FuturePrediction 是封装；candidate_actions 仍保留且拒绝启用。
9. **声明不等于权重证据。** checkpoint_ref 等是身份声明，框架未校验真实文件/机器人能力。Stage 0 文献核实不替代原生运行证据。
10. **平台验证范围有限。** Python >=3.10 是包声明，目前只在 3.13.5/macOS arm64 验证；GPU 上需记录 Linux/Python/依赖和复测结果。

## 9. 表格核实状态与首个候选

原始表格：`模型×Benchmark_分工表.xlsx`，来源工作表 `模型×Benchmark`，模型在 `B3:B18`，具身列在 `AB2:AH18`。已标记组合为 55 个：VLABench 9、VidMan 7、LIBERO 7、RoboTwin 6、RoboDojo 6、DreamZero Protocol 6、World-in-World 14。

阶段 0 研究记录：5 项需要明确转换，4 项在已检查材料中缺少匹配权重，46 项协议/版本/角色待确认；真实运行验证 0 项。这些是证据充分程度，不能解读为其余组合一定不可用。完整条目、来源单元格、官方链接、权重 revision 在复制的兼容性清单中。

特别保留三个开放问题：VidMan 是否指 CALVIN/OXE 或内部协议；DreamZero R021/R022 的具体映射及 DROID/AGIBOT 区别；World-in-World 被测 policy、world_model 或完整系统，并固定辅助模型。名称不清楚时不能创造一个新 Benchmark。

### OpenWAM-α × LIBERO：研究候选，尚未确认

这是先前推荐候选，不是已经选定的试点。以下是 **2026-10-04 的核实记录**，本次交接没有重新访问上游；进入阶段 3 时继续确认版本可获取性和原生要求。

| 项目 | 已记录的候选基线 |
| --- | --- |
| 被测角色 | action_policy / WAM，首轮只执行动作输出 |
| OpenWAM 源码 | https://github.com/OpenWAM-Official/OpenWAM ，快照 `48bd67b89d489b14d03b8d92bc66e65d306df32e` |
| checkpoint 仓库 | https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-LIBERO |
| checkpoint revision | `110b07e4d5581c81520a77419cbb0e9371df231b` |
| 实际文件名 | `checkpoint_step_10690.safetensors`，旧参考示例的 10000 不是已核实的公开文件名 |
| 权重大小 | 元数据 24,813,767,464 bytes；尚未下载，不能据此推算推理显存 |
| 配套资产 | config.yaml、normalization_stats.npy、UMT5 tokenizer；未下载/加载 |
| LIBERO revision | 原生客户端要求 `8f1084e3132a39270c3a13ebe37270a43ece2a01`；依赖/补丁尚未复现 |
| 第一任务 | 候选 libero_spatial task 0；安装后核对任务名和初始化文件 |
| 原生说明 | https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/README.md |

研究发现原生输出是 10D EEF 表达，LIBERO 是 7D OSC 命令，涉及 rot6d → rotation vector、夹爪方向和控制器域。必须查原生实现，不能把前三维直接标 m 或重复施加控制器已有尺度。权重文件体积也不代表所需显存，先取得实际硬件和原生推理证据。

## 10. GPU 接手后下一步的阶段 3 确认材料

完成轻量基线复现后，在 `docs/stage3-review.md` 中补齐下列信息，再提交用户确认：

| 决策 | 需要落实的值与证据 |
| --- | --- |
| 模型 | 官方仓库、固定 revision、被测角色、原生推理调用与状态规则 |
| 权重 | 已有路径或下载地址、revision/hash、任务与机器人匹配依据、normalization/tokenizer |
| Benchmark | 固定 revision、suite/task/init/seed、原生入口、官方指标来源 |
| 机器人/控制 | 相机与状态、关节/EEF/TCP/frame、夹爪、控制器域、频率、warm-up |
| 时序/预算 | chunk 归属、返回长度与执行 horizon、历史更新规则、步数/时间/调用预算 |
| 资源/依赖 | GPU/显存/磁盘、模型和仿真运行位置、Python/依赖隔离、渲染方式 |
| 对照/验收 | 原生命令、相同初始化、观测/动作采样点、容差、终止/评分顺序和产物 |
| 修改范围 | 新 Adapter、注册/配置、版本化 Profile、真实性与产物记录；公共契约缺口及影响 |

推荐实施顺序是：**只读核实原生入口和双向映射 → 确认阶段 3 具名方案 → 复现官方原生入口 → 实现单一配对 → 单任务单 episode 对照 → 提交验收**。安装、下载和原生运行的具体范围亦随该阶段确认。若需先安装小型依赖才能核实映射，先说明具体需要，不把准备工作扩展为未经确认的整阶段实现。

真实验收基本门槛是获得有效官方判定，并能解释与原生实现的差异；不要求第一条 episode 必然成功。不能只展示“模型返回动作、仿真没报错”就认定 Adapter 正确。数值容差、控制 horizon 和 trials 以选定原生实现为依据，不从 mock 默认值继承。

每次实施完保存：原生与新框架命令、版本与配置、输入/动作对照、原生评分、轨迹、资源/依赖信息、回归输出及遗留问题。用户确认后再进入下一阶段。

## 11. 继续工作时的工程约定

- 对接口查源码和官方文档，不从模型名、benchmark 名或向量维度猜测。
- 只在获确认范围内逐步修改，复用现有类型/校验/工具；模型/Benchmark 专属逻辑放各自 Adapter。
- 不改原工作簿和只读研究快照。更新判断时写新证据，保留核实时间、版本及旧来源。
- 大文件下载优先只对当前命令临时绕过 HTTP(S) 代理直连，失败再用原代理；不能永久修改全局代理。固定 revision，保存实际文件名/hash，使用明确存储目录。
- 不把原生实现的不明语义用自动缩放、自动限幅、补零或私自改 success 条件掩盖。
- 不把运行异常、任务失败、主动停止和无效评分混记；不把 mock 数据用于模型排名。
- 涉及原 XPolicyLab 代码时另读其 AGENTS.md；其 RGB 编解码、父 workspace env_cfg、checkpoint resolver 和 deploy.yml 规则适用于该工程。本项目没有继承它的运行时依赖。
- 不添加凭据到代码、配置、轨迹或交接包。远端服务身份只记录模型 ID、协议和非敏感配置。

## 12. 接手完成的判定

接手者至少报告：交接包/文件摘要通过；当前项目路径；GPU、Python 与磁盘；85 项测试和五个 CLI 用例的实际结果及新产物路径；仍待决定的试点、权重、机器人和依赖；下一阶段具名配置与需要确认的变化。

交接完成不会改变 `stage2.acceptance_confirmed=false` 或 `stage3.details_confirmed=false`。只有后续人类确认才更新阶段授权状态。
