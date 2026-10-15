# 阶段 4：第二模型、第二 Benchmark 与全组合覆盖路线（已批准实施）

初次检查时间：2026-10-05 UTC。阶段 3 的 OpenWAM-α×LIBERO 单次真实闭环已获用户工程验收。用户于 2026-10-06 确认本方案，并调整为“先把所有模型和bench的adapter都搞定之后，最后再一起测试”。本文件保留下方具名设计及历史研究证据；阶段4实现回执另见更新章节，仍**不是阶段4真实闭环验收**。GPU 按八卡总体时间平均，目标至少20%。原始分工表、阶段0快照和相邻 XPolicyLab 保持只读。

本文记号：`AB18` 等是原电子表格的单元格地址；`revision` 是固定源码提交或模型仓库版本；SHA-256 是用于核对文件内容的 256 位摘要，HF LFS 是 Hugging Face 大文件存储的文件元数据，GCS 是 Google Cloud Storage 对象存储。GB/PB 是十进制十亿/千万亿字节量级，MiB 是 1,048,576 字节；RGB 是红绿蓝图像。EEF10 指末端执行器位置 3 维、旋转表示 6 维、夹爪 1 维的原生 10 维数组；IK 是根据末端目标求关节命令的逆运动学；EGL 是无显示器的图形渲染接口。FP32/BF16 分别是 32 位浮点和 16 位脑浮点推理精度；NaN 表示不可用的数值，不可解释为零分；CLI 是命令行入口。本文的“单次”只表示一个固定初始化状态的完整尝试。

## 范围与逐项门禁

### 2026-10-08 运行推进回执（阶段4未正式验收）

以下是新的运行层证据，下方10月7日下载/导入失败描述保留为历史，不是最新状态。用户已同意NVIDIA Omniverse条款，仅用于本项目运行进程；未设置全局环境、代理或升级共享驱动。

- 六类固定清单共494文件、116,987,749,975字节全部摘要核验；[本轮完整资产清单](../outputs/mn-runtime-preparation-20261008-run1/assets.json)。它不包含原生构造器的全部额外基础组件、RoboDojo完整任务或全部USD素材引用。
- [OpenWAM ARX真实推理及转换](../outputs/mn-runtime-openwam-arx-request-20261008-run4/component-result.json)：GPU2加载真实匹配权重，使用合成相机/状态，原生一次生成32条20维双臂动作，Adapter取32行；输入像素/状态/指令、全部32条公共动作转换回原生后的数值与类型逐项一致。比较使用固定源码纯转换函数，不导入/实例化XPolicyLab组合模型；未比较独立重新推理或仿真执行。初次CUDA13 PyTorch不能用当前驱动，仅把本任务新环境约束到源码允许的torch2.10.0+cu128/torchvision0.25.0+cu128，依赖检查及真实推理通过。
- [Cosmos真实推理](../outputs/mn-runtime-cosmos3-request-20261008-run2/component-result.json)：GPU4真实权重和合成观测推理通过。原生CUDA库查找只指定当前环境已安装的库路径；另按固定源码版本921dbaf3补齐2,818,839,170字节Wan2.2 VAE并核验SHA256，用源码支持的配置覆盖仅改资产路径。403包元数据检查的NumPy/openpi-client与Jupyter查看器/Python3.13两项偏差仍保留，不能由本路径运行通过推定训练/全部工具环境完全复现。
- [DROID两条真实保持动作](../outputs/mn-runtime-droid-hold-20261008-run3/component-result.json)：场景1的三个720×1280相机、实测关节和夹爪可取得；两条动作实际16物理步、0.1333333333秒，隔离后子进程与外层正常退出。先前缺GLU/OpenGL库用本项目私有库解决；未安装系统包或改驱动。
- 原生快速关闭曾在汇总前结束Python，关闭快速模式后又出现[退出段错误](../outputs/mn-runtime-droid-hold-20261008-run2/receipt.json)。当前环境Adapter独占一个通过spawn启动的Python进程，所有初始化/控制器/观测/物理执行都在其中运行；原生环境和应用关闭返回后直接退出，避免插件再次进行不安全的解释器收尾。父进程检查关闭回复及退出码。没有修改Runner，也没有并行trial、硬超时、自动重试、恢复或执行状态猜测。
- [首次Cosmos／DROID有限步链路](../outputs/mn-runtime-cosmos3-droid-bounded-20261008-run1/receipt.json)完成4动作、Runner错误0，但退出码-11，整轮标失败并保留。[隔离后的复测](../outputs/mn-runtime-cosmos3-droid-bounded-20261008-run2/receipt.json)正常退出0，Runner错误0；原生服务1次请求、Adapter4次predict、实际4动作/32物理步/0.2666666667秒。模型服务GPU4、仿真GPU0，场景1/seed0/4动作预算。原生DROID没有官方任务成功评分，success=null且valid=false；不是全任务/全suite验收，不抵扣原表R021/R022，不是独立原生重新推理严格对照。
- 四套本轮环境（轻量框架、DROID、OpenWAM ARX、Cosmos）均320项测试、零跳过及CLI退出码0/0/0/1/2。新增14项为诊断安全/私有进程边界合成测试；保留原八套环境306项历史回执，不回写为320项。[轻量回归](../outputs/mn-runtime-baseline-20261008-final2/baseline-check.json)、[DROID回归](../outputs/mn-runtime-baseline-droid-20261008-final2/baseline-check.json)、[OpenWAM回归](../outputs/mn-runtime-baseline-openwam-20261008-final2/baseline-check.json)、[Cosmos回归](../outputs/mn-runtime-baseline-cosmos3-20261008-final2/baseline-check.json)。
- DreamZero主权重15/15已核验，策略导入和GPU矩阵运算通过，但还没有DreamZero权重加载/推理证据。固定原生构造器额外先读取Wan2.1基础组件，共82,243,108,389字节；固定版本6b73f84e。仅在当前下载命令临时绕过代理，失败回退，旧分块保留；[本轮下载清单](../outputs/mn-runtime-dreamzero-backbone-20261008-run2/manifest.json)尚未完整，下载结束不会自动启动推理或仿真。
- RoboDojo仍待私有IsaacSim5.1/IsaacLab2.3/CuRobo、原生控制器和素材引用验证；共享驱动570.124.06低于文档580.65.06要求，不擅自升级。原49格分类不变，不用新DROID替换未定义原表协议，不承诺跨机器人任务成功。

[本轮网页报告](../outputs/mn-runtime-delivery-20261008-final/M+N-runtime-progress-and-evidence.html)已按静态模板生成，预览只监听实例127.0.0.1:8765；24个报告/直接证据链接HTTP核验通过，未链接文件返回404。未证明桌面端口转发，不能声称用户本机浏览器已可访问。35份阶段2历史证据与原M+N报告摘要未变，0.2公共契约/接口/Runner四项摘要与10月7日封存值完全相同。阶段4未正式验收，契约未冻结，不进入下一阶段。

[阶段 0 兼容性清单](handoff/source-materials/兼容性清单.md)按原表有 55 个被分工单元格，分属 VLABench 9、VidMan 7、LIBERO 7、RoboTwin 6、RoboDojo 6、DreamZero 协议 6、World-in-World 14。单元格指原表中的一个模型与一个 Benchmark 名称的交叉格；它**不等于**已定义、可运行或已评分的协议。World-in-World 的每格又可能拆 ImageNav 与 Manipulation 两种子协议，不能在原 55 上简单翻倍。阶段 0 的 46 项“协议待确认”、4 项“缺少匹配权重”、5 项“需要明确转换”是当时检查快照；下述新发现只更新研究判断，不回写历史记录，也不把未实测项目算作“可直接接入”。

为逐项推进，每一单元格先核对：被测角色（动作策略、Agent、世界模型或完整系统）、固定模型及源码 revision、匹配权重和归一化、Benchmark 源码/子协议/机器人/控制器、原生入口与任务预算、公开与隐藏信息边界、原生逐步对照与官方评分。只有这些齐备，才提出该组合的 `AdaptationProfile`（模型和 Benchmark 的具名映射配置），执行单次真实闭环，提交逐步证据；单次工程通过不等于正式全量成功率。无法取得服务端点、专用权重或协议定义的格子保持“待决定”，不得用相近模型、随机动作、mock 或离线 VLM 分数代替。

[55 格机器可读覆盖账本](../outputs/stage4-coverage-20261005T1818Z.json)从只读阶段 0 JSON 逐格派生，独立记录当前研究状态：1 格真实单 trial 工程验收（AD18），5 格找到专用匹配资产但未运行（AD14、AB18、AE4、AE18、AF18），其余 49 格仍待协议、身份、权重或端点核实。账本不是运行配置，也不是 55 格适配已完成；其 SHA-256 为 `19b877372792134e301fcbd494964128d24773b47fa5a0f80fe782e2a5ecf712`。可通过 `docs/handoff/build_stage4_coverage.py` 从原快照重新生成到一个新路径。

## 2026-10-05 研究轮已核对资源和候选次序

`Model/` 位于 `/inspire/hdd3/project/agent-driven-world-model/zhouzhixiang-240107010008/Model`，不是框架工作区内的目录。已有 OpenWAM-Alpha-Sim-LIBERO 约 24 GB、Qwen3.8-27B 约 52 GB、Cosmos3-Nano 约 33 GB、Hy-Embodied-VLM-1.0 约 57 GB。Qwen 本地模型卡说明其为图文模型，配置架构名沿用 `Qwen3_5ForConditionalGeneration`；Cosmos 本地配置对应 Nano-Reasoner。两者存在本地权重，不证明具有对应机器人闭环动作策略。2026-10-05 的 `df` 显示共享文件系统可用 11 PB，但这不证明用户配额；八张 RTX 4090 各报告 49,140 MiB 显存。GPU 后台合成占用与真实评测必须分开记账。

本轮未改 Runner 或真实 Adapter。只给原有 GPU 采样脚本增加向后兼容的 `--target-percent` 参数，然后以 20% 为门槛连续每秒采样 60 次：[完整 GPU 原始记录](../outputs/stage4-gpu-target20-20261005T1812Z.json)的八卡总体时间平均为 **26.3521%**，记录 SHA-256 `28c903d8f2007efcea9c2023518f11b7c4e2bd2a9819d27b49d14a08ecb1bcb0`。采样期间 GPU 2–7 的约 35% 是现有轻量合成占用程序，不是模型/Benchmark 算力；GPU 0/1 的采样均值均为 0。故只能证明共享资源利用率达标，不能作为新模型效率或阶段 4 真实运行证据。[本轮基线回放](../outputs/stage4-preflight-baseline-20261005T1815Z/baseline-check.json)为 93/93 测试、8 个契约示例通过、五组 mock CLI 退出码 `0/0/0/1/2`，全部仍是原合成基线；回执 SHA-256 `828b1bf30a4ee1db4f6ad1b8a81eb6dd1805d987fc5598dfabc97521834a50a8`。

| 顺序 | 原表单元格 | 对象与官方匹配资产 | 当前证据/欠缺 |
| --- | --- | --- | --- |
| 1：第二模型 | AD14，π0.5×LIBERO | [openpi 官方 LIBERO 示例](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)明确给出 `gs://openpi-assets/checkpoints/pi05_libero/`，不是 `pi05_base`。GCS 公开对象清单有 16 个对象、共 12,439,085,481 bytes，包含 `assets/physical-intelligence/libero/norm_stats.json`。openpi 代码固定 `215abfb217dbac7d5f1273282331b9b1866c0479`；其 LIBERO submodule 指向 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。 | 源码已只读检出；权重尚未下载/校验，openpi 原生服务与客户端未启动。现有阶段 3 LIBERO 为另一提交 `8f1084e...`，不能把两种原生协议混成一个结果。 |
| 2：第二 Benchmark | AB18，OpenWAM-α×VLABench | OpenWAM 源码 `48bd67b89d489b14d03b8d92bc66e65d306df32e` 的[原生说明](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/vlabench/README.md)指定 `OpenWAM/OpenWAM-Alpha-Sim-VLABench`。权重仓库 revision `480d3bc77b298434e69400a884a416d7196e677c`，checkpoint 为 24,813,767,464 bytes、HF LFS SHA-256 `6352f34db37f0a44ff8262f4d401d85d2a7b036799d00c6402ff81363684eb5e`，归一化文件 SHA-256 `6434b2f23e8406d224424b57cca5e13527777dc260de446ec2bdf4876c29af8f`。VLABench 源码只读检出 `cf588fe60c0c7282174fe979f5913170cfe69017`。 | 权重与约 17 GB 场景/物体资源尚未下载/校验；VLABench 独立 Python 3.10 环境尚未建；EGL 与原生单例未实测。不能沿用 LIBERO 专用 checkpoint。 |
| 3：双新对象 | AE4，LingBot-VLA 2.0×RoboTwin 2.0 | [LingBot-VLA 2.0 官方仓库](https://github.com/Robbyant/lingbot-vla-v2)与旧 `lingbot-va` 不同；源码只读检出 `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`。官方专用权重 `robbyant/lingbot-vla-v2-6b-robotwin` revision `0451855729ec904f970600e0aec8b84661423afe`，六片 safetensors 合计 25,503,889,772 bytes。官方入口要求 `checkpoints/global_step_50000/hf_ckpt`、Qwen3-VL-4B-Instruct backbone、FP32 发布验证，并明确 clean/randomized 模式。 | 权重、backbone 和 RoboTwin 环境未取得，机器人动作域/原生固定 task 尚未核对。官方 launcher 会向 RoboTwin 工作树拷贝客户端并修补配置，不能在只读上游或 XPolicyLab 中直接执行；要用隔离工作副本。 |

以上顺序是建议，不是用户已经选定的正式批量评测。第三条不能使用旧 LingBot-VA 1.0 权重冒充 2.0。其余组合按原表逐格继续核验；R009/R010、R021/R022、R054/R055 的定义尚未提供，VidMan 本身有 CALVIN/OXE 实验而非自动可注册的仿真环境，World-in-World 仍需选择被测角色与子协议。DreamZero 的 DROID 仿真与真实 RoboArena 亦不能混淆，真实机器人仍在本版范围外。

## 两条优先路线的原生条件

**π0.5×LIBERO**：固定 openpi 自带 LIBERO 原生协议，先在 `libero_spatial` task 0、首个初始状态做单次工程对照。其[原生客户端](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py)默认 seed 7、先静置 10 个环境步、每次推理取动作 chunk 的前 5 步执行后重规划、spatial 最多 220 个策略步；正式每任务 50 次。它把两路 256×256 图像旋转 180°、带 padding 缩放到 224×224，末端位置 3 维 + 四元数转 axis-angle 3 维 + 两指位置 2 维组成 8 维 state，模型输出前 7 维直接交给 LIBERO `env.step`。axis-angle 是轴角旋转向量，不是四元数；chunk 是一次推理给出的动作序列。此配置与 OpenWAM 阶段 3 的 seed 42、静置 30 步、10 步服务缓存、600 步预算不同。openpi 自带 LIBERO submodule 与 Python 3.8 仿真依赖另建环境，不更改现有阶段 3 LIBERO 检出或历史证据。模型服务端还需验证 checkpoint 归一化资产、显存和完整原生输出。

**OpenWAM-α×VLABench**：建议只先选 `track_1_in_distribution` 的 `select_fruit` 第 0 个冻结 episode；这里的 track 是 VLABench 已存档的固定场景集合，故不把 seed 42 误记为此 episode 的唯一初始化依据。固定 track JSON SHA-256 `61b07440c6f4272705bea0c8d721bc196e744232f48408075613cc9b6f7c8c04`，原生 `policy_config.yml` SHA-256 `f1b51222e5f68d5a5d7dbd27e2c2948720c4af2d36411cf3fc2acdbb56`。原生每任务 50 episode；`select_fruit_series` 的评测上限为 300 个策略决策，原生 `max_substeps=1`，但仿真控制周期仍需运行时验证。[原生 VLABench 客户端](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/vlabench/openwam2vlabench_interface.py)使用相机索引 2/3/0、世界与机器人基座坐标转换、原生 EEF10 状态/动作、模型服务拥有 chunk 缓存；Evaluator 执行逆运动学、关节命令和 success/progress/intention 评分。`get_ee_open_state` 的状态极性与动作极性相反，不能在 runner 中纠正。原生代码对部分任务的 intention `KeyError` 使用显式 NaN 保护；若首例触发，需说明此运行时修补并保留该指标无效状态，不能伪造零分。一个模型服务端口只服务一个并行仿真实例，以免缓存串话。全量 track 评分在单次逐步对照通过后另行验收。

## 实施前必须确认的公共边界

1. **新增适配器与 CLI 选择**：现有 `config.py` 只注册 mock 和 OpenWAM×LIBERO，并把真实试点限制为唯一 episode、seed 42、600 步。为两条新路线，需要具名 `openpi_libero`、`openwam_vlabench` 与相应 Benchmark Adapter、严格 pair 白名单、各自预算和来源证明。修改会改变新配置可接受的 Adapter 名称及新产物的 `synthetic=false` 来源，但不重写旧产物。`server_identity_verified=false` 必须保留，除非服务端握手实际支持可验证权重身份。
2. **VLABench 模型可见输入**：现有 `libero.native_proprio.v1` 不能拿来标记 VLABench 的 `ee_state`、机器人基座和夹爪极性。建议新增 `vlabench.native_proprio.v1`（固定数组形状、世界/基座 frame、夹爪语义），由 `SpaceSpec` 声明、`Observation` 提供，`AdaptationProfile.view_policy` 显式放行。核心 dataclass 可不变；新增公开扩展会扩大模型可见字段，需要审查隐藏任务信息隔离和示例/校验。
3. **VLABench 图像载荷**：三个 480×480 RGB 相机若全程 inline 进入 JSON 轨迹，会显著放大单次 300 步产物；现有 `DataRef` 有声明却无读取/摘要验证/作用域规则。建议先明确是否接受单次 inline 的大产物，还是另批准文件载荷解析与哈希校验；后一选项会影响序列化、可移植性、路径安全、校验及恢复，不在未确认时偷偷开启。模型专用 resize 仍由 Model Adapter 执行。
4. **原生评分和时间**：VLABench 的一条策略动作对应默认一个 `env.step`，当前 Runner 可接受；若实际 `max_substeps` 改为大于 1，现有 Runner 对 `ActionStep` 强制消费恰好一个 tick，就会产生公共执行语义缺口，必须先另议。LIBERO 两种原生协议也不能共用静置/预算/初始化配置。官方 success、progress、intention 均由 Benchmark Adapter 提供，不由通用 Runner 计算。
5. **55 格覆盖口径**：建议“全部适配”指对 55 个原表单元格逐项给出可运行证据或明确的缺权重/端点/协议阻塞；不承诺将不相容模型强行闭环运行。若要求 55 格都输出成功率，必须先提供上述未定义编号、专用权重/可用端点和被测角色，并另定相应评分协议。服务密钥只应通过受控环境变量/本地密钥管理接入，不写入配置、日志或对话。

上述两条优先路线及第1–4点已由用户2026-10-06回复确认，按新增执行顺序先实现具名 Adapter，再集中原生与框架对照。第5点的缺失协议、服务身份及新增公共核心缺口仍不自行假定。进入后续 Agent/世界模型或全量评分前，仍需确认具体评测细节。

## 2026-10-06 实现批次更新

- 新注册 `openpi_libero×libero_openpi`：原生图像旋转与224 padding、state8、5步重规划缓存；独立 LIBERO 版本、seed7、10步静置、220步预算；官方 GCS 对象校验与归一化摘要。`padding` 是保持宽高比缩放后补齐画布，state8 是上述8维原生状态。
- 新注册 `openwam_vlabench×vlabench`：480×480相机2/3/0、当步公开指令、世界/基座转换、原生IK及一步控制周期、官方 success/progress/intention。intention 原生 KeyError 保留指标无效/null，不伪造0分；inline像素载荷获批准，但 DataRef读取仍未建设。
- 新注册 `openpi_vlabench×vlabench_openpi`（原表 AB14）：新核实的 [官方 π0.5 primitive-10task 专用发布](https://huggingface.co/VLABench/pi05-primitive-10task/blob/f658a8bec801aab960fe0ceab370a123b31c1837/README.md)，权重固定上述 HF revision，推理所需19个文件共12,440,875,959 bytes；匹配归一化 SHA-256 `bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f`。采用模型卡指向的 [openpi fork 固定版本原生客户端](https://github.com/Shiduo-zh/openpi/blob/788ef1f20e655de5bb40869fb7b7fcdd49732387/examples/vlabench/eval.py)，实际脚本已改名为 `vlabench_scripts`，不执行模型卡旧拼写路径。此版本是三相机、state7、5步缓存、动作7维；不是 VLABench 仓库里另一份2相机/4步客户端，更不是 LIBERO 配置。服务端固定 `pi05_ft_vlabench_primitive`，保留原生 padding、归一化和相对/绝对动作变换。共用已批准的 VLABench 公开扩展和原生评分，不新增核心字段。仍只覆盖冻结 track1/select_fruit/episode0 单例；没有声称12种任务或所有track已测。
- 新完成 OpenWAM、LingBot-VLA 2.0 的 RoboTwin 模型侧原生桥接；仅返回原生数组，没有完整 Benchmark 执行注册或成绩。
- [最新55格实现覆盖账本](../outputs/stage4-implementation-coverage-20261006T0830Z-final.json)：1格历史真实单次验收、3格实现待集中真实对照、2格仅模型侧桥接、49格仍列明协议/身份/权重/运行时缺口。16个模型名称和7个Benchmark组全部逐项记录，**不是55格适配完成**；早先0740Z/0755Z账本保留其当时2格/50格状态，不覆盖；0810Z保留修复初始化加载前的代码摘要。
- [最新接口与基线回归](../outputs/stage4-interface-regression-20261006T0830Z-final/baseline-check.json)：125项全部通过且无跳过（85原基线+8阶段3+32本轮），8个原契约示例通过；五组mock CLI仍为0/0/0/1/2。全部是软件接口/合成检查，没有新的真实模型/Benchmark评分。先前118项、124项回执仍保留，不把后来新增检查写进旧证据。
- [最新GPU60次采样](../outputs/stage4-gpu-20261006T0810Z.json)：八卡总体时间平均69.81042%，超过20%；主要来自现有合成占卡，不能计作真实评测算力，也不证明未采样时段的全程平均。此前0730Z的69.38125%记录保持不变。
- 三组专用权重均通过完整性核对：π0.5×LIBERO 的16个固定GCS对象 [回执](../outputs/stage4-download-pi05_libero-20261006T0735Z.json)，OpenWAM×VLABench 的7个文件 [续传回执](../outputs/stage4-download-openwam_vlabench-20261006T0757Z-retry.json)，π0.5×VLABench 的19个推理文件 [续传回执](../outputs/stage4-download-pi05_vlabench-20261006T0823Z-retry.json)。后者首次18/19，剩余对象出现HTTP/2传输错误和回退超时；保留部分文件后以当前命令HTTP/1.1重新续传，最终19/19通过。每次先临时直连、失败回退原代理环境；已有完整文件不覆盖、失败回执不改写；均没有加载模型或启动评测。
- VLABench官方Google Drive资源触发下载限额。尝试 [LeRobot推荐的资产镜像路线](https://github.com/huggingface/lerobot/blob/main/docker/Dockerfile.benchmark.vlabench)，固定 `lerobot/vlabench-assets@08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4`，清单35,680个文件、12,020,412,514 bytes。首次两条网络路线 [回执](../outputs/stage4-vlabench-mirror-20261006T0806Z.json)均失败，仅保留可续传缓存，**没有发布完整场景资源**；继续诊断和尝试普通git文件批量获取。即使镜像全部通过校验，也不能证明其与无官方哈希的Drive原包完全相同，来源字段必须保留该限制；不采用LeRobot的缺资源空目录补丁。
- openpi LIBERO 服务、openpi VLABench 服务各自用固定源码锁建立独立 Python3.11 环境，VLABench 用 Python3.10 仿真环境，LIBERO openpi 框架侧另建 Python3.10 环境。后者与官方Python3.8依赖锁有明确兼容性移植差异，见 `requirements-libero-openpi-stage4.txt`；不能声称依赖逐项相同，需集中原生对照时核验。四份环境的依赖一致性检查均通过；LIBERO 新环境也独立 [回放通过125项及五组CLI](../outputs/stage4-interface-regression-libero-env-20261006T0830Z-final/baseline-check.json)。PyTorch2.7的默认安全加载不兼容原生NumPy初始化文件，故Adapter在注册工厂已核对文件与固定git blob摘要后，显式 `weights_only=False` 保留历史原生加载方式；不修改f78abd6上游源码，也不对未知pickle文件启用这条路径。现有阶段3环境和Mac历史绝对路径均未改变。
- [最新环境/源码/资源盘点](../outputs/stage4-preparation-inventory-20261006T0825Z-final.json)：Linux x86_64、128个逻辑CPU、约1.08TB物理内存，8张系统报告49,140MiB显存的RTX4090，驱动570.124.06；共享磁盘可用不等于用户配额。八份上游固定检出的tracked源码均干净；盘点只读取版本/包元数据，不导入真实模型或环境；0812Z早先盘点保留。

## 2026-10-06 可变时长批准后更新

用户在白话解释后回复“我认为可以，继续”，批准 [具名可变时长协议](variable-transition-review.md)。已实施通用类型/协商/校验/累计计数与新模式预算拒绝，旧固定模式保留；增加RoboTwin只读原生执行计数和LingBot全序列缓存组件，尚未注册完整RoboTwin pair。

- [VLABench侧环境回放](../outputs/stage4-native-timing-regression-20261006T1030Z/baseline-check.json)、[LIBERO侧环境回放](../outputs/stage4-native-timing-regression-libero-env-20261006T1030Z/baseline-check.json)：两套均163项全部通过、无跳过，包含原85+8阶段3+32既有阶段4+22时间核心+16原生组件检查；8个原契约示例与五组CLI0/0/0/1/2均保留。是软件/合成证据，没有新增真实成绩。
- [新55格实现账本](../outputs/stage4-implementation-coverage-20261006T1033Z.json)：仍1格历史验收、3格注册待真实对照、2格RoboTwin组件尚未完整注册、49格其他缺口。新增组件不抬高“可运行pair”数，旧0830Z账本不回写。
- [60次GPU采样](../outputs/stage4-gpu-20261006T1026Z.json)：八卡总体平均70.16042%，仍主要来自现有占卡；没有修改后台负载，不是本次模型效率或完整时段平均。
- [新只读盘点](../outputs/stage4-preparation-inventory-20261006T1033Z.json)保留环境/资源及源码检查；VLABench镜像0823Z尝试已于08:42结束，两条路线均超时失败。10:27又按当前命令直连优先、原代理回退开始有界续传（8线程，每路线最多600秒），输出待写到 `outputs/stage4-vlabench-mirror-20261006T1040Z-resume.json`；产物标签不是实际开始时间，回执中时间为准。完成全量校验前继续视为资源未就绪。

完整RoboTwin接入的新具体缺口已提交在 [原生公开状态映射方案](variable-transition-review.md#新核对出的原生观测缺口提交下一步确认尚未实施)：原生关节向量是驱动目标，不是实测角度；LingBot不是逐行取图；OpenWAM helper与RoboTwin实际四元数顺序存在冲突。建议保留原生行为、明确原始字段和采样时间，以新公开扩展隔离，不私下修正规则。该扩展尚待用户确认，时间模式无需再次确认；集中真实测试、未知编号协议及RoboDojo解耦门禁仍保留。

## 2026-10-06 自主完成源码明确适配后的集中验证更新

本节替代上文对应的“未注册／资源未齐／尚未实施”当前状态，但不改写历史快照。用户要求不再逐项问细节，继续完成已有范围并交付架构说明；不据此编造未知协议或默认阶段4已验收。

- 六组完整入口已注册：历史 OpenWAM×LIBERO，以及新增 π0.5×LIBERO、OpenWAM×VLABench、π0.5×VLABench、OpenWAM×RoboTwin、LingBot-VLA 2.0×RoboTwin。原55格的其余49格仍是具名协议／身份／端点／权重／角色缺口，没有占位Adapter冒充已完成。[新覆盖账本](../outputs/stage4-implementation-coverage-20261006T1545Z.json)分别列注册状态、实际完整试验和原生对照门禁。
- 三套仿真侧环境均 **200项通过、无跳过**：85原基线＋8阶段3＋32既有阶段4＋22可变时长核心＋18RoboTwin执行组件＋28完整RoboTwin接口＋7内嵌像素严格性检查。原8个契约示例和五组mock退出码0/0/0/1/2保留：[VLABench侧](../outputs/stage4-runtime-regression-vlabench-stage4-20261006T1533Z/baseline-check.json)、[LIBERO侧](../outputs/stage4-runtime-regression-libero-openpi-stage4-20261006T1533Z/baseline-check.json)、[RoboTwin侧](../outputs/stage4-robotwin-runtime-root-regression-20261006T1530Z/baseline-check.json)。软件与合成检查不是Benchmark成绩。[无NumPy的轻量CLI](../outputs/stage4-lightweight-cli-20261006T1430Z/summary.json)仍能注册／运行mock。
- 三条新可运行路线均已真实重置／取图并完成一次模型请求：[重置门禁](../outputs/stage4-native-env-smoke-20261006T1418Z-fixed/receipt.json)、[π0.5×LIBERO请求](../outputs/stage4-native-model-probe-20261006T1423Z/receipt.json)、[OpenWAM×VLABench重试](../outputs/stage4-native-model-probe-20261006T1426Z-openwam-retry/receipt.json)、[π0.5×VLABench重试](../outputs/stage4-native-model-probe-20261006T1432Z-openpi-vla-port-fixed/receipt.json)。首次缺包、冷启动超时、脚本端口不一致回执全部保留。
- **π0.5×LIBERO真实完整单次成功**：103策略动作、5.15秒仿真、103框架预测调用、21次原生服务请求，官方success=true且有效，环境正常终止。[真实回执](../outputs/stage4-closed-loop-pi05-libero-20261006T1436Z/receipt.json)。独立原生环境重放同103个已记录动作，104份原始状态、两相机图像和224像素预处理快照全部精确一致：[最终原生重放](../outputs/stage4-native-action-replay-pi05-libero-20261006T1511Z-final/receipt.json)。该重放没有重新调用模型，不是完整新原生策略试验，也不代表全suite成功率。
- **OpenWAM×VLABench真实完整单次成功**：139原生动作、13.9秒仿真，官方success=true、progress=1、intention=1且全部有效，环境正常终止。[真实回执](../outputs/stage4-closed-loop-openwam-vlabench-20261006T1450Z-optimized/receipt.json)。π0.5×VLABench完整试验仍在运行，只有写出最终回执后才能记录成绩；原有两份慢速尝试中断且没有成绩，[中断记录](../outputs/stage4-delivery-20261006T1435Z/vlabench-interrupted-attempts.json)保留。原生VLABench动作重放另行运行，不复用BenchmarkAdapter执行方法，不将重放冒充新模型推理。
- inline内嵌像素仍逐值验证，不缓存可变对象的“已验证”状态。仅削减合法不可变标量的递归／路径拼接成本；非法值仍走精确错误路径。三次CPU微测量480×480×3图像解码约0.36–0.37秒、含解码校验约0.64–0.65秒：[实际记录](../outputs/stage4-delivery-20261006T1435Z/inline-tensor-performance.json)。不是全链路速度或GPU效率声明。DataRef实际载荷读取仍未实施。
- RoboTwin原始公开状态扩展按用户自主推进授权落实；14维驱动目标不冒充实测关节角，相机／自身状态缓存保留真实采样时间，原生数值类型和控制／评分不改写。OpenWAM四元数顺序冲突明确保留；LingBot原未固定Python指令随机数，本版单列带种子移植协议。详见 [时间与公开映射说明](variable-transition-review.md) 和 [契约增补](contract-v0.1.md)。旧固定周期保留，首版不接硬仿真时间预算，契约没有正式冻结。
- 新RoboTwin环境 torch2.4.1、SAPIEN3.0.0b1、mplib0.2.1、CuRobo0.7.8、warp1.13.0 已构建并通过包依赖检查；独立下载校验CUDA12.1编译器，不改系统CUDA。只按原固定安装脚本调整新环境UTF-8读取和mplib条件，原生tracked源码不改。基础真实空场景3步与CUDA规划器导入通过：[基础门禁](../outputs/stage4-robotwin-runtime-gate-20261006T1518Z-fixed/receipt.json)。**它不是机器人动作路径计数已经验证**。SAPIEN实际返回0.004000000189989805秒，先协商实际周期再建manifest；任务启动和结果目录独占顺序已修复并回归验证。
- LingBot独立Python3.12.15／torch2.8.0环境、官方匹配FlashAttention wheel、LeRobot v0.4.2已安装；[原生推理导入门禁](../outputs/stage4-lingbot-env-gate-20261006T1532Z/receipt.json)通过。官方路线的torch／torchvision／datasets等依赖上界冲突单列，pip check没有通过，不能声称逐项一致。固定源码以Qwen配置构造网络后加载完整VLA权重，不需要另下一套大backbone；[9个配置／分词／processor文件](../outputs/stage4-download-qwen3-vl-processor-20261006T1505Z.json)全量校验，所固定观察到的Qwen revision不等于发布方未提供的评测身份。
- RoboTwin机器人本体压缩包已校验并发布223个原文件，另生成两份仅替换资产根占位符的规划配置，原文件不改。匹配大权重、背景与物体尚有不完整项；原Xet多次失败留下的是累计残片，不是可靠续传前缀。新增有界HTTP分段路线仅保存完整小段，验证Content-Range（服务器实际返回的字节范围），完整对象仍须固定大小和SHA256；每命令先临时直连再回退原代理环境，未改全局代理。新回执在1541Z-ranges目录标签下，实际开始／结束以回执为准，未写最终回执不宣称就绪。
- [最新源码／七环境／资源盘点](../outputs/stage4-preparation-inventory-20261006T1533Z.json)为Linux x86_64、128逻辑CPU、约1.08TB物理内存、8张系统报告49,140MiB的RTX4090，驱动570.124.06；共享磁盘可用不是用户配额。10份固定原生tracked源码干净。60次窗口总体GPU均值 [69.44%](../outputs/stage4-gpu-20261006T1500Z.json)，主要包含现有合成占卡，不是本次评测效率；另启40分钟连续采样，最终回执完成前不提前计算全程值。背景占卡未改动。

静态架构／职责边界／55格细节／证据索引集中在 [EmbodiedEval架构汇总HTML](../outputs/stage4-delivery-20261006T1435Z/EmbodiedEval-architecture.html)。没有新全量Benchmark榜单；阶段4正式验收和契约冻结未自动发生。硬超时、通用并发／恢复和DataRef读取不因外部检查脚本有界运行而自动具备。历史Mac绝对路径、原表、阶段2历史产物和XPolicyLab保持不变。

### 16时补充：资源门禁、浮点精度与GPU实测

- LingBot专用17个文件、25,518,316,625字节已完整校验：[原传输回执](../outputs/stage4-download-lingbot-robotwin-20261006T1514Z-xet.json)、[独立复核](../outputs/stage4-download-lingbot-robotwin-20261006T1541Z-ranges.json)。三份RoboTwin压缩包全部通过：[回执](../outputs/stage4-download-robotwin-assets-20261006T1514Z-xet.json)；本体223、[物体9368](../outputs/stage4-robotwin-objects-20261006T1547Z.json)、[背景11000](../outputs/stage4-robotwin-background-20261006T1554Z.json)个原始文件均校验解包，两份规划配置另列派生摘要。
- [官方LingBot FP32实际权重加载／服务启动](../outputs/stage4-lingbot-checkpoint-startup-20261006T1552Z-client-env/receipt.json)通过，未发送模型请求。FP32指32位浮点推理。[真实机器人场景重置／渲染／公开映射](../outputs/stage4-robotwin-env-smoke-lingbot-20261006T1557Z/receipt.json)通过：原生候选种子100000因UnStableError（场景不稳定）被拒绝，100001专家可行；三路240×320图像和14维驱动目标有效。专家动作只作初始化可行性筛选，不当策略成绩或模型输入。
- [OpenWAM×VLABench独立原生动作重放](../outputs/stage4-native-action-replay-openwam-vlabench-20261006T1544Z/receipt.json)完成：139动作、140份自身状态／三相机快照全部精确一致，success/progress/intention三项分数相同。历史Python随机数状态未保存，重放不重采样指令、不重新调用模型，仍不能代替完整新原生策略对照。
- 修正π0.5×VLABench动作输出的原生精度：固定Pi0.predict复制动作位置切片后原地加基座偏移，32位动作按32位舍入；原适配器的非原地相加会提升为64位。现在保留原生数据类型计算后再无损写公共数字，增加32／64位精确检查；不改变公共契约、控制器、预算或评分。[细节和新旧代码摘要](../outputs/stage4-delivery-20261006T1435Z/pi05-vlabench-precision-review.json)。1450Z旧试验不能作精度修正后证据，完成后仍保留，新代码须另跑。
- 三套环境回归增至 **202项，无跳过**，原85项、8契约示例、五组CLI0/0/0/1/2不变：[VLABench](../outputs/stage4-native-f32-regression-vlabench-stage4-20261006T1606Z/baseline-check.json)、[LIBERO](../outputs/stage4-native-f32-regression-libero-openpi-stage4-20261006T1606Z/baseline-check.json)、[RoboTwin](../outputs/stage4-native-f32-regression-robotwin-stage4-20261006T1606Z/baseline-check.json)。新增2项是Model侧原生精度检查，不是GPU／Benchmark成绩。
- LingBot首次完整闭环在begin（连接模型的阶段）因仿真环境缺websockets而停止，模型预测0、策略动作0：[失败证据](../outputs/stage4-closed-loop-lingbot-robotwin-20261006T1600Z/receipt.json)。补固定websockets15.0.1、msgpack1.1.1；147包依赖检查和原生float32消息往返检查通过，1611Z新目录重跑。该旧错误虽由原生评分器返回有效false，不是实际模型任务失败率。原生自带TOPP失败处理保留，只禁止额外环境开关替换规划规则；不宣称原生内部没有回退路径。
- [1200次长窗口GPU采样](../outputs/stage4-gpu-continuous-20261006T1520Z.json)：实际15:20:10至16:03:03，八卡逐样本总体时间平均 **69.9723958333%**，高于20%；每卡采样均值也均约69–71%。实际跨度约43分钟，2秒为设定间隔，查询开销计入墙钟跨度；不能称精确40分钟或任务从开始至今的全程平均。含现有合成占卡，不作为模型效率。后续900次采样继续覆盖本批真实运行，背景负载未改变。

### 16时后续：四组新真实成功，严格复现差异单列

- OpenWAM RoboTwin专用7文件／24,835,226,524字节完整核验并发布：[最终回执](../outputs/stage4-download-openwam-robotwin-20261006T1541Z-ranges.json)。当前六组所需具名资产全部就绪，不能继续称“匹配大权重尚未齐”。已固定校验的原包、失败缓存和原生tracked源码均未改写。
- **LingBot×RoboTwin真实完整单次成功**：[回执](../outputs/stage4-closed-loop-lingbot-robotwin-20261006T1611Z-transport-fixed/receipt.json)，111原生动作、9648物理步、38.59200183302164仿真秒、111框架预测调用／3次原生策略请求。独立原生关节动作重放112份快照、指令和最终成功结果精确一致，实际PhysxCpuSystem调用另计9648步：[重放](../outputs/stage4-native-action-replay-lingbot-robotwin-20261006T1633Z/receipt.json)。未重新调用模型，不是正式验收或全任务计数证明。
- **OpenWAM×RoboTwin真实完整单次成功**：[回执](../outputs/stage4-closed-loop-openwam-robotwin-20261006T1618Z/receipt.json)，113原生动作、11425物理步、45.700002170633525仿真秒。独立原生末端动作重放严格失败：同设备第一次状态差异在动作1，113动作最终10571物理步，仍成功：[完整差异回执](../outputs/stage4-native-action-replay-openwam-robotwin-20261006T1650Z-same-device/receipt.json)；不同设备先前尝试的112步提前结束错误也保留，不认定完全一致或单独归因于硬件。
- **OpenWAM物理计数专项通过，但轨迹对照仍失败**：[同时观察回执](../outputs/stage4-native-meter-crosscheck-openwam-robotwin-20261006T1655Z/receipt.json)。新执行112动作、10627步，生产Scene.step计数器与底层PhysX观察器逐动作一致。`simultaneous_meter_cross_check_passed=true`只说明当前计数正确；整体`passed=false`仍保留历史轨迹复现失败。没有把10627当成历史11425，更没有改原回执计数或容差。
- **π0.5×LIBERO重新推理原生试验成功，但严格对照失败**：[第一次](../outputs/stage4-fresh-native-policy-pi05-libero-20261006T1647Z/receipt.json)、[重复试验](../outputs/stage4-fresh-native-policy-pi05-libero-20261006T1652Z-repeat/receipt.json)。两次都直接调用固定原生环境／官方WebsocketClientPolicy／新模型服务，102策略动作、21请求、成功；未使用记录动作控制环境。均从第一动作与103步框架历史试验有差异，重复试验首动作最大绝对差0.0008084111328124921。原生两次初始快照均与历史精确一致，其第一步图像本身也有跨进程差异；完整原因尚未定位，不宣称全部由模型随机性造成。首次原生模型输入和Adapter输出、两套消息打包后数值／类型全部精确一致：[输入专项](../outputs/stage4-delivery-20261006T1435Z/pi05-libero-native-input-check.json)。这些新结果不撤销此前“103记录动作原生重放精确一致”的特定结论，也不能把旧重放充作新推理严格通过。
- π0.5×VLABench旧试验正常300动作／30仿真秒结束：有效success=false、progress=0.5、intention=1，退出0／预算耗尽；[旧回执](../outputs/stage4-closed-loop-pi05-vlabench-20261006T1450Z-optimized/receipt.json)及[旧原生动作重放](../outputs/stage4-native-action-replay-pi05-vlabench-20261006T1620Z-superseded/receipt.json)保留但被浮点精度修正后的新试验替代。修正后的1618Z-native-f32完整闭环尚在运行，不提前给成绩。
- 最终RoboTwin仿真环境补齐147包后的 [202项重回归](../outputs/stage4-final-runtime-regression-robotwin-stage4-20261006T1640Z/baseline-check.json)仍无跳过，8契约示例和CLI0/0/0/1/2不变。[1700Z覆盖账本](../outputs/stage4-implementation-coverage-20261006T1700Z.json)明确分开真实任务成功、原生严格差异、同时计数专项和验收，49外部缺口未抹掉。[后续900次GPU窗口](../outputs/stage4-gpu-continuous-20261006T1607Z.json)总体约69.63%，又启动1800次继续覆盖等待阶段，不更改现有占卡。

当前能够自主继续的是完成修正后的π0.5×VLABench试验、相关原生对照和证据收尾。若后续为严格重复性引入新的随机数／确定性开关、控制器版本或数值容差，须先说明具体影响；本批没有悄悄改评测规则，也没有开始阶段5。

## 17时最终集中试验与交付更新

本节替代上文“π0.5×VLABench仍在运行”等当前状态描述，历史证据不改写。六组源代码明确的适配器均已完整注册，五组新增固定单次真实闭环均完成且成功；其余49格保留外部缺口，不用占位实现充数。

| 新增模型／Benchmark | 原生动作 | 仿真秒数 | 官方单次成功 | 独立记录动作重放 |
| --- | ---: | ---: | --- | --- |
| π0.5／LIBERO | 103 | 5.15 | 有效true | 103动作／104快照精确一致 |
| OpenWAM／VLABench | 139 | 13.9 | 有效true，progress和intention均1 | 139动作／140快照与三项评分一致 |
| π0.5／VLABench | 165 | 16.5 | 有效true，progress和intention均1 | 165动作／166快照与三项评分一致 |
| OpenWAM／RoboTwin | 113 | 45.700002170633525 | 有效true | 严格失败，首次轨迹差异在动作1 |
| LingBot-VLA 2.0／RoboTwin | 111 | 38.59200183302164 | 有效true | 111动作／112快照及9648物理步精确一致 |

- [π0.5×VLABench新真实回执](../outputs/stage4-closed-loop-pi05-vlabench-20261006T1618Z-native-f32/receipt.json)与[新独立原生重放](../outputs/stage4-native-action-replay-pi05-vlabench-20261006T1716Z/receipt.json)已完成；165次框架预测实际33次原生服务请求。原生累计16.499999999999737秒与固定周期账本16.5秒的微小表示差保留，不改回执。旧300步有效失败及其精确重放单列；一次新旧结果变化不能证明精度修正改善了模型质量。VLABench重放不复现缺失的历史指令随机数状态，不重新调用模型。
- **LingBot重新推理的严格对照未通过**：[第一次](../outputs/stage4-fresh-native-policy-lingbot-robotwin-20261006T1722Z/native/receipt.json)112动作／9757物理步，[重复试验](../outputs/stage4-fresh-native-policy-lingbot-robotwin-20261006T1730Z-repeat/native/receipt.json)111动作／9634物理步，均3次原生策略请求且任务成功。完整按原生历史筛选100000→100001，不用记录动作控制环境。初始公开状态／三相机像素与框架完全一致，首个预测动作即不同。重复试验首轮官方请求与Adapter输入映射、双方消息编码后的数值和类型全部一致；两次原生首动作本身也不同，最大混合动作分量差0.0006865592004032806，不是统一米制误差。[只读比较摘要](../outputs/stage4-delivery-20261006T1435Z/lingbot-fresh-repeat-check.json)。完整原因尚未定位，不能把所有差异归为随机性或认定Adapter全无问题。此前记录动作精确重放仍是另一项有效结论。
- **OpenWAM RoboTwin重放的种子历史专项也未消除轨迹差异**：[再次重放](../outputs/stage4-native-action-replay-openwam-robotwin-20261006T1702Z-seed-search/receipt.json)精确重现100000原生UnStableError拒绝、100001通过及指令，仍从动作1出现差异；113动作、10583物理步、42.33200201066211秒、成功。不是历史11425物理步复现通过。此前同次生产计数器／PhysX观察器112动作10627步精确一致的专项保持独立。
- 当前三处已观察到的严格复现缺口分别是π0.5×LIBERO重新推理、LingBot×RoboTwin重新推理、OpenWAM×RoboTwin原生轨迹重放。VLABench两条路线只有记录动作重放通过，完整新模型控制的原生策略对照仍未做。没有放宽容差、改控制器、改随机数开关或官方评分来消除失败。
- [VLABench](../outputs/stage4-delivery-regression-vlabench-stage4-20261006T1740Z-final/baseline-check.json)、[LIBERO](../outputs/stage4-delivery-regression-libero-openpi-stage4-20261006T1740Z-final/baseline-check.json)、[RoboTwin](../outputs/stage4-delivery-regression-robotwin-stage4-20261006T1740Z-final/baseline-check.json)最终三套回放均202项通过、0跳过；原85项、8契约示例、五组CLI0/0/0/1/2完整保留。它们仍是软件／合成证据。

当前资源、55格覆盖和全部新真实／严格差异回执统一索引于[stage-state.json](stage-state.json)，离线架构与细节见[HTML报告](../outputs/stage4-delivery-20261006T1435Z/EmbodiedEval-architecture.html)。原迁移86/86摘要通过后，收尾又单独核对原35份阶段2历史产物与验收材料均未变化；当前源码与文档另建新摘要，不改旧清单，也没有伪造git提交。阶段4尚未正式验收，契约未冻结；没有进入阶段5。大权重完整发布与场景解包就绪，失败缓存保持原样，模型试验进程只清理本次所有进程组，原占卡未动。最后长窗口GPU证据在完成后写入索引，均值包含占卡，不是评测效率。

最终长窗口已完成：[1800次采样](../outputs/stage4-gpu-continuous-20261006T1640Z.json)，实际16:39:37.584605至17:43:53.548285，八卡总体均值69.69916666666667%。前1200次69.97239583333334%、后900次69.62555555555555%的两个窗口也均超过20%。从每份原始八卡读数独立重算与回执一致；包含原有合成占卡，不能称为模型评测效率或未采样时段全程平均。本次模型服务／试验进程已全部正常回收，原占卡父进程及八子进程保持原样。最终资源与干净的10份原生tracked源码见[1745Z盘点](../outputs/stage4-preparation-inventory-20261006T1745Z-final.json)，新代码摘要和单次／严格验证门禁见[1745Z覆盖账本](../outputs/stage4-implementation-coverage-20261006T1745Z-final.json)。

## 2026-10-07：已批准M+N独立0.2通路实施与证据

本节是新增架构的交付索引，不修改上文0.1历史记录。M指模型数量，N指仿真Benchmark数量；M+N表示双方各实现自己的边界，通过接线数据组合，不按每一对新增Python适配器。用户明确要求实施该方案，批准本轮公共0.2扩展；正式阶段4验收和契约冻结仍未确认。

已完成独立注册、参数/资产/来源检查、带含义的公共观测与动作、接线匹配、GPU-free预检查（不启动GPU推理或仿真）、通用串行Runner与六组迁移。0.1配置、五组CLI、历史Mac路径和历史验收产物保留原行为。新通路不调用旧组合工厂；模型不接收环境实例或私有任务，环境不接收模型配置/端点/权重。控制器、逆运动学、物理推进和评分仍由环境执行边界负责。

实现拆成3个模型家族的6个具名输入输出配置，以及3个环境家族的5个独立协议。训练覆盖只作提示，缺输入、单位/动作模式不符、坐标不明、遗漏双臂/关节或原生扩展签名不符仍拒绝。真实配置和组件资产检查见[独立预检查回执](../outputs/mn-preparation-20261007-final-verified/receipt.json)，完整语义见[0.2契约](contract-v0.2.md)。

四套Linux环境最终软件回放均263项通过、0跳过，即原202项加61项新检查；旧CLI退出码仍0/0/0/1/2：[轻量环境](../outputs/mn-baseline-20261007-final/baseline-check.json)、[LIBERO](../outputs/mn-baseline-libero-20261007-final/baseline-check.json)、[VLABench](../outputs/mn-baseline-vlabench-20261007-final/baseline-check.json)、[RoboTwin](../outputs/mn-baseline-robotwin-20261007-final/baseline-check.json)。新测试包括两个不同模型和两个不同仿真的四种组合，以及单独注册第三模型/第三环境，均明确为合成架构证据：[保存的六组示例](../outputs/mn-synthetic-architecture-20261007-final/receipt.json)。原生模型边界另用合成服务验证π0.5前5行动作缓存与LingBot整段缓存/取图时机，不代表真实推理。

### 新真实单次与独立原生执行重放

| 新0.2入口 | 动作次数 | 物理步 | 仿真秒数（表内四舍五入） | 官方单次结果 | 记录动作原生重放 |
| --- | ---: | ---: | ---: | --- | --- |
| OpenWAM / LIBERO | 74 | 未单列 | 3.700000 | 有效成功 | 74动作、75快照精确一致 |
| π0.5 / LIBERO | 102 | 未单列 | 5.100000 | 有效成功 | 102动作、103快照精确一致 |
| OpenWAM / VLABench | 139 | 未单列 | 13.900000 | 成功，progress/intention均1 | 139动作、140快照精确一致 |
| π0.5 / VLABench | 153 | 未单列 | 15.300000 | 成功，progress/intention均1 | 153动作、154快照精确一致 |
| OpenWAM / RoboTwin | 113 | 10685 | 42.740002 | 有效成功 | 严格失败：动作1首差异，10502物理步 |
| LingBot-VLA 2.0 / RoboTwin | 112 | 9743 | 38.972002 | 有效成功 | 112动作、113快照及9743物理步精确一致 |

六组原始目录通过[新的只读索引](../outputs/mn-real-index-20261007-final/receipt.json)关联，没有改写或复制旧trace。单次不等于全suite或泛化能力；表中舍入只为阅读，原始秒数和比较规则保留。固定周期路线没有单列物理步，不能推造。每个新真实manifest依然保留`server_identity_verified=false`：本地权重摘要和受监督启动命令不是服务端可信权重身份回执。

全部6组转换夹具通过：[原生输入/动作转换回执](../outputs/mn-native-mapping-20261007-final-direct-source/receipt.json)。历史真实公开观测配合固定合成输出；原生客户端函数/未改写的纯源码片段直接执行，记录来源和摘要；不调用新模型或计入评分。独立记录动作重放另见：[OpenWAM/LIBERO](../outputs/mn-native-replay-20261007T1020Z-openwam-libero/receipt.json)、[π0.5/LIBERO](../outputs/mn-native-replay-20261007T1019Z-openpi-libero-retry/receipt.json)、[OpenWAM/VLABench](../outputs/mn-native-replay-20261007T1053Z-openwam-vlabench/receipt.json)、[π0.5/VLABench](../outputs/mn-native-replay-20261007T1122Z-openpi-vlabench/receipt.json)、[OpenWAM/RoboTwin](../outputs/mn-native-replay-20261007T1122Z-openwam-robotwin/receipt.json)、[LingBot/RoboTwin](../outputs/mn-native-replay-20261007T1120Z-lingbot-robotwin/receipt.json)。VLABench重放固定记录指令，未重采样Python指令随机数，仍不代表完整新策略对照。

### 已发现并明确处理的边界，以及未消除的差异

- LingBot首轮0.2试验在第一条动作执行前因越界夹爪请求被公共开度校验拒绝，执行0动作；[失败保留](../outputs/mn-real-lingbot-robotwin-20261007T1043Z/lingbot_robotwin/framework/summary.json)。固定RoboTwin源码是先插值原始请求、再每物理步截断；模型端提前截断会改变轨迹。新增同签名具名原生夹爪请求，12个关节仍用公共物理命令；公共夹爪开度的0到1范围不放宽。
- 固定RoboTwin夹爪访问器返回控制目标，不是测得的手指位置。两个夹爪状态改标`gripper_drive_target/controller_target`，关节驱动目标也不当实测；两条路线新目录重跑，不改旧记录。动态公开指令更新保留公开目标图像/位姿和采集时间，不泄露私有评分字段。
- [新旧重新推理首输入/首动作专项](../outputs/mn-fresh-differences-20261007-final.json)显示六组首输入数值/类型全部相同，π0.5两条路线和LingBot首动作不同；OpenWAM首动作相同也不说明全程一样。没有受控实验不能确定是采样、后端数值还是服务状态，不能直接全归因随机性。
- OpenWAM/RoboTwin新重放同设备、相同种子筛选与指令，仍从动作1出现差异。[同轮计数专项](../outputs/mn-meter-crosscheck-20261007T1130Z-openwam-robotwin/receipt.json)新执行113动作，生产计数器与独立PhysX观察器逐动作都计10518步。它证明该轮计数正确，但不能把10518或10502当成框架历史10685，也不消除轨迹复现失败。PhysX具体指底层物理仿真系统。
- GPU3的π0.5首推理发生CUDA非法访存，未执行动作；[失败保留](../outputs/mn-real-trials-20261007T1005Z-retry/openpi_libero/framework/summary.json)。小型Torch运算两设备均通过，仍不能认定硬件坏了或排除特定JAX工作负载问题。转GPU2完成验证；未改官方评分、确定性开关、占卡或全局代理。

### 覆盖与交付边界

原49缺口重新分类：31格不在直接动作模型范围，5格动作型号未定，4格确切LingBot-VA身份未实现，8格仿真协议未实现，1格模型输入输出配置缺失。原表是LingBot-VA，而本次迁移的是LingBot-VLA；不能用另一个模型抵扣VA缺口。没有占位Adapter或“全部55格完成”的声明。6个模型配置×5个环境协议的30格显式候选接线中10格结构通过，未真实运行的交叉项仍未验证；不是运行时组合白名单。

完整架构、人话说明、逐格覆盖、资源及GPU原始窗口见[新增HTML报告](../outputs/mn-delivery-20261007-final/M+N-architecture-and-evidence.html)和[覆盖账本](../outputs/mn-delivery-20261007-final/coverage-ledger.json)。新摘要清单位于[本轮源码/交付封存回执](../outputs/mn-integrity-20261007-final/receipt.json)，不改旧SHA256SUMS、不伪造Git提交。总体GPU采样均值含既有占卡；占卡单独记录，不计评测成绩或工作量。不新增大权重下载，没有使用Mac虚拟环境。DataRef读取、硬超时、并发Runner、自动恢复与视频转动作规划器均未建设。上述交付仍待正式阶段4验收及契约冻结，不进入阶段5。

本轮最后GPU窗口已结束：[1800次原始采样](../outputs/mn-gpu-window-20261007T1035Z.json)，实际10:35:35.018878至11:39:56.959057，八卡采样均值60.89319444444445%。另外两个完成窗口均为约60.76–60.87%，所有窗口样本合并均值60.85033333333333%；部分窗口重叠，不能称为整轮唯一时间加权平均。三个窗口都超过20%，但仍含原有占卡，不证明评测本身单独达到20%。本轮自有模型服务和试验进程已清理，占卡未动；新HTML通过静态校验、全部本地证据链接检查及35份历史阶段2摘要核对，未执行浏览器视觉检查。

## 2026-10-07追加：暂缓LingBot-VA，推进其他具名组件

用户要求“不管lin'bot va，把别的先都搞定，然后继续推进”。本轮按暂缓LingBot-VA处理；既有LingBot-VLA是不同模型，继续保留。新增直接输出机器人动作的型号和源代码已明确的仿真协议；不新增视频转动作规划器，不猜原表R009/R010、R021/R022或WIW角色。没有自动通过阶段4验收或冻结契约。

### 实现范围与边界

现有5个真实模型家族、10份输入输出配置：OpenWAM原3份加ARX绝对双臂位姿，π0.5/openpi原2份加ARX双臂关节，LingBot-VLA保留，新增Cosmos3-Nano-Policy-DROID与DreamZero-DROID。现有5个仿真家族、7份具名协议：LIBERO两份、VLABench一份、RoboTwin两份，新增DROID sim_evals与RoboDojo。没有占位Adapter抵扣缺口。

新增DROID具名协议覆盖源代码的场景1/2/3，配置示例选择场景1，没有官方任务成功评分，结果应标不可取得而不是编造成功率。新增RoboDojo限stack_bowls、ARX-X5双臂、布局集合seed0/index0，不代表42个任务全部接入。只提取固定环境源码中的12个环境方法，去掉模型reset调用；不执行上游策略客户端、调度、恢复或写结果代码。任务初始化、观测、原生控制与评分仍归环境侧。

π0.5 ARX输出12个有单位的关节目标加两个具名夹爪原始请求。原始夹爪请求表示要求的开度，可暂时超出0到1，具体截断与插值顺序由环境协议决定；公共合法开度范围没有放宽。OpenWAM ARX把20维权重表示解码为有坐标含义的双臂绝对位姿与夹爪，不给旧含义冲突的原生数组贴上通用标签。模型仍不接收环境对象，环境仍不接收模型配置；Runner未增加任何模型或控制器分支。

RoboTwin为新物理输入采集实际关节角，不把驱动目标冒充实测；夹爪目标继续明确标控制目标。新物理执行边界观察原生IK、位姿规划和轨迹计时结果：失败或空轨迹明确反馈，不把没执行的手记成成功；不改变原生规划结果，不自动重试或回滚，保留实际动作/物理步/仿真秒数。旧0.1和具名旧原生数组路径保留原行为。

相机接线也已逐个核对固定源码。Cosmos按实际左右相机位置接线；DreamZero仿真客户端却把external_cam叫right_image，所以两者的示例数据接线不同。这里是原生变量命名差异，不是在环境Adapter里按模型选择实现。旧示例保留，最新结构与源函数回执注明修正。

### 本轮证据及明确失败

- 软件回归：306项，零跳过，原85项、202项以及旧CLI退出码0/0/0/1/2无回归。五套既有环境与新增Cosmos/DreamZero/OpenWAM依赖环境里的三套软件诊断均通过，后者不等于原生运行依赖已认证。[最新结构/回归索引](../outputs/mn-extension-preparation-20261007-final3/receipt.json)；9项[固定源函数比较](../outputs/mn-extension-native-mappings-20261007-final2.json)全部使用合成输入，没有模型加载、仿真或任务成绩。6份新配置结构通过，仍不是运行时组合名单。
- π0.5 ARX：[固定18文件权重与normalization核验](../outputs/mn-extension-preparation-20261007-final3/assets.json)。原声明NumPy低于2与LeRobot0.4.4间接要求至少2冲突；新独立兼容环境采用源码支持的旧LeRobot导入分支0cf8648、rerun-sdk0.21.0及源锁optax/chex。[209包版本/依赖检查](../outputs/mn-extension-pi05-arx-package-versions-20261007.json)通过，不声称训练环境已复现。
- [GPU2真实模型、合成观测检查](../outputs/mn-extension-pi05-arx-gpu-smoke-20261007-gpu2/model-result.json)：匹配权重加载后产生50条14维动作，1次原生请求、50次Adapter取行、缓存为0。没有仿真或官方成绩。[GPU3原生CUDA非法访存观察](../outputs/mn-extension-pi05-arx-gpu3-abort-20261007.json)保留，不能据此断定硬件坏了。
- π0.5 ARX／RoboTwin两次真实有限步诊断：[第一轮](../outputs/mn-extension-pi05-arx-robotwin-runtime-20261007-run1/receipt.json)50动作、1486物理步、5.944000282324851秒；[重复](../outputs/mn-extension-pi05-arx-robotwin-runtime-20261007-final/receipt.json)50动作、1523物理步、6.092000289354473秒。两次框架错误0、有效官方success=false、预算耗尽，均使用原生筛选种子100001。不是全任务/全suite验证，不证明ARX训练权重能成功迁移，也没有原生新策略严格对照或可信服务端权重身份。
- [只读重复比较](../outputs/mn-extension-pi05-arx-bounded-repeats-20261007.json)：首轮state/images/prompt的数值和类型完全一致，首动作不同，最大单关节目标差0.003164157221350772弧度，夹爪请求也不同。没有受控实验不能区分随机采样、后端计算或服务状态，不把差异全部归因随机性，不拿任务成败代替映射正确。
- Cosmos/DreamZero/OpenWAM ARX的新独立原生依赖环境安装完成，DreamZero/OpenWAM模块导入通过；没有启动这些新权重的真实推理。Cosmos的403包检查仍有openpi-client需要NumPy低于2、jupyter-compare-view不支持Python3.13两处不兼容，配置引入Megatron路径还抛RuntimeError，不能称为运行就绪。
- DROID源锁固定IsaacLab2.2.0/IsaacSim5.0.0.0。[首次失败](../outputs/mn-extension-droid_sim-env-20261007-run1/receipt.json)、[具体缺模块诊断](../outputs/mn-extension-droid-sim-env-20261007-build-compatible-run4/receipt.json)保留。flatdict4.0.1构建引用已删除的pkg_resources；固定setuptools80.9.0并核验原锁sdist摘要后引导构建通过，但[完整环境两路线均超时](../outputs/mn-extension-droid-sim-env-20261007-build-bootstrap-run5/receipt.json)，继续有界续装。只改生成的安装输入和本轮自有环境，未改参考仓库、运行协议、共享驱动或旧环境。
- DROID的8个文件、RoboDojo所选412个文件摘要通过不等于场景完整：USD根文件还引用远端资产或运行时材质模块。[只读引用检查](../outputs/mn-extension-usd-audit-20261007-run2.json)不替代原生Isaac渲染。DROID scene1引用的my_droid.usdz所在robot节点未启用，不能单凭该引用判运行必失败。RoboDojo要求的IsaacSim5.1文档驱动580.65.06高于当前570.124.06，当前未验证兼容；未擅自升级共享驱动。

### 剩余范围与交付

原49格保留为31格超出直接动作模型范围、4格用户暂缓VA、9格原表协议/角色不明、2格Cosmos动作/观测结构不满足，以及3格新增接线通过但完整运行验收未完成。后3格是π0.5／RoboTwin、π0.5／RoboDojo、OpenWAM／RoboDojo；π0.5／RoboTwin的有限步诊断另列，不将该格改成全任务完成。全部55格中VA暂缓共5格，包含曾误把VLA算作VA的旧格。新增DROID不自动替代R021/R022。

Cosmos/DreamZero/OpenWAM ARX大权重仍在固定摘要分块下载；原下载结束后各最多2轮受监督续传，失败残片保留，只在整文件核验后发布，不自动开始模型/仿真。新的完整说明、架构边界、原49格逐项人话说明、资产和失败证据见[本轮HTML报告](../outputs/mn-extension-20261007-final2/M+N-adapter-extension-and-gaps.html)及[覆盖账本](../outputs/mn-extension-20261007-final2/coverage-ledger.json)。报告遵循单页静态HTML模板，回环预览只提供报告与明确链接的证据；远程浏览需实际端口转发，不上传公网。历史报告、35份阶段2证据和Mac绝对路径不回写。

最新完成[180次GPU采样](../outputs/mn-extension-gpu-20261007-run4.json)八卡总体均值61.06319444444444%，另3个本轮窗口也均超过20%。窗口可能重叠，只保证所采窗口，不声称未采样时段全程达标；现有占卡负载单列，不计评测工作量或成绩。两次π0.5自有模型服务均已回收，原占卡未动。DataRef读取、Runner硬超时/并发/恢复、硬仿真时间预算和视频转动作规划仍未建设。继续新增组件真实闭环与逐项原生对照；阶段4尚未正式验收，契约未冻结，不进入阶段5。

## 2026-10-08：继续准备正式验收（本节仍是进行中记录）

用户要求“继续推进，直到可以正式验收”。继续已批准的阶段4和独立M+N范围，LingBot-VA仍暂缓；不将准备工作解释为验收已通过，不改历史报告、评分规则或共享驱动。最新事实以stage-state.json的mn_acceptance_preparation_20261008为准，早期320项测试等快照保留，不回写成新数字。

### 软件与模型运行隔离

最新五套环境均347项通过、零跳过，原85项、202项以及五组CLI退出码0/0/0/1/2保留：[轻量](../outputs/mn-acceptance-baseline-runtime-0-20261008-run3/baseline-check.json)、[DROID](../outputs/mn-acceptance-baseline-runtime-1-20261008-run3/baseline-check.json)、[OpenWAM ARX](../outputs/mn-acceptance-baseline-runtime-2-20261008-run3/baseline-check.json)、[RoboDojo](../outputs/mn-acceptance-baseline-runtime-3-20261008-run3/baseline-check.json)、[Cosmos](../outputs/mn-acceptance-baseline-runtime-4-20261008-run3/baseline-check.json)。这些是软件与合成架构证据，不是347个真实任务成绩。

原生websockets12与框架16的参数不同，运输边界改为按实际API传递参数，不改消息编码或采样顺序，也无自动重试。两个版本各一次真实本机socket测试通过，传输合成float32数组、数值/类型完全一致：[版本12](../outputs/mn-acceptance-socket-ws12-20261008-run1/receipt.json)、[版本16](../outputs/mn-acceptance-socket-ws16-20261008-run1/receipt.json)。它们不是策略服务已验证。

OpenWAM ARX需要transformers5.5以上，固定RoboDojo/IsaacLab需要4.57.6，不能强塞进同一个Python环境。模型组件增加runtime_python（选择原生解释器）与runtime_gpu（只选择该模型进程使用的GPU）配置；只用于此具名模型配置，不改变公共观测/动作或Runner。私有本机管道只传公开数据契约对象，模型预处理、推理、动作解码和缓存全部在模型自己的进程中；不接收环境对象，不按Benchmark名称分支，不新增远程协议、重试、恢复、硬超时或并发trial。

[真实权重隔离检查](../outputs/mn-acceptance-openwam-model-process-20261008-run1/receipt.json)通过：GPU4原生生成一次，顺序消费32条动作，全部通过公共动作校验并保留float32；主进程未导入Torch或Isaac，原生模型关闭回复及退出均检查。输入是合成图像/状态，所以还不是机器人场景闭环。此前直接加载模型的32条原生源码映射对照仍独立保留，不能拿这次合成检查代替真实观测对照。

### π0.5首动作差异的受控调查

在同一份已保存的真实LIBERO起始观测上，完成三个独立原生模型进程、共九次真实推理，未新增仿真任务。在每个进程内恢复相同的原生随机数状态，整段10×7动作逐值完全一致；换为下一份随机数状态，动作发生改变。这里“随机数状态”是推理所用随机数生成器的内部记录，不是更改任务seed。

但两次独立进程即使首输入数值/类型、随机数状态、包版本、GPU及源码摘要都相同，首动作最大分量仍相差0.0016450205078125668，整段也不精确一致：[受控比较](../outputs/mn-acceptance-pi05-controlled-comparison-20261008-run1.json)。因此可以证明随机性影响输出，不能证明历史差异全由随机噪声变化造成；具体跨进程非确定性计算环节尚未定位。没有启用新的生产确定性开关、改官方采样或放宽容差。首轮缺jsonschema失败及其独立诊断依赖层均保存，历史原生环境未改。

### RoboDojo原生部署与尚未通过的场景门禁

固定IsaacLab与CuRobo两份gitlink源码已取得；gitlink指父仓库明确指定的子仓库提交，不是追踪变化的main：[来源回执](../outputs/mn-acceptance-robodojo-sources-20261008-run3/receipt.json)。新独立环境的基础和原生安装命令已完成，但固定IsaacLab的starlette0.49.1声明与RoboDojo安装脚本最终0.45.3冲突，沿用脚本最终值并公开偏差，不宣称全部依赖一致或直播功能已验证。

真实场景启动依次暴露并保留了以下失败：AppLauncher参数对象类型不符；核心模块导入缺h5py；机器人配置只有curobo_tmp.yml而没有按原生规则生成的curobo.yml；普通CuRobo wheel未带原生YAML配置。启动参数已修正；h5py按固定源码补入独立环境；路径模板只在本次产物目录生成，不改参数和参考资产；CuRobo按原生editable方式安装，971份tracked安装副本文件与固定参考摘要一致。editable表示运行时读取该安装副本，避免普通wheel漏掉配置资源，不表示改写原仓库。

另一次失败在原生应用关闭后的解释器收尾发生段错误，退出码-11。RoboDojo环境应用已采用自己的子进程隔离，必须完成原生关闭、返回回复并验证子进程退出0；不能仅凭exit0、生成一个中间文件或成功导入认定场景通过。动作执行失联仍记已开始且状态未知，不自动重试。当前第五轮真实reset/两条保持动作诊断仍在进行；前四份失败回执保留，尚无通过的场景重置或模型控制结论。

### DreamZero下载与GPU记账

DreamZero主checkpoint已完整取得，原生构造器还必须先加载固定Wan2.1-I2V-14B-480P的15份组件，共82,243,108,389字节。不能跳过构造器步骤或拿别的Wan版本代替。先前整文件及Xet下载失败/超时保留；新单个8MiB（8×1024×1024字节）范围通过了严格HTTP范围检查，临时直连失败后使用原代理：[小范围回执](../outputs/mn-acceptance-dreamzero-range-20261008-run1/receipt.json)。现继续固定版本的持久分块下载，残片和局部块不是完整资产，整文件SHA256通过前不启动模型。两卡真实推理检查脚本已准备，尚未执行；不能宣称DreamZero权重加载或真实闭环已完成。

本轮已完成两个600次GPU采样窗口，八卡总体均值分别60.620625%与60.82958333333333%，均超过20%；第三窗口继续。包含原有占卡，不能当成评测工作量、效率或未采样时段的全程保证。占卡未停、全局代理未改、共享驱动未改。后续继续真实场景/模型运行和逐项原生对照，完整材料另行提交用户正式验收。

### 2026-10-08后续：RoboDojo真实重置通过，动作验证仍在继续

第五轮失败定位为Adapter错误地将应用的CUDA渲染设备传入了物理仿真配置。固定原生main.py并不做这个覆盖；固定sim_config.yml省略device，BaseEnv因此选择CPU物理，而渲染和CuRobo仍使用GPU。已撤销错误覆盖，不改原生NumPy评分、控制器或判定门槛。CPU物理不是失败后另造的回退方案，而是恢复固定来源的原行为。

[第六轮实际场景记录](../outputs/mn-acceptance-robodojo-reset-20261008-run6/component-result.json)中，重置已通过，三路相机均为480×640的RGB图像，双臂六个实测关节和末端位姿有效；夹爪仍明确记录控制目标，不能当作实测位置。这轮整体仍失败：首条保持动作发现遗漏了have_empty/get_control_empty这两个原生控制队列方法。已增加原文方法提取及两项合成防回归检查，14个环境方法均仍禁止模型依赖，第七轮真实重置和两条保持动作正在重跑。重置通过不等于动作通过，也不等于模型任务通过。

DreamZero备用整文件任务因回执子目录未及时建立而退出，部分对象回执及残片保留，没有最终通过回执；已修复后续任务的目录创建及单对象安全错误记录。当前持久分块和另一条Xet备用传输继续，任何局部传输量均不计为完整已校验资产。

### 后续原生条件登记修正与通过的保持动作

第七轮首次保持动作后出现“已结束”，诊断仍请求第二动作被拒绝；第八轮诊断正确遵循结束标志退出，但随后查明这是Adapter遗漏了原生run_eval开头的run_reward/get_score登记调用。原生RewardManager把空条件视为条件已经完成，因此这个异常成功不能用作任务结果；第八轮文件保留，不作为修正后证据。

已在原生reset清空状态后、任何动作前调用原生任务条件和过程分数登记；登记后检查此具名任务必须有一组成功条件、两级transition过程分数。没有改成功判断、评分函数或阈值。结束收分时不再重新调用Task.get_score登记函数，避免清空过程分数；只读取原生RewardManager累计分数。增加三项相应合成检查。

[第九轮修正后实际保持动作](../outputs/mn-acceptance-robodojo-reset-20261008-run9/receipt.json)完整通过：两次原生动作、20个物理步、0.08秒仿真时间，结束标志为false，正常关闭回复及子进程退出0均已检查。三路图像和双臂实测状态有效；没有策略模型推理或任务成绩。OpenWAM／RoboDojo四动作真实模型试验另行进行，不能用保持动作抵扣。

### RoboTwin原生规划差异进一步定位

[两次原生首动作比较](../outputs/mn-acceptance-robotwin-planning-comparison-20261008-run1.json)均仅使用同一条保存的真实动作，不重新调用模型，也不经过Benchmark Adapter。目标数组、类型、两臂完整实测关节位置和速度、包版本、机器人源文件完全一致；两次均94个物理步。但左臂原生left_plan_path输出不同：轨迹位置最大分量差0.00004646368324756622，速度最大分量差0.002752333879470825；右臂规划输出本次完全一致。实际送出的控制序列也随之不同。这里只将差异定位到可观察的原生规划边界，规划器内部随机状态/缓存或具体非确定性计算原因尚未识别。

其中一次首步完整公开快照与历史精确一致，另一次不一致；不能挑选通过的一次撤销历史完整113动作重放失败。Adapter路径与这两次原生运行的首动作观察仍在继续；没有修改规划种子、物理设置或比较容差。

### 2026-10-08最新实测：真实接线通过，完整任务仍需补齐

五套环境最新均360项测试通过、零跳过，原85项、202项及五组CLI退出码0/0/0/1/2保持不变：[轻量](../outputs/mn-acceptance-baseline-runtime-0-20261008-run7/baseline-check.json)、[DROID](../outputs/mn-acceptance-baseline-runtime-1-20261008-run7/baseline-check.json)、[RoboDojo](../outputs/mn-acceptance-baseline-runtime-2-20261008-run7/baseline-check.json)、[OpenWAM ARX](../outputs/mn-acceptance-baseline-runtime-3-20261008-run7/baseline-check.json)、[RoboTwin](../outputs/mn-acceptance-baseline-runtime-4-20261008-run7/baseline-check.json)。新增检查是交换记录安全与原生重放的完整双臂、原生类型、重复/额外目标拒绝，仍全部是软件或合成检查，不是360个真实任务。

[OpenWAM／RoboDojo](../outputs/mn-acceptance-openwam-robodojo-bounded-20261008-run1/receipt.json)与[π0.5／RoboDojo](../outputs/mn-acceptance-pi05-robodojo-bounded-20261008-run1/receipt.json)分别完成4次真实动作、40物理步、0.16秒仿真时间，正常关闭、退出0；官方success=false、score=0均有效。这里“通过”只表示这两次有限步接入没有框架错误，不是任务成功。恢复后的任务条件登记和累计过程分数读取均参与实测。

使用保存的真实观测，各完成一次新的原生模型推理：[OpenWAM整段32动作映射](../outputs/mn-acceptance-openwam-real-input-mapping-20261008-run2/receipt.json)、[Cosmos整段32动作映射](../outputs/mn-acceptance-cosmos-real-input-mapping-20261008-run1/receipt.json)均通过。相机像素、状态、指令及整段动作转换后的数值和类型，与固定原生纯函数逐项精确相同；顺序消费完整32行缓存。这两项没有新增仿真任务，也没有把另一轮重新推理当作确定性参考。OpenWAM首次诊断的合成形状构造错误保留，不能把失败回执改成通过。

π0.5以800动作为上限的新试验实际在一小时墙钟预算结束时停止：[真实回执](../outputs/mn-acceptance-pi05-robodojo-native-limit-20261008-run1/receipt.json)。实际165动作、1650物理步、6.600000000000005秒仿真时间，官方success=false、过程score=15；最后原生terminated/truncated均为false。因此不能宣称跑完800步或原生任务自然结束。这里“墙钟时间”指现实中经过的时间，和仿真中的6.6秒不是同一个量。

[这165步的原生交换与缓存只读对照](../outputs/mn-acceptance-pi05-robodojo-native-exchange-comparison-20261008-run1.json)通过：4次实际请求输入与固定源码相同，返回的4段各50行动作全部核对，共200行；实际执行顺序是50+50+50+15，共165行，剩余35行不冒充执行量。全部数值和类型精确相同，没有额外推理、网络重试或放宽容差。此项证明边界映射及缓存顺序，不替代完整任务或独立环境重放。

RoboTwin进一步的[Adapter与两次原生首动作对照](../outputs/mn-acceptance-robotwin-planning-comparison-20261008-run2.json)已完成：Adapter与其中一次原生执行的两臂规划结果、全部288条控制操作完全一致；另一原生执行自身左臂规划仍有差异。两次均94物理步。保留完整历史轨迹严格失败，不把首步精确的一次包装成113步全部复现，也不宣称已经定位某个随机数或GPU内核根因。

### DreamZero全套辅助资产完成，新实例上的验证继续

[整套复核](../outputs/mn-acceptance-dreamzero-backbone-verified-20261008-run1/receipt.json)已通过：固定Wan2.1版本6b73f84e的15份辅助文件、82,243,108,389字节均为完整已核验文件，并建立仅此产物目录使用的离线缓存。原生构造器查询main时，该私有缓存明确解析到固定revision；revision指文件仓库的固定版本，不是追随远端最新版本。此前HTTP、Xet及局部修复失败回执保留，完整通过来自新的全套复核，不能回写旧下载任务为通过。主DreamZero checkpoint另有15文件，不能把两套各15文件混成一次模型加载证明。原生双卡权重加载、真实观测推理和输入/动作对照正在启动；尚无正式闭环或官方成绩。

本次续接观察到八张GPU的UUID（每张物理卡的标识）与早前资源快照全部不同，按当前实例重新盘点，不声称跨续接仍为同一张物理卡。[按既有授权释放占卡](../outputs/mn-acceptance-occupancy-release-20261008-run1.json)只停止已核实GPU0、1、2、4上的四个占卡子进程，保留控制父进程和其余四张卡的占卡，不修改程序源码或全局设置。新窗口继续按八卡总体平均至少20%采样；占卡仍不是评测工作量。原五个600次窗口均高于20%，第五窗口60.94833333333333%；只保证已采窗口，不推定未采时段或实例切换期间的全程平均。

独立原生RoboDojo重放的首轮在原生CuRobo/Warp初始化失败，第二轮则是诊断观察器装得过早：固定BaseEnv在reset时才创建仿真，不能在构造后直接读取尚未建立的仿真对象。两份失败回执保留；仅将只读物理计数观察器移到原生reset完成后，第三轮继续验证。未改变原生重置、控制、评分或物理参数。OpenWAM的800动作上限试验使用4小时的现有协作式墙钟预算继续；这是声明的预算配置，不新增Runner硬超时、并发trial、重试或恢复。

阶段4仍未正式验收，契约仍未冻结。本节不将局部映射、有限步接入、文件下载或新实例续接解释为用户已通过阶段4。

### 2026-10-08继续收尾：363项回归、依赖修复与新增原生执行对照

最新五套环境各363项测试通过、零跳过，八份旧契约示例及五组CLI退出码0/0/0/1/2保持不变：[轻量](../outputs/mn-acceptance-baseline-runtime-0-20261008-run8/baseline-check.json)、[DROID](../outputs/mn-acceptance-baseline-runtime-1-20261008-run8/baseline-check.json)、[RoboDojo](../outputs/mn-acceptance-baseline-runtime-2-20261008-run8/baseline-check.json)、[OpenWAM ARX](../outputs/mn-acceptance-baseline-runtime-3-20261008-run8/baseline-check.json)、[RoboTwin](../outputs/mn-acceptance-baseline-runtime-4-20261008-run8/baseline-check.json)。新增3项检查独立DROID记录动作重放的完整七关节、夹爪、原生数值类型以及重复/缺少目标拒绝，仍是合成诊断测试。当前[四种合成交叉及第三模型/环境新增](../outputs/mn-acceptance-synthetic-architecture-20261008-run1/receipt.json)全部通过，无需改Runner或其他Adapter。

DreamZero双卡首轮在实际加载前失败，定位到固定Transformers4.51.3与安装时自动选择的DeepSpeed0.19.7循环导入，不能认为模型权重坏了或已经加载过。固定源码只声明deepspeed、不锁版本；在本项目独立环境中选用官方0.17.6，先校验发布源码包摘要。初次安装还缺CUDA编译器，失败回执保留；[项目内CUDA12.9工具链](../outputs/mn-acceptance-cuda129-private-toolchain-20261008-run1/receipt.json)来自NVIDIA固定发布清单的三个完整校验包，编译检查通过，不安装共享驱动、不改全局CUDA。[依赖修复与原生导入检查](../outputs/mn-acceptance-dreamzero-import-compatibility-20261008-run2/receipt.json)通过，Torch2.8.0+cu129及Transformers4.51.3保持不变。双卡GPU1/6的新原生加载检查继续，尚无模型加载完成、动作推理或闭环通过回执。

独立RoboDojo原生工厂重放缺ffmpeg的失败保留，使用项目内已安装且RECORD内容摘要核验过的FFmpeg7.0.2提供私有PATH，原生录像代码不变。[完成4动作重放](../outputs/mn-acceptance-pi05-robodojo-native-replay-20261008-run4/native-result.json)：五次关节/末端/夹爪目标快照、40个独立观察物理步、0.16仿真秒与原生score/success精确相同；三路相机像素不同，严格比较仍失败。没有拿状态或任务评分相同撤销图像失败。同一当前GPU5上两次原生重复已执行，保存原始RGB、相机外参和渲染设置，继续只读比较，不放宽容差。

[DROID独立原生4动作执行重放](../outputs/mn-acceptance-cosmos3-droid-native-replay-20261008-run1/native-result.json)也已完成：原生双重reset后，直接执行记录关节动作，未调用Model/Benchmark运行方法或新模型推理。状态、32个独立观察物理步、固定周期、结束标志与公开指令相同；相机像素不同，严格比较失败。官方任务成功评分仍不可取得，没有自行添加评分规则。两项图像差异继续调查，不能仅因早前GPU硬件标识不同就宣称根因已查清。

按已有授权另释放已核实GPU6的单个占卡子进程，保留占卡控制器及GPU3/5/7的三个子进程。第六个已完成600次采样窗口八卡总体均值34.916875%，高于20%；下一窗口仍在采样。占卡不计评测工作量或成绩。OpenWAM和π0.5的RoboDojo800动作上限试验仍在原4小时协作式现实时间预算内执行，未取得最终回执，不预记完整任务通过。没有改变公共核心契约、Runner、控制器、评分、共享驱动或全局代理。

另出保留历史内容的[当前验收准备网页快照](../outputs/mn-acceptance-progress-20261008-run1/M+N-acceptance-preparation-and-evidence.html)，包含独立架构、配置、实际回归、复现差异和原49格逐项说明。静态报告不是实时面板，也不代表阶段4已通过。报告按html-report技能保留静态单页模板、明确证据边界及逐项人话解释，受限回环服务需实际端口转发才能在远程电脑访问。

### 北京时间2026-10-08 22:10：双卡加载完成，图像差异出现原生复查证据

DreamZero的[第二轮双卡检查](../outputs/mn-acceptance-dreamzero-request-20261008-run2/receipt.json)正常写出两卡权重加载完成记录：[第一卡](../outputs/mn-acceptance-dreamzero-request-20261008-run2/rank-0-model-loaded.json)、[第二卡](../outputs/mn-acceptance-dreamzero-request-20261008-run2/rank-1-model-loaded.json)。但这一轮仍失败，没有产生真实动作：合并的诊断进程先导入环境已安装的openpi_client，随后要求导入配置中固定源码副本，来源保护正确拒绝，失败位置在输入构造之前；第二卡因第一卡未发送请求而随后停止。已调整诊断入口，让配置声明的客户端源码先解析；没有删除来源保护或替换已加载模块，没有改模型权重加载、采样、后端或公共契约。第三轮另建目录继续，旧失败不回写。

[RoboDojo两次纯原生重复的只读比较](../outputs/mn-acceptance-robodojo-native-render-comparison-20261008-run1.json)已经完成。两次均使用当前同一物理GPU5，执行同一四条保存动作；相机外参、仿真配置、公开状态、评分及计数完全相同，但原始RGB像素仍不同。起步三路图像的平均每通道绝对差分别约0.435、0.373、0.415，通道值范围仍是0到255。这个实验说明原生执行本身也有画面非精确复现，不能看到图像差异就认定Adapter映射有误；它没有唯一解释历史差异，也没有确定渲染内核或时间累积状态的根因。所有旧严格重放失败保留，像素容差和渲染开关未改。

为补齐更严谨的来源边界，新原生重复会在应用启动之前再次核验全部412项已选资产，先前两轮没有声称做了这一步，不能事后回写。DROID的独立原生重放已确认状态/32物理步/固定周期/公开指令一致但图像不同，继续用同一当前GPU做纯原生重复并保存相机外参、内参和原始RGB，不修改原生渲染行为。

第七个600样本窗口八卡总体均值28.444583333333334%，达到20%目标；[采样证据](../outputs/mn-acceptance-gpu-20261008-run7.json)。占卡仍单列，不当评测工作量。两条800动作上限试验仍在执行，不能把中间轨迹数量当作最终计数或任务自然结束。阶段4未正式验收，契约未冻结。

### 北京时间2026-10-08 23:05：两条RoboDojo原生任务正常结束，DreamZero按用户安排后置

七套独立环境各372项软件检查全部通过、零跳过：五套主运行环境见`outputs/mn-acceptance-baseline-runtime-{0..4}-20261008-run11/baseline-check.json`，另外两套见[DreamZero](../outputs/mn-acceptance-baseline-dreamzero-20261008-run3/baseline-check.json)、[Cosmos3](../outputs/mn-acceptance-baseline-cosmos3-20261008-run4/baseline-check.json)。花括号中的0到4表示五份不同目录，不是文件名的一部分。原85项与202项检查、八份旧契约示例、CLI退出码0/0/0/1/2均无回归；仍是软件/合成检查，不是372次真实任务。

[OpenWAM/RoboDojo](../outputs/mn-acceptance-openwam-robodojo-native-limit-20261008-run1/receipt.json)最终319次动作、3190物理步、12.76仿真秒；[π0.5/RoboDojo](../outputs/mn-acceptance-pi05-robodojo-native-limit-20261008-run2/receipt.json)最终280次动作、2800物理步、11.2仿真秒。两条均原生环境正常终止、官方success=true、score=100，框架错误0、关闭检查通过且客户端退出0。800只是原生动作上限，不是实际次数；各仅验证一个固定stack_bowls任务、一个layout和一个seed，不宣称全套任务或成功率。旧165次、墙钟预算截断的回执保留，不回写成完整成功。

[π0.5这280次动作的只读原生边界对照](../outputs/mn-acceptance-pi05-robodojo-native-exchange-comparison-20261008-run3.json)通过：六次实际请求的输入、六段各50行共300行生成动作的转换，以及实做280行缓存顺序均精确一致，包括数值类型；未用掉20行不计动作。没有额外推理、仿真、重试或放宽容差。OpenWAM另有保存真实观测的一整段32行映射证据；本次319动作没有全程捕获原生模型输出，不能把前一段证据说成全程逐行原生输出对照。

[DROID同次观测数组复核](../outputs/mn-acceptance-droid-observation-array-verification-20261008-run1.json)和[RoboDojo修复后同次观测数组复核](../outputs/mn-acceptance-robodojo-observation-array-verification-20261008-run2.json)均通过：重置加四条保存动作各五份快照，三路图像、全部实测/控制目标状态、坐标、时间、公开指令和来源标记均精确相同。原始与公共数组都已保存并重新只读核验，不额外拍照或推理。独立重开环境的严格像素重放失败、同一卡原生重复也有像素变化，均继续保留，根因没有唯一定位；同次映射通过不撤销历史复现差异。

π0.5/RoboTwin长程首轮第95次动作的左臂原生轨迹规划失败，夹爪和右臂仍有部分执行；原生记录95动作/2891物理步，Runner因不合法的错误编号只能校验到94动作/2841物理步，因此该轮状态未知、成功值无效。错误不是给失败动作加重试：RoboTwin与RoboDojo两处Adapter失败分支已改用公共协议原来就允许的`planning_failed`，意为规划失败；仍返回执行失败状态、实际计数和仅已确认的命令ID，不改Runner、公共枚举、原生规划器或控制参数。七套372项回归包含此修复。新[RoboTwin400动作上限试验](../outputs/mn-acceptance-pi05-robotwin-native-limit-20261008-run2/config.json)独立重新运行中，不预记任务成功，也不重写首轮失败。已经完成的两条RoboDojo长程试验在修复前加载了代码，不能称它们验证过这个错误分支。

DreamZero默认设置、限制大块拆分、允许分配块增长的三次实际双卡推理均显存不足，最后一种分配下两卡碎片计数已经为0，仍无动作输出：[默认失败](../outputs/mn-acceptance-dreamzero-request-20261008-run3/receipt.json)、[大块分配失败](../outputs/mn-acceptance-dreamzero-request-20261008-run4/receipt.json)、[增长式分配失败](../outputs/mn-acceptance-dreamzero-request-20261008-run5/receipt.json)。没有量化、改变图像尺寸、精度、采样、原生编码或权重。用户回复：“可以安排，但是这个要留到后面进行，我会和你对接，你到时候要提醒我”。故两张每卡80GB的H100真实验证明确后置；其余证据就绪或用户资源对接时提醒，不自动租用、不声称已设定时提醒，也不保证换卡必然通过。DreamZero主/辅助资产与双卡加载已验证，真实动作及仿真闭环仍未验证；准备好的入口不记为运行通过。

最新第九个600样本窗口八卡平均26.103958333333335%，达到20%：[采样证据](../outputs/mn-acceptance-gpu-20261008-run9.json)。第十窗口继续。占卡负载单独记录、不计评测工作量或成绩，未采时段和实例切换没有全程平均保证。阶段4仍待用户验收，契约未冻结。

### 北京时间2026-10-08 23:27：非后置运行收齐，提交带明确限制的验收材料

七套环境最新各376项软件/合成测试通过、零跳过，CLI仍0/0/0/1/2。五套主环境见`outputs/mn-acceptance-baseline-runtime-0-20261008-run12`至`runtime-4-20261008-run12`各目录中的`baseline-check.json`；追加两套见[DreamZero软件回归](../outputs/mn-acceptance-baseline-dreamzero-20261008-run4/baseline-check.json)与[Cosmos软件回归](../outputs/mn-acceptance-baseline-cosmos3-20261008-run5/baseline-check.json)。新增四项只检查保存回执的计数、防漏手、非法错误编号及失败后的已知状态，不是新真实任务。

[π0.5/RoboTwin第二轮终局](../outputs/mn-acceptance-pi05-robotwin-native-limit-20261008-run2/receipt.json)达到固定来源的400动作上限，实际12967物理步、51.868002463597804仿真秒，框架错误0、关闭后退出0，原生truncated=true、terminated=false，官方success=false且有效。truncated表示因原生上限结束，terminated表示任务自然成功结束；这里任务没有成功。初始化仍为原生seed-index-0，原生专家有效性筛选淘汰100000，实际采用100001；不是声称场景物理seed为0，也不是Runner重试。

[全部400行原生边界对照](../outputs/mn-acceptance-pi05-robotwin-native-exchange-comparison-20261008-run1.json)通过：八次实际请求、八段各50行全部输入与转换、400行缓存顺序的数值和类型精确一致。未新增推理/仿真、没有以任务结果代替映射正确。[逐条计数核验](../outputs/mn-acceptance-pi05-robotwin-accounting-20261008-run1.json)也通过，回执、Runner和原生执行器自记录一致；这是保存证据的一致性检查，不是本次重新独立测量物理步。

错误编号修复仅使原生失败能按合法协议报告，不改变原生规划或执行。新运行重新进行了推理，与第一轮并非逐动作相同；第二轮没有触发相同规划失败，不意味着原生规划失败被修好了。第一轮95/94动作计数分界、状态未知和失败回执继续保留，未新增重试或强迫缺失左臂执行。

两条RoboDojo原生任务各自正常结束且成功（OpenWAM319动作、π0.5 280动作）的[逐条回执核验](../outputs/mn-acceptance-openwam-robodojo-accounting-20261008-run1.json)及[π0.5核验](../outputs/mn-acceptance-pi05-robodojo-accounting-20261008-run1.json)均通过。均只验证一个任务/场景，不是全suite或成功率。原有六组执行对照五组严格通过、OpenWAM/RoboTwin整段严格未通过；新增DROID/RoboDojo独立重放状态和计数一致但像素不一致，原生重复和同次映射调查分别留证。π0.5跨进程重新推理差异也保留，内部非确定性根因尚未唯一定位。所有“映射精确”“任务成功”“严格复现”必须分开读。

第十个600样本窗口26.198541666666667%，[第十采样](../outputs/mn-acceptance-gpu-20261008-run10.json)；最后180样本窗口26.216666666666665%，[第十一采样](../outputs/mn-acceptance-gpu-20261008-run11.json)。全部11个已采窗口达20%，不覆盖未采时段或保证未来。保留的三张卡占卡负载单列，不计评测工作量。全部本轮真实模型/仿真工作进程已关闭，仅报告预览服务与用户已有占卡保留。

本轮材料汇总为[正式验收提交报告](../outputs/mn-acceptance-delivery-20261008-final/M+N-acceptance-preparation-and-evidence.html)、[验收清单](../outputs/mn-acceptance-delivery-20261008-final/acceptance-checklist.json)及[当前覆盖账本](../outputs/mn-acceptance-delivery-20261008-final/coverage-ledger.json)，使用html-report静态模板，保留49格逐项人话解释与当前运行补充。覆盖账本的旧字段保留10月7日来源状态，新`current_validation`记录10月8日实际进展；该字段意为“当前验证情况”，不改写历史账本。网页完整性、HTTP检查及源码摘要封存另有回执，未完成的检查不会预写成通过。

待用户正式确认的是本轮架构交付和上述明确限制，不重复要求已批准的实现细节。DreamZero仍按用户安排留待后续两张每卡80GB的H100对接，届时提醒，不计真实动作/闭环通过、不保证换卡必过。LingBot-VA暂缓；范围外模型、未知协议和结构不足格不以占位完成。阶段4未由助手代替通过，契约未冻结，阶段5未开始。

### 最终交付核验与网页入口

[报告完整性核验](../outputs/mn-acceptance-delivery-20261008-final/verification.json)通过：197份Python源码语法可解析、原49格逐项解释完整、55格历史来源字段未改、35份阶段2历史证据和旧M+N报告未变、0.2公共核心/Runner四项摘要未变。全部非后置新真实试验已有终局，严格失败、DreamZero后置和原生非确定性调查分别保留，不把文件完整性通过当作任务成功或阶段验收通过。

受限网页服务仅监听实例本机127.0.0.1:8767：[预览回执](../outputs/mn-acceptance-preview-20261008-run3.json)。[HTTP核验](../outputs/mn-acceptance-http-check-20261008-run3.json)确认全部68份显式报告/证据内容与预期摘要一致，根路径只跳转选定报告，目录、私有文件与路径穿越请求返回404。未向公网发布、未验证桌面端口转发，远程电脑需平台转发或SSH隧道。报告是生成时的静态快照，其阶段记录先于本节最终HTTP与摘要封存，不回写快照。

最终源码及交付内容指纹由`docs/handoff/seal_mn_delivery.py`记录到[摘要封存回执](../outputs/mn-acceptance-integrity-20261008-final/receipt.json)及`SHA256SUMS`；“摘要封存”只表示这些文件内容被记录，不是正式冻结契约、Git提交或阶段4已批准。具体操作是否完成以该独立回执为准，封存后不再修改所封文件。

### 北京时间2026-10-09：按用户要求补齐独立重新推理闭环和对比视频（进行中）

用户明确要求“对现有框架补齐同条件的原生对照，以及记录下对比视频”。本次在现有可运行真实链路内补齐：双方固定源码/权重/normalization、任务/初始化/公开指令、机器人/控制器、取图与缓存节奏、原生动作上限、解释器和GPU条件。使用独立新模型服务重新推理；不是把框架保存动作重放当作原生策略闭环。DreamZero和LingBot-VA仍后置，不推定阶段4验收或契约冻结。

首组[π0.5 ARX / RoboTwin完整原生对照](../outputs/native-comparison-pi05-robotwin-20261009-run1/native-result.json)已完成：独立原生直接调用固定来源输入编码、输出拆分、原生WebSocket客户端和原生take_action；八次新模型请求、400次动作、12850个独立观察PhysX物理步、51.400002441369仿真秒，官方eval_success=false。对照的现有框架终局为400动作、12967物理步、51.868002463597804秒、success=false。两侧起始公开状态、三路RGB及第一份实际模型输入完全相同；第一次动作已经不同，最大绝对差0.0019155965973975952。该值是混合关节/夹爪14维数组的差，不应统一称为米或某个角度误差。完整逐步精确比较未通过，不拿两边都失败撤销这个差异。

这里是**同一具名ARX输入输出协议的原生组件参考**，不是未经改动的RoboTwin官方π0.5入口。官方入口使用另一种状态定义和模型配置，不满足“相同权重和输入”的条件，不能混为一条原版结果。参考保留原生专家有效性筛选100000失败、100001有效以及实际指令；没有强制换成功seed、IK回退、调参或重新评分。[监督回执](../outputs/native-comparison-pi05-robotwin-20261009-run1/receipt.json)记录同一当前GPU5/7及固定硬件标识、独立新服务、客户端退出0；服务器仍没有可信权重身份握手，保留server_identity_verified=false。

两侧原视频及[时间对齐并排视频](../outputs/native-comparison-pi05-robotwin-videos-20261009-run1/comparison.mp4)已生成，[录像回执](../outputs/native-comparison-pi05-robotwin-videos-20261009-run1/receipt.json)记录摘要和时轴。图像来自正常执行已消费观测；无新增get_obs、相机拍摄或物理步。保存原始PNG与每个相机采集时间；30帧/秒是观看视频的编码频率，不是机器人控制频率。两侧均按实际仿真时刻显示最近一份观测，旧图像保持时明确标注采集时间，短的一侧结束后显示结束标记；不是连续物理相机录像，也不把有损MP4像素用来证明数值一致。三路画面均保留，不静默丢弃一只手。

新工具的379项软件/合成检查已通过、零跳过，其中新增3项检查时间选帧、原图与时间保留及双臂命令完整性；CLI新回归和其余配对真实任务继续执行，未提前写成通过。公共核心契约、Runner和生产Adapter未修改，旧10月8日交付目录/快照不回写；新的说明附加在本节，旧源码封存摘要作为旧时点证据保留。

#### 北京时间2026-10-09 12:00补充：七组终局、三个待补链路

当前轻量环境实际384项测试全部通过、零跳过，五组CLI实际仍0/0/0/1/2，见`outputs/native-comparison-baseline-20261009-run3/baseline-check.json`。新增检查是合成录像、受限预览与独立进程间数组保真检查，不是增加真实任务成绩。实际NumPy 2.3.3／1.26.0／2.4.6间私有数组封装往返数值、类型与字节完全一致，见`outputs/native-comparison-array-wire-20261009-run1.json`；其目的仅是防止NumPy2数组pickle在NumPy1解释器内无法导入，不改公共协议或模型数值。

六组迁移链路均已分别启动相同命令的新模型服务，双方重新推理完毕，并生成三种视频。OpenWAM／LIBERO两侧74动作成功，整段逐项精确一致；OpenWAM／VLABench两侧139动作成功，整段逐项精确一致。π0.5／LIBERO两侧103动作成功但第一条动作已有差异；LingBot-VLA／RoboTwin两侧112动作成功但第一条动作已有差异、物理步分别9743与9755；OpenWAM／RoboTwin框架112动作／10444物理步成功、原生111动作／10213物理步成功，第一条动作及类型、94物理步精确相同，但其后的实测状态与原始相机像素开始不同，不能称为完整严格复现。

**π0.5／VLABench必须单独保留反例：框架211动作后成功，原生达到300动作上限仍未成功，原生progress为0.5、intention_score为1。** progress意为官方任务进度，0.5不能改写为成功；intention_score意为官方意图检查得分。起始实测状态、三路原图及第一份模型请求打包精确一致；第一条新动作已出现目标位置约0.000029931米、欧拉角分量约0.000876820弧度差异。差异入口在重新推理输出，后续成功差异的完整因果仍待调查，不用成功结果取代映射正确。最初口头更新曾误述双方成功，已向用户明确更正；实际回执和视频从未回写。

RoboDojo两个原生组件对照尚未执行第一条模型动作：原生相机在reset内返回shape为0的空CPU缓冲，固定来源图像内核仍以1×480×640画面启动，卡在CPU读取函数。第一轮同名utils包冲突、OpenWAM诊断NumPy传输风险与随后空图像卡住的尝试均保留失败／取消回执。后续只对新诊断脚本添加传递原参数的边界日志、空缓冲安全拒绝和对齐生产环境已有root启动标志，未改只读源码、控制器、渲染节奏或填充假图。模型加载先后顺序变更未证明能解决相机问题，不将其记成已修复。

Cosmos／DROID本次改为双方均在模型GPU6、仿真GPU0新运行、原四动作预算不变。首轮框架完成，原生诊断误用六组旧模型的资产校验入口，已改为Cosmos自身verify_policy校验入口重跑，保留首轮失败回执。仍是有限四动作／32物理步，不是450动作完整任务；原环境无可用官方任务成功评分，不新增自定义评分。

四个已完成600样本的八卡总体平均窗口分别约26.19%、26.59%、26.21%、26.66%；第五窗口继续采样。3／5／7卡原有占卡负载未修改，平均利用率包含这些负载，不能解释为模型有效工作量或成绩，也不覆盖未采样时段。当前七组完整对照与21个MP4均留新路径，历史报告不回写；其余三组未补齐前不声称十组全部完成。

#### 北京时间2026-10-09 12:44补充：八组与24视频核验完成，RoboDojo两组仍在排查

Cosmos／DROID原生新推理已经完成：本轮框架第二轮四动作运行与原生第三轮均模型GPU6、仿真GPU0，原生1次模型请求、4动作、32物理步、0.26666666666666666仿真秒，正常关闭。起始实测七关节／夹爪精确相同，但三路原始画面不同，第一条动作也不同；不能将后续差异单独归因于模型推理。没有官方成功评分，不记成任务成功或完整450动作评测。原生前三轮错误与运行记录不回写：第二轮新诊断把七关节无批次数组错误切成标量，修复仅在新对照脚本，生产Adapter未改。对应两项合成测试检查保留全部关节、原始float32字节及拒绝不完整状态。

最新[轻量Linux回归](../outputs/native-comparison-baseline-20261009-run6/baseline-check.json)388项、零跳过，CLI仍0/0/0/1/2。新增两项启动边界测试验证复用代码只执行原生reset前缀，绝不执行旧保存动作循环；它们不是GPU或真实任务测试。[前七组媒体](../outputs/native-comparison-video-verification-20261009-run1.json)及[DROID媒体](../outputs/native-comparison-video-verification-20261009-run2.json)合计6,884张原始PNG文件与像素指纹、24个MP4完整解码和实际帧数均核验通过，检查本身无新增推理／仿真。

两个RoboDojo原生新推理对照仍未补齐。root标志、隔离Kit启动参数、继承代理并排除本机地址、延迟模型／辅助库导入、复用独立原生reset前缀均未确立解决办法；最近卡住进程仅按明确PID和独立进程组取消，保留未收到原生关闭确认的事实。原有独立原生脚本在GPU5和GPU0各跑四条保存动作证明工厂可以启动，但只属于启动排查，不能抵扣新模型推理对照。当前另用最小入口先执行原生reset，之后才导入新推理／比较工具；不填补图像、不多推进物理、不修改只读来源仓库或公共协议，不能提前记录完成。

第六个600样本窗口八卡平均26.26%，第七窗口采集中。原有3／5／7卡占卡未修改，不计入评测成绩。八组已完成、两组待补与严格差异会分别展示；仍不代替用户通过阶段4或冻结契约。

#### 北京时间2026-10-09 13:16终局补充：十组原生重新推理与30视频收齐

查实先前RoboDojo相机阻塞的一处新增诊断问题：私有图形库目录以相对路径写入子进程的库搜索环境，子进程切入产物目录后该路径不存在；原有独立脚本先转换成绝对路径。两个已取消进程的[π0.5路径实测](../outputs/native-comparison-pi05-robodojo-20261009-run9/cancellation.json)与[OpenWAM路径实测](../outputs/native-comparison-openwam-robodojo-20261009-run10/cancellation.json)均保留。修复仅在新诊断监督脚本：四个运行路径先解析成绝对路径，启动前验证图形库目录确实存在，没有改共享库、驱动、渲染参数、模型或生产Adapter。修复后的两组原生reset均正常返回，随后新推理完整执行；前述猜测与失败尝试不回写成成功。

[π0.5／RoboDojo原生终局](../outputs/native-comparison-pi05-robodojo-20261009-run10/native-result.json)：六次新请求、289动作、2890物理步、11.56仿真秒，官方成功且score=100，正常关闭。旧框架为280动作、2800步、11.2秒成功。初始全部实测关节／末端姿态及夹爪控制目标相同、三路原始RGB均不同；第一份实际请求状态和指令相同、图像不同。不能将这组后续动作差异单独归因于模型随机性或Adapter。

[OpenWAM／RoboDojo原生终局](../outputs/native-comparison-openwam-robodojo-20261009-run11/native-result.json)：十次新请求、319动作、3190物理步、12.76秒，官方成功且score=100，原生模型关闭确认及环境关闭均通过。旧框架也是319动作成功，仍不等于轨迹精确相同。初始全部状态相同、三路RGB不同；原生第一次输入的无损数组保存与框架CPU打包比较，状态和指令一致、图像不同。该无损数组保存不是网络抓包，不声称捕获了旧框架全部原生输出。

两组新原生组件参考明确排除旧保存动作循环，直接执行固定来源Model纯函数／真实推理及原生环境初始化、控制和评分。原生新推理不实例化生产Model／Benchmark Adapter，不调用Runner；单独的CPU打包检查会使用生产Model纯转换函数，这两个边界不混淆。[运行中硬件身份](../outputs/native-comparison-robodojo-hardware-20261009-run1.json)记录当前实际GPU UUID和自有进程，不事后冒充启动时抓取。[26轮尝试汇总](../outputs/native-comparison-attempt-ledger-20261009-final.json)区分错误、取消、真实新推理完成及两次仅保存动作的启动短测。

现在十组各有框架、原生、并排三种视频，共30个MP4。四份只读媒体核验累计10,517张原始PNG文件及像素指纹，30个MP4完整解码、实际帧数匹配。七组／DROID核验保持原回执；追加[π0.5媒体](../outputs/native-comparison-video-verification-20261009-run3.json)和[OpenWAM媒体](../outputs/native-comparison-video-verification-20261009-run4.json)。录像只复制正常执行已消费的观测，没有新增取图、物理步、推理或强行对齐动作数。

最新[390项轻量Linux软件回归](../outputs/native-comparison-baseline-20261009-run8/baseline-check.json)通过、零跳过，原85与202项基线及五CLI退出码0/0/0/1/2无回归。不冒称七套环境都重跑了390项；七套各376项是10月8日保留的旧时点证据。八个已完成采样窗口共4380样本，八卡样本加权平均26.408133561643837%，每窗均超过20%，包含原有3／5／7卡占卡负载，不代表模型有效工作量或未采时段。评测模型／仿真进程均已关闭，网页预览和用户已有占卡保留。

汇总网页将另存到`outputs/native-comparison-delivery-20261009-final/native-closedloop-comparison-and-videos.html`。其中只有OpenWAM／LIBERO与OpenWAM／VLABench整段动作／状态／原图一致；时钟仅有已声明的1e-8秒浮点舍入容差，不用于动作、状态、图像。其他严格差异原样保留，尤其π0.5／VLABench框架成功、原生300动作仍未成功，π0.5 ARX／RoboTwin两侧400动作仍未成功。材料收齐不是全部严格门禁通过，不是多任务／多起点成功率，也不替用户验收阶段4或冻结契约；DreamZero／LingBot-VA仍按用户安排后置。

安全说明：一次过宽进程盘点的工具输出误包含其他Notebook／隧道服务认证参数，已告知用户建议轮换；这些值未写入报告和证据，本文不复述。后续盘点只输出本项目明确自有脚本进程的允许字段，不擅自改动相关服务。

#### 最终网页读取核验与摘要封存

[最终网页](../outputs/native-comparison-delivery-20261009-final/native-closedloop-comparison-and-videos.html)使用html-report静态模板，225份Python源码可解析、35份阶段2历史文件及0.2公共核心／Runner四项指纹未变，完整性检查通过。实例本机127.0.0.1:8766已启动[受限预览](../outputs/native-comparison-preview-20261009-final.json)，只提供报告显式链接文件、无目录浏览、未上传公网。[HTTP内容核验](../outputs/native-comparison-http-check-20261009-final.json)确认114份文件内容摘要正确，根目录跳转、私有文件及路径越界拒绝通过；[实际视频分段读取](../outputs/native-comparison-video-http-check-20261009-final.json)确认30视频的开头／中间／末尾字节与原文件一致、HEAD及越界416响应正确。没有浏览器工具可实测桌面端口转发，远程电脑需平台转发8766或SSH隧道，不声称已在用户浏览器播放。

报告是生成时静态快照；后续HTTP与封存以独立回执为准，不回写报告。源码及交付指纹将保存到`outputs/native-comparison-integrity-20261009-final/receipt.json`和`SHA256SUMS`；以该回执及实际摘要复核为准，不提前声称封存完成。它不是Git提交、协议冻结或用户阶段验收；封存后不再修改其覆盖文件。
