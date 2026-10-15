# 阶段 3：首个真实闭环的确认材料

状态：具名 OpenWAM-α × LIBERO 单 trial 及第 6 节最小协议方案已获“继续推进”授权；真实框架闭环与原生逐步对照已完成。用户随后回复“继续推进，总体平均就可以,但是还是最好到20%比较保险,可以作为验收阶段”，据此将阶段 3 **单 trial 工程验收**记为通过。此确认不是 50-trial 官方统计结果，也不冻结公共契约。阶段 4 的具体方案另见 [stage4-review.md](stage4-review.md)。阶段 2 已验证 mock 执行器；阶段 3 的目标是验证一组匹配权重的真实动作策略和仿真 Benchmark，并与原生 scorer 对照。

## 1. 立项前检查清单（下表保留初稿状态；实测更新见第 5–6 节）

| 项目 | 需要的具体信息 | 当前状态 |
| --- | --- | --- |
| 模型 | 仓库、提交版本、动作策略角色、原生推理入口 | 待确定 |
| 权重 | 本地路径或官方地址、版本/摘要、匹配任务与机器人、归一化资产 | 待确定 |
| Benchmark | 仓库/版本、子 suite、首个任务、初始化 seed、官方评测入口 | 待确定 |
| 机器人与控制器 | 相机、关节/EEF、TCP、动作域、夹爪、控制周期 | 原生代码核对后提交映射 |
| 运行资源 | GPU 机器/型号/显存、访问方式，或现有模型服务地址与协议 | 待确定 |
| 试点预算 | 原生 warm-up、执行 horizon、最大步数、终止和评分条件 | 必须沿用该协议，待核实 |

优先从兼容性清单中选择已有匹配权重、原生评测入口可运行的一组操作任务。先确认实际对象，不从模型名称或向量形状推断兼容。若尚无现成试点，可先做仓库、权重和资源的只读核实，再提交具名配置；不直接把未核实组合视为可测。

## 2. 确认对象后的拟实施内容

1. 在确定的计算环境复现官方原生入口，记录代码、权重、归一化和依赖版本。
2. 写该 Benchmark 的 SpaceSpec / TaskSpec、Observation 映射、控制执行与官方 scorer 适配。
3. 写模型的 CapabilitySpec、原生预处理、输出解析与状态重置。
4. 提交具名 AdaptationProfile，包括原生图像尺寸、相机映射、关节/EEF 顺序、动作单位/域、frame、夹爪、chunk 归属及实际预算。
5. 配置真实 Adapter 注册与 CLI 专属参数；维持通用执行循环。
6. 用同一初始化和 seed 比较原生入口与新框架的观测、动作、环境步和评分。

不把模型专属 resize、tokenizer、去归一化、IK 或评分条件放入 runner.py。RGB 或动作只在有证据的 Adapter 边界转换。既有预处理的顺序和参数需要逐项对照原生代码。

## 3. 拟验收门槛

- 原生入口可运行，匹配权重和机器人有证据，结果/失败路径有官方定义。
- 第一轮先完成单任务单 episode 的逐步对照；任务成功不是必须条件，有效官方判定才是基本门槛。
- 图像/状态关键字段与原生输入对齐；几何、动作缩放、夹爪与控制器映射有明确配置。
- 动作数值比较的容差、确定性条件和重新推理时刻，在原生实现核对后确认，当前不预设通用容差。
- 任务初始化、warm-up、最大步数、终止标志和 scorer 调用顺序与官方协议一致。
- 保存输入/输出对照、轨迹、版本、配置和结果；评测隐藏信息不进入模型。
- 现有 85 项测试保持通过，真实 Adapter 加入必要的映射和原生对照检查。
- 若首轮需要新数据载荷、未知统计表达或专属时间协议，先说明公共契约缺口及影响，再确认修改。

## 4. 实施前将提交的最终配置

~~~text
模型 / checkpoint / normalization：
Benchmark / suite / task / seed：
机器人 / 相机 / 控制器：
Model Adapter 原生调用与状态规则：
Benchmark Adapter reset / step / score：
观测和动作双向字段映射：
chunk ownership / horizon / period：
计算机器、模型服务与依赖：
原生对照入口、采样点、容差及产物：
~~~

上述信息补齐并确认后，仅实施这一组真实试点。第二个模型或 Benchmark、Agent、导航扩展和批量并发仍按后续阶段分别确认。

## 5. 2026-10-05 Linux 接手后的具名候选（提交确认，未获批准）

本节是对 **OpenWAM-α × LIBERO / `libero_spatial` task 0** 的具名建议，不表示已选定试点或已运行真实评测。原先只读核实后，用户追加授权检查 `Model/` 并下载模型与 Benchmark；权重、归一化和任务资产的本地 SHA-256 已核对，LIBERO 环境烟测通过，详见[候选资产预检](stage3-preflight-20261005.md)。候选依据还包括[OpenWAM 固定提交原生说明](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/README.md)、[客户端源码](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/openwam2libero_interface.py)、[评测循环](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/single_eval.py)和[权重仓库固定 revision 文件列表](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-LIBERO/tree/110b07e4d5581c81520a77419cbb0e9371df231b)。模型加载与原生闭环仍未验证。

下文的一个控制 `tick` 是一次 `env.step` 对应的逻辑控制步；`trial` 是一次固定任务初始化后的完整尝试；`horizon` 是生成动作序列后连续执行的最大步数。`rng_mode=environment` 表示随机数状态由仿真环境管理，`reseed_each_trial=false` 表示不同 trial 之间不再逐次重设种子。EEF 指机器人末端执行器，TCP 指工具中心点；OSC_POSE 指在末端位姿空间接受命令的控制器；`controller_native` 表示数值保留在该控制器的原生命令域。`site` 是 MuJoCo 模型中的具名空间点；`m` 表示米，`rad` 表示弧度。EGL 是仿真的 GPU 离屏渲染接口，BDDL 是 LIBERO 的任务定义文件，`PONG` 是模型服务回复的能力握手消息。HF LFS SHA-256 是 Hugging Face 大文件元数据列出的内容摘要，下载后仍需计算本地文件摘要核对。

| 决策项 | 建议固定值与现有依据 |
| --- | --- |
| 被测模型 | OpenWAM-α，角色 `action_policy`，首轮只评动作；源码提交 `48bd67b89d489b14d03b8d92bc66e65d306df32e`。不从相邻 XPolicyLab 导入运行时代码。 |
| 匹配权重 | `OpenWAM/OpenWAM-Alpha-Sim-LIBERO`，revision `110b07e4d5581c81520a77419cbb0e9371df231b`；`checkpoint_step_10690.safetensors`，本地 24,813,767,464 bytes、SHA-256 `34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a`，均与 HF 元数据相符。该模型卡称在 LIBERO 仿真上微调；原生评测要求 checkpoint 的 `eef` 表示。文件大小不能推算实际推理显存。 |
| 配套资产 | 同一 revision 的 `config.yaml`、`normalization_stats.npy`（HF LFS 与本地 SHA-256 均为 `f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e`）和 `tokenizer/google/umt5-xxl/` 已在本机模型目录。固定配置含 `dataloader.type=libero`、`action_mode=eef`、`num_frames=33`、`height=384`、`width=320`、多视角及 `min-max` 归一化；权重目录加载器从同目录加载归一化资产。文件匹配已核验，实际模型加载尚未完成。 |
| Benchmark | [LIBERO 提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01)；suite `libero_spatial`，task ID `0`，该提交的任务名为 `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`；具体 BDDL 和初始状态文件安装后记录 SHA-256。 |
| 固定试验条件 | 原生 `policy_config.yml` 固定 seed `42`、`rng_mode=environment`、`reseed_each_trial=false`，trial `0` 对应第一个 init state；每 trial 先 reset / 设置 init state，再执行 30 个 settle tick，原生 settle action `[0,0,0,0,0,0,-1]`，然后重置策略缓存。策略执行最多 600 tick。原生完整协议为每 task 50 trials；单 trial `--smoke` 仅用于工程对照，不能报告为官方汇总成功率。该客户端在 `env.step` 返回 `done` 时记录成功，同时保存 `reward`。 |
| 机器人与控制 | [LIBERO 环境包装器](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)默认单臂 `Panda`、`OSC_POSE`、`control_freq=20`，故一个控制 tick 为 `0.05 s`。已安装 robosuite 1.4.0 的 `osc_pose.json` 确认 `control_delta=true`、输入 `[-1,1]`、位移输出幅度 `0.05 m`、旋转输出幅度 `0.5 rad`；其 `single_arm.py` 把控制器 `eef_name` 设为夹爪 `grip_site`，观测位置取该 site，四元数取机器人 EEF body，不可把二者未经核验地合并成新位姿。相机 `agentview_image`、`robot0_eye_in_hand_image`，原生评测以 256×256 渲染后旋转 180°，再按 checkpoint 配置组装视图；第三视角为空。末端位置、xyzw 四元数、两指关节位置用于构造 10 维已实现状态；还需原生运行时核对命名、参考坐标和 TCP。 |
| 动作映射 | 模型返回原生 10 维：3 个**归一化控制命令**、由旋转向量指数映射得到的 6 维旋转表示、1 个“+1 张开”的夹爪命令。原生[转换函数](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/openwam2libero_interface.py)解码旋转为 3 维 rotation vector、夹爪变号并裁到 `[-1,1]`，传 7 维 OSC 命令；不组合当前末端位姿，也不再次应用控制器的 0.05 m / 0.5 rad 输出尺度。该 3 维位移绝不能在标准层冒充米。建议用 `ControllerCommand` 的 `controller_native` 域表达已核对的参数映射，由 Benchmark Adapter 调用原生控制器。 |
| 时序与预算 | 原生[同步执行器](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/openwam/deploy/executors/sync_executor.py)内部缓存 chunk；[评测调度器](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/scheduler.py)默认 `inference_horizon=10`，即生成一个 chunk 后最多执行 10 个动作，丢弃尾部再推理。建议首轮固定同步推理、同步去噪、horizon 10；在本框架采用 `adapter_owned`、Profile horizon 1，每次 WebSocket 请求只向 Runner 暴露一个动作，缓存由模型服务管理并在 episode 边界 reset。每 tick 都发送观测；服务在缓存非空时可以不重推理。`model_forward_count` 若服务未给可靠值，记 `null`。预算是 settle 后最多 600 个策略 tick；调用数和墙钟只记录实际值，不沿用 mock 默认预算。 |
| 资源与依赖 | 本机 Ubuntu 22.04 / x86_64，驱动 570.195.03；`nvidia-smi` 报告 8 张 RTX 4090、每张 49,140 MiB 总显存，当时约 47,833–47,841 MiB 空闲，但 GPU 利用率较高，资源独占性和模型可装载性未验证。候选先申请 GPU 0 给模型、GPU 1 给 EGL 渲染，实际分配待确认。[OpenWAM 安装说明](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/README.md)推荐模型环境 Python 3.10、PyTorch 2.7.1/CUDA 12.8；其 [LIBERO 独立环境](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/environment.yml)固定 Python 3.10、robosuite 1.4.0、MuJoCo 3.3.2、PyTorch 2.7.1 CPU 等。本机已分别建立并通过依赖检查，LIBERO 一步 EGL 烟测也通过；尚未加载模型。原生入口采用模型服务与仿真客户端两个进程/环境，WebSocket 连接。阶段 2 的 Python 3.13 轻量环境不直接代替这两个环境。 |

### 原生对照及验收提案

固定源码的原生 `setup_env.sh` / `run_smoke.sh env` 建环境、任务名和首个任务 BDDL/init 摘要核对已完成。确认候选与资源后，再在独立的模型环境加载 checkpoint、normalization 与 tokenizer。官方[原生入口说明](https://github.com/OpenWAM-Official/OpenWAM/blob/48bd67b89d489b14d03b8d92bc66e65d306df32e/benchmarks/libero/README.md)提供 `run_eval.sh`；首轮建议限定 `--suites spatial --smoke --inference-horizon 10 --compile-enabled false --render-gpus 1`，使用 GPU 0 运行模型、GPU 1 渲染，记录完整命令和新输出目录。`--smoke` 是 task 0 / trial 0 / 一次试验；不要与 50 trials 的官方统计混用。只在原生闭环正常后实施一组真实 Adapter。

框架与原生入口用相同 checkpoint、配置、seed、task、init state、30 个 settle tick 和策略 600 tick 上限。逐步保存原生 256×256 两相机图像摘要、10 维已实现状态、模型输入、10 维原始动作、7 维 OSC 命令、环境步号、`reward`/`done` 和策略重推理时刻；先核对完全相同的观测和 7 维控制输入，再比较状态轨迹与原生判定。图像的像素一致性、状态/动作的数值容差须在两边实际 dtype 与原生打印结果核对后确定，不能预先声明通用容差。原生单 trial 若任务失败但 scorer 给出有效 `success=false`，仍可用于映射验收；真实结果需有原生来源、非 synthetic 标记和可追溯的文件摘要。

### 确认前的缺口、影响与待决定项

1. **请确认是否选择本候选及单 trial 范围。** 阶段 0 记录的建议不是选择；后续授权已覆盖本机查找、下载模型/Benchmark 和环境预检，但未单独确认本候选为阶段 3 试点，也未授权公共契约变化。若团队已有另一组已匹配权重，请提供其路径/版本/机器人，先对比原生入口。大文件下载命令已临时绕过代理直连，未修改全局代理配置。
2. **请确认 GPU 使用位置和可用空间配额。** `nvidia-smi` 和共享文件系统空闲量不能证明可独占两卡或可用磁盘配额。LIBERO 已在 GPU 0 上完成一步 EGL 烟测，但模型装载与模型/渲染双进程并行仍未实测。候选所需的 OpenWAM/LIBERO 资产现已另行取得，路径和摘要见预检记录。
3. **轻量框架缺真实注册和真实性记录。** 阶段 3 若批准，需要新增 Model/Benchmark Adapter 注册、专属配置、版本化 Profile、权重与资产 SHA-256、原生依赖和明确 synthetic=false 的证据链；不能仅翻转布尔值。此前未形成可审阅实现方案，开始实现前先确定具体文件与配置边界。
4. **公共契约潜在缺口。** `ControllerCommand` 现有 `controller_native` 域可表达 7 维原生命令，初步不需要修改公共消息。现有 DataRef 只能校验形状和摘要文本，不能读取图像载荷；若 inline 图像成本不可接受，需先提交载荷解析/传输的具体契约变化与兼容影响。30 个原生 settle tick 拟在 Benchmark `reset` 内执行，标准 `env_step=0` / `sim_time_s=0` 以首个策略观测为原点，同时另记原生预热计数；若官方对照要求暴露绝对仿真时钟，则需提议增加时钟/预热字段，并说明对校验、Runner 与产物的影响，获得确认后修改。WebSocket 执行响应丢失时不能盲目重试；硬超时/恢复仍非现有能力。
5. **仍需实测的证据。** checkpoint、normalization、task 0 的 BDDL/init 摘要及 EGL 一步烟测已取得；控制器版本化配置已从安装包核对。配置实际加载、`PONG` 的 `eef` 声明、真实前向显存/延迟、末端参考 frame/TCP 与控制器参数的运行时核对、原生首个 episode 的逐步输入和评分尚未取得。若其中任何一项与本节建议冲突，先更新具名方案并重新提交确认。

阶段 2 的 Linux 复现证据见 [接手复现记录](stage2-linux-replay-20261005.md)。上述第 5 节是提交确认时的候选快照；用户随后回复“继续推进”，按当时同时提出的两个确认问题，现将其解释为阶段 2 复现验收和本具名试点的单 trial 原生验证授权。原生结果及边界见下一节；历史 macOS 证据不因此改写。

## 6. 已确认范围内的原生结果与下一道协议门禁

2026-10-05 在本机依固定源码执行了官方 `run_eval.sh --suites spatial --smoke --inference-horizon 10 --compile-enabled false --render-gpus 1`，模型 GPU 0、EGL 渲染 GPU 1，退出码 0。`libero_spatial` task 0 / trial 0 / seed 42 在 30 个 settle 步后执行 73 个策略步，原生客户端记录 `success=true`、末步 `reward=1.0`；服务识别本地 checkpoint 的 `min-max`、`eef`、10 维归一化资产。完整的输入摘要、原生 `manifest.json`、结果与日志摘要见[单 trial 原生证据](stage3-native-evidence-20261005.md)。这是一次真实原生烟测，不是框架 Adapter 已验证，也不能据 `1/1` 报告 50-trial 官方成功率。

原生代码进一步揭示一个表示缺口：已安装 robosuite 1.4.0 的 `single_arm.py` 使 `robot0_eef_pos` 取夹爪 `grip_site` 的世界位置，而 `robot0_eef_quat` 取 `robot_model.eef_name` 刚体的世界四元数；Panda 的 `eef_name` 是 `right_hand`，`grip_site` 位于挂载其下、带自身旋转的夹爪 XML 中。两值不是同一具名体的直接完整位姿，未经显式坐标转换就拼成标准 `Pose` 会误报参考点。模型仍须收到这两个原生值及两指关节位置，原生 `libero_obs_to_eef10` 才能构造匹配训练的 10 维状态。此处的 EEF body 是机器人模型中的末端刚体，`grip_site` 是 MuJoCo 定义的夹爪作用点。

| 待确认的最小变更 | 具体做法与影响 |
| --- | --- |
| 具名公开输入扩展 | 仅为此试点增加 `libero.native_proprio.v1`：`eef_position_m`（`grip_site` 世界位置，3 个米数值）、`eef_quaternion_xyzw`（EEF body 世界四元数，4 个分量）、`gripper_qpos_m`（两指关节位置，2 个米数值）。在 `SpaceSpec.extensions` 声明，在 `Observation.extensions` 按原生采集，在 `AdaptationProfile.view_policy` 明确放行；Model Adapter 严格核对形状、有限值并调用固定上游转换函数。标准 `Pose` 不伪造，隐藏任务状态仍不进模型。公共核心 dataclass 字段不变；这是新增一个有版本的模型可见扩展名及其载荷规则。 |
| CLI 真实注册 | 在现有严格配置中增加 `openwam_libero` / `libero_spatial` Adapter 选择，以及固定源码/权重/归一化路径和本地 WebSocket 端点字段；仅允许这一组配对、suite/task/seed 42 和单 episode。现有 mock 配置与五组退出码保持兼容。模型服务仍由官方脚本独立启动，Runner 不托管 GPU 服务。 |
| 图像载荷 | 首个单 trial 先使用现有 `Tensor.data` 的 `uint8`、高×宽×通道 RGB 数组；两路 256×256 图像进入公开观测，模型 Adapter 复用上游 180° 翻转、L 形视图缩放与编码。这样不假设尚未实现的 DataRef 文件读取，但 JSON 轨迹体积和序列化开销较大；超出单 trial 前再单独审议 DataRef 读取、校验及路径规则。 |
| 真实产物与来源 | `manifest.json`、`summary.json` 中的既有 `synthetic` 字段按已注册且实际执行的 Adapter 配对确定：mock 为 `true`，此真实配对为 `false`；新增源码提交、checkpoint/normalization/BDDL/init SHA-256 与原生协议摘要等 provenance 元数据。改动只影响新运行的 CLI 产物，不重写旧 synthetic 产物或历史 macOS 路径。若预检失败或未执行，不能生成 `synthetic=false` 的“成功结果”。 |
| 时序与评分 | 30 个 settle 步在 Benchmark Adapter `reset` 内执行，标准 `env_step=0` / `sim_time_s=0` 从首个策略观测起算，并另记原生 settle 计数；每个 `advance` 只调用一次原生 `env.step`。单 trial 预算显式设为策略 `env_steps=600`、`sim_time_s=30`、`model_calls=600`，不沿用 mock 默认值；这是 20 Hz 下最多 600 次策略控制，非硬墙钟超时。成功仅由 LIBERO 原生 `done` 给出，Runner 不实施控制器缩放、图像处理或评分。模型服务拥有 10 步 chunk 缓存，Profile 为 `adapter_owned`、对 Runner 只暴露 1 步。 |

批准这些具体扩展和 CLI/产物语义后，再实施两个真实 Adapter 及 Profile，保留 85 项测试并新增输入映射、动作 10→7 维、公开信息白名单、原生对照与错误路径测试。首个框架 episode 须与原生同任务/seed/init 状态比对；原生当前未保存逐步图像哈希，故先对照可取得的原生状态与动作，再通过同一上游预处理函数和追加调试证据核对图像。硬超时、并发和恢复不在这一次协议变更内；现有墙钟预算仍只有协作式检查，WebSocket 响应不明时不得盲目重试。阶段 3 只有在框架真实闭环、逐步对照及新证据通过后才提交验收。

## 7. 真实框架闭环与单 trial 逐步对照（提交阶段 3 验收）

用户随后回复“可以的，继续推进”并要求控制 GPU 平均利用率不低于 15%，据此实施第 6 节最小扩展。`ModelAdapter` 复用固定 OpenWAM 原生 LIBERO 客户端的图像预处理、状态构造和 10→7 维动作转换；`BenchmarkAdapter` 复用固定 LIBERO 环境、初始状态、30 步 settle 与官方 `done` 判定。Runner 只按既有通用循环调度，没有加入 resize、去归一化、控制器、逆运动学或评分规则。执行前核对两份源码提交和 checkpoint、normalization、BDDL、`.pruned_init` 四个 SHA-256；产物中的 `server_identity_verified=false` 如实表明 WebSocket `PONG` 不能证明服务端所载权重，独立启动命令及服务加载日志提供额外来源证据。

固定最终源码运行 `examples/stage3-openwam-libero-spatial-pilot.json`，结果 `outputs/stage3-framework-20261005T1729Z-final/`：CLI 退出 0、`synthetic=false`、单 episode `completed`，官方 `success=true`，策略步 73、模型请求 73、逻辑仿真时间 3.65 s；服务不提供可信前向次数，故 `model_forward_count=null`。同一服务随后执行原生 `single_eval.py`、打开逐步调试，结果 `outputs/stage3-native-debug-20261005T1724Z/`：trial 0 成功、73 步。`outputs/stage3-compare-20261005T1734Z.json` 对两侧逐步状态 10 维、原生模型动作 10 维和 OSC 控制命令 7 维做了 73/73 步对比，三者每步精确相等，最大绝对差均为 0；这比预设一个未经论证的浮点容差更直接。原生 debug 文件没有图像字节，故不能声称跨入口像素哈希相等。两侧调用的是同一固定上游图像处理函数，且状态和动作全程吻合，但图像像素一致性仍是未直接观测项。详细路径、摘要、基线与限制见[阶段 3 框架证据](stage3-framework-evidence-20261005.md)。

原有 85 项和新增 8 项测试全部通过，8 个契约示例与五组 mock CLI 退出码 `0/0/0/1/2` 保持不变。180 次 GPU 采样的 8 卡总体时间平均利用率为 22.35%，满足按总体平均解释的 15% 下限，但 GPU 0/1 各自未达 15%，统计口径待用户确认。第一次框架诊断运行在 33 步后为固定源码而主动中止，不具备 `summary.json`，不得算作成功运行。正式运行的 inline 图像产物约 246 MB；DataRef 读取与压缩仍未建设。该结果仍只是单 trial 工程验收，不是官方每任务 50 次统计；阶段 3 未经用户确认前，不实施后续阶段、扩大任务或修改公共核心契约。
