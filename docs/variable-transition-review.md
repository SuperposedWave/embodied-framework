# RoboTwin 可变时长原生动作：已批准设计与实施记录

2026-10-06 用户在白话解释及确认请求后回复“我认为可以，继续”，批准本文具名字段和最小执行语义。以下保留提交时的设计依据；批准允许实施，不等于实现完成或真实原生验收。旧固定周期路径继续保留，契约仍未正式冻结。

## 已核对的具体缺口

OpenWAM 对应 RoboTwin 源码固定 `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4`；LingBot-VLA 2.0 对应另一固定版本 `13c3c47ff4312dd62484bcd51be034af55c062d1`。两份 `envs/_base_task.py:take_action` 都先增加一次原生动作计数，再按双臂规划轨迹执行多次 `scene.step()`。前者的默认物理时间步是 1/250 秒；动作轨迹长度由规划结果或 TOPP 决定，并可能因官方成功条件提前结束。TOPP 指按速度/加速度约束对关节路径进行时间参数化，不是一个固定长度动作。

当前公共校验要求 ActionStep.duration_s 等于 SpaceSpec.control_period_s，并要求反馈仿真时间等于消费的逻辑步数乘固定周期。Runner 还要求一个已执行 ActionStep 恰好消费一个逻辑步。把原生一次 take_action 记为一周期会伪造时间；拆成物理步并重新询问模型会改变原生策略/缓存与成功检查节奏。

具体依据是只读固定版本的 `_base_task.py:1479–1665` 与现有 `validation.py:340–342、416–427`、`runner.py:205–207`。行号只帮助定位，不作为内容摘要或版本身份；实现覆盖账本记录实际源码摘要。

## 建议的最小新增协议

1. SpaceSpec 增加具名时间模式，默认仍为固定周期；新模式表示“一次原生动作、实际测量时长”。名称和字段需经本设计审查后再固定，不复用现有 duration_s 表示未知时长。
2. 新增专用于原生宏动作的 decision 类型，明确机器人/控制器/动作域，但不由模型声称预计仿真时长。宏动作是一次原生调用内部包含多个物理步的动作。原有 ActionStep、mock、LIBERO 和 VLABench 全部保留固定周期规则。
3. `env_steps` 在新模式明确记录原生策略动作次数；另外记录实际物理步数和累计仿真秒数。Benchmark 在原生执行边界计数/测量，不改原生规划、IK、轨迹、夹爪或成功规则。计数方法必须在真实单例中验证，不能根据墙钟时间估算仿真时间。
4. 初版只接收原生动作次数和模型调用次数上限，拒绝 `sim_time_s` 上限，因为不透明规划调用尚不能安全中断以保证硬仿真时间预算。墙钟上限仍只能沿用当前边界检查，不把它宣传成硬超时。以后若需要硬预算/取消，另建设原生执行器及部分执行反馈。
5. 观测只在原生策略决策边界更新，不声称拥有每一个物理步的图像或中间观测。LingBot 客户端返回的整个 joint14 动作序列按原顺序逐行动作执行，每行之间检查官方成功/动作预算；OpenWAM 保持服务端缓存和 EE20→EE16 转换。
6. 来源记录分别固定两种 RoboTwin revision、专用 checkpoint/归一化、task、clean/randomized 配置、机器人、专家可行 seed 筛选、上游任务步数限制和评分；桥接与执行层不能把两版本拼成一个结果。

joint14 指双臂各 6 个关节角与 1 个夹爪值，共 14 个值。EE20 指双臂各位置 3 维、旋转 6 维、夹爪 1 维；EE16 指双臂各位置 3 维、四元数 4 维、夹爪 1 维。

## 影响与验收

### 提交确认的具体字段草案

本次已批准的具体设计如下：

- `SpaceSpec.timing_mode` 取 `fixed_period`（固定周期，旧配置默认值）或 `measured_native`（一次原生调用，完成后报告实际时长）。`control_period_s` 在固定模式保持正数，在原生模式必须为 `None`，避免它继续被误当动作时长；新增 `physics_period_s` 只声明仿真物理步长。两种周期都是秒，不是墙钟时间。
- `CapabilitySpec.supported_timing_modes` 默认只支持固定周期，`AdaptationProfile.timing_mode` 默认固定周期；三方协商必须一致。未经扩展的旧模型不能误接变时长 Benchmark。
- 新 `NativeAction` decision（模型一次决策）含 `kind="native_action"`、现有 `action_spec_ref` 和现有 `ControllerCommand` 的 `commands`，**不含**预计 `duration_s`。它只表达一次原生控制请求，坐标/动作转换仍由适配器做。接口 `advance` 增加该类型，不删改原 `ActionStep`。
- `ExecutionFeedback.consumed_physics_steps` 与 `ExecutionStatistics.physics_steps` 新增可空计数；旧固定周期产物可留 `None`。原生模式必须同时提供真实原生动作次数、物理步数及 `elapsed_sim_time_s`。物理时间由逐物理步计数与固定步长核验；不能用运行耗时替代。原生规划失败后可能已经增加动作次数却尚未步进物理，三项必须分别记录。
- 新模式拒绝配置 `sim_time_s` 预算；`env_steps` 限制原生动作次数，缓存每行执行前检查上限。观测时间累加执行反馈中的实际仿真时间，而不是动作次数乘常数。完整原生计数若在异常后不可取得，结果标记统计不完整，不填虚假精确值。

`None` 表示没有可声明的值，并不表示零秒；`kind` 是新增 NativeAction 的序列化类型标签；`timing_mode` 是时间计量模式。这些名字与向后兼容默认值已经获得用户确认。尚未验证的物理步计数包装会在隔离运行时做，不修改上述只读上游源码；若无法可靠计数，不注册完整可运行 pair。

这是公共执行语义扩展，影响类型/序列化、校验、能力与 profile 协商、预算账本、失败后的统计完整性和结果解读；不是只在 adapter 中绕过一条校验。默认分支必须保留原85项基线和已有阶段3行为。新分支需要动作次数/物理步/时间一致性、变长动作、提前成功、预算拒绝、失败后的已执行统计、缓存/跨 episode 隔离测试，随后做固定原生种子的逐动作对照。

批准时已完成 OpenWAM 和 LingBot-VLA 2.0 的模型侧原生桥接（`native_robotwin_models.py`），未注册完整 RoboTwin pair，也没有生成真实成功率。现进入公共核心与 Benchmark 执行层实施；真实闭环与原生对照仍按阶段4约定最后集中进行。

RoboDojo 另有运行时解耦问题：本次只读检出 `266130a3ec41ba9b20b0e5648d2f3542f219c06c`，其 `src/eval_client/eval_env.py` 直接导入 XPolicyLab 的 WsModelClient。不能把这个客户端带入独立框架，也不能把上游结果表当作本次成绩。其默认 `arx_x5` 配置是双 X5、0.004 秒物理步、25 Hz 观测采样，并以控制序列插值执行；独立重置/观测/控制/评分和资源来源仍需完成。本次没有初始化或修改其 XPolicyLab submodule。

仍需提供原分工表 R009/R010、R021/R022、R054/R055 的含义，以及 World-in-World 的子协议与被测角色。上述定义缺失不能通过下载更多权重解决。

## 本次批准后的实施与证据

已实施具名时间模式、NativeAction、能力/profile协商、物理计数与实际时间校验、Runner累计账本和原生模式仿真秒数预算拒绝。固定模式缺省值保持，原85项迁移检查和已有阶段3/4检查继续保留。新增22项核心/Runner检查、16项RoboTwin低层组件检查，均用合成数据/替身，不加载真实模型。

- [VLABench侧环境回放](../outputs/stage4-native-timing-regression-20261006T1030Z/baseline-check.json)与 [LIBERO侧环境回放](../outputs/stage4-native-timing-regression-libero-env-20261006T1030Z/baseline-check.json)均为163项全部通过、无跳过，8个原契约示例通过、五组CLI退出码0/0/0/1/2。
- `native_robotwin_executor.py` 实现暂时代理 `env.scene` 以计数**成功返回**的原生 `scene.step()`；控制、规划、渲染、官方成功判断不改写，不添加 `get_obs/check_success` 调用。正常结束与可以测量的部分错误返回实际动作/物理步/秒数；物理调用异常、物理步长或环境计数变化时标记状态未知并禁止继续消费。真实SAPIEN中代理是否覆盖全部物理步尚未验证，诊断明确为false。
- 同文件的LingBot缓存组件保留整个joint14序列、原顺序和数值类型，episode结束清除尾部，只有缓存耗尽才请求新序列；当前不是已注册ModelAdapter/BenchmarkAdapter完整pair。
- 完整RoboTwin pair仍未注册，两个候选格只增加了组件，不能把163项通过解读为真实成绩或全部55格已适配。

## 新核对出的原生观测缺口：提交下一步确认，尚未实施

以下不是已批准时间模式的反悔或重复确认，而是完整公开观测映射所需的新具体事实。

1. LingBot固定客户端 `experiment/robotwin/eval_policy_client_lingbotvla.py:311–362` 只在一个推理动作序列开始时调用 `get_obs`；逐行执行时取图调用被注释。若新框架逐行额外取图，会改变原生图像获取次数，随机光照模式还可能改变随机数消耗。不能为了给每次反馈填“新图”而改原生节奏；`SensorSample.captured_at_s` 必须如实表示实际取图时刻，不能把复用图像标成新采样。
2. 两份RoboTwin `envs/robot/robot.py:get_left/right_arm_jointState` 返回的是 `joint.get_drive_target()`（关节控制目标），不是 `get_*_arm_real_jointState` 的实际关节位置。原生LingBot正是读取这个目标向量。因此不能将它直接冒充当前 `RobotState.joints.position` 测量；需要具名原生状态字段。
3. OpenWAM固定 `action_conversion.py:271–296` 明确将原始末端四元数当作xyzw；RoboTwin固定 `robot.py:588–603` 实际直接使用 `transforms3d.quaternions.mat2quat` 返回值，其 [官方库说明](https://matthew-brett.github.io/transforms3d/reference/transforms3d.quaternions.html#mat2quat)顺序是wxyz。xyzw是三个向量分量在前、实部在后；wxyz是实部在前。这里的源码冲突不能靠统一写一个字段名消除。原生模型桥接目前保留官方helper调用，不私下重排；也不宣称该数组已是标准Pose。

建议下一步采用**原生忠实复现**，不额外逐行取图、不自行修正上游旋转转换。另提交公开扩展 `robotwin.native_proprio.v1`（名字表示RoboTwin原始自身状态版本1），仅保存原生公开状态与真实获取时刻，不包含专家轨迹、目标实体、分割标签或隐藏评分：

| 拟新增字段 | 具体含义 |
| --- | --- |
| `joint_drive_targets_and_grippers` | 14个有限实数，依原生双臂顺序各6个关节驱动目标（弧度）加1个夹爪开度；不是实测关节位置 |
| `left_endpose_world_wxyz` / `right_endpose_world_wxyz` | 每臂7个有限实数，原生变换后的末端世界位置3维（米）与wxyz四元数4维；不重排供官方helper读取的原始值 |
| `left_gripper_open_fraction` / `right_gripper_open_fraction` | 原生公开夹爪开度，0闭合、1张开；仅记录，不改变原生夹爪控制 |
| `captured_at_sim_time_s` | 上述状态真实获取时的累计仿真秒数；缓存动作期间不得改写成新测量时刻 |

该扩展会增加模型可见输入，须由SpaceSpec声明、Observation提供、能力与profile显式放行；不会把原生驱动目标装进实测JointState，也不会把旋转顺序有争议的数组伪装成标准Pose。拟由具名原生控制器保存/执行原始EE16或joint14数值，并在来源记录显式标记约定冲突；修正旋转将是另一个评测变体，需另行确认与原生对照，不能混入本版。

本轮只提交此映射方案，没有增加该扩展/注册完整pair，也没有修改只读上游。待确认后继续原生初始化、专家可行seed筛选、公开观测映射与双版本各自注册；物理计数最终须在集中真实测试中验证。

## 2026-10-06 后续自主推进记录

用户随后要求“不再问细节问题，先全部推进完成，再总结架构和细节”。据此在既有范围内落实上节具名原生映射，不再逐项请示。上文“尚未实施/待确认”是当时状态，本节更新为已实施；不回写历史回执。

已新增 `robotwin_spec.py`、`robotwin_model.py`、`robotwin_benchmark.py`、`robotwin_pairs.py`。两个固定版本各自注册，采用 demo_clean/adjust_bottle/episode0/unseen、种子编号0、最多100候选的原生专家可行性筛选，动作上限400；真实选中种子另记，起始候选是100000。公开相机保留原生D435的240×320 RGB；关节驱动目标仅进入具名扩展，不装作实测JointState；控制器载荷保持原生EE16或joint14数值与数据类型。

LingBot缓存动作之间复用上一次相机/状态，并保留真实采样时间；只在一批动作结束且还能继续决策时取图。OpenWAM按原生逐次请求节奏取图。两者都不在结束/预算耗尽后追加取图，也不额外调用官方成功判断。OpenWAM旋转helper顺序冲突原样保留并标记，不宣称已纠正。

OpenWAM初始化指令采用其固定源码的sealed manifest函数：隔离的Python/NumPy/Torch随机数状态与固定候选种子决定文本。sealed manifest指先固定初始化与指令再执行。LingBot原脚本没有给Python文本生成随机数设种子，本次明确补种子并隔离随机数；协议名为 `lingbot-seeded-instruction-port-1`，即“指令带种子的兼容移植第1版”，不能宣称与未固定Python随机状态的旧脚本逐字/逐像素相同。视频/数据采集诊断输出关闭，不改变动作规划或官方评分，差异进入诊断。原生源码及XPolicyLab没有改动。

新增25项替身接口/串行Runner检查；两套独立环境累计188项、8个契约示例、五组CLI0/0/0/1/2通过。真实SAPIEN计数覆盖与逐动作原生对照仍未完成，完整注册只表示源代码和入口存在，不表示真实可运行环境已就绪或阶段4验收。

后续集中真实依赖检查发现SAPIEN3周期返回值的浮点表示：源码1/250秒在接口中为0.004000000189989805秒。describe先获取实际值，再创建profile和manifest；后续记账保留实际值且执行中不允许周期改变。只容许初始化时的32位浮点表示误差，不放宽真正的控制周期变化、不制造原生宏动作固定时长。新增4项周期反例/接线检查。基础真实SAPIEN空场景3次物理步及CUDA/CuRobo导入已通过，见 `outputs/stage4-robotwin-runtime-gate-20261006T1518Z-fixed/receipt.json`；这还不是机器人任务重置或完整原生动作计数覆盖。

### 真实机器人执行与两层对照（16时更新）

LingBot真实单次111动作／9648物理步／38.59200183302164仿真秒，官方成功且有效。独立进程直接重放原生关节动作，在实际PhysxCpuSystem完成调用处另计9648步，112份公开状态与三相机快照、指令和最终成功结果精确一致：[回执](../outputs/stage4-native-action-replay-lingbot-robotwin-20261006T1633Z/receipt.json)。不是新模型推理或所有任务的计数证明。

OpenWAM真实单次113动作／11425物理步／45.700002170633525仿真秒也成功。但同设备独立原生末端动作重放，从第一步出现状态与图像差异，113动作最终10571步／42.28400200838223秒，仍成功；**严格轨迹对照未通过**：[回执](../outputs/stage4-native-action-replay-openwam-robotwin-20261006T1650Z-same-device/receipt.json)。早先不同设备重放的失败也保留。原因尚未完全定位，不能直接归结为计数器漏记或据此放宽规则。

另作同时观察的计数专项：同一次新真实末端执行中，生产Scene.step计数器与更底层PhysX观察器逐动作精确一致，实际112动作／10627步：[回执](../outputs/stage4-native-meter-crosscheck-openwam-robotwin-20261006T1655Z/receipt.json)。该回执的整体 `passed=false` 指历史轨迹复现失败；独立 `simultaneous_meter_cross_check_passed=true` 只证明当前执行的物理计数相符。两种结论不可混用，不回写原运行诊断中的未验证标记。

三套环境已202项软件回归、8契约示例、五组mock退出码0/0/0/1/2通过，最新索引见stage-state.json。公共协议仍按已批准方案，不擅自修改控制器、旋转顺序或重复性容差；阶段4验收与契约冻结未自动发生。

### 17时严格复现补充

OpenWAM原生重放增加完整候选种子历史后，100000拒绝和100001接受及指令一致，仍在动作1出现轨迹差异，113动作实际10583物理步：[回执](../outputs/stage4-native-action-replay-openwam-robotwin-20261006T1702Z-seed-search/receipt.json)。没有把不同轨迹的步数差当作计数器漏记，也不将此前同时计数专项通过充作历史轨迹通过。

LingBot另做两次由新原生模型推理控制的完整试验，任务都成功，但分别112动作／9757物理步与111动作／9634物理步，不同于框架111动作／9648步；严格首动作即有差异。第二次首次原生模型输入与Adapter映射在消息编码后数值／类型精确一致，原生重复预测本身仍不同：[重复回执](../outputs/stage4-fresh-native-policy-lingbot-robotwin-20261006T1730Z-repeat/native/receipt.json)。这是完整模型／环境重复性缺口，不撤销已有记录动作重放的计数与快照一致结论；完整原因未定位，不新增未经确认的确定性开关或容差。
