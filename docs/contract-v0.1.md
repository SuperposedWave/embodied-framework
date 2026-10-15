# 数据契约 v0.1：实现说明

状态：阶段 1 契约已实现并用于阶段 2 的 mock 执行器；尚未正式冻结。Python 类型定义以 embodied_eval/contracts.py 为准，实际 JSON 格式见 examples/contract_examples.json。本说明将此前讨论稿落实为可校验的数据结构。

## 1. 三方职责与数据流

| 事项 | 模型侧 Adapter | Benchmark 侧 Adapter | 框架 |
| --- | --- | --- | --- |
| 身份与能力 | 模型源码、权重、归一化、支持的机器人、输入与输出能力 | Benchmark 协议、机器人、传感器、控制器、工具 | 按版本化 Profile 检查声明 |
| 观测 | 标准输入 → 原生输入；选取、排序、resize、tokenize、归一化、原生旋转表示 | 原生传感器与状态 → 标准 Observation；声明真实时间、单位和坐标 | 校验、白名单过滤、快照隔离 |
| 动作 | 原生输出 → 标准 Prediction；区分物理量与控制器命令 | 标准决策 → 原生控制；负责 IK、控制器、限位和实际反馈 | 校验、执行时序和缓存编排 |
| 工具 | 推理、对话状态、工具调用参数和结果消费 | 工具实现、参数语义、计划与执行、每步预算 | 统一反馈，禁止隐藏评分字段进入模型 |
| 生命周期 | 清空模型历史、KV cache、内部 chunk 缓存 | 初始化和结束环境、官方判定、评分 | 组织调用、错误收尾和释放资源 |
| 结果 | 可提供前向次数和原始输出引用等诊断 | 官方指标及评分来源 | 身份、结束原因、执行统计及产物引用 |

~~~text
Benchmark 原生观测
  → BenchmarkAdapter → 标准 Observation
  → 白名单快照 → ModelInput
  → ModelAdapter → 模型原生输入

模型原生输出
  → ModelAdapter → 标准 Prediction
  → 校验与编排 → BenchmarkAdapter
  → 机器人原生控制输入
~~~

模型专属预处理随 Adapter 源码和 Profile 版本管理。标准空间不要求模型原生图像尺寸、关节排列、夹爪取值或动作向量一致。模型声明的 required_inputs 描述标准输入；原生需求由模型 Adapter 转换。隐含坐标变换、自动缩放、自动限幅不属于框架行为。

## 2. 结构层次及必填语义

| 契约 | 必填身份与内容 | 可选部分及限制 |
| --- | --- | --- |
| SpaceSpec | spec_id、revision、协议、control_period_s、frames、robots、sensors | 命名 action_specs、controllers、discrete_specs、tools 和有命名空间的扩展 |
| CapabilitySpec | model_id、source_revision、checkpoint_ref、normalization_ref、roles、required_inputs、supported_embodiments、observation_update、chunk_mode | 输出规格引用、停止能力、预测模态、扩展；batch_supported 仅为声明，尚无 batch 执行器 |
| AdaptationProfile | 配置身份与版本、双方规格身份、角色、robot_ids、input_bindings、输出引用、chunk 归属、执行 horizon、view_policy | frame_conversions、必需扩展、unresolved；unresolved 非空即阻塞 |
| TaskSpec | 任务和 episode ID、协议、public、初始化引用、可见性策略引用、budgets、评分配置引用 | evaluation_private 始终留在 Benchmark/框架内部 |
| Observation | Context、observation_id、spec_ref、env_step、sim_time_s、命名 sensors / robots | 动态坐标变换和明确允许的扩展 |
| Prediction | Context、request_id、based_on_observation_id、profile_ref | decision 与 predictions；角色决定何者必须存在 |
| StepResult | Context、新观测、执行反馈、terminated、truncated | 中间观测、结束原因、内部评测信息；结束原因与官方指标分开 |
| EpisodeResult | episode_id、manifest 引用、运行状态、结束原因、官方 metrics、执行统计 | 轨迹、视频和日志等产物引用 |

Context = run_id / env_id / episode_id / message_id / contract_version。接口采用 0.1；无法识别的版本或字段会拒绝。Python 构造和缺省字段使用 0.1，正式跨版本迁移尚未实现。

没有数据用缺失字段或 null；失效传感器另用 validity=invalid 与原因。有效零值保留。缺失必需输入报错，不补零。预算中的 null 表示该项无上限，0 表示没有剩余预算。所有数值须有限；JSON 不接受 NaN、Infinity、重复键或把 bool 当作整数。

## 3. 空间和载荷

### 机器人与命名

robot0/left_j0、robot0/left_eef、robot0/left_gripper、robot0/base 是动作实体 ID。机器人 ID、实体局部 ID 不得包含斜线，base 是保留局部名。

RobotSpec 定义关节、末端和夹爪。JointSpec 的位置/速度/力单位分别为 rad / rad/s / Nm 或 m / m/s / N，包含可用的位置范围。末端定义 TCP 和局部夹爪 ID。多个机器人、多个机械臂、多个相机通过字典的实体名区分。

GripperSpec 声明支持的模式：

- open_fraction：0 闭合、1 张开。
- width_m / force_n：米 / 牛顿，支持明确宽度上限。
- joint_targets：指定 RobotSpec 内 joint_ids，数值是对应关节的位置，沿用这些关节的单位与限位。

Observation 内机器人状态可以缺少部分测量。JointState 对不同字段分别使用 null；整个关节测量失效时，所有数值须为 null。机器人状态在 v0.1 中视为该 Observation 时刻的同步快照；异步关节与状态采样需要后续明确协议。

### 传感器与张量

SensorSpec 明确 modality、role、mounted_on、frame_id、TensorSpec、单位、颜色顺序和可选采样频率。RGB 使用 RGB，HWC 或 CHW，3 个通道；depth 使用米。

SensorSample 明确 captured_at_s、frame_id、validity、provenance=sensor、payload。失效样本没有 payload。未来生成的图像不会装进 SensorSample。

Tensor 包含 dtype、shape、layout；data 与 data_ref 二选一。固定形状按每维检查；规格允许 -1 代表动态维，实际载荷各维必须为正数。inline data 是有嵌套形状的 JSON 数组，不接受未声明的 ndarray/bytes。

DataRef 为未压缩、紧密排列的逻辑张量缓冲区：storage / key / byte_length / sha256。dtype、shape、layout 来自 Tensor；以 little-endian、C 连续顺序解释多字节元素。当前只检查长度和 SHA-256 字段格式，不访问文件或验证实际内容。压缩 JPEG/PNG 需要 Benchmark Adapter 解码成标准张量；编解码和传输实现不在本阶段。

### 坐标与旋转

位置用米，旋转用单位 quaternion_xyzw，角速度用 rad/s。Transform 是 T_parent_child，把 child 坐标转到 parent：

~~~text
p_parent = R_parent_child × p_child + translation_m
T_A_C = T_A_B × T_B_C
~~~

FrameSpec 有 parent_id；非根坐标系必须恰好声明 static_transform 或 dynamic。根没有父变换，图无环。动态变换由 Observation.frame_transforms 返回，带独立采样时间。不同坐标根之间没有隐含关系。

Pose.frame_id 表示该位姿数值使用的参考坐标系；EndEffectorSpec.frame_id 表示 TCP 的坐标节点，两者不要求相同。共享 frame_transform 按声明的坐标树计算；没有动态样本或存在过期样本时拒绝。

quaternion ↔ rot6d 的 rot6d 约定是旋转矩阵前两列顺次拼接。退化表示报错。单位或旋转转换必须由 Adapter 显式调用，不因字段形状相同而自动执行。

## 4. 模型输入绑定与信息边界

InputRequirement 为模型输入 key 指定 kind，可选约束为模态、dtype、shape、layout、单位、坐标系和传感器 max_age_s。InputBinding 将 key 绑定到明确实体：

| kind | source_id | field_name |
| --- | --- | --- |
| sensor | head_rgb | null |
| joint | robot0/left_j0 | position / velocity / effort |
| end_effector | robot0/left_eef | pose / gripper_state |
| base | robot0/base | pose / twist |
| task | instruction / goal_image / goal_location | null |

view_policy 同时规定公开的 task_fields、sensor_ids、robot_fields、frame_ids、feedback_fields 和 extension_names。robot_fields 使用完整路径，例如 robot0/joints/left_j0/position，只暴露位置时不会把速度、力顺带暴露。公开坐标树需包含所需的父节点。

模型收到：

- begin_episode：Context、过滤后的 PublicTask、过滤后的 SpaceSpec、已校验 Profile。
- observe：ModelInput，仅包含公开任务、公开观测和结构化执行反馈。
- end_episode：ModelEndContext，只包含身份与 environment_terminated / budget_exhausted / model_stop / run_error / aborted 之一。

初始化引用、评分配置、evaluation_private、隐藏场景坐标、未允许的状态字段均不进入模型输入。失效原因可能含内部诊断，发送给模型前替换为通用 unavailable。数组和字典生成深拷贝，模型修改副本不会污染原始环境状态。

信息白名单由 Benchmark 官方协议和人工核对决定。框架无法识别被错误标记为公开的图像、扩展或指令里是否包含答案；这一点必须在真实接入时审计。模型初始化只收到公共机器人元数据；不接收完整 TaskSpec 或 StepResult。

## 5. 决策与时间

Prediction.context.message_id = request_id；based_on_observation_id 必须对应源观测。校验可接收待完成请求的 Context，拒绝旧请求响应。模型、环境和 episode 身份不匹配即拒绝。

### 动作序列

固定周期模式下，ActionSequence = action_spec_ref + execution_horizon_steps + 有序 steps。ActionStep.index 从 0 连续编号，每步一个逻辑控制 tick，duration_s 必须等于 control_period_s。同一步一个目标只能有一个命令；整个 chunk 的 command_id 唯一。

命令为：

| 类型 | 内容 |
| --- | --- |
| JointTarget | target、position / velocity / effort、值及明确单位 |
| PoseTarget | target、米和 xyzw 绝对位姿、参考 frame |
| PoseDelta | target、Transform、frame、reference_policy、composition |
| TwistCommand | base / eef、线速度与角速度、frame |
| GripperCommand | gripper target、明确模式与值 |
| ControllerCommand | target、controller_profile_ref、命名参数、controller_native 域 |

PoseDelta 的 reference_policy 为 chunk_start / step_observed / previous_command。chunk_start 必须引用产生该 chunk 的 observation ID；另外两种由执行器在对应步骤解析，不能偷用 chunk 初始位姿。composition 明确是 pre_multiply 或 post_multiply，不能仅根据“delta”命名猜测。增量 frame 是进行该矩阵计算的参考坐标系；参考位姿先显式转换到它。IK 和参考位姿解析在 Benchmark Adapter，框架不得把增量改写为归一化控制器向量。

runner_owned：模型返回整个 chunk；执行器只执行本次 horizon，丢弃余下预测，再根据当前观测重新推理。adapter_owned：模型 Adapter 自持缓存，每次必须只返回一个 ActionStep，Profile horizon=1。episode 边界都必须清空缓存；消费、清空和历史隔离已在阶段 2 的 mock 闭环验证。

ControllerSpec 声明 frame、参数 JSON Schema 和每个参数的单位/域。ControllerCommand 保留原生控制域，例如 normalized-controller@1 中 [-1,1] 的 delta；不伪装成米或 rad。标准层既不隐含缩放，也不把控制器参数送到共享物理单位转换。

### 离散动作、工具和停止

DiscreteSpec 定义词表、目标和 duration_s；持续时间必须是完整控制 tick 的整数倍。ToolSpec 定义 argument_schema、目标和 per_control_step 预算约定。参数使用完整 JSON Schema Draft 2020-12 校验；只允许本地 $ref / $dynamicRef，不联网取 schema，无法解析的引用报错。

每次工具调用包含 call_id / tool_ref / target_id / arguments。Benchmark 必须在每个环境 tick 前检查剩余步数与模拟时间预算，返回实际消耗。规划/参数拒绝在执行前发生时不消耗环境步；已执行后的错误保留实际消耗。

长工具和多 tick 离散动作可声明 intermediate_observations_supported。若模型需要 every_env_step，必须提供从执行起点后第一步至最终观测前一刻的完整中间观测。StepResult.intermediate_observations 顺序连续，最终观测仍在 observation。工具原子调用结束后按时间顺序更新模型历史；该字段不意味着支持工具运行中的模型抢占或再次决策。

StopRequest 表示模型主动停止，不能据此记 success=true。success 仍由 Benchmark 官方 scorer 返回。

### 未来预测

FuturePrediction 有 modality、relative_time_s、Tensor 和 provenance=model_prediction。纯 world_model 角色没有机器人 decision，必须有预测。未来状态 Tensor 的具体字段含义和评分协议尚未定义，须在阶段 5C 确认。ModelInput.candidate_actions 是预留字段，当前不允许启用，避免未确认的规划输入混入评测。

### 2026-10-06 已批准的阶段4公开扩展

核心类型及固定周期执行规则不变，契约仍未正式冻结。新增 `vlabench.native_proprio.v1` 服务已确认范围内的 VLABench 固定场景策略协议，OpenWAM 和 π0.5 分别注册映射，不替代 LIBERO 扩展：

| 字段 | 形状及具体含义 |
| --- | --- |
| `ee_state_world_wxyz` | 8 个有限实数：世界坐标末端位置 3 维（米）、按 w/x/y/z 顺序的原生四元数 4 维、原生夹爪 closed 标记 1 维；保留上游 `get_ee_open_state` 的已知反向极性，不把它改成动作极性 |
| `robot_base_world_m` | 3 个有限实数：世界坐标机器人基座位置（米）；原生模型客户端用它做世界/基座平移转换 |
| `instruction` | 当步原生 `task.get_instruction()` 抽取的非空公开任务文本；不包含目标实体 ID、私有条件或评分信息 |

`wxyz` 表示四元数实部 w 在前、三个虚部 x/y/z 在后。此扩展由 SpaceSpec 声明、Observation 提供，并由 AdaptationProfile 的公开视图显式放行；Model Adapter 只读取这些指定字段。VLABench 当步公开措辞可能改变，因此初始 TaskPublic.instruction 不代替当步 extension 文本。原生控制器和 IK、图像 resize、状态/动作转换及官方评分分别留在 Benchmark/Model Adapter，Runner 不含这些专属逻辑。IK 是从末端目标求关节位置的逆运动学，resize 是图像尺寸变换。

阶段4首轮使用已批准的 inline RGB 图像，即 Tensor.data 内保存完整像素；不意味着 DataRef 已具备载荷读取。RGB 指红、绿、蓝三通道。π0.5 的 `openpi-native-1` 与阶段3 OpenWAM 的 LIBERO 协议分别登记静置、缓存、seed 和预算，不能合并原生结果。可变时长核心扩展随后单独获批并实施，见下节；不改变这些旧固定周期路径。

2026-10-06 增加独立 `openpi_vlabench×vlabench_openpi` 注册，复用上述已公开字段，不新增核心数据字段。固定 openpi fork `788ef1f20e655de5bb40869fb7b7fcdd49732387`、专用权重 `VLABench/pi05-primitive-10task@f658a8bec801aab960fe0ceab370a123b31c1837`，不复用 LIBERO 权重/归一化/相机旋转。原生三相机输入顺序为前视、右侧、腕部；原始8维状态在 Model Adapter 转成位置3维、欧拉角3维、夹爪标记1维。原生输入位置减固定 `[0,-0.4,0.78]` 米，输出位置加当步公开基座位置；两者按该版本原生客户端分别保留，不自行统一。服务端负责32维补齐、归一化及训练定义的相对/绝对动作转换，客户端缓存前5个7维动作，Benchmark 执行同一原生 IK 和评分。上述规则的实现检查不代表真实原生对照已通过。

### 2026-10-06 单独批准并实施的可变时长核心

用户在白话解释后回复“我认为可以，继续”，批准 [具体设计](variable-transition-review.md)。`timing_mode` 是时间计量模式：`fixed_period` 表示固定周期（旧消息缺省值），`measured_native` 表示一次原生调用完成后报告实际时长。

- `SpaceSpec.control_period_s` 在固定模式必须是正数；原生模式必须为 `null`，同时给出正数 `physics_period_s`，即每个实际物理步的仿真秒数。`null` 不是零秒。
- `CapabilitySpec.supported_timing_modes` 缺省只支持固定周期；`AdaptationProfile.timing_mode` 缺省固定周期，双方及映射必须一致。原生模式首版要求 Adapter 自持缓存、每次执行一个原生动作、仅决策边界观测，不接离散动作/工具执行。
- 新 `NativeAction` 用 `kind="native_action"` 标明类型，含 `action_spec_ref` 和 `ControllerCommand` 数组，**没有** `duration_s`。每个目标仅一个命令、命令 ID 唯一、原生控制参数仍经既有 JSON Schema 校验。它不能送到固定模式，原 ActionSequence 也不能送到原生模式。
- `ExecutionFeedback.consumed_env_steps` 在新模式记录原生动作次数（每次至多一次），`consumed_physics_steps` 必须报告完成的物理步数；`elapsed_sim_time_s` 等于该计数乘物理步长。`ExecutionStatistics.physics_steps` 累计真实物理步。旧固定模式的两个新增物理计数字段可以为 `null`。
- 原生规划失败可能已消费一次动作，即使物理步为零；不套用旧模式“规划失败消费零逻辑步”的规则。参数拒绝仍不消费动作或物理步。提前成功、已知部分执行失败保留实际计数；无法确认执行状态时沿用“最后确认的统计”，不估计完整消耗。
- 新模式拒绝任何非 `null` 的 `sim_time_s` 预算（包括零）；动作次数/模型调用次数限额仍可用。墙钟限额只在调用边界检查，不是硬超时。`Observation.sim_time_s` 使用累计实际仿真时间，不用原生动作次数乘一个虚构周期。
- 物理计数不自动扩大模型可见反馈：原有 `feedback_fields` 白名单保持不变。观测只表示已声明的决策边界，不伪造每物理步图像。

本轮新读取器可解码省略新字段的旧固定周期消息，并保留旧执行行为；新产物会序列化新增字段，旧严格读取器会拒绝未知字段，**不是**双向跨版本兼容保证。契约仍为未冻结的0.1；正式版本迁移需另行设计。RoboTwin完整pair入口已注册，低层物理计数、公开状态映射和缓存已有替身检查；尚未证明真实 SAPIEN 计数或原生逐步对照一致，注册不等于真实环境已就绪。

## 6. 执行反馈、评分与错误

ExecutionFeedback 分开表达 request_id / call_id / command_ids、执行状态、真实环境步、真实模拟时间和结构化 code。没有自由诊断文本或任务成功字段。StepResult 的内部 evaluation_private 永不传给模型。

terminated、truncated 保留原生两个标志，允许二者同时为 true；有任一结束标志时须给 end_reason。同时发生任务终止和预算耗尽时，官方成功判定以 scorer 为准。运行异常用 EpisodeResult.run_status=error，不能默认当作任务失败。

Metric 必须带 source 与 unit。获得的失败是 value=false / valid=true；无法评分是 value=null / valid=false / unavailable_reason，不能记为 false 或 0。ExecutionStatistics 区分 model_calls 和实际 model_forward_count；未知前向次数用 null，不估算。

ContractError 携带 code、path、message。AdapterError 分类覆盖模型输入/输出、服务、工具参数、规划、执行、Benchmark、评分和基础设施等；execution_started / state_known / retryability 帮助后续执行器判断恢复策略。当前没有实现自动重试。

## 7. 接口落实与验收说明

ModelAdapter：describe → begin_episode → observe / predict → end_episode → close。
BenchmarkAdapter：describe / enumerate_tasks → reset → advance / execute_tool → score → close。

相对此前讨论稿，落实时明确了以下接口细节：

1. 模型 begin_episode 增加过滤后的 SpaceSpec，提供单位和坐标解释依据。
2. advance / execute_tool 显式接收请求 Context，反馈可以准确关联请求。
3. StepResult 增加中间观测序列，支持长工具后完整更新模型历史。
4. ModelEndContext.reason 使用闭合枚举，避免把评分或内部错误文本带给模型。
5. 夹爪 joint_targets 要求声明关节身份；离散持续时间按逻辑控制 tick 校验。

这些决定均以合成数据验证，尚未用真实模型或 Benchmark 验证。阶段 2 的 Runner 验收见 stage2-acceptance.md；阶段 3 在确定真实组合后验证原生控制与官方评分。执行状态未知时保留最后确认的统计并显式标记，不能解读为完整实际消耗。

## 8. RoboTwin 原生公开扩展（阶段4自主推进落实）

`robotwin.native_proprio.v1` 表示RoboTwin原始自身状态版本1。扩展经SpaceSpec声明、CapabilitySpec支持、AdaptationProfile白名单放行；不扩大隐藏信息。`joint_drive_targets_and_grippers` 是双臂各6个驱动目标角（弧度）加1个夹爪值，共14个数，不是实测关节角；RobotState不借此填测量值。左右 `endpose_world_wxyz` 各是位置3维（米）加标量在前的旋转四元数4维。`captured_at_sim_time_s` 是真实取图/状态采样时刻，缓存期间原样保留；不是每次反馈都产生新测量。

原生ControllerCommand参数含 `native_row`（原生控制数组）、`native_dtype`（数值类型）与 `acquire_observation_after`（在原生批次边界取图的调度标志）；后两者只作传输/调度元数据，不传给机器人控制器。原生行不缩放、裁剪或静默改类型。OpenWAM每次请求一行EE16；LingBot完整消费joint14序列，只有批次边界需要新图。EE16是每臂位置3维、四元数4维、夹爪1维，共16维；joint14含义如上。首次不声明未知相机外参，各相机坐标系独立为根，不虚构校准。

OpenWAM原生转换函数与Benchmark源四元数顺序存在冲突，本版保留官方调用并标注，不当作标准Pose、不私下修正。具体默认初始化、种子筛选、取图节奏和移植差异见 `variable-transition-review.md` 后续记录；正式冻结与全量成绩仍未批准。

实际SAPIEN3的物理周期通过32位浮点接口返回：源码设置0.004秒，运行时返回0.004000000189989805秒。本轮在describe阶段先读取实际返回值并填入既有physics_period_s，后续计数和时间都用该值，不把累积结果舍入成名义值。初始化仅接受名义1/250秒的浮点表示误差；执行中仍严格检查周期不变。没有新增核心字段、改变原生控制或扩大预算能力。基础真实场景计数已检查；完整RoboTwin动作路径覆盖仍待真实任务验证。
