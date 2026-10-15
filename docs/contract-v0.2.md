# 0.2：独立模型与仿真环境的公共契约

状态：2026-10-07 用户批准实施的新增通路，尚未正式冻结；正式阶段4验收另行提交。0.1保持原有含义、配置、CLI行为及历史证据。本文是技术接口说明，不是全量评测成绩。

M表示模型数量，N表示仿真Benchmark数量。M+N的具体含义是：模型只实现自己的输入输出，环境只实现自己的观测与执行；组合时写相机、控制目标和坐标接线数据，不为每对组合新增Python适配器。它不表示任意权重都能完成任意环境的任务。

## 两条互不改写的通路

`contract_version="0.2"`的配置由`embodied_eval.mn.config`解码，经独立注册表进入`mn.runner`。省略版本或版本为0.1的既有配置继续走原配置和执行循环。旧六组注册名和输出目录不改名、不回写。

0.2复用0.1已有的SI单位、张量、坐标树、命令叶子类型、预算、执行反馈和指标类型；新的顶层上下文、能力、空间、观测、动作、预测和任务是独立类型。不能把0.2消息作为0.1消息解码。严格JSON解码继续拒绝未知字段、重复键、类型错误和非有限数值。

SI单位在这里指米、弧度、秒等具有明确物理含义的单位；四元数`xyzw`表示四个分量依次为x、y、z和标量w，不是原生`wxyz`的另一种写法。

## 数据流和权限边界

~~~text
仿真原始观测 → Benchmark Adapter → 公共观测
公共观测 → 输入接线/公开投影 → Model Adapter → 模型推理
模型输出 → Model Adapter → 公共动作
公共动作 → Benchmark Adapter → 原生控制器/IK/物理推进/官方评分
~~~

IK指逆运动学，即从末端目标位置求机器人关节目标；它属于环境执行边界。

模型收到公开的任务文字、空间定义、已解析接线和公开输入；不收到环境实例、内部评分条件、专家动作、私有初始化内容或Benchmark配置。环境工厂只收到自己的配置和产物路径，不收到模型源仓库、端点、权重或模型配置。注册、预检查和Runner均不枚举可用组合白名单。

Runner只编排生命周期、检查消息、计算预算、累加实际执行反馈并写证据。图像翻转/缩放、状态排列、归一化与去归一化、原生采样顺序、动作缓存和模型调用属于模型。机器人控制、IK、物理计数与官方评分属于环境。

## 独立注册与生命周期

`AdapterRegistry.register_model(id, ModelRegistration)`和`register_benchmark(id, BenchmarkRegistration)`分别注册。重复或未知名称明确拒绝。添加第三个组件只注册该组件，不修改Runner或另一侧Adapter。

每个注册记录提供：参数校验、静态能力/空间声明、工厂、资产校验、就绪检查、来源说明。环境还提供指标模板。工厂只在运行时调用；静态预检查不得导入Torch、JAX、NumPy或仿真/GPU运行库。

模型接口：`begin_episode(context, task_public, bound_profile, public_space)`、`observe(ModelInput)`、`predict(context)`、`end_episode(context)`、`close()`、`diagnostics()`。

环境接口：`describe()`、`enumerate_tasks(episodes)`、`reset(task, context)`、`advance(action, context, remaining_budgets)`、`score(context)`、`close()`、`diagnostics()`。

现有真实环境协议仍限制每个进程一个固定试点任务；0.2独立架构不是同时完成全部suite或新任务的承诺。

## 公开观测：数值含义、来源和采集时间

`SpaceSpec`包含版本/修订、具名环境协议、时间模式、坐标树、机器人、传感器、状态字段、可执行动作格式和控制器。

`StateSpec`定义状态的`semantic`（具体含义）、`unit`（单位）、`frame_id`（坐标系）、`tensor`（维数/类型/排列）和`origin`（实际采集部位或执行器）。支持末端位置/方向、夹爪位置/开度、机器人底座位置、实测关节位置和关节驱动目标。

`StateSample`包含张量、`captured_at_s`（采集时对应的仿真秒数）、`provenance`（measurement实测或controller_target控制目标）。关节驱动目标必须标为控制目标，不能冒充当前关节角度。图像同样保留采集时间和传感器坐标系。观测当前仿真时刻与旧图像采集时刻可以不同。

`Observation.public_instruction`是环境当前公开指令；有该字段时替代任务初始化文字。私有评分字段不会因此进入模型输入。

逐步公开投影保留任务已有的公开目标图像和目标位姿，只替换动态指令；不把替换指令误当作删除其他公开目标。投影仍深拷贝，模型不能改写环境原始样本。

`frame_transforms`保留声明为动态的坐标变换及其采集时间。输入要求坐标转换时，必须有显式接线和可连通坐标树；缺少动态变换时拒绝。坐标转换不是对不同TCP位置、机器人形态或关节语义的猜测补齐。TCP指模型/控制器定义的末端工具参考点；来源不同不能仅靠更名视为同一点。

原生RoboTwin的14维驱动目标状态明确注明：左6关节弧度、左夹爪开度、右6关节弧度、右夹爪开度；其单位为具名混合字段`rad_and_open_fraction`，不是14个实测关节角。动作则拆成12个独立关节命令和2个夹爪命令。

RoboTwin的`get_left_gripper_val`/`get_right_gripper_val`也返回记录的截断控制目标，而非夹爪实测关节位置。因此两个单独夹爪状态明确为`gripper_drive_target`和controller_target；不改其数值，也不冒充measurement。需要实测开度的模型不能把这个字段作为相同语义输入。

## 模型声明与匹配

`CapabilitySpec`包含模型、源码修订、权重与normalization身份、输入要求、输出说明、支持的时间模式、观测更新方式和训练覆盖标签。Normalization在这里指权重对应的输入输出标准化统计，不可用另一权重的统计替换。

每个`InputSpec`有模型逻辑槽名、输入种类sensor/state/task、所需含义、单位、张量要求、最大采集年龄和可选坐标系。图像维数描述可接受的公共输入；例如`(-1,-1,3)`表示高度和宽度可变、RGB三通道，不要求环境先缩成模型内部224×224。模型负责最终排列和缩放。

`OutputSpec`声明命令类型、逻辑控制槽及其目标种类。关节输出还声明position/velocity/effort，分别是位置、速度和力矩/力目标；夹爪输出逐槽声明open_fraction/width_m/joint_targets/force_n，分别是归一化开度、物理宽度、夹爪关节目标和夹持力。

匹配检查依据相机/状态实际结构、含义、单位、维数、坐标、动作类型、执行模式和必要目标。`SpaceSpec.required_action_targets`表示该执行模式要求同时提供的全部目标；双臂执行不能只送一只手。`action_joint_modes`表示环境实际实现的关节执行模式；位置控制不自动支持速度或力矩。

训练过的机械臂名称不再是拒绝门槛；未确认训练覆盖返回`training_coverage_unverified`提示。缺相机、缺关节、夹爪单位不符、坐标不明、重复覆盖同一目标或原生协议不符仍明确拒绝。结构匹配不等于存在跨机械臂有效权重。

## 接线配置只声明数据

`BindingConfig.inputs`把模型逻辑输入槽映射到环境公开字段；`targets`把逻辑输出槽映射到具体机器人目标；`action_spec_ref`选择环境已声明的执行格式；`frame_conversions`声明源坐标系到目标坐标系的转换。

输入和输出槽必须全部且仅被绑定一次；不能静默忽略额外相机槽或第二只手。目标标识例如`robot0/left_j0`具体表示robot0的左臂第0个已声明关节，不凭字符串猜关节次序。

绝对位姿转换由Model Adapter调用公共几何工具后输出目标坐标中的动作，Benchmark按声明的坐标接受；接线不含Python函数、模型分支、IK、评分或组合工厂。静态预检查检查坐标路径；运行时检查实际动态变换。

## 公共动作与具名原生扩展

`Action.commands`是一组带类型命令，而不是所有模型共用一个无含义数组：

| 类型 | 明确表达的含义 |
| --- | --- |
| JointTarget | 哪个关节、位置/速度/力矩模式、数值与单位 |
| PoseTarget | 哪个末端、绝对位置米、xyzw方向、坐标系 |
| PoseDelta | 哪个末端、相对位姿、参考观测/参考策略、坐标系 |
| GripperCommand | 哪个夹爪、宽度/开度/力等模式及数值 |
| TwistCommand | 线速度、角速度及坐标系 |
| ControllerCommand | 具名控制器、声明的参数结构、原生协议 |

公共类型存在不代表每个真实环境已实现全部执行模式。现有VLABench执行绝对位姿和夹爪宽度；RoboTwin执行完整双臂关节目标或绝对位姿/夹爪，也保留具名旧原生动作。未声明的相对位姿、速度或力矩仍拒绝。

`native_protocol`及签名仅允许原生数组进入明确声明同一协议的执行端。LIBERO旧归一化OSC动作包含环境缩放、世界坐标增量和20Hz定义；OSC指原生操作空间控制器，不能把其7个归一化数直接当米/弧度。

OpenWAM旧双臂转换存在原生四元数约定冲突，因此保存在`dual-eef16-legacy-quaternion@1`具名扩展；不将其强行标为通用物理位姿或宣称可跨环境使用。后续修正需新的具名模型配置/协议及独立对照，不能改写旧数值以获得一致。

LingBot首轮真实输出揭示夹爪请求可以略超0到1。固定RoboTwin源码先将原值与当前值插值，再在每个物理步的`set_gripper`中截断，提前截断会改变轨迹。因此joint14配置保留12个公共JointTarget，两个夹爪使用`gripper-open-fraction-raw-request@1` ControllerCommand，参数`requested_open_fraction`是未截断的有限归一化请求，不是合法公共开度。模型/环境共同声明`gripper-open-fraction-interpolate-then-clamp@1`签名，接线选择`joint-absolute-native-gripper@1`；无相同声明的环境拒绝。通用GripperCommand.open_fraction仍严格限定0到1；通用公共双臂关节/夹爪执行格式继续存在，没有放宽范围或修改原生插值/评分。

`native_dtype`保留原生数值精度。`source_euler_xyz_rad`是绝对PoseTarget的可选无损源表示：保留模型原生欧拉角弧度，避免四元数转回欧拉角时产生新的舍入。校验其与公共四元数代表同一方向后环境才可使用；做坐标变换时移除旧提示。它不能改变物理含义或官方评分。

`acquire_observation_after`保留模型缓存的原生取图节奏：例如LingBot完整chunk（一次返回的连续动作段）执行完再取图。它不是Runner拥有的动作缓存。每次执行均返回含时间/计数的公开观测，图像可能保持旧采集时间。

## 预算、时间和错误

fixed_period表示每次动作有固定控制周期；measured_native表示一次原生动作可推进不同数量物理步。

原生模式分别记录：`env_steps`原生动作调用数、`physics_steps`实际完成物理步数、`sim_time_s`这些物理步对应的仿真秒数。计数来自环境，不用动作数乘固定时长冒充物理计数。预算保留动作次数、模型调用次数及协作式wall_time_s；原生模式拒绝任何sim_time_s硬上限，包括零。wall_time_s仅在调用边界检查，不能中断阻塞推理/执行；诊断脚本的外部进程监督不等于Runner硬超时、并发或恢复。

`Prediction`绑定请求和来源观测，跨episode或旧观测预测拒绝。模型服务器内部前向次数无法核实则为null，不能用请求次数冒充；模型缓存和原生请求另有诊断记录。

执行后状态不确定时不评分，计数不假装完整；评分失败指标为null/invalid，不伪造success=false。运行完成和任务成功不同。连接/工厂失败可以发生在manifest创建前，此时监督日志记录该门禁失败，不编造完整评测结果。

环境报告executed时必须确认一次动作及所有命令；rejected、planning_failed或部分执行反馈保留已确认计数后停止，不自动重试，避免只有动作预算时零执行循环无限继续。

## 配置、检查与证据

纯合成示例：`examples/mn_synthetic_run.json`。真实配置由`docs/handoff/check_mn_preparation.py`将旧模型配置与旧环境配置分别翻译到新输出目录；运行时不调用旧组合注册或工厂。

~~~sh
.venv/bin/python -m embodied_eval check --config examples/mn_synthetic_run.json
.venv/bin/python -m embodied_eval run --config examples/mn_synthetic_run.json --output outputs/my-mn-run
.venv/bin/python -m embodied_eval check --config PATH_TO_REAL_CONFIG --verify-assets
~~~

PATH_TO_REAL_CONFIG表示新生成的某份0.2真实配置路径。未加`--verify-assets`时只报告声明与本地路径，不启动GPU、加载权重或创建仿真。加该选项分别核验模型/环境资产摘要，仍不做推理。缺依赖或资产校验失败单列，不能称为结构不兼容。

输出同时区分：结构匹配、资产检查、服务权重身份和真实运行。当前服务器没有可信权重身份回执，`server_identity_verified=false`始终保留。本地文件摘要通过、受监督的启动命令和synthetic=false不能将该字段改为true。

所有新产物写新目录；manifest保存双方独立配置、能力、空间、接线、来源和资产核验；每个episode保存公开输入、内部执行trace、指标和诊断。历史Mac绝对路径、旧验收产物和SHA256SUMS保持原样。

DataRef只是可声明的文件载荷引用，读取器仍未实现。需要实际图像/状态时无inline载荷明确拒绝，不用零值代替。并发、自动恢复、画面转动作规划器及未知Benchmark编号解释均不在本轮实现范围。
