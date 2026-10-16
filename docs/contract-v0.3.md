# 0.3 公共数据契约

用户于2026-10-16明确批准“直接实施0.3，全部迁移”。这是当前运行协议，不是正式冻结或阶段4验收结论。正常 `run`、`check`、Python顶层配置／契约／接口／校验入口只接受0.3；不再隐式识别旧版。历史0.1、0.2契约和证据保持原样。

## 核心规则

环境原生观测 → 环境Adapter → **统一命名的公共观测** → 模型Adapter → 私有输入编码／推理 → **公共动作** → 环境Adapter → 原生控制与官方评分。

Adapter指负责某一侧格式转换的适配器。Runner指只负责调用顺序、预算、计数与保存证据的通用执行循环。M+N表示M个模型与N个环境分别维护自己的适配器，不维护模型与环境的逐对名单。

模型只接收公开数据，不接收环境对象或环境配置；环境不知道当前模型。相机重命名和状态提取由各自Adapter负责；普通运行配置没有 `binding`，即不再要求用户逐对填写字段对应关系。空间声明里的 `native_source` 仅用于追溯原生来源，不是公开数据查找路径。

## 类型和入口

公共入口：`embodied_eval.contracts`、`embodied_eval.interfaces`、`embodied_eval.config`、`embodied_eval.validation`、`embodied_eval.runner`。公共实现与注册：`embodied_eval/v03/`；版本无关的物理类型、结构校验、坐标路径与注册容器：`embodied_eval/core/`；原生模型编码、环境执行封装及资产核验：`embodied_eval/native/`。`native/contracts.py` 与 `native/environment_records.py` 中的冻结私有消息仅衔接原生组件和历史解码，不是旧版运行入口。`mn/` 中迁出组件的旧路径保留导入别名；历史Runner、配置和协议校验不参与当前通路。此处仅更新内部源码归属，公共字段、默认值和协议语义不变。

消息严格校验未知字段、版本、重复JSON键、非有限数、跨轮次和不完整目标。JSON是配置与证据使用的文本数据格式。版本字符串必须为 `"0.3"`。

| 记录 | 关键字段与含义 |
| --- | --- |
| `RunConfig` | `contract_version/run_id/model/benchmark/episodes/control_mode`；分别选择模型、环境和任务，不含逐对接线 |
| `SpaceSpec` | 空间身份、机器人、坐标树、传感器、状态、控制器、动作格式、参考部位、视角别名和时间模式 |
| `CapabilitySpec` | 固定源码、权重、归一化统计、具名 `io_profile_id`、输入要求、输出目标和支持时间模式 |
| `ResolvedProfile` | 自动匹配回执：实际公共输入路径、完整动作目标、选中控制方式、坐标转换路径和警告；不是用户手填配置 |
| `Observation` | 上下文、观测编号、动作数、仿真秒数、公共 `sensors/states`、动态公开指令、采样坐标变换 |
| `ModelInput` | 按模型声明提取的公共样本；保留公共键名，并附状态／传感器声明、视角别名、采样变换 |
| `Prediction` | 来源观测编号、当前请求上下文、公共动作、可核实的前向次数与转换说明 |
| `StepResult` | 新观测、执行反馈、终止／预算结束标记；反馈保留动作、物理步和实际仿真时间 |

`io_profile_id` 表示一份权重的具名输入输出规则，不是环境名称。`control_mode` 是可选的具名环境执行格式；有多个同样匹配的执行格式时必须明确选择，不能自动取第一项。

## 相机公共字段

RGB是红绿蓝三个图像通道，原图保留实际分辨率、数值类型及采集时间；模型需要的缩放和排列由模型Adapter完成。公共键和 `SensorSpec.role` 必须一致。

| 公共键 | 明确含义 |
| --- | --- |
| `head` | 工作区总览的逻辑角色，不保证相机物理安装在机器人头部 |
| `wrist` | 单臂腕部视角 |
| `left_wrist/right_wrist` | 双臂各自腕部视角，不能用单腕冒充另一只手 |
| `external_front/external_left/external_right` | 安装方向经核实的外部相机，不按模型私有变量名猜左右 |
| `external_0/external_1` | 外部相机稳定来源身份；暂未核实物理左右时保留编号，不伪造方位 |

`view_aliases.main` 是环境明确声明的默认主视角，可供需要 `views.main` 的模型选择；实际输入数据的键仍为 `head` 等公共字段。它不是额外一份重复图像；没有声明就拒绝必要主视角请求。模型需要 `sensors.head` 时，不能擅自用腕部图像代替。

声明还保留 `modality/mounted_on/frame_id/tensor/unit/color_order/available_rate_hz/native_source`，依次说明模态、安装位置、坐标、形状与类型、单位、颜色顺序、采集频率及固定来源。深度、点云等保留相应类型和角色后缀的表达能力，但现有五类环境没有因此新增真实深度、点云、音频或触觉读取器。

## 状态公共字段

单臂使用下列基础键；双臂按 `left_`、`right_` 前缀分别声明；`robot_base_*` 是整台机器人基座，不加手臂前缀。不把两只手或夹爪混装成无说明的14维、55维或80维公共数组。

| 基础键 | 单位与来源 |
| --- | --- |
| `joint_positions` | 实测关节位置；转动关节为rad，即弧度；移动关节为m，即米 |
| `joint_velocities/joint_efforts` | 实测速度／力矩或力；按每个物理关节声明单位 |
| `joint_drive_targets` | 控制器设置的关节目标，不是实测位置 |
| `eef_position/eef_quaternion` | 已核实部位的末端实测位置／朝向；分别标明部位，不能自动拼成同一工具位姿 |
| `tcp_position/tcp_quaternion` | 工具中心点的实测位置／朝向；工具中心点指实际控制或接触使用的参考部位 |
| `gripper_joint_positions` | 夹爪各关节实测位置，必须列出真实关节顺序和单位 |
| `gripper_width/gripper_open_fraction` | 实测总开口宽度（米）／实测开度（0至1） |
| `gripper_open_fraction_target/gripper_width_target` | 夹爪控制目标，不是实际测量值 |
| `gripper_joint_drive_targets/gripper_force_target` | 各夹爪关节控制目标／夹持力目标（N，即牛顿） |
| `robot_base_position/robot_base_quaternion` | 基座实测位置／朝向，不等于默认世界原点 |

`StateSpec` 的 `semantic/unit/frame_id/tensor/robot_id/provenance/site_id/joint_ids/component_units/origin/native_source` 分别说明物理含义、单位、坐标、形状、机器人、测量或目标来源、参考部位、关节顺序、逐分量单位和来源依据。不同种类关节混合时用 `per_component` 并逐项列出单位，不虚称整段都是弧度。

`StateSample` 保留 `captured_at_s/payload/provenance/validity/invalid_reason`：采集仿真时间、数据、来源、有效性和无效原因。无效状态必须没有伪造数据并说明原因；必要输入无效、缺失或过期就拒绝。

RoboTwin原生14维驱动目标拆成左6关节、左夹爪、右6关节、右夹爪；LingBot需要的原生14维排列仅在模型私有编码中重建，要求四段采集时间和精度一致。环境已经分别提供夹爪目标时，两份原生值必须相符，不能挑一份凑数。

## 坐标和参考部位

位置用米，时间用秒；四元数使用 `xyzw`，即向量分量x、y、z在前，实数分量w在后。四元数整体取反仍表示同一朝向。表示排列变化、坐标变换和工具参考点变化是三件不同的事。

`world` 是各具名环境协议核实的世界坐标，不保证不同环境拥有同一物理原点。未知机器人基座外参时，`robot0/base` 保留为不连通坐标，不填假单位变换。没有完整变换路径就拒绝跨坐标请求；动态变换必须有真实采样值和时间。单位四元数沿用平方长度距1的绝对容差0.000001，不为迁移放宽；合成旋转往返测试位置误差要求小于0.000000000001米，允许四元数整体反号。

`reference_sites` 是参考部位目录；每项记录机器人、测量位置／测量朝向／控制目标／工具中心点类别及固定来源。状态 `site_id` 和末端 `control_site_id` 指向该目录。坐标旋转和平移不能凭空补出两个不同工具点之间的偏置。

LIBERO现有位置来自grip_site、朝向来自right_hand，它们保持两个部位，不谎称天然就是一个经过标定的工具位姿。原生OSC指归一化末端控制器扩展，不因此宣称LIBERO已经实现公共绝对位姿执行器。

## 公共动作与执行

动作结构包含类型化命令、是否采集下一张图、具名原生协议、原生精度以及有依据的无损欧拉角提示。目标必须使用空间声明中的公共ID，例如 `robot0/left_eef` 表示机器人0的左侧末端。双臂、关节和夹爪目标必须完整。

命令保留关节目标、绝对位姿、相对位姿、速度、夹爪和具名控制器类型。结构能够表达某种动作，不等于某个环境已经实现该执行器；仅当空间动作格式实际声明支持时才能运行。当前通用几何边界工具实现有明确坐标路径的绝对位姿转换，不新增相对动作积分器、逆运动学或速度控制器。

原生数组只有在双方协议名和结构签名一致时才能执行；它不会因为包在一个对象里就变成公共位姿。旧OpenWAM双臂含糊编码保留具名原生扩展；另一个 `eef20_dual_wxyz_encoded_absolute` 配置通过已核验的特殊训练编码进入公共物理位姿，二者不能互相冒充。

固定周期分别核对动作数与时间；`measured_native` 表示变时长原生控制，分别记录动作数、真实物理步数、实际仿真秒数。仍不支持硬仿真时间上限、并发评测、重试或恢复。

## 配置与显式迁移

```sh
.venv/bin/python -m embodied_eval check --config examples/mn_synthetic_run.json
.venv/bin/python -m embodied_eval run --config examples/mn_synthetic_run.json --output outputs/my-v03-run
.venv/bin/python -m embodied_eval migrate --config /path/to/old.json --output outputs/migrated-v03.json
```

预检查不创建模型／仿真工厂，不下载权重。`--verify-assets` 仅核验本地固定资产。结构可组合、运行资产就绪、真实运行、严格原生一致和正式验收是不同状态。

离线迁移支持已有原生0.1配置及可核实的0.2对应配置，保留权重、归一化、环境协议、种子和预算；自定义接线、坐标转换、未知字段或已退役工具行为无法证明等价时拒绝，不静默丢掉。输出不覆盖已有文件。

五个旧mock运行示例退出当前examples目录，原文件留在 `tests/fixtures/legacy-configs` 和 `pre-v03-20261016` Git标签。当前五组CLI用例为单步、缓存、变时长、评分错误和结构拒绝；预期仍为0/0/0/1/2，但不冒充旧工具／Runner持有动作段的同一测试语义。0.1合同样本只是历史测试夹具，不是当前可运行配置。

历史诊断脚本若依赖旧配置或内部格式，应在 `pre-v03-20261016` 检出到独立工作目录后运行，不将历史证据就地转成新版本。当前脚本为 `replay_v03_baseline.py`、`check_v03_saved_boundaries.py`、`run_v03_bounded.py`；旧脚本中纯来源、视频、硬件和原生函数仍可按其明确输入范围复用。

## 不在本轮完成范围

DreamZero真实动作仍待后续双H10080GB资源；LingBot-VA暂缓。没有新增全Benchmark任务初始化／评分协议、全任务多种子统计、DataRef载荷读取、硬超时、全身机器人或MassWorld整合。DataRef指对外部数据的引用，当前仍不把其声明能力当作已实现文件读取。契约未冻结，阶段4未自动接受。
