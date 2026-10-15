# 阶段 1 验收说明

历史验收记录：以下描述 2026-10-04 的阶段 1 交付和当时的范围。后续用户“继续推进”授权实施阶段 2；当前进展见 stage2-acceptance.md，原 53 项契约检查仍全部通过。validation-evidence.json 的摘要对应阶段 1 时的源码，不表示后续重构后的当前源码。

## 交付与证据

- 模型侧与 Benchmark 侧 Protocol 接口；初始化和模型结束信息明确分开。
- SpaceSpec、CapabilitySpec、AdaptationProfile、TaskSpec、Observation、Prediction、StepResult、EpisodeResult 及其组成类型。
- 严格 JSON 编解码与语义校验；单位、坐标、目标实体、时间、chunk 归属、工具 schema 和指标可用性检查。
- 静态兼容性报告及动态必需输入检查，返回 code / path / message。
- 公开空间与 ModelInput 白名单副本，排除任务初始化、评分、隐藏坐标及未授权状态。
- 显式物理单位、坐标树、SE(3)、quaternion / rot6d 转换。
- 8 组实际线格式决策示例：单步、chunk、双臂、控制器原生命令、离散动作、工具、停止与世界模型预测。
- 53 项 unittest 检查通过；完整输出和环境记录见 test-output.txt / validation-evidence.json。

这些检查包含错误路径及非零旋转几何计算。测试不会启动模型、下载权重、调用模型 API 或安装仿真器；工具的三步结果是合成观测序列，尚未运行执行循环。

## 接口落实时明确的细节

模型初始化接收经过过滤的 SpaceSpec；Benchmark 执行方法显式关联请求 Context；多步工具可返回中间观测序列；模型结束原因使用闭合枚举；夹爪关节模式明确单位来源。详见 contract-v0.1.md。

原讨论稿保留为历史材料；代码与 examples/contract_examples.json 是本次待验收版本。通过后才可将 v0.1 标为冻结。

## 未完成范围

CLI、Runner、动作缓存消费、episode 资源释放和调用顺序尚未实现，因此本阶段没有运行“完整 episode”。单机并发、服务传输、超时/重试/恢复、DataRef 内容解析及真实指标对照也未实现。

尚未确认首个真实模型/权重、Benchmark 版本/子任务、计算资源与原生对照标准；阶段 0 的实际试点核实仍为开放项。它们不影响契约和 mock 设计，但必须在阶段 3 真实接入前确认。

世界模型预测仅验证封装和角色隔离，尚无规划或官方预测评分。任意扩展的业务含义需要人工协议核对，不能靠 namespace 或 schema 判断是否泄漏答案。

## 下一阶段确认

stage2-review.md 已给出具体模块、JSON CLI 配置、生命周期、统计预期和异常验收场景。依用户阶段约定，等待阶段 1 验收与阶段 2 细节确认后，再实现执行器。
