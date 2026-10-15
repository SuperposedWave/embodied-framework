# 阶段 2 验收说明

完成日期：2026-10-05。用户“继续推进”授权按已提交的阶段 2 默认方案实施。CLI、串行 Runner 和合成 Adapter 已实现；当前等待本阶段验收与下一阶段配置确认。

## 1. 已实现的行为

- CLI 使用严格 JSON 配置，只启用内置 mock 注册项，拒绝未知字段及不兼容组合。
- 完成 reset → begin_episode → observe / predict → execute → score → end_episode / close 的闭环。
- Runner chunk 只消费本次 horizon，丢弃余下预测；Adapter 内部缓存跨 episode 清空。
- 长工具每步检查剩余预算，返回实际步数及中间观测，按顺序更新模型历史。
- 正常连续 episode 复用实例并重置状态；发生异常后关闭两侧，再为下一条 episode 建立新实例。
- 私有轨迹与模型公开输入分别记录。初始化、评分及结束钩子的私有信息不进入模型。
- 主动停止、任务成功、预算耗尽和运行异常分开。无法评分用 null，未知前向次数用 null。
- 输出目录存在时拒绝覆盖。单进程串行执行，无自动重试及中途恢复。

通用循环只消费标准消息；位姿参考、归一化控制器映射、离散导航及工具执行的合成逻辑均位于 Benchmark Adapter。合成规格从阶段 1 示例抽取到 adapters/mock_spec.py 复用，公共数据契约结构未修改。

## 2. 验证结果

85 项 unittest 全部通过：原有 53 项契约测试，加 32 项执行循环/CLI 测试。测试文件含多组参数化场景，覆盖正常路径及异常注入。完整输出见 stage2-test-output.txt；代码摘要、CLI 退出码及实际统计见 stage2-evidence.json。

| 保存的 CLI 用例 | 结果 | 关键统计 |
| --- | --- | --- |
| Chunk 8 / horizon 2 | completed，budget_exhausted | 6 ticks、3 次 predict、7 次 observe、模拟时间 0.30 s |
| 工具请求每次 3 ticks | completed，budget_exhausted | 实际两次执行 3 / 2 ticks，predict=2、tool_calls=2、总计 0.25 s |
| 内部缓存，两个 episode | 两条均 completed | 各 3 ticks、3 次 predict、1 次 mock forward；第二条从新缓存开始 |
| 合成 scorer 故障 | error | success=null、valid=false、scorer_failed，CLI 退出码 1 |
| chunk 归属不兼容 | reset 前拒绝 | CLI 退出码 2，未创建结果目录 |

默认合成 scorer 不将动作完成等同于任务成功，所以正常运行用例的 success=false。这是明确的 mock 规则，没有实际 Benchmark 成功率含义。

还验证了双臂同 tick 计一步、多相机、非零旋转增量、三种增量参考、模型调用预算、零预算、过期响应、历史缺帧、重复观测身份、迟到预测、超预算执行、初始化/结束/评分/释放异常和执行状态未知。资源释放时出现第二个异常，不会抹掉执行状态未知的标记。

## 3. 查看和重跑

验收产物根目录：

/Users/boboyuyong/Library/CloudStorage/OneDrive-Personal/WorkSpace/1TempFiles/EmbodiedEval/outputs/stage2-20261005T063033Z-final

~~~sh
.venv/bin/python -m embodied_eval run --config examples/mock_chunk_run.json --output outputs/my-chunk-run
.venv/bin/python -m embodied_eval run --config examples/mock_tool_run.json --output outputs/my-tool-run
.venv/bin/python -m embodied_eval run --config examples/mock_cache_run.json --output outputs/my-cache-run
.venv/bin/python -m unittest discover -s tests -v
~~~

每次选新的 output 目录。项目已有 .venv 可直接运行；新环境可用 uv sync --frozen 安装锁定依赖。当前验证为 Python 3.13.5、jsonschema 4.26.0，无新增运行依赖。

每条 episode 的 task.internal.json、trace.internal.jsonl、model_input.public.jsonl、diagnostics.internal.json、result.json 保存在 episodes 目录。manifest 包含源码摘要、依赖版本、完整配置、seed 和 synthetic=true；summary 区分完成和异常，并标记统计是否完整。

## 4. 限制及下一阶段需核实的内容

本阶段只运行纯 Python 合成环境与策略，没有真实权重、模型 API、GPU 或仿真器调用。Mock 动态学仅用于验收数据和控制语义，不能用于机器人效果比较。seed 被传入、记录；当前 mock 的固定状态转换不依赖随机数。

墙钟预算采用同步调用前后的协作检查，不能打断卡住的推理或执行。多进程隔离、硬超时和恢复尚未实现。DataRef 的文件/共享内存/远端载荷解析尚未实现；测试图像为小型 inline RGB 数组。

执行响应丢失时，现有 EpisodeResult 的整数统计只能记录最后确认的进度：termination_reason=execution_state_unknown，summary 中 execution_statistics_complete=false / scope=last_confirmed；此时指标不可用。不能把这些数值解读为完整实际消耗。若真实协议需要独立表示未知/部分控制 tick 及精确物理时间，必须先讨论公共契约扩展。

CLI 当前配置仍专用于 mock。真实接入时需明确注册名、专属配置归属及映射 Profile；如需改变公共消息或已确认接口，先提交影响说明。

下一步先确认阶段 3 的真实模型/权重、Benchmark 版本与子协议、机器人、资源和原生对照配置，见 stage3-review.md。它们仍是阶段 0 的开放项，本阶段测试不构成真实兼容性证据。
