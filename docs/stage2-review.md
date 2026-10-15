# 阶段 2：实施前确认材料

状态：用户“继续推进”授权按本材料默认方案实施。CLI、Runner 和测试用 Adapter 已实现；验收证据见 stage2-acceptance.md。以下保留实施前的范围和验收约定。本阶段只验证执行语义，不依赖真实模型、GPU 或仿真器。

## 1. 范围与环境

继续使用本项目目录。Python、CLI、一个进程串行运行一个环境和一个模型实例；同一 worker 同时只持有一个 episode。单机多进程、硬超时、远端服务和恢复属于阶段 6 或之后的单独确认，不在本阶段悄悄加入。

新增模块建议：

| 模块 | 用途 |
| --- | --- |
| embodied_eval/runner.py | 预检查、episode 生命周期、决策执行、预算和结果 |
| embodied_eval/config.py | 严格运行配置、内置 Adapter 注册、配置校验 |
| embodied_eval/__main__.py | CLI 入口和退出码 |
| embodied_eval/adapters/mock_model.py | 单步、chunk、内部缓存、Agent 和停止等测试用策略 |
| embodied_eval/adapters/mock_benchmark.py | 纯 Python 可控状态、逐步执行工具和合成评分 |
| embodied_eval/artifacts.py | manifest、内部轨迹、episode 结果和汇总 |
| tests/test_runner.py | 生命周期、预算、异常和隐私验收 |

不增加模型 SDK 或仿真依赖。现有契约如有缺口，先提交具体缺口与受影响对象确认，再调整。

## 2. 运行入口和配置

建议命令：

~~~sh
uv run --frozen python -m embodied_eval run \
  --config examples/mock_chunk_run.json \
  --output outputs/mock-chunk
~~~

CLI 配置使用 JSON；只接受内置注册的 mock Adapter，不从任意路径动态导入代码。建议默认配置：

~~~json
{
  "run_id": "mock-chunk-run",
  "benchmark": {"adapter": "mock_benchmark", "control_period_s": 0.05},
  "model": {
    "adapter": "mock_model",
    "scenario": "chunk",
    "chunk_length": 8,
    "observation_update": "every_env_step"
  },
  "profile": {
    "chunk_mode": "runner_owned",
    "execution_horizon_steps": 2
  },
  "episodes": [
    {
      "episode_id": "ep-0",
      "seed": 0,
      "budgets": {
        "env_steps": 6,
        "sim_time_s": 0.3,
        "model_calls": 6,
        "tool_calls": 0
      }
    }
  ]
}
~~~

配置不是完整 profile 的替代品；mock 工厂根据场景生成 SpaceSpec / CapabilitySpec / AdaptationProfile / TaskSpec，仍逐项走阶段 1 检查。默认任务不因合成动作达成成功；成功场景必须单独指定明确的 mock 成功条件。所有结果注明 synthetic=true。

默认不用墙钟预算触发不稳定验收；可选 wall_time_s 在同步调用返回后检查，不声称能打断卡死调用。进程硬超时方案留到资源与并发阶段确认。

输出目录已存在时默认拒绝覆盖；本阶段不做断点恢复。--output 必须传入或使用生成的唯一目录。配置错误与声明不兼容在环境 reset 前发现。

## 3. 生命周期和执行循环

1. 验证配置、规格、任务和 Profile；记录快照及版本。
2. Benchmark reset；校验观测。模型 begin_episode 仅接收公开任务、公开空间和 Profile。
3. 构造、校验 ModelInput；初始观测发送一次。
4. 预算允许时 predict；将其 Context 关联到唯一待完成 request_id。
5. 校验输出。runner_owned 执行前 horizon 个动作，每个动作前检查剩余预算；未执行尾部直接丢弃。adapter_owned 每次只取一个动作。
6. Benchmark 返回 StepResult；对照实际环境步、时间、请求、命令和观测历史。
7. every_env_step 按执行顺序观察每一步，包括最终观测；decision_boundary 仅在下一次 predict 前观察最新状态。一次观测 ID 不重复 observe。
8. 工具传入剩余预算，由 Benchmark 每步检查；长工具返回所有中间观测。工具执行后才能发起新决策。
9. 结束后执行 Benchmark score，形成 EpisodeResult；模型收到不带指标的 ModelEndContext。
10. 无论正常或异常，都完成模型结束钩子与最终资源释放。单个 episode 报错后，必须重建不确定的 Adapter 状态才可继续后续 episode。

模型主动停止或环境结束后不继续消费 chunk。预算为 0 时不再 predict；wall time 无法强制终止同步调用的限制写入 manifest。

模拟步、模拟时间、模型调用和工具调用分别计数。只调用一次 predict 不等于只执行一次模型前向；mock 能提供真实已计数前向次数，其他模型未知时保留 null。

## 4. 产物、错误和退出码

~~~text
outputs/<run>/
  manifest.json
  summary.json
  episodes/<episode>/
    task.internal.json
    trace.internal.jsonl
    model_input.public.jsonl
    result.json
~~~

manifest 保存配置、契约和 profile 版本、源码内容摘要、Python 与依赖版本、seed 及 synthetic 标记。内部轨迹可以保存官方评分状态，仅写本地；公开模型输入日志只记录已过滤视图。结果引用相应产物，之后的真实接入可增加视频。

退出码建议：0=全部计划 episode 完成且没有运行/评分异常（任务成功与任务失败均可能）；2=配置/兼容性拒绝；1=存在运行或评分异常。run_status 与 success 分开。正常主动停止由 mock scorer 判定，不自动记成功。

不自动重试执行动作或工具。部分执行错误记录实际消耗；score 失败时指标 null 并记录来源与不可用原因，不能伪装成任务失败。

## 5. 验收场景与明确预期

| 场景 | 输入约定 | 通过标准 |
| --- | --- | --- |
| 单步动作 | 预算 3 ticks，every_env_step | predict=3、observe=4，env_steps=3，sim_time=0.15 |
| Runner chunk | chunk=8、horizon=2、预算 6 ticks | predict=3、observe=7、env_steps=6；每个 chunk 只执行前 2 步 |
| Adapter 缓存 | 内部 chunk=8、horizon=1，两个 episode 各预算 3 ticks | 每个 episode predict=3、forward=1；第二个 episode 不消费第一条缓存 |
| 双臂与多相机 | 两个 EEF 命令同 tick，3 个 RGB 相机 | 无串臂/串相机；同 tick 两条命令计 1 个环境步 |
| 增量动作 | 非零旋转参考位姿、pre / post multiplication | 最终位置符合矩阵计算；chunk_start / step_observed / previous_command 使用约定参考 |
| 离散动作 | 词表 forward、duration=0.05 | 返回 1 tick；未知 token 在执行前拒绝 |
| 长工具与预算 | 每次请求 3 ticks，总预算 5 | 两次工具调用分别执行 3、2 ticks；第二次 partial + budget_exhausted，总计 5 / 0.25 |
| 长工具历史 | every_env_step、完整中间观测 | 逐步 observe；缺一帧提前报执行契约错误，不继续推理 |
| 主动停止 | 初始决策 StopRequest | 无环境执行；run completed、reason=model_stop、mock success=false |
| 隐藏评分 | 初始化、轨迹含人工 secret 字段 | ModelInput、初始化和结束钩子无 secret；修改公开副本不影响环境 |
| 过期/跨 episode 响应 | 返回旧 observation/request/episode ID | 作为输出错误拒绝，不执行任何命令 |
| 执行/评分异常 | 显式注入错误 | 分类准确、实际消耗保留、资源释放，无法评分用 null |
| 两个 episode 状态隔离 | 相同 seed、不同 episode ID | 模型历史、KV/模拟缓存与动作缓存重置；统计不串用 |

测试验收报告会给出每条场景的断言、实际统计与产物路径。实际墙钟推理速度、并发吞吐和真实成功率不在本阶段验收。

## 6. 待确认决定

建议采用本文件的默认值：JSON CLI、串行单环境、Runner 持有 chunk、显式预算、合成工具逐步历史、只用 mock Adapter、不覆盖已有输出、不自动重试。

收到确认后仅实施阶段 2。首个真实模型、权重、Benchmark 子协议、机器和 GPU 仍须另行核实；不会凭本阶段的合成结果认定真实组合兼容。
