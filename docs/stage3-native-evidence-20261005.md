# 阶段 3：OpenWAM-α × LIBERO 单 trial 原生闭环证据

日期：2026-10-05 UTC。用户在阶段 2 Linux 复现与具名阶段 3 候选提交后回复“继续推进”；据此仅执行已提交的 `libero_spatial` task 0、seed 42、trial 0 原生烟测。`trial` 指一次固定任务初始化后的完整尝试。**本次 1/1 成功只证明这一轮真实原生闭环可运行，不是每任务 50 trials 的官方汇总成功率，也不是 EmbodiedEval Adapter 评测结果。**

## 固定输入和资源

- OpenWAM 源码提交 `48bd67b89d489b14d03b8d92bc66e65d306df32e`；LIBERO 源码提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，仅应用 OpenWAM 官方 `libero-pytorch-load.patch`。
- checkpoint：`/inspire/hdd3/project/agent-driven-world-model/zhouzhixiang-240107010008/Model/OpenWAM/OpenWAM-Alpha-Sim-LIBERO/checkpoint_step_10690.safetensors`，24,813,767,464 bytes，SHA-256 `34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a`。配套 `normalization_stats.npy` SHA-256 `f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e`。
- OpenWAM 模型环境：Python 3.10、PyTorch 2.7.1+cu128、Transformers 5.18.0、Diffusers 0.40.0、Deepspeed 0.18.9；LIBERO 环境：Python 3.10、PyTorch 2.7.1+cpu、robosuite 1.4.0、MuJoCo 3.3.2。模型 GPU 0，EGL 离屏渲染 GPU 1。EGL 是仿真器使用的 GPU 离屏渲染接口。
- 原生协议文件 `upstream/OpenWAM/benchmarks/libero/policy_config.yml` SHA-256 `d4211e25b46ec5e164b3662f4ed06743bef2be9ca8243f7c41b14e631f1a89cf`；`action_mode=eef`、seed 42、30 个 settle 步、最多 600 个策略步。`eef` 表示末端执行器动作表示；settle 步是策略开始前让仿真状态稳定的控制步。

执行前检查 `gpu_usage.py --util 70` 的 PID 62026、工作目录及子进程后，依用户许可只对该主进程发送 SIGTERM 正常退出信号；其子进程随后退出，GPU 0/1 空闲。未终止其他进程。SIGTERM 是请求进程自行清理后退出的信号。

## 执行与产物

先用相同参数 `--dry-run` 预检，显示只调度 1 个任务、1 个 trial、模型 GPU 0 / 渲染 GPU 1，未启动服务或客户端；产物在 `outputs/stage3-native-20261005T1554Z-dry-run/`。随后从固定 OpenWAM 仓库运行原生 `benchmarks/libero/run_eval.sh`，checkpoint 路径与文件名显式传入，参数为 `--suites spatial --smoke --inference-horizon 10 --compile-enabled false --render-gpus 1`，环境变量 `GPUS=0`、`REPLICAS_PER_GPU=1`。仅对该命令设置 `HF_HUB_OFFLINE=1`、`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`；没有输出密钥，也没有改全局网络设置。`--smoke` 将范围固定为首个任务的首个 trial。

原生产物目录：`outputs/stage3-native-20261005T1557Z-smoke/`。`manifest.json` 确认同步推理、同步去噪、`inference_horizon=10`、10 次去噪迭代、有效 seed 42、trial 范围 `0:1`，以及模型 GPU 0 / 渲染 GPU 1。原生服务日志确认从本地 checkpoint 加载，启用 `min-max` 归一化、`action_mode=eef`、10 维原始末端动作，随后监听本地 WebSocket；客户端打印通过 `PONG` 能力检查后的 `action_mode=eef model_action_dim=10 runtime_action_dim=7`。WebSocket 是本机客户端与模型服务之间的双向连接协议。服务日志中一次“opening handshake failed”来自调度器的原始 TCP 端口就绪探测，随后真实客户端 WebSocket 连接成功，不是 trial 故障。

加载完成、客户端开始前的一次 `nvidia-smi` 读数为 GPU 0 已用约 21,512 MiB、GPU 1 已用 9 MiB；这是采样时刻而非推理显存峰值。模型在载入阶段的进程驻留内存曾约 48 GB，本机 `free -h` 当时报告约 907 GiB 可用内存。MiB/GiB 分别是按 2 的幂计的内存单位。

启动器退出码 **0**，本轮只一次尝试，无重试。`results.json` 与 `summary.json` 均记录 `libero_spatial` task 0：`success=true`、`policy_steps=73`、`last_reward=1.0`、`1/1`。执行后模型服务 PID 3438859 已退出，端口 8920 无监听，GPU 0/1 显存回到空闲基线。

| 产物 | SHA-256 |
| --- | --- |
| `manifest.json` | `08ca86c042c49f5be0ccda5068fe967ed6a7a1e3dc78741f64c3090772aa278c` |
| `summary.json` | `b71482fc9a21be5de4b7c228689b7d5590db4d1b60dc2141703d167a32935085` |
| `videos/libero_spatial/task_00/trials_000_000/attempt_01/results.json` | `25c1b1c3a1e5cdee827f4b78a863d5a8895fb202826a21f254504958855eb24a` |
| `logs/clients/libero_spatial_task00_trials000-000_attempt01.log` | `5b865e4044305f17205d2fe7a013f3f32294b57339d290ea8941e180908f5ca3` |
| `logs/servers/gpu0_replica0_port8920.log` | `e2ec4ace2dd517beabc03067541ed294b58b0216cf74f71141ad14a69964e1a3` |

这些文件保留原生启动器的路径和协议记录，不能作为框架已接入真实 Adapter 的证明。仍需在框架中复现同一初始状态与公开观测、记录原始 10 维动作及实际发送给控制器的 7 维命令，逐步对照原生数据。原生默认日志只给最终步数/判定与服务推理进度，未提供逐步图像哈希；不能声称像素轨迹已对齐。公共契约与 CLI/产物协议的待确认修改见[阶段 3 审查材料](stage3-review.md)。

原生执行后另将当前 EmbodiedEval 以可编辑方式装入独立 LIBERO Python 3.10 环境。在项目根目录执行 `../upstream/envs/libero/bin/python -m unittest discover -s tests -q`，实际 **85 项测试全部通过**。这只验证现有 synthetic 框架在该环境中的兼容性，不表示真实 Adapter 已通过。首次从工作区父目录运行发现 `examples` 样例包不在导入路径，33 项已发现测试中出现一个导入错误；改为项目根目录运行后 85/85 通过，未修改代码或测试来规避错误。

随后在该 Python 3.10 环境执行 `docs/handoff/replay_baseline.py --output outputs/stage3-runtime-baseline-20261005T1619Z`，收据 `baseline-check.json` 记录 `passed=true`、85 项测试及 8 个契约样例通过；五组 synthetic CLI 的实际退出码仍为 `0/0/0/1/2`。这是第三套独立 Linux 环境复现产物，不和上述真实原生 trial 的结果混合。
