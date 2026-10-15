# 阶段 3：OpenWAM-α × LIBERO 单 trial 框架闭环证据

状态：已完成已授权的最小单 trial 实施与对照，**待用户确认阶段 3 验收**。不代表完整 50-trial 官方成功率，也不授权下一阶段。本文中的 trial 是一次固定任务初始化后的完整尝试；策略步是一次 `env.step`；`synthetic=false` 指真实 Adapter 配对执行，并不单独证明服务端权重身份。

## 固定对象与运行边界

- 模型源码 OpenWAM 提交 `48bd67b89d489b14d03b8d92bc66e65d306df32e`；Benchmark 源码 LIBERO 提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。模型资产 revision `110b07e4d5581c81520a77419cbb0e9371df231b`。运行前程序对两份源码 `HEAD`、checkpoint SHA-256 `34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a`、normalization SHA-256 `f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e`、BDDL SHA-256 `9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`、初始化 `.pruned_init` SHA-256 `cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a` 逐项校验；不匹配则拒绝运行。
- [框架试点配置](../examples/stage3-openwam-libero-spatial-pilot.json)：`libero_spatial` task 0 / trial 0 / seed 42，第一个 init state、30 步 settle、策略最多 600 步 / 30 逻辑仿真秒 / 600 次模型请求；服务同步去噪 10 步、推理 horizon 10，模型端持有 chunk 缓存，Runner 每次只执行一个控制步。两个 256×256 RGB 相机和具名 `libero.native_proprio.v1` 扩展进入公开视图，评分隐藏字段不进入模型。末端 `grip_site` 位置与 `right_hand` 四元数保持分立，不拼成虚假的标准 Pose。
- 框架进程使用独立 LIBERO Python 3.10 环境、GPU 1 EGL；模型服务使用独立 OpenWAM Python 3.10 / PyTorch 2.7.1+cu128 环境、GPU 0。服务启动命令与既有[原生证据](stage3-native-evidence-20261005.md)相同：本地 checkpoint、`--denoise-steps 10 --denoise-mode sync --inference-mode sync --inference-horizon 10 --compile-enabled false`，端点 `127.0.0.1:8920`。服务日志确认加载本地 checkpoint、`min-max` 归一化、`eef` / 10 维状态；客户端 `PONG` 确认 `eef` 表示和 10→7 维映射。WebSocket 握手不返回 checkpoint 摘要，因此 manifest 如实标记 `server_identity_verified=false`；独立进程命令与加载日志是补充证据。运行结束后服务停止，端口 8920 已释放。

## 结果与逐步对照

| 证据 | 实测 |
| --- | --- |
| [框架正式产物](../outputs/stage3-framework-20261005T1729Z-final) | CLI 退出码 0；`summary.json` 为 `synthetic=false`、`completed=1`、`errors=0`。官方 `done` 判定 `success=true`；策略步 73、模型请求 73、逻辑仿真时间 3.65 s。服务没有可信的 forward 计数，记录为 `null`。`manifest.json` 与当前 `embodied_eval/*.py` 源码摘要全部一致。 |
| [原生逐步调试产物](../outputs/stage3-native-debug-20261005T1724Z) | 固定上游 `single_eval.py`、[调试配置](../examples/stage3-native-debug-pilot.yml)，同一模型服务顺序运行；trial 0 成功，73 步，末步 reward 1.0。只跑一次，不报告完整官方统计成功率。 |
| [逐步比较 JSON](../outputs/stage3-compare-20261005T1734Z.json) | 两侧各 73 份调试记录。状态 10 维、原生模型动作 10 维、LIBERO 控制命令 7 维在 73/73 步逐项精确相等；三类数据最大绝对差均为 0，首次数值分歧不存在。原生调试元数据未保存图像字节，故无法直接比较两侧图像像素哈希。 |
| [最终回归复现](../outputs/stage3-final-regression-20261005T1736Z/baseline-check.json) | 原有 85 项 + 新增 8 项测试共 93/93 通过；8 个契约示例通过；五组 mock CLI 退出码仍为 `0/0/0/1/2`。`replay_baseline.py --additional-tests 8` 保持默认的 85 项交接门槛不变，并显式计入新增测试。 |

关键摘要：正式 `manifest.json` 为 `562586f132661f8fac12138d45df6b017a139d8486bc94ca60ae43008be0a191`；`summary.json` 为 `12c2809339fe1204c784617f59ba65236cb83445a615de3645008cfd4056f668`；episode `result.json` 为 `c121fe22da7dc7372ea59ff411bee721953db8615f837cd7e24c8ecf55297093`；原生调试 `results.json` 为 `25c1b1c3a1e5cdee827f4b78a863d5a8895fb202826a21f254504958855eb24a`；逐步比较 JSON 为 `3d7fab32942c6d650bc47d3e1301f6ff296daaf1995afe8f94e427f3dbf2f3e3`；最终回归 receipt 为 `756cccce2f21933f78a88180c77e4f027d11202b189c3c1d5bc5076de990e10c`。

第一次框架诊断目录 `outputs/stage3-framework-20261005T1718Z-trial0/` 在 33 步后因实施中补强来源核验而主动中断，没有 `summary.json`，**不计入通过**。曾按未扩展的交接脚本运行 89 项时因其固定期待 85 项而退出 1；不是测试失败。随后给脚本增加显式 `--additional-tests` 参数、默认值仍为 0，最终 93 项和五组 CLI 均通过。

## GPU 利用率、资源及剩余界限

[GPU 连续采样原始记录](../outputs/stage3-gpu-util-20261005T1730Z.json)覆盖 `2026-10-05T17:27:23Z` 至 `17:33:47Z`，每 2 秒一次、共 180 次。统计口径是每次对 8 张卡的 `nvidia-smi` 利用率做算术平均，再对 180 次取时间平均；实测 **22.3458% ≥ 15%**。GPU 2–7 用已授权的轻量占用程序保持约 30% 目标，模型占 GPU 0、EGL 渲染占 GPU 1；采样窗内 GPU 0/1 的各卡时间平均分别约 0.74% / 0.12%。因此**仅满足“8 卡总体平均”口径，不满足“每张卡平均都 ≥15%”口径**；用户尚未确认口径。GPU 2–7 的占用程序在结束本次核验后仍运行，以维持总体平均目标；没有修改全局代理。采样 JSON SHA-256 为 `451f49e87765f4cfd4c08f66fec9c8892ea7cdb36c60942138da6cd13266e585`。

机器 `nvidia-smi` 枚举 8 张 RTX 4090，每卡 49,140 MiB；MiB 是二进制兆字节。模型服务运行时 GPU 0 约占 24,650 MiB，GPU 1 EGL 约占 792 MiB。共享文件系统报告 13 PiB 总量、约 11 PiB 空闲；PiB 是二进制拍字节，但共享空闲量不证明用户配额。正式 inline 图像产物约 246 MB；DataRef 真实读取、压缩、硬超时、并发及恢复均未建设。`model_forward_count=null` 不应推成 73 次前向；73 是 WebSocket 模型请求数，服务内部可能复用 chunk。框架仍独立于 XPolicyLab，未修改历史 Mac 绝对路径或旧证据。

待用户确认：本单 trial 是否作为阶段 3 验收；GPU 15% 指 8 卡总体时间平均，还是每张卡各自平均。确认前不进入下一阶段或扩大试验范围。
