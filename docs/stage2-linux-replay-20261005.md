# 阶段 2 Linux/GPU 实例接手复现记录

日期：2026-10-05 UTC。范围：交接包完整性、资源/资产盘点、轻量框架环境和既有 synthetic 基线复现。未调用真实模型、仿真器或评分器；本记录不改变阶段 2/3 的确认状态。

## 交接完整性

- 原始包：`../../EmbodiedEval-GPU-handoff-20261005.tar.gz`；本机计算的 SHA-256 为 `36196e8702dc34af9ff63d03d581ec8676e53163617e397e354db03b77ec80d9`。本次传入目录未附带包外 `.sha256` 文件，因此无法与发送方的包级摘要独立比较。
- 解包前检查了归档成员，无绝对路径、`..` 路径段或特殊文件；解包在当前工作目录的 `EmbodiedEval/` 下。解包后、修改本项目文档前运行 `sha256sum -c SHA256SUMS`：86/86 个包内文件通过，包括 `outputs/stage2-20261005T063033Z-final` 的历史产物。原有 `docs/stage3-review.md` 摘要为 `089885d20b010d49ac061a33133c426c31d8f3ccef2143ce63b9748737c5386b`；本次按要求补充该文件后，原始交接清单自然不再是当前工作区的活清单，保留原包可复查初始状态。
- 交接包没有 `.git`；解包后 `git status` 确认不是 Git 仓库。没有提交或 remote 可沿用。历史证据中的 macOS 绝对路径未修改。

## 资源与资产（09:46 UTC 左右读取）

| 项目 | 本机观察 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.4 LTS，Linux 5.15.0-141-generic，x86_64 |
| Python | `python3` 为 3.13.5；另有 `/usr/bin/python3.10` 为 3.10.12；`conda` 可用，`uv` 未在 PATH |
| GPU | `nvidia-smi` 报告 8 × NVIDIA GeForce RTX 4090，各 49,140 MiB 总显存，查询时空闲约 47,833–47,841 MiB；驱动 570.195.03。查询时 GPU 利用率约 62%–81%；是否可独占、能否装载 OpenWAM 和 EGL 渲染尚未实测。 |
| 磁盘 | 项目所在 `gpfs_hdd03` 显示总 13 PB、空闲 11 PB；根 overlay 显示空闲约 1.2 TB。共享文件系统的个人配额未核实。 |
| 已有资产 | 在本用户的 `Project/` 与 `Model/` 路径按 OpenWAM、LIBERO、`normalization_stats.npy` 名称检索，未发现对应目录/文件；`Model/` 中有其他项目权重，不代表本试点可用。交接包自身无权重、仿真器或场景资产。 |

## 轻量环境与实际复现

先用本机 Python 3.13.5 新建 `.venv`，常规 `pip install -e .` 因当前内部 PyPI 索引找不到构建依赖 `setuptools>=68` 而失败。随后在同一新建 Linux `.venv` 启用 `--system-site-packages`，执行 `pip install --no-build-isolation --no-deps -e .` 成功；此环境借用主机 setuptools 78.1.1、jsonschema 4.26.0 等依赖，首次基线复现通过。

为验证不依赖主机 site-packages，又新建 `outputs/.venv-isolated`，仅对此次安装命令指定公开 PyPI 索引，`pip install -e .` 成功。实际包版本为 embodied-eval 0.1.0、jsonschema 4.26.0、attrs 26.1.0、jsonschema-specifications 2025.9.1、referencing 0.37.0、rpds-py 2026.9.1。该隔离环境也复现了完整基线。两者都没有使用 macOS `.venv`；由于 `uv` 不可用、pip 采用 `pyproject.toml` 范围而非 `uv.lock` 的全量锁定版本，仍需在阶段 3 所选 Python/原生依赖环境中另做回归。

分别运行 `.venv/bin/python docs/handoff/replay_baseline.py` 与 `outputs/.venv-isolated/bin/python docs/handoff/replay_baseline.py`，两次均退出 0，收据 `passed=true`。两次各 85 项 unittest 全部通过；8 个契约消息样例核验通过。五组 CLI 的实得退出码均为：chunk 0、tool 0、cache 0、score-error 1、rejected 2，与历史基线一致。chunk 为 6 tick/3 次模型调用；tool 为 5 tick/2 次工具调用；cache 的两条 episode 各 3 tick/1 次 mock forward；scorer 故障被记作不可用指标，预检拒绝未生成结果目录。

首次产物位于 `outputs/handoff-baseline-20261005T094006Z-13ce0b2c/`；独立环境产物位于 `outputs/handoff-baseline-20261005T095212Z-df9e83de/`，以后者作为主要 Linux 复现证据。两处均有 `baseline-check.json`、`unittest.txt`、`contracts.txt`、五组 CLI 输出与各自产物。全部是 synthetic；本次复现只支持阶段 2 迁移验收，不支持真实模型/Benchmark 结论。阶段 3 具名建议见 [阶段 3 确认材料](stage3-review.md)；后续单独授权下载及原生环境预检的实际状态见[候选资产预检](stage3-preflight-20261005.md)。
