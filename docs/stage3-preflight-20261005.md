# 阶段 3 候选资产与原生环境预检（2026-10-05 UTC）

本记录依据用户后续授权检查本机 `Model/`、下载模型和 Benchmark，并继续准备原生环境。**资产准备不等于选择阶段 3 试点或验收阶段 2。** 本次尚未运行 OpenWAM 推理、完整 LIBERO trial 或框架真实 Adapter；未产生真实模型评分。下载命令只临时清除代理环境变量并尝试直连，没有修改全局代理设置。下述 `SHA-256` 是文件内容摘要。

## 固定来源与本地资产

| 资产 | 固定版本、路径与核验 |
| --- | --- |
| OpenWAM 源码 | `upstream/OpenWAM/`；提交 `48bd67b89d489b14d03b8d92bc66e65d306df32e`，工作树无修改。 |
| LIBERO 源码与场景 | `upstream/LIBERO/`；提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。原生 `setup_env.sh` 已应用 OpenWAM 自带的 `libero-pytorch-load.patch`，只改动 `libero/libero/benchmark/__init__.py`，`git diff --check` 通过。 |
| OpenWAM-α 的 LIBERO checkpoint | `/inspire/hdd3/project/agent-driven-world-model/zhouzhixiang-240107010008/Model/OpenWAM/OpenWAM-Alpha-Sim-LIBERO/checkpoint_step_10690.safetensors`；固定 Hugging Face revision `110b07e4d5581c81520a77419cbb0e9371df231b`，24,813,767,464 bytes；本地 SHA-256 `34c9a6c9af67dda9aaffa2edc0efa29b28eeee1c26dea419f98065c80508c96a`，与发布方大文件元数据一致。未将权重复制到框架源码目录。 |
| 配套文件 | 同一模型目录有 `config.yaml`、`normalization_stats.npy` 和 `tokenizer/google/umt5-xxl/`。归一化文件本地 SHA-256 `f3fd7daa340ee6cbcb0d6a489a130f1e866c2688fcc40ffd98cae506b4d0cc8e`，与发布方元数据一致。尚未验证模型实际加载。 |
| LIBERO 首任务 | `libero_spatial` task 0 为 `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`。BDDL SHA-256 `9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`；原生 Benchmark 使用的 `.pruned_init` SHA-256 `cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`。BDDL 是任务定义文件，`.pruned_init` 是固定初始化状态文件。 |

原先 `Model/` 中的 nvidia/Cosmos3-Nano、tencent/Hy-Embodied-VLM-1.0 和 Qwen/Qwen3.8-27B，没有查到与本候选原生 LIBERO 评测相匹配的 checkpoint、归一化资产及配置；不据名称擅自替代 OpenWAM 权重。现有模型目录里另有一次已取消串行下载留下的 `checkpoint_step_10690.safetensors.partial`，不是可用 checkpoint。

## 独立环境与仿真预检

- `upstream/envs/libero/` 是独立的 Python 3.10 环境，不复用 macOS `.venv`。运行固定 OpenWAM 提供的 `benchmarks/libero/setup_env.sh`，其包版本检查、`pip check` 与导入检查均通过，最终打印 `LIBERO evaluation environment ready with official requirements and MuJoCo 3.3.2`。实装关键版本为 PyTorch 2.7.1 CPU、robosuite 1.4.0、MuJoCo 3.3.2、NumPy 1.22.4。
- 安装中 `egl_probe` 首次因缺 CMake 失败；在此独立环境安装 CMake 4.4.4 后发现它移除了该旧源码所用的低版本兼容配置，故仅在该环境降为 CMake 3.31.10 并构建 `egl_probe` 1.0.2。原生脚本重新执行后成功。此处没有修改上游 `egl_probe` 或系统 CMake。
- 运行固定上游 `run_smoke.sh env`，限定 `LIBERO_SMOKE_GPU=0` 和项目内 `LIBERO_CONFIG_ROOT`，退出码 0；输出核对 `libero_spatial` task 0、任务文本、BDDL 路径，成功建立 EGL 离屏仿真环境并执行一步，打印 `env_smoke=ok reward=0.0 done=False`，观测包含两路相机、末端位姿、关节和夹爪字段。`reward=0.0` 和 `done=False` 只是烟测一步，不是模型评测结果。
- 模型侧另建 `upstream/envs/openwam-py310/`，安装官方推荐的 PyTorch 2.7.1/CUDA 12.8、TorchVision 0.22.1/CUDA 12.8、TorchAudio 2.7.1/CUDA 12.8，并以可编辑方式安装固定 OpenWAM 源码及其声明依赖。三个 PyTorch wheel（Python 安装包文件）均取自官方 CUDA 12.8 索引，实际 SHA-256 分别为 `d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3`、`538f4db667286d939b4eee0a66d31ed21b51186668006b0e0ffe20338ecc7e00`、`2ba0816eee659e343851a9c5dc60c8e1eb819a3969b29268fab27d3143273d78`，均与该索引的文件摘要一致。wheel 存在 `upstream/envs/wheels/`；NVIDIA 运行库从公开 PyPI 直连安装。实装关键版本为 OpenWAM 0.1.0、PyTorch 2.7.1+cu128、Transformers 5.18.0、Diffusers 0.40.0、Deepspeed 0.18.9。`torch`/`torchvision`/`torchaudio` 导入成功，`torch.cuda.is_available()` 为 `True`，可枚举 8 张设备。`pip check` 输出 `No broken requirements found.`，`openwam.deploy.server` 导入成功。安装时设置 `DS_BUILD_OPS=0`，只跳过 Deepspeed 可选算子编译，不改变 OpenWAM 源码。checkpoint 的可加载性、模型服务握手、前向显存和延迟均未验证。首轮用当前占卡程序未中断也完成了仿真烟测；推理若需空卡，再仅核对并终止该占卡程序。
- 此次磁盘占用的 `du -sh` 读数：OpenWAM 源码约 36 MB、LIBERO 约 659 MB、LIBERO 环境约 3.4 GB、模型环境约 8.4 GB、三个 PyTorch wheel 约 1.0 GB、模型目录约 24 GB；共享盘个人配额仍未核实，以上是目录占用而非可用配额。

## 阶段门禁

阶段 2 的 Linux 85 项测试及五组 CLI 退出码见 [迁移复现记录](stage2-linux-replay-20261005.md)。本记录只描述用户确认前的候选预检快照；后续“继续推进”后的[具名阶段 3 配置](stage3-review.md)和[单 trial 原生结果](stage3-native-evidence-20261005.md)另行记录。原生成功不等于框架 Adapter 验收；新增公开扩展与 CLI/产物语义仍有下一道确认门禁。
