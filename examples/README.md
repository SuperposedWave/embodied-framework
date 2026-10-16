# 当前示例：0.3

正常运行JSON均显式包含 `contract_version: "0.3"`，没有逐对 `binding`。模型和环境分别选择自己的Adapter、具名权重输入输出配置和环境协议。

- `mn_synthetic_run.json`、`synthetic_*_run.json`：合成架构／CLI检查，不是真实成绩。其中score_error预期退出1，rejected预期退出2。
- `stage3-*.json`、`stage4-*.json`：保留易识别文件名的六条既有原生入口，内容已迁为0.3；阶段名不表示本次重新验收通过。
- `mn_openwam_dual_wxyz_encoded_absolute.json`：OpenWAM旧双臂权重的具名标准化物理位姿入口，不是ARX配置的替身。
- `v03_*.json`：ARX／DROID的六份独立模型与环境配置。DreamZero仍待后续双H10080GB真实验证；配置存在不是推理已通过。

`contract_examples.json`、`contract_sample.py`、`validate_contracts.py` 是保留给旧测试的历史消息序列化样本，**不是0.3运行配置**；当前run/check不接受它们。`stage3-native-debug-pilot.yml` 是原生调试材料，也不是当前框架RunConfig。

旧五份mock运行配置及旧原生配置的原件在 `tests/fixtures/legacy-configs/` 和Git标签 `pre-v03-20261016`。不要直接运行或改写历史证据中的旧文件来冒充新结果。

从项目根目录运行：

```sh
.venv/bin/python -m embodied_eval check --config examples/mn_synthetic_run.json
.venv/bin/python -m embodied_eval run --config examples/mn_synthetic_run.json --output outputs/my-v03-run
```

真实配置里的服务地址不是常驻服务；先核验源码、资产及独立运行环境，再用具名监督工具启动。接入细节见 `docs/adapter-onboarding.md`；协议字段见 `docs/contract-v0.3.md`。
