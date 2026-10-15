"""New template-based HTML and evidence snapshot; never rewrites old delivery."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from build_mn_report import resources, table
from prepare_mn_extension import read, save, digest, historical_integrity, asset_inventory


LABELS = {"model_present_action_structure_mismatch": "模型组件已有，当前环境的动作/状态结构不满足",
    "protocol_identity_still_undefined": "原表环境编号或被测角色仍未定义",
    "user_deferred": "用户明确暂缓LingBot-VA",
    "outside_action_model_scope": "本轮不做画面/文字结果转动作的规划系统",
    "historically_implemented_components": "历史具名组件已有实际证据",
    "new_structure_ready_runtime_unverified": "新增组件接线通过，运行验证另列"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    prep = args.preparation.resolve()
    trial = args.trial.resolve()
    trial_receipt = read(trial / "receipt.json")
    summary_path = trial / "pi05_arx_robotwin/framework/summary.json"
    summary = read(summary_path) if summary_path.is_file() else None
    result_path = trial / "pi05_arx_robotwin/framework/episodes/ep-0/result.json"
    result = read(result_path) if result_path.is_file() else None
    first_trial = ROOT / "outputs/mn-extension-pi05-arx-robotwin-runtime-20261007-run1/receipt.json"
    repeat_comparison = ROOT / "outputs/mn-extension-pi05-arx-bounded-repeats-20261007.json"
    repeat_check = read(repeat_comparison)
    assert result and repeat_check["repeat_statistics"] == result["execution_statistics"]
    gpu_smoke = ROOT / "outputs/mn-extension-pi05-arx-gpu-smoke-20261007-gpu2/model-result.json"
    smoke = read(gpu_smoke)
    assert smoke["passed"] and smoke["model_loaded"] and smoke["native_chunk_rows"] == 50
    prep_receipt, ledger = [read(prep / filename) for filename in ("receipt.json", "coverage-ledger.json")]
    assets = asset_inventory()
    # This copy is a NEW audit layer, not a rewritten original ledger or trace.
    ledger["new_real_benchmark_runtime_diagnostics_attempted"] = 2
    ledger["new_real_benchmark_runtime_diagnostics_completed_without_framework_error"] = bool(summary and summary["errors"] == 0)
    ledger["runtime_diagnostic_ref"] = str(trial.relative_to(ROOT))
    ledger["new_real_trials"] = 2 if summary else 1
    ledger["runtime_diagnostic_receipts"] = [str(first_trial.relative_to(ROOT)), str(trial.relative_to(ROOT)) + "/receipt.json"]
    ledger["new_full_task_or_suite_acceptance_trials"] = 0
    for row in ledger["rows"]:
        if row["source_cell"] == "AE14":
            row["bounded_real_runtime_diagnostic"] = {"completed": bool(summary and summary["errors"] == 0),
                "action_budget": 50, "receipt": str(trial.relative_to(ROOT)) + "/receipt.json", "full_task_acceptance": False}
            row["plain_language_reasons"].append("另做50动作上限的真实运行诊断；不是全任务验证、原生新策略严格对照或该机械臂训练证明。")
    save(target / "coverage-ledger.json", ledger)
    save(target / "assets.json", assets)
    resource = resources()
    resource["new_large_weights_downloaded_in_this_architecture_batch"] = True
    resource["independent_environments_created"] = ["upstream/envs/pi05-arx-mn", "upstream/envs/pi05-arx-mn-inference", "upstream/envs/mn-usd-audit",
        "upstream/envs/cosmos3-mn-native", "upstream/envs/dreamzero-mn-native", "upstream/envs/openwam_arx-mn-native", "upstream/envs/droid_sim-mn-native"]
    resource["pi05_declared_training_dependency_stack_install_failed"] = True
    resource["pi05_compatible_inference_stack_smoke_passed"] = True
    resource["downloads_at_snapshot"] = {}
    for component in ("cosmos3", "dreamzero", "openwam_arx"):
        directory = Path(assets["components"][component]["directory"])
        chunks = list((directory / ".persistent-http-ranges").glob("*/*.chunk"))
        resource["downloads_at_snapshot"][component] = {"complete_range_files_with_metadata": sum(p.with_suffix(".json").is_file() for p in chunks),
            "complete_range_cache_bytes": sum(p.stat().st_size for p in chunks if p.with_suffix(".json").is_file()),
            "range_cache_is_not_published_checkpoint": True, "cache_hashes_not_recomputed_by_this_snapshot": True}
    resource["asset_continuation_max_rounds_after_initial_download"] = 2
    resource["asset_continuation_never_starts_models_or_simulators"] = True
    save(target / "resources.json", resource)
    paths = [ROOT / f"outputs/mn-extension-gpu-20261007-run{i}.json" for i in (1, 2, 3, 4)]
    windows = [read(path) for path in paths]
    for window in windows:
        recalculated = sum(sum(d["utilization_percent"] for d in r["devices"]) / 8 for r in window["rows"]) / window["samples"]
        assert abs(recalculated - window["aggregate_mean_percent"]) < 1e-10
        assert window["aggregate_mean_percent"] >= 20
    report_path = target / "M+N-adapter-extension-and-gaps.html"
    def link(path, label):
        relative = os.path.relpath(path.resolve(), target)
        return '<a href="' + quote(relative, safe="/") + '">' + escape(label) + '</a>'
    sections = []
    def section(identifier, title, body):
        sections.append('<section aria-labelledby="' + identifier + '"><h2 id="' + identifier + '">' + title + '</h2>' + body + '</section>')
    date = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M 中国时间")
    section("outcome", "本轮结论", f'''<div class="callout"><strong>LingBot-VA暂缓，LingBot-VLA保留。</strong>
      源代码能明确界定的新增动作模型与仿真执行边界已写入独立注册表，现有5个模型家族、10份模型输入输出配置、5个仿真家族和7份具名环境协议。
      这不是“所有原表组合都已真实跑通”。下载、依赖、闭环与原生对照分别记账，阶段4未正式验收，契约未冻结。</div>
      <p>软件回归：5套环境均306项、零跳过，旧五组CLI退出码仍0/0/0/1/2。新增43项测试在原263项之上；都是软件/合成边界检查。
      9项固定源码函数对照全部通过，输入是合成数据，不是模型成绩。π0.5 ARX真实GPU推理已产生50条14维动作，但使用合成观测。</p>
      <p>另外完成两次50动作上限的π0.5 ARX／RoboTwin真实运行诊断，实际结果见下文；不将结构接线、合成观测推理或有限步诊断当作全任务验收。</p>
      <p>{link(target / 'coverage-ledger.json', '本轮逐格覆盖账本')}；{link(prep / 'receipt.json', '结构与回归证据索引')}；
      {link(ROOT / 'outputs/mn-delivery-20261007-final/M+N-architecture-and-evidence.html', '上一轮完整架构与六组历史证据（原文件不改）')}。</p>''')
    section("models", "模型接入情况", table(("模型家族", "具名输入输出配置", "本轮状态"), [
        ("OpenWAM-alpha", "原3份＋eef20_arx_absolute", "新增ARX双臂绝对位姿配置；32条动作与源函数一致；ARX权重下载/真实推理未完成"),
        ("π0.5 / openpi", "原2份＋arx_joint14", "ARX权重18/18核验，兼容环境、真实GPU50条动作检查通过；新增RoboTwin有界诊断另列"),
        ("LingBot-VLA 2.0", "joint14", "既有配置及证据保留；没有用它代替LingBot-VA"),
        ("Cosmos 3", "nano-policy-droid-joint8", "明确选定动作Policy型号；输入/32动作源函数对照通过；大权重及服务运行未完成"),
        ("DreamZero-DROID", "droid-joint8-causal", "输入/8动作源函数对照通过；因果会话和缓存由模型侧处理；大权重及服务运行未完成"),
        ("LingBot-VA", "无新增", "按用户最新要求暂缓，不标记已适配")]) + '''<p>“配置”指一份特定权重所需的输入排列、状态含义和动作解码规则，不是为每个环境再写一套模型Adapter。
      Cosmos家族的Nano-Reasoner不能当作这次的Nano-Policy；纯画面或文字输出不自动变成机器人动作。训练覆盖仍是提示，不阻断结构明确的接线。</p>''')
    section("benchmarks", "仿真接入情况", table(("仿真家族", "已具名协议", "边界和验证范围"), [
        ("LIBERO", "spatial-30settle / spatial-10settle", "保留历史空间任务0和不同起步等待；不能把关节目标直接当原生末端控制"),
        ("VLABench", "frozen-track1", "保留select_fruit的冻结场景、原生控制及官方指标；不是全suite"),
        ("RoboTwin", "sealed-instruction / seeded-instruction", "保留adjust_bottle/demo_clean；新增实测关节公开字段和双臂规划失败检查"),
        ("RoboDojo", "stack-bowls-arx-x5-native-v1", "新独立执行边界：ARX双臂、stack_bowls、布局集合seed0/index0；412文件核验≠场景/依赖全部就绪"),
        ("DROID sim_evals", "droid-three-scenes-native-v1", "源代码的场景1/2/3；示例选场景1；没有官方任务成功评分，不自动映射原表R021/R022")]) + '''<p>没有把VIDMAN的R009/R010猜成CALVIN，也没有将WIW的ImageNav/Manipulation和世界模型规划角色自行定下来。
      新DROID组件是明确的新增环境，不是旧表中含义未定项目的换名替身。</p>''')
    config_rows = [(key.replace("_", " / "), "通过", "检查不是运行") for key in ledger["additional_examples_not_original_cell_claims"]]
    section("wiring", "接线与职责", '''<p>M是模型数量，N是仿真环境数量。M+N的意思是各写自己的Adapter，新增一方时不逐个修改另一方。</p>
      <p>数据依次经过：环境原生观测 → 环境Adapter → 公共观测 → 模型Adapter → 原生推理 → 模型Adapter → 公共动作 → 环境Adapter → 仿真。</p>
      <p>接线配置只写相机、状态字段、左右手目标和明确坐标转换；没有新增每对组合的Python工厂。Runner仍只管通用顺序、校验、预算、计数和证据。
      模型侧负责图像处理、排列、归一化、解码与缓存；环境侧负责控制、逆运动学、物理推进和官方评分。</p>''' + table(("六份新增示例", "结构预检查", "证据含义"), config_rows) +
      '<p>' + '；'.join(link(prep / (key + ".json"), key + "配置") + ' / ' + link(prep / (key + "-preflight.json"), "预检查") for key in ledger["additional_examples_not_original_cell_claims"]) + '</p>' +
      '''<p>ARX权重接RoboTwin时保留未核实训练覆盖的提示，不据此拒绝，也不声称该权重已经在AgileX机械臂训练过。
      缺必要相机、漏一只手或关节、数组维数不符、坐标和动作含义不明仍拒绝，不能靠补零或改名称冒充通用性。</p>''')
    section("semantics", "已处理的关键细节", '''<h3 id="state-meaning">测量值与控制目标不能混用</h3>
      <p>RoboTwin新增的六个关节位置来自原生机械臂实际位置访问器，不再用“刚才要求关节去哪里”冒充“关节现在在哪里”。
      RoboTwin和RoboDojo的夹爪输入在此原生路径上仍来自控制目标，因此明确标为控制目标。DROID则提供实际关节/夹爪测量。
      采集时间、图像和动态公开指令保留；不额外取图、消耗随机数来伪造对齐。</p>
      <h3 id="gripper-and-dtype">夹爪方向、截断顺序与数值类型</h3>
      <p>DROID原生数值越大越接近关闭，公共开度越大越接近打开，双方明确取反。原生阈值严格大于0.5才关闭，等于0.5不随意改规则。
      Dream原生客户端把单精度动作拼上双精度夹爪数组后，整行变成双精度；这次照原样保留，而不是只保留数值、忽略类型。</p>
      <p>RoboTwin先沿原始夹爪请求插值，再逐物理步截断；RoboDojo先截断后插值。提前统一截断会改变动作过程，因此由各自环境协议决定。
      新增具名“原始开度请求”扩展只承载一个含义清晰的夹爪数值；12个关节仍是公共弧度目标，不用不透明14维数组包办一切。
      公共夹爪开度仍是0到1，公共核心结构和0.1未改。</p>
      <h3 id="partial-execution">不能把没执行的手记成执行成功</h3>
      <p>RoboDojo的逆运动学失败、RoboTwin的位姿规划失败，以及关节轨迹计时失败/空轨迹，都可能造成手臂不动而夹爪仍推进。
      新物理动作路径观察原生返回值，不改变规划结果或轨迹；失败明确返回执行错误，只确认实际接受的目标，保留已发生的动作/物理步/时间，不做自动重试或回滚。
      旧0.1及具名旧原生EE16保留历史行为，历史轨迹和报告不回写。</p>
      <h3 id="independent-dojo">RoboDojo如何与模型分开</h3>
      <p>只复用固定源码中12个环境方法，移除那一条属于模型生命周期的reset调用；不执行原生模块里的模型客户端/工厂/恢复逻辑。
      任务构造、观测、控制、计时和评分依旧来自原生环境。编译检查验证所选方法不残留模型客户端和策略配置访问，但没有替代真实Isaac运行检查。</p>
      <h3 id="camera-wiring">别被原生相机变量名误导</h3>
      <p>固定DROID场景的external_cam在正Y侧，external_cam_2在负Y侧。Cosmos客户端按RoboLab的左右相机位置取图，
      示例接线是exterior_left→external_cam、exterior_right→external_cam_2。DreamZero固定仿真客户端却将external_cam叫right_image：
      它的exterior_right→external_cam、exterior_left→external_cam_2。这里槽位名是原生客户端的输入名字，不等于真实左右位置。
      两份接线是明确数据配置，模型和环境实现不互相分支。旧示例保留，最新预检查和源函数回执记录了修正后的接线；没有将左右图偷偷互换来提升成绩。</p>
      <h3 id="frames-and-timing">坐标和时间分开记</h3>
      <p>OpenWAM新增ARX配置使用具名的权重输入校准，将有含义的世界位姿映射到模型使用的双臂基座表示，再转回公共世界位姿。
      没有把旧四元数含义冲突的16维数组重新贴成“通用位姿”。RoboDojo/RobotTwin分别记录原生动作次数、物理步数和实际仿真秒数；仍不提供硬仿真秒数上限。</p>''')
    phase = result["execution_statistics"] if result else {}
    metrics = result["metrics"] if result else {}
    section("evidence", "本轮实际验证", '<h3 id="regression">软件与源函数检查</h3>' +
      table(("环境", "测试数", "跳过", "CLI退出码"), [(row["receipt"].split("/")[1], row["tests"], row["skips"], "0/0/0/1/2") for row in prep_receipt["baselines"]]) +
      '<p>' + '；'.join(link(ROOT / row["receipt"], "环境回归 " + str(i+1)) for i, row in enumerate(prep_receipt["baselines"])) + '</p>' +
      f'''<p>9项源函数检查覆盖Cosmos图像组合/单精度状态/32动作，Dream相机顺序/端点/8动作双精度提升，π0.5 ARX双臂输入/50动作，
      RoboDojo环境方法编译，以及OpenWAM ARX校准输入/32动作。全部为合成输入直接调用固定源函数，不计真实任务成绩：
      {link(ROOT / prep_receipt['native_function_receipt'], '源函数对照回执')}。
      {link(ROOT / 'outputs/mn-extension-wheel-20261007-final/verification.json', '安装包中6份资产清单的字节校验')}也通过，避免只在源码目录能找到资产声明。</p>
      <h3 id="gpu-smoke">真实模型、合成观测</h3><p>π0.5 ARX权重及匹配normalization（归一化参数）核验后，在2号卡加载原生策略，产生50条14维动作。
      Adapter请求1次、取出50行、缓存剩余0，全部通过公共动作校验及纯执行数组转换。没有启动仿真或评分：{link(gpu_smoke, 'GPU推理与50行动作检查')}。
      3号卡原生CUDA非法访存另记：{link(ROOT / 'outputs/mn-extension-pi05-arx-gpu3-abort-20261007.json', '3号卡故障观察')}，不归结为已证实硬件故障。</p>
      <h3 id="runtime-diagnostic">真实模型接真实RoboTwin：有限步诊断</h3><p>这次使用ARX权重、RoboTwin原有adjust_bottle/demo_clean协议以及50动作预算。
      框架错误数：{summary['errors'] if summary else '无结果文件'}；原生动作：{phase.get('env_steps', '未取得')}；物理步：{phase.get('physics_steps', '未取得')}；
      实际仿真秒数：{phase.get('sim_time_s', '未取得')}。任务指标原始值与有效性：<code>{escape(json.dumps(metrics.get('success', {}), ensure_ascii=False))}</code>。</p>
      <p>这是跨权重/环境的真实运行证据，不是全任务测试；没有任务成功不能单凭一次判断模型完全不能迁移，更不能用任务成功替代映射正确。
      没有独立新策略逐项对照、服务端可信权重身份或全suite统计，因此不标为完整验收。
      {link(trial / 'receipt.json', '监督回执')}；{link(summary_path, '框架摘要') if summary else '无框架摘要'}；
      {link(result_path, '原始动作计数与官方指标') if result else '无episode结果'}。</p>''')
    sections[-1] = sections[-1].replace('</section>', f'''<p>第一次运行50动作、1486物理步、5.944000282324851秒；此次重复50动作、1523物理步、6.092000289354473秒，
      两次均是有效success=false与预算耗尽，框架错误为0。首次回执保持原样：{link(first_trial, '首轮有限步诊断')}。</p>
      <p>只读比较两次保存数据：首轮原生state/images/prompt（状态、图像、指令）的值与类型全部一致，首动作关节目标不一致，
      最大单关节差{repeat_check['maximum_first_joint_target_difference_rad']}弧度，夹爪原始请求也不同。它是重新推理差异，不是已定位的映射错误或硬件故障。
      没有受控随机数/后端实验，不能全部归因随机性；更不能把两次物理步的不同抹平。{link(repeat_comparison, '重复运行首输入与首动作比较')}。</p></section>''')
    asset_rows = [(name, f"{r['verified_files']}/{r['required_files']}", f"{r['verified_bytes']/1e9:.3f} / {r['required_bytes']/1e9:.3f}",
        "固定文件集已核验；依赖/引用另查" if r["complete_pinned_file_set"] else "未完成；分块缓存不计完整权重") for name, r in assets["components"].items()]
    section("assets-runtime", "资产、环境与真实缺口", table(("组件", "核验文件数", "核验/所需GB（十进制）", "含义"), asset_rows) +
      f'''<p>{link(target / 'assets.json', '固定资产逐文件盘点')}。大文件当前命令先临时去代理直连，失败才回退原代理；未改全局代理，未保存密钥。
      超时残片和完整分块保留，可继续下载；只有整文件匹配发布者摘要才发布为权重。Cosmos已有Reasoner文件只有摘要完全相同的VAE等5项复用，
      不拿同名、同大小但摘要不同的策略权重冒充完成。</p>
      <h3 id="python-dependencies">依赖不是“装过包就算好”</h3><p>π0.5源声明NumPy低于2，LeRobot 0.4.4间接要求至少2，原依赖安装失败。
      本轮新建的兼容推理环境采用源码明确支持的旧LeRobot导入分支，固定到0cf8648版本，配rerun-sdk 0.21.0；optax/chex使用源锁版本。
      JAX0.5.3、Flax0.10.2、Torch2.10.0、NumPy1.26.4保留具名记录；209个安装包依赖检查通过。
      这是公开记录的推理依赖偏差，不宣称原训练环境可复现，也未修改参考仓库或旧环境。
      {link(ROOT / 'outputs/mn-extension-pi05-arx-env-20261007-run1/receipt.json', '原依赖安装失败')}；
      {link(ROOT / 'outputs/mn-extension-pi05-arx-env-20261007-compatible-run2/receipt.json', '兼容环境安装')}；
      {link(ROOT / 'outputs/mn-extension-runtime-probe-pi05-arx-20261007-compatible-run3.json', '具名配置导入')}；
      {link(ROOT / 'outputs/mn-extension-pi05-arx-package-versions-20261007.json', '209包版本与只读依赖检查')}。</p>
      <h3 id="new-native-environments">其他新环境：装好、能导入、能推理不是一回事</h3>
      <p>Cosmos使用固定源码支持的Python3.13/CUDA12.8/Torch2.10组，区别于文档的CUDA13/Torch2.13默认。
      安装完成但403包检查仍有两处不兼容：openpi-client声明NumPy低于2而源覆盖安装了2.2.6；jupyter-compare-view要求Python低于3.13。
      原生配置引入Megatron路径时抛RuntimeError，按要求先调用原生初始化后CPU导入仍失败，未定位完整原因；不是证明GPU硬件坏了。
      {link(ROOT / 'outputs/mn-extension-cosmos3-env-20261007-run1/receipt.json', 'Cosmos安装与所选源组')}；
      {link(ROOT / 'outputs/mn-extension-cosmos3-package-versions-20261007.json', 'Cosmos两处包约束冲突')}；
      {link(ROOT / 'outputs/mn-extension-cosmos3-native-import-startup-20261007.json', 'Cosmos原生导入失败位置')}。</p>
      <p>DreamZero声明依赖已安装，依赖检查和原生GrootSimPolicy类导入通过；OpenWAM ARX声明依赖安装、114包检查和原生服务模块导入也通过。
      这些只是环境/导入证据，没有实例化新权重、做GPU推理或仿真。
      {link(ROOT / 'outputs/mn-extension-dreamzero-env-20261007-run1/receipt.json', 'DreamZero安装')}；
      {link(ROOT / 'outputs/mn-extension-dreamzero-package-versions-20261007.json', 'DreamZero安装包检查')}；
      {link(ROOT / 'outputs/mn-extension-dreamzero-native-import-20261007.json', 'DreamZero原生类导入')}；
      {link(ROOT / 'outputs/mn-extension-openwam_arx-env-20261007-run1/receipt.json', 'OpenWAM ARX安装')}；
      {link(ROOT / 'outputs/mn-extension-openwam_arx-package-versions-20261007.json', 'OpenWAM ARX包检查')}；
      {link(ROOT / 'outputs/mn-extension-openwam-arx-native-import-20261007.json', 'OpenWAM原生模块导入')}。</p>
      <p>DROID的源锁是IsaacLab2.2.0/IsaacSim5.0.0.0。flatdict4.0.1的构建脚本调用pkg_resources，而当前setuptools已移除此模块。
      首次构建失败；仅在本轮环境中按原锁源包摘要引导构建并固定setuptools80.9.0后，该包成功。
      整体原生安装在直连/原代理各120秒的窗口都超时，仍在续装；未接受运行时EULA（软件使用条款）或启动仿真，不改全局驱动和原仓库。
      {link(ROOT / 'outputs/mn-extension-droid-sim-env-20261007-build-compatible-run4/receipt.json', '缺pkg_resources具体失败')}；
      {link(ROOT / 'outputs/mn-extension-droid-sim-env-20261007-build-bootstrap-run5/receipt.json', '原摘要构建通过、全环境超时')}；
      <a href="https://setuptools.pypa.io/en/latest/deprecated/pkg_resources.html">setuptools移除模块说明</a>；
      <a href="https://docs.astral.sh/uv/reference/settings/">uv构建依赖约束说明</a>。</p>
      <h3 id="usd-runtime">场景文件齐了，内部引用未必齐</h3><p>独立usd-core只读检查打开DROID机器人/三场景/桌面及RoboDojo ARX根文件。
      DROID还有远端物体、材质和贴图引用；核心USD工具未解析这些URL，不等于远端文件一定不存在，仍需原生Isaac运行时验证。
      scene1还引用my_droid.usdz，但该robot节点已处于关闭状态，不能以这一条就判运行必失败。
      RoboDojo ARX根检查的两条未解析项是可能由运行时提供的材质模块；本检查不覆盖所有碗/房间/材质，也没有启动渲染。
      {link(ROOT / 'outputs/mn-extension-usd-audit-20261007-run2.json', '逐根引用与启用状态检查')}；
      <a href="https://openusd.org/24.08/api/dependencies_8h.html">OpenUSD依赖检查说明</a>。</p>
      <p>当前驱动570.124.06低于RoboDojo所用Isaac Sim5.1文档列出的Linux580.65.06要求，当前未验证可运行；不能擅自升级共享驱动。
      DROID的固定兼容运行环境仍未装完。Cosmos和Dream所需额外编码器、完整权重与GPU推理也未全部核验。
      不能简单用CUDA版本号判所有路线都不能运行：次版本兼容仍有条件限制。
      <a href="https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html">Isaac Sim5.1要求</a>；
      <a href="https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html">NVIDIA CUDA次版本兼容</a>。</p>''')
    section("gaps", "原49缺口逐项保留", table(("原49格的本轮分类", "格数"), [(LABELS[key], value) for key, value in ledger["scope_counts_original_49_only"].items()]) +
      '''<p>合计仍是原来的49格，没有删除不方便的行。31格属于本轮不做的画面/文字转动作规划，4格VA按用户暂缓，9格环境编号/角色仍未明确，
      2格Cosmos当前动作结构不满足，3格已有新接线但完整运行验收未完成。这里3格是π0.5／RoboTwin、π0.5／RoboDojo和OpenWAM／RoboDojo。
      原表AE4虽然曾被旧账本误记“实现”，实际上仍是VA而不是VLA，本轮继续单列纠正，因此全部55格中VA暂缓为5格。</p>''' +
      table(("原表格", "原模型 / 环境", "当前分类", "人话说明"), [(r["source_cell"], r["original_model"] + " / " + r["benchmark"], LABELS[r["scope_status"]], " ".join(r["plain_language_reasons"])) for r in ledger["rows"] if r["was_one_of_original_49_gaps"]]) +
      '<p>' + link(target / 'coverage-ledger.json', '完整55格机器可读账本') + '。源码有足够定义的组件继续推进；未知角色不会用占位Adapter、猜测协议或另一个模型抵扣。</p>')
    section("resources", "资源与GPU目标", f'''<p>Linux x86_64，{resource['logical_cpus']}逻辑CPU，主机总内存约{resource['host_memory_bytes']/2**30:.1f}GiB。
      8张RTX4090，实例报告每张49140MiB，驱动570.124.06。共享盘的空闲量不等于个人配额。</p>''' +
      table(("采样窗口", "样本数", "八卡总体均值", "目标"), [(str(i+1), w["samples"], f"{w['aggregate_mean_percent']:.2f}%", "≥20%，通过") for i, w in enumerate(windows)]) +
      '<p>' + '；'.join(link(path, '原始GPU窗口 ' + str(i+1)) for i, path in enumerate(paths)) + '；' + link(target / 'resources.json', '当前资源、进程与分块下载快照') + '</p>' +
      '''<p>口径是每次八卡使用率先取平均，再对该窗口样本平均。窗口可能重叠，不拼成“整个未采样时段也达标”。
      现有占卡进程保留，单列说明，不计评测工作量或成绩。真实模型推理与占卡负载是两回事。</p>''')
    section("next", "接下来怎么推进", '''<ol><li>完成未发布大权重的分块下载及整文件摘要核验；检查模型额外编码器、normalization与端点身份，不以有文件夹代替就绪。</li>
      <li>准备并固定Cosmos/Dream推理环境与DROID/RoboDojo仿真栈；原生Isaac依赖、共享驱动兼容和材质引用是独立门禁，不更改共享驱动。</li>
      <li>依次完成新组件真实运行、同环境原生输入/动作转换和逐步执行对照；任务效果差与映射错误分开调查。有限步测试不扩充为全suite成绩。</li>
      <li>旧OpenWAM/RoboTwin轨迹与π0.5/LingBot重新推理差异继续保留；不改容差、评分或历史记录。未知环境编号维持具体缺口；不逐项追问已明确的工程默认值。</li>
      <li>提交完整证据后仍由用户确认阶段4验收和契约冻结；不自动进入阶段5。</li></ol>
      <p>DataRef载荷读取、Runner并发/恢复/硬超时、视频转动作规划器仍未建设。本次下载续传与诊断进程外部上限不等于Runner具备这些能力。</p>''')
    template = Path("/root/.codex/skills/html-report/assets/report-template.html").read_text()
    title = "M+N适配扩展：暂缓VA，推进其他具名组件"
    prefix = template.split('<h1 id="report-title">', 1)[0].replace('<html lang="en">', '<html lang="zh-CN">').replace('<title>Report title</title>', '<title>' + title + '</title>')
    html = prefix + '<h1 id="report-title">' + title + '</h1><p class="lead">报告生成：' + date + '。资产核验UTC时间：' + escape(assets['at_utc']) + '。代码、资产、真实运行和阶段验收分层说明；历史证据保持原样。</p>' + '\n'.join(sections) + '</main></div></body></html>\n'
    report_path.write_text(html, encoding="utf-8")
    validate = subprocess.run([sys.executable, "/root/.codex/skills/html-report/scripts/validate_report.py", str(report_path)], capture_output=True, text=True)
    from serve_report import catalog
    route, exposed, hashes = catalog(report_path)
    verification = {"kind": "mn_extension_delivery_integrity_not_acceptance", "at_utc": datetime.now(timezone.utc).isoformat(),
        "delivery_integrity_passed": validate.returncode == 0, "html_validation_output": validate.stdout,
        "report_sha256": digest(report_path), "linked_local_file_count": len(exposed), "linked_local_file_sha256": hashes,
        "baselines_all_306_passed": all(row["passed"] for row in prep_receipt["baselines"]),
        "source_function_checks_passed": 9, "real_model_synthetic_observation_smoke_passed": True,
        "bounded_real_runtime_diagnostic": trial_receipt, "visual_browser_check_performed": False,
        "stage4_formally_accepted": False, "contract_frozen": False, **historical_integrity()}
    save(target / "verification.json", verification)
    if not verification["delivery_integrity_passed"]:
        raise ValueError("HTML validation failed")
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
