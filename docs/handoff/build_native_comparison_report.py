"""Fresh immutable report/index for real native closedloops and passive videos.

Terminal execution, strict equality, task scores, and user acceptance remain
separate. No historical outputs, public core, simulator or inference are changed.
"""
import argparse
import ast
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import subprocess
from urllib.parse import quote
from zoneinfo import ZoneInfo

from prepare_mn_extension import ROOT, read, save, digest, historical_integrity
from build_mn_acceptance_report import resource_snapshot


def cases():
    result = []
    for key, label in (("openwam_libero", "OpenWAM／LIBERO"), ("openpi_libero", "π0.5／LIBERO"),
            ("openwam_vlabench", "OpenWAM／VLABench"), ("openpi_vlabench", "π0.5／VLABench"),
            ("openwam_robotwin", "OpenWAM／RoboTwin（旧双臂位姿协议）"), ("lingbot_robotwin", "LingBot-VLA／RoboTwin")):
        run = ROOT / f"outputs/native-comparison-{key}-20261009-run1"
        result.append({"id": key, "label": label, "run": run, "framework": run / "framework", "native": run / "native",
            "videos": ROOT / f"outputs/native-comparison-{key.replace('_', '-')}-videos-20261009-run1", "both_sides_new_this_round": True})
    for key, label, old, attempt in (
            ("pi05-robotwin", "π0.5 ARX／RoboTwin", "mn-acceptance-pi05-robotwin-native-limit-20261008-run2", "run1"),
            ("pi05-robodojo", "π0.5 ARX／RoboDojo", "mn-acceptance-pi05-robodojo-native-limit-20261008-run2", "run10"),
            ("openwam-robodojo", "OpenWAM ARX／RoboDojo", "mn-acceptance-openwam-robodojo-native-limit-20261008-run1", "run11")):
        run = ROOT / f"outputs/native-comparison-{key}-20261009-{attempt}"
        result.append({"id": key, "label": label, "run": run, "framework": ROOT / "outputs" / old / "framework", "native": run,
            "videos": ROOT / f"outputs/native-comparison-{key}-videos-20261009-run1", "both_sides_new_this_round": False})
    run = ROOT / "outputs/native-comparison-cosmos3-droid-20261009-run3"
    result.append({"id": "cosmos3-droid", "label": "Cosmos3 Nano Policy／DROID", "run": run,
        "framework": ROOT / "outputs/native-comparison-cosmos3-droid-20261009-run2/framework", "native": run,
        "videos": ROOT / "outputs/native-comparison-cosmos3-droid-videos-20261009-run1", "both_sides_new_this_round": True})
    return result


def link(path, label):
    path = path.resolve(); path.relative_to(ROOT)
    if not path.is_file():
        raise ValueError("report_link_requires_existing_file")
    return '<a href="' + quote('/' + path.relative_to(ROOT).as_posix(), safe='/') + '">' + escape(label) + '</a>'


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + cell + '</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def score(metric):
    value = metric.get("success")
    if isinstance(value, dict):
        value = value["value"] if value["valid"] else None
    return "成功" if value is True else "未成功" if value is False else "没有官方成功评分"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/native-comparison-baseline-20261009-run8/baseline-check.json")
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly report uncompleted chains, never mark them passed")
    args = parser.parse_args()
    output = args.output.resolve(); output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc)
    inventory, rows, complete_items, incomplete = [], [], [], []
    for item in cases():
        required = [item["native"] / "native-result.json", item["run"] / "receipt.json", item["videos"] / "receipt.json"]
        if any(not path.is_file() for path in required):
            if not args.allow_incomplete:
                raise ValueError("selected_comparison_evidence_missing")
            incomplete.append({"id": item["id"], "label": item["label"], "run": str(item["run"].relative_to(ROOT)),
                "completed": False, "reason": "native_execution_or_comparison_video_not_completed", "formal_acceptance": False})
            continue
        manifest = read(item["framework"] / "manifest.json")
        cfg = manifest["config"]
        result = read(item["framework"] / "episodes" / cfg["episodes"][0]["episode_id"] / "result.json")
        native, supervision = read(item["native"] / "native-result.json"), read(item["run"] / "receipt.json")
        video = read(item["videos"] / "receipt.json")
        if not native["completed"] or not supervision["completed"] or len(video["videos"]) != 3:
            if not args.allow_incomplete:
                raise ValueError("all_selected_real_comparisons_and_three_videos_must_be_terminal")
            incomplete.append({"id": item["id"], "label": item["label"], "run": str(item["run"].relative_to(ROOT)),
                "completed": False, "reason": "native_execution_failed_or_videos_incomplete", "formal_acceptance": False})
            continue
        for row in video["videos"]:
            if digest(item["videos"] / row["file"]) != row["sha256"]:
                raise ValueError("video_sha256_mismatch")
        nr = native.get("native_metrics", {"success": native.get("native_success")})
        native_actions = native["native_actions"] if "native_actions" in native else native["native_policy_actions"]
        native_requests = native["native_requests"] if "native_requests" in native else native["native_policy_requests"]
        first = native.get("first_divergence")
        first_row = next((r for r in native["rows"] if r["env_step"] == first), None)
        changed = [] if first_row is None else [key for key, value in first_row["fields"].items() if not value["exact_equal"] or not value.get("dtype_equal", True)]
        initial = native["rows"][0]
        camera_names = set(manifest["space"]["sensors"])
        image_same = all(initial["fields"][name]["exact_equal"] for name in camera_names)
        state_same = all(v["exact_equal"] for name, v in initial["fields"].items() if name not in camera_names)
        request_check = item["run"] / "first-native-request-check.json"
        request_check_data = read(request_check) if request_check.exists() else None
        record = {"id": item["id"], "label": item["label"], "synthetic": False,
            "both_framework_and_native_new_this_round": item["both_sides_new_this_round"],
            "framework": str(item["framework"].relative_to(ROOT)), "native": str(item["native"].relative_to(ROOT)),
            "video_directory": str(item["videos"].relative_to(ROOT)), "source_config": cfg,
            "capability": manifest["capability"], "benchmark_space": manifest["space"], "assets": manifest["provenance"]["verified_assets"],
            "framework_statistics": result["execution_statistics"], "framework_metrics": result["metrics"],
            "native_actions": native_actions, "native_requests": native_requests,
            "native_physics_steps": native.get("native_physics_steps"), "native_sim_time_s": native["native_sim_time_s"],
            "native_metrics": nr, "strict_equal": native["strict_equal"], "initial_states_and_non_image_fields_equal": state_same,
            "initial_all_original_camera_pixels_equal": image_same, "first_divergence_action": first, "first_changed_fields": changed,
            "first_native_request_packing_check": request_check_data, "native_supervision": supervision,
            "model_boundary_identity_handshake_verified": False, "formal_acceptance": False}
        inventory.append(record)
        complete_items.append(item)
        rows.append([escape(item["label"]), str(result["execution_statistics"]["env_steps"]) + "／" + str(native_actions),
            score(result["metrics"]) + "／" + score(nr), "一致" if native["strict_equal"] else "不一致",
            "无" if first is None else "起点" if first == 0 else "第" + str(first) + ("条新动作" if set(changed) & {"action", "position", "euler", "gripper_width"} else "条动作执行后"),
            link(item["videos"] / "comparison.mp4", "并排视频")])
    baseline = read(args.baseline)
    cli = [r["actual_exit_code"] for r in baseline["checks"] if "actual_exit_code" in r]
    if not baseline["passed"] or cli != [0, 0, 0, 1, 2] or baseline["checks"][0]["tests_run"] < 202:
        raise ValueError("original_software_and_CLI_regression_required")
    history = historical_integrity()
    previous = read(ROOT / "outputs/mn-acceptance-delivery-20261008-final/historical-integrity.json")
    core = previous["public_v02_core_and_runner_unchanged"]
    if any(digest(ROOT / name) != value["sha256"] for name, value in core.items()):
        raise ValueError("public_contract_and_Runner_must_remain_unchanged")
    history["public_v02_core_and_runner_unchanged"] = core
    save(output / "historical-integrity.json", history)
    resources = resource_snapshot()
    # Current UUIDs are compared to actual framework run receipts, not old
    # earlier-instance assumptions in the historical resource helper.
    resources["same_physical_GPU_claim"] = "per-comparison receipt only; no cross-instance blanket assertion"
    resources.pop("current_gpu_uuids_differ_from_earlier_instance", None)
    resources.pop("same_physical_gpu_across_continuation_claimed", None)
    resources["occupancy_processes_not_evaluation"] = {"3": 26963, "5": 26965, "7": 26967}
    save(output / "resources.json", resources)
    windows = sorted((ROOT / "outputs").glob("native-comparison-gpu-20261009-run*.json"))
    gpu = []
    for path in windows:
        value = read(path)
        recomputed = sum(row["eight_gpu_mean_percent"] for row in value["rows"]) / len(value["rows"])
        if abs(recomputed - value["aggregate_mean_percent"]) > 1e-10:
            raise ValueError("GPU_window_recalculation_mismatch")
        gpu.append({"file": str(path.relative_to(ROOT)), "samples": value["samples"], "aggregate_mean_percent": recomputed,
            "start": value["rows"][0]["at_utc"], "end": value["rows"][-1]["at_utc"], "occupancy_included": True})
    mean = sum(row["samples"] * row["aggregate_mean_percent"] for row in gpu) / sum(row["samples"] for row in gpu)
    media_proofs = []
    media_directories = set()
    for path in sorted((ROOT / "outputs").glob("native-comparison-video-verification-20261009-run*.json")):
        proof = read(path)
        if not proof["passed"] or proof["model_or_environment_calls"] != 0:
            raise ValueError("completed_read_only_media_verification_required")
        names = {row["directory"] for row in proof["directories"]}
        if media_directories & names:
            raise ValueError("duplicate_media_verification_must_not_double_count")
        media_directories.update(names)
        media_proofs.append((path, proof))
    if any(str(item["videos"].relative_to(ROOT)) not in media_directories for item in complete_items):
        raise ValueError("all_delivered_videos_must_have_full_decode_verification")
    png_count = sum(proof["PNG_files_verified"] for _, proof in media_proofs)
    mp4_count = sum(proof["MP4_files_fully_decoded"] for _, proof in media_proofs)
    snapshot = {"kind": "native_comparison_and_video_delivery_not_stage_acceptance", "at_utc": now.isoformat(),
        "comparisons": inventory, "selected_real_comparisons": len(inventory),
        "requested_comparisons": len(cases()), "incomplete_comparisons": incomplete,
        "requested_work_complete": not incomplete,
        "strict_equal_comparisons": sum(r["strict_equal"] for r in inventory), "software_tests": baseline["checks"][0]["tests_run"],
        "CLI_exit_codes": cli, "GPU_windows": gpu, "GPU_sample_weighted_mean_percent": mean,
        "GPU_average_not_unsampled_whole_day_claim": True, "occupancy_is_not_evaluation_work_or_score": True,
        "stage4_accepted": False, "contract_frozen": False, "DreamZero_and_LingBot_VA_deferred": True,
        "all_tasks_or_multiple_seeds_evaluated": False, "native_reference_is_fixed_source_components_not_unmodified_every_official_dispatcher": True}
    snapshot.update(original_PNG_files_verified=png_count, MP4_files_fully_decoded=mp4_count,
        media_verification_receipts=[str(path.relative_to(ROOT)) for path, _ in media_proofs])
    save(output / "snapshot.json", snapshot)
    save(output / "stage-state-snapshot.json", read(ROOT / "docs/stage-state.json"))
    title = "EmbodiedEval：同条件原生对照与比较视频"
    sections = ['<h1 id="report-title">' + title + '</h1><p class="lead">' + now.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y年%m月%d日 %H:%M（北京时间）") + '。本次补真实重新推理，不把旧动作重放充作原生政策运行。</p>']
    sections += ['<h2 id="summary">先说结论</h2><div class="callout">计划补齐' + str(len(cases())) + '组，目前' + str(len(inventory)) + '组已有原生重新推理对照和两侧／并排视频；其中' + str(snapshot["strict_equal_comparisons"]) + '组完整记录逐项一致。' + ('其余' + str(len(incomplete)) + '组尚未完成，不能标成通过。' if incomplete else '') + '其他差异原样保留。<strong>材料补齐不代表所有严格复现门槛通过，也不替你正式验收。</strong></div>',
        '<p>你问的π0.5 ARX／RoboTwin：这次原生组件参考也执行了400条动作，仍未成功；框架侧同样未成功。所以不能把该次失败直接归咎于框架，但也不能断言模型在所有起点都做不了。双方第一条新动作已经不同，尚不能用这两条不同轨迹排除全部适配问题。</p>',
        '<p>“原生组件参考”指固定版本的原生模型输入／输出函数、真实模型服务和原生仿真控制／评分直接连接。六组迁移链路和Cosmos／DROID另起同命令模型服务，两侧各自重新推理；三组ARX扩展链路用新的原生运行对照10月8日保存的框架运行，不重写旧结果。具名ARX协议不是未经修改的RoboTwin官方π0.5整套入口，DROID接线也不是RoboLab原生环境本身。</p>',
        table(["组合", "框架／原生动作数", "官方任务结果：框架／原生", "完整逐项对照", "首次差异", "观看"], rows)]
    sections += ['<h2 id="conditions">怎样保证比较有意义</h2><p>固定源码版本、对应权重与归一化、任务、场景／初始状态编号、种子、机器人控制器、输入相机、动作缓存长度及停止预算。两侧不重试推理、不改量化或采样参数；原生侧不执行我们的Runner或Adapter。Runner只是框架的通用调度循环。具体配置、来源文件指纹和实际GPU身份都在每组回执里。</p>',
        '<p>“同条件”首先是配置相同，不承诺渲染和GPU计算自动逐位相同。下表逐项列起点检查；起始画面不同的运行不能称为同一个像素输入。部分服务的握手只返回普通元信息，不能独立证明服务载入了什么权重；本次用自有新服务进程、固定启动命令和加载前资产校验加强来源证据，未把这个限制删除。</p>']
    condition_rows = []
    for r in inventory:
        cap, cfg = r["capability"], r["source_config"]
        condition_rows.append([escape(r["label"]), escape(cfg["model"]["options"]["io_profile"] + "；" + cfg["benchmark"]["options"]["protocol"]),
            "一致" if r["initial_states_and_non_image_fields_equal"] else "不一致", "一致" if r["initial_all_original_camera_pixels_equal"] else "不一致",
            str(cfg["episodes"][0]["seed"]) + "；动作上限" + str(cfg["episodes"][0]["budgets"]["env_steps"])])
    sections.append(table(["组合", "具名模型输入输出配置／环境协议", "起始状态与非图像字段", "起始原始画面", "种子／预算"], condition_rows))
    sections += ['<p>输入输出配置是该权重需要的相机、状态和动作规则，不是环境白名单。ARX是这里使用的双臂机器人配置名称；π0.5的ARX权重与LIBERO／VLABench权重不是同一份权重。LIBERO两种协议分别先稳定仿真10步、30步，再从评测时钟零点开始；VLABench保留冻结场景、两次原生reset、逐步公开指令采样。RoboTwin保留专家种子筛选过程，最终场景种子和物理步长另见回执；它们不等同于配置中的外层seed。</p>',
        '<h2 id="details">每组差异与视频</h2>']
    for item, r in zip(complete_items, inventory):
        first = r["first_divergence_action"]
        body = "从开始到停止的动作、公开状态、原始相机像素和已记录执行字段逐项一致。" if r["strict_equal"] else (
            "起始原始观测已不完全一致，不能把后续新动作差异单独归因于模型或Adapter。" if first == 0 else
            "起始观测一致；首次差异出现在第" + str(first) +
            ("条新动作及其执行记录。" if set(r["first_changed_fields"]) & {"action", "position", "euler", "gripper_width"} else "条动作执行后的记录。") +
            "涉及字段：" + escape("、".join(r["first_changed_fields"])) + "。")
        sections += ['<h3 id="case-' + r["id"].replace('_', '-') + '">' + escape(r["label"]) + '</h3><p>' + body + '</p>',
            '<p>框架' + score(r["framework_metrics"]) + '，原生' + score(r["native_metrics"]) + '。原生实际发起' + str(r["native_requests"]) + '次模型请求，执行' + str(r["native_actions"]) + '条动作；仿真累计' + format(r["native_sim_time_s"], '.9g') + '秒。' +
            ('独立观察到' + str(r["native_physics_steps"]) + '次物理推进。' if r["native_physics_steps"] is not None else '该原生诊断未另设物理步计数器，不由动作数猜物理步数。') + '</p>']
        check = r["first_native_request_packing_check"]
        if check:
            subject = "第一次原生推理输入的无损数组保存" if check.get("native_request_is_lossless_payload_array_archive_not_network_wire") else "第一份原生实际请求"
            sections.append('<p>' + subject + '与框架公开观测的CPU打包检查：' + ("所有字段精确一致。" if check["passed"] else "存在差异，详见专项回执。") + '此项检查的是转换函数，不是补造历史框架网络抓包。</p>')
        if r["id"] == "pi05-robotwin":
            sections.append('<p>第一份实际原生请求与既有框架请求也精确一致；第一动作最大分量差0.0019155966。这是关节角与夹爪原生分量混合数组中的差值，不能统一解释成米或弧度。原生12850物理步／51.400002441秒，框架12967步／51.868002464秒；两侧均未成功。它只回答这个固定任务和起点，不给总体成功率。</p>')
        elif r["id"] == "openpi_libero":
            sections.append('<p>两侧均103条动作成功，第一动作仍有约0.0008225103的混合分量差。输入打包精确一致，因此这份证据把差异入口定位到重新推理的返回值；具体随机状态或跨进程GPU数值计算原因尚未完全定位。不能说“任务成功了，所以复现通过”。</p>')
        elif r["id"] == "openpi_vlabench":
            sections.append('<p>这组任务结果也不一致：框架211条动作后成功；原生跑满300条动作仍未成功，原生进度为0.5，意图得分为1。进度0.5只表示官方检查认定完成了一部分，不能改写为任务成功。起始画面、实测状态、第一份请求打包均精确一致，但第一条新动作的目标位置已相差约0.000029931米，三个欧拉角分量最大差约0.000876820弧度。欧拉角是一种用三个转动角表达方向的方式。差异入口在新推理返回动作；最终成功与否怎样受到后续推理／执行影响，仍需单独调查。</p>')
        elif r["id"] == "lingbot_robotwin":
            sections.append('<p>两侧均112条动作成功。第一动作最大混合分量差约0.0000904627，原生9755物理步与框架9743步不同。相机在缓存动作段沿用上次采集的图像，因此画面短暂一致不等于机器人实测关节完全相同。</p>')
        elif r["id"] == "openwam_robotwin":
            sections.append('<p>这组第一条模型动作及其原生数值类型精确一致，第一条执行也同为94物理步，但执行后的右臂实测位置／关节和三个相机像素开始不同。差异入口在执行后的状态，不是第一条模型采样；具体原生规划／物理计算的跨进程重复性原因尚未完全定位。框架112条动作、10444物理步成功；原生111条动作、10213物理步成功，不能拿双方成功替代轨迹精确复现。</p>')
        elif r["id"] == "cosmos3-droid":
            sections.append('<p>这里只补与既有框架相同的4动作／32物理步有限闭环，不是450动作完整任务评测。固定DROID环境没有可用的官方任务成功评分，本次仍不自造评分。Cosmos服务不读取session_id字段；这个字段只是每次运行的独立会话编号，不把新UUID声称为历史相同UUID。</p>')
        elif r["id"] == "pi05-robodojo":
            sections.append('<p>框架280动作／2800物理步／11.2秒成功；原生289动作／2890物理步／11.56秒成功。两侧起始实测关节、末端姿态及夹爪控制目标精确相同，三路原始画面不同。第一份请求的状态和指令一致、图像不同。没有把新运行强行截到旧280动作，也不能拿双方成功否认轨迹差异。</p>')
        elif r["id"] == "openwam-robodojo":
            sections.append('<p>两侧均319动作／3190物理步／12.76秒成功，但不表示同一条轨迹。起始实测关节、末端姿态及夹爪控制目标精确相同，三路原始画面不同；第一次推理输入存档的状态和指令一致、图像不同。保留原生动作数值类型与32行缓存顺序；私有数组传输不改变数值、不经过公共协议扩展。</p>')
        sections.append('<p>' + ' · '.join(link(item["videos"] / filename, label) for filename, label in (("comparison.mp4", "并排视频"), ("framework.mp4", "框架侧视频"), ("native.mp4", "原生侧视频"))) + '</p>')
        evidence = [(item["run"] / "receipt.json", "自有进程运行回执"), (item["native"] / "native-result.json", "逐动作对照"),
            (item["framework"] / "manifest.json", "框架配置／来源"), (item["videos"] / "receipt.json", "视频文件指纹与时间轴"),
            (item["native"] / "frames/frames.json", "原生无损画面与采集时间索引")]
        if check:
            evidence.append((item["run"] / "first-native-request-check.json", "第一模型请求专项"))
        if r["id"] in ("pi05-robodojo", "openwam-robodojo"):
            evidence += [(item["run"] / "private-runtime-paths.json", "实际绝对图形库路径"),
                (ROOT / "outputs/native-comparison-robodojo-hardware-20261009-run1.json", "运行中的实际GPU身份")]
        sections.append('<p>' + ' · '.join(link(path, label) for path, label in evidence) + '</p>')
    if incomplete:
        sections += ['<h3 id="incomplete">尚未完成的组合</h3><p>' + escape('、'.join(r["label"] for r in incomplete)) + '：原生环境首次相机采集遇到空缓冲／卡住，尚无完整新推理闭环和比较视频。不是任务失败分数，也不是Adapter验收通过。正在排查，不能用旧保存动作重放抵扣。</p>']
    sections += ['<h2 id="video-method">视频怎么读</h2><p>已完成的每组提供框架、原生及上下并排三个MP4；保留所有协议需要的相机，不丢腕部或另一只手。横向同一行是同一侧的各相机，上下两行按实际仿真秒数对齐，不把第100条动作硬对到另一侧第100条。</p>',
        '<p>30 fps表示成片每秒30帧，不是额外采集了30次相机。没有新观测时保留上一张真实图像；标签中的captured是实际采集时间，action是已经执行的动作序号。FINISHED表示这一侧先结束，后面保留最后画面。对照采集新增的get_obs、渲染、物理推进均为零；录像未改变原生采样顺序。</p>',
        '<p>所有原始RGB保存为无损PNG，附像素与文件SHA256。RGB就是红、绿、蓝三个颜色通道；SHA256是文件内容指纹。MP4为有损压缩，观看方便但不能拿视频像素当精确一致判据。原始PNG和完整实际请求仍在本机相应产物目录；预览仅开放报告显式链接的文件，不开放目录浏览。</p>',
        '<p>LIBERO／VLABench的仿真时间单独允许0.00000001秒的源浮点时钟舍入差：例如累加每步0.1秒与直接读取原生时钟，二进制浮点末位可能不同。这项容差已在逐动作回执声明，不适用于动作、实测状态或图像；“完整逐项一致”不等于连时钟计算的最后一个二进制位也相同。</p>',
        '<p>共' + format(png_count, ',') + '张原始PNG逐一校验文件和像素指纹，' + str(mp4_count) + '个MP4完整解码、实际帧数匹配。' + ' · '.join(link(path, "媒体核验" + str(i+1)) for i, (path, _) in enumerate(media_proofs)) + '。这些检查不调用模型或仿真，不把视频完整性解释为任务成功。</p>',
        '<h2 id="regression-resources">回归、资源与保留证据</h2><p>最新轻量Linux回归' + str(snapshot["software_tests"]) + '项测试通过，包含原85项和202项基线，五组CLI退出码仍为0／0／0／1／2。CLI就是命令行入口；其中1和2分别是预期的评分失败与配置拒绝，不是漏修的异常。本轮新增检查覆盖录像、网页安全、数组传输和原生诊断输入／启动边界，仍是软件／合成检查，不能算真实模型或任务成绩；也不冒称七套环境均重跑了本轮测试数。</p><p>' + link(args.baseline, "实际软件回归回执") + ' · ' + link(output / "resources.json", "当前实例资源") + ' · ' + link(output / "historical-integrity.json", "历史文件与公共核心指纹检查") + '</p>',
        '<p>本实例为8张系统报告的RTX 4090，单卡报告49140 MiB，驱动570.124.06；MiB等于1024×1024字节。这是实例的实际报告，不推断等同官方模型的推荐硬件。物理GPU以UUID识别，而不是只看卡号。共享磁盘剩余空间不等于个人配额。没有改共享驱动、全局代理或只读参考仓库，也没有保存API密钥。</p>',
        '<p>已完成采样窗口的样本加权八卡总体平均为' + format(mean, '.2f') + '%。计算方式：每个采样点先对八卡利用率取平均，再对已完成窗口的所有采样点平均；没有把未采样时段说成全程达标。占卡在3、5、7号卡，且7号卡可能与渲染重叠；不按GPU总利用率伪造模型有效工作量或任务成绩。占卡程序本轮未改动。</p><p>' + ' · '.join(link(ROOT / row["file"], "GPU窗口" + str(i + 1)) for i, row in enumerate(gpu)) + '</p>',
        '<p>π0.5／RoboDojo原生首轮在创建环境工厂前遇到同名utils工具包加载顺序冲突，失败回执保留；修正的是新诊断脚本的来源搜索顺序，不是改控制器或模型。' + link(ROOT / "outputs/native-comparison-pi05-robodojo-20261009-run1/receipt.json", "未完成首轮回执") + '。</p>',
        '<p>RoboDojo后续新诊断启动出现相机返回空CPU缓冲、原生图像内核仍按完整画面读取的卡住问题。保留失败／取消与安全拒绝记录；补root标志、隔离Kit启动参数、对齐代理及调整模型／辅助库加载顺序均未证明解决。另用原有独立原生脚本在GPU5和GPU0执行四条保存动作，验证原生工厂可以初始化；这两次仅是启动排查，不是新的模型推理或完整任务。之后查实：新诊断把私有图形库目录用相对路径传给子进程，子进程进入产物目录后该路径实际不存在；原脚本先转成绝对路径。修复新诊断的路径后，两侧原生reset均返回并取得真实图像，再接新推理。仍只复用原生reset前缀，明确排除保存动作循环，不包装旧组合Adapter、不填假图或多加仿真步。</p>',
        '<p>' + link(ROOT / "outputs/native-comparison-attempt-ledger-20261009-final.json", "26轮诊断／启动尝试原样汇总") + '：失败、取消、新推理完成与仅保存动作启动短测分别标注。' + link(ROOT / "outputs/native-comparison-pi05-robodojo-20261009-run9/cancellation.json", "π0.5旧错误路径实测") + ' · ' + link(ROOT / "outputs/native-comparison-openwam-robodojo-20261009-run10/cancellation.json", "OpenWAM旧错误路径实测") + '。</p>',
        '<p>Cosmos／DROID原生诊断前两轮分别误选资产校验入口、把未带批次维度的七关节数组错误切成一个标量，回执及实际请求均保留。修复只在新增对照脚本，完整七关节与夹爪的保真／拒绝测试已加入；生产Model／Benchmark Adapter没有修改。最终框架取本轮GPU6／0的第二轮成功四动作运行，原生取同GPU、同服务命令的第三轮，不重复写已有框架证据。</p>',
        '<h2 id="limits">还不能据此承诺什么</h2><p>这些是固定单任务、单起点的对照，不是全任务、多种子成功率，更不承诺任何权重能在任意机器人上成功。严格差异未消失的组合仍保留门禁失败；任务成功、映射正确、重新推理逐位一致是三个不同问题。</p>',
        '<p>DreamZero仍等你后续安排两张每卡80GB的H100，再验证真实动作和闭环；现在不会擅改量化、画面尺寸或采样绕过显存要求。LingBot-VA继续暂缓，不把已有LingBot-VLA拿来抵扣。原49组缺口及范围划分不回写，不以本次原生对照标成“全部模型／Bench已完成”。</p>',
        '<p>阶段4尚待你正式确认，公共契约仍未冻结。本轮不新增Runner并发、重试、恢复、硬仿真时间上限或DataRef载荷读取。DataRef指外部数据引用，当前不会把一个路径自动当成已经读入的观测。</p>',
        '<p>' + link(output / "snapshot.json", "本轮覆盖与结果索引") + ' · ' + link(output / "stage-state-snapshot.json", "交付时阶段状态快照") + '</p>']
    template = Path("/root/.codex/skills/html-report/assets/report-template.html").read_text()
    shell = template[:template.index("<body>")].replace('lang="en"', 'lang="zh-CN"').replace("Report title", title)
    report = output / "native-closedloop-comparison-and-videos.html"
    report.write_text(shell + '<body><div class="report-shell"><main class="report-content">' + ''.join(sections) + '</main></div></body></html>', encoding="utf-8")
    validation = subprocess.run([__import__("sys").executable, "/root/.codex/skills/html-report/scripts/validate_report.py", str(report)], capture_output=True, text=True, timeout=30)
    if validation.returncode:
        raise ValueError("HTML_skill_validation_failed")
    from serve_comparison_report import serve_report
    serve_report.SUFFIXES = serve_report.SUFFIXES | {".mp4"}
    _, exposed, hashes = serve_report.catalog(report)
    source = {}
    for folder in ("embodied_eval", "tests", "docs/handoff"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" not in path.parts:
                ast.parse(path.read_text()); source[str(path.relative_to(ROOT))] = digest(path)
    save(output / "evidence-sha256.json", hashes)
    save(output / "verification.json", {"kind": "native_comparison_delivery_integrity_not_acceptance", "delivery_integrity_passed": True,
        "HTML_skill_validation_passed": True, "selected_file_count": len(exposed), "source_python_files_parsed": len(source),
        "source_python_sha256": source, "software_tests": snapshot["software_tests"], "CLI_exit_codes": cli, **history,
        "stage4_accepted": False, "contract_frozen": False, "strict_equal_comparisons": snapshot["strict_equal_comparisons"],
        "real_or_strict_pass_inferred_from_integrity": False})
    print(json.dumps({"report": str(report.relative_to(ROOT)), "comparisons": len(inventory), "strict_equal": snapshot["strict_equal_comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
