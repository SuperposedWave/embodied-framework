"""Audit existing real runs only; do not infer, simulate, retry, or download.

Two CPU readers may run concurrently; this is OFFLINE log analysis, never
Runner concurrency or parallel trials. Old versions stay old and read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docs.handoff.execution_audit import analyze_directory, digest_file


def inventory():
    old = []
    for name in ("openwam_libero", "openpi_libero", "openwam_vlabench", "openpi_vlabench", "openwam_robotwin", "lingbot_robotwin"):
        old.append({"id": "historical-" + name, "route": name, "cohort": "historical",
                    "directory": ROOT / f"outputs/native-comparison-{name}-20261009-run1/framework"})
    for name, relative in (
        ("pi05_robotwin", "mn-acceptance-pi05-robotwin-native-limit-20261008-run2/framework"),
        ("pi05_robodojo", "mn-acceptance-pi05-robodojo-native-limit-20261008-run2/framework"),
        ("openwam_robodojo", "mn-acceptance-openwam-robodojo-native-limit-20261008-run1/framework"),
        ("cosmos3_droid", "native-comparison-cosmos3-droid-20261009-run2/framework"),
        ("openwam_dual_physical", "openwam-dual-pose-live-20261015-run1/recorded-framework/framework")):
        old.append({"id": "historical-" + name, "route": name, "cohort": "historical", "directory": ROOT / "outputs" / relative})
    index = ROOT / "docs/handoff/qz-runtime-repair-20261019-final-index.json"
    latest = json.loads(index.read_text())
    recent = []
    for case in latest["cases"]:
        base = (ROOT / case["receipt"]).parent
        for mode, suffix in (("direct", "direct"), ("bridge", "bridge/backend")):
            recent.append({"id": "recent-" + case["case"] + "-" + mode,
                "route": case["case"], "cohort": "recent_4_actions", "mode": mode,
                "directory": base / suffix})
    return old + recent


def array_difference(left, right, *, image=False):
    import numpy as np
    a, b = np.asarray(left), np.asarray(right)
    result = {"shape_equal": a.shape == b.shape, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    if a.shape != b.shape:
        result["exact"] = False
        return result
    mask = a != b
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    result.update(exact=not bool(mask.any()), values=int(a.size), changed_values=int(mask.sum()),
                  max_abs_difference=float(delta.max()) if delta.size else 0.)
    if image and a.ndim == 3:
        changed = np.any(mask, axis=2)
        coords = np.argwhere(changed)
        result["changed_pixels"] = int(changed.sum())
        result["changed_pixel_bbox_yx"] = [coords.min(axis=0).tolist(), coords.max(axis=0).tolist()] if len(coords) else None
        result["exact_after_flip"] = {"vertical": bool(np.array_equal(a, b[::-1])),
            "horizontal": bool(np.array_equal(a, b[:, ::-1])), "both": bool(np.array_equal(a, b[::-1, ::-1]))}
    return result


def first_record(directory, filename):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    path = directory / "episodes" / m["config"]["episodes"][0]["episode_id"] / filename
    with path.open("rb") as stream:
        line = stream.readline()
    data = json.loads(line)
    return data, {"path": str(path), "first_line_sha256": hashlib.sha256(line).hexdigest()}


def compare_recent(left, right):
    a, ap = first_record(left["directory"], "trace.internal.jsonl")
    b, bp = first_record(right["directory"], "trace.internal.jsonl")
    a, b = a["data"], b["data"]
    cameras, states = {}, {}
    for key in sorted(set(a["sensors"]) | set(b["sensors"])):
        x, y = a["sensors"].get(key), b["sensors"].get(key)
        if x is None or y is None or x["payload"].get("data") is None or y["payload"].get("data") is None:
            cameras[key] = {"exact": None, "reason": "missing_sensor_or_inline_pixels"}
        else:
            cameras[key] = array_difference(x["payload"]["data"], y["payload"]["data"], image=True)
            cameras[key]["dtype_equal"] = x["payload"]["dtype"] == y["payload"]["dtype"]
    for key in sorted(set(a["states"]) | set(b["states"])):
        x, y = a["states"].get(key), b["states"].get(key)
        states[key] = {"exact": False} if x is None or y is None else {
            **array_difference(x["payload"]["data"], y["payload"]["data"]),
            "full_sample_exact": x == y}
    ai, aip = first_record(left["directory"], "model_input.public.jsonl")
    bi, bip = first_record(right["directory"], "model_input.public.jsonl")
    initial_input_exact = ai == bi
    actions = []
    for n, (x, y) in enumerate(zip(left["actions"], right["actions"]), 1):
        actions.append({"index": n, "complete_action_exact": x == y})
    transitions = []
    for x, y in zip(left["transitions"], right["transitions"]):
        transitions.append({"index": x["action_index"], "left_physics_steps": x["physics_steps"],
            "right_physics_steps": y["physics_steps"], "physics_steps_equal": x["physics_steps"] == y["physics_steps"],
            "left_elapsed_s": x["elapsed_s"], "right_elapsed_s": y["elapsed_s"],
            "elapsed_equal_at_numeric_tolerance": abs(x["elapsed_s"] - y["elapsed_s"]) <= 1e-8})
    first_action = next((x["index"] for x in actions if not x["complete_action_exact"]), None)
    first_physics = next((x["index"] for x in transitions if not x["physics_steps_equal"]), None)
    all_images = all(x["exact"] is True and x["dtype_equal"] for x in cameras.values())
    all_states = all(x["exact"] and x.get("full_sample_exact", False) for x in states.values())
    if not initial_input_exact:
        boundary = "saved_public_input_differs_before_first_returned_action"
    elif first_action == 1:
        boundary = "same_first_public_input_but_first_returned_action_differs"
    elif first_physics is not None and (first_action is None or first_physics < first_action):
        boundary = "same_public_actions_up_to_first_different_execution_physics_count"
    elif first_action is not None:
        boundary = "later_action_difference_requires_prior_state_and_request_telemetry"
    else:
        boundary = "no_difference_in_reviewed_initial_inputs_actions_and_execution_counts"
    return {"route": left["route"], "kind": "saved_direct_vs_MassWorld_not_official_native_reference",
        "initial_camera_pixels_all_exact": all_images, "initial_measured_and_target_state_samples_all_exact": all_states,
        "first_saved_public_model_input_exact": initial_input_exact,
        "first_action_difference": first_action, "first_physics_count_difference": first_physics,
        "camera_differences": cameras, "state_differences": states,
        "actions": actions, "execution": transitions, "first_localized_boundary": boundary,
        "root_cause_proven": False, "server_weight_identity_verified": False,
        "training_or_sampling_rate_cause_proven": False,
        "source_first_records": [ap, bp, aip, bip]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("all", "recent_4_actions", "historical"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("one_to_four_offline_CPU_readers_required")
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    selected = [x for x in inventory() if args.cohort == "all" or x["cohort"] == args.cohort]
    selected.sort(key=lambda x: (x["cohort"] != "recent_4_actions", x["id"]))
    receipt = {"kind": "read_only_saved_execution_clock_and_input_audit", "passed": False, "completed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "selected_runs": len(selected), "runs": [],
        "git_commit_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "new_GPU_inference": False, "new_simulation": False, "new_platform_jobs": 0,
        "historical_evidence_modified": False, "public_contract_changed": False,
        "action_or_observation_schedule_changed": False, "formal_acceptance": False}
    source_paths = [ROOT / "docs/handoff/execution_audit.py", ROOT / "docs/handoff/audit_saved_execution.py"]
    receipt["audit_source_sha256"] = {str(x.relative_to(ROOT)): digest_file(x) for x in source_paths}
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(analyze_directory, row["directory"]): row for row in selected}
            for future in as_completed(futures):
                row = futures[future]
                value = future.result()
                value.update(id=row["id"], route=row["route"], cohort=row["cohort"], mode=row.get("mode"))
                path = output / (row["id"] + ".json")
                with path.open("x") as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                receipt["runs"].append({"id": row["id"], "route": row["route"], "cohort": row["cohort"],
                    "mode": row.get("mode"), "passed": value["passed"], "evidence": path.name,
                    "sha256": digest_file(path), "actions": value["executed_actions"], "quality_issues": value["quality_issues"]})
                print(json.dumps({"completed": len(receipt["runs"]), "total": len(selected),
                    "id": row["id"], "passed": value["passed"], "actions": value["executed_actions"],
                    "requests": value["recorded_native_requests"], "issues": value["quality_issues"]}), flush=True)
        compared = []
        for route in sorted({x["route"] for x in selected if x["cohort"] == "recent_4_actions"}):
            pair = [json.loads((output / ("recent-" + route + "-" + mode + ".json")).read_text()) for mode in ("direct", "bridge")]
            compared.append(compare_recent(*pair))
        with (output / "recent-boundary-localization.json").open("x") as f:
            json.dump({"kind": "seven_saved_direct_vs_bridge_boundary_localizations", "comparisons": compared,
                       "new_inference": False, "root_causes_proven": False}, f, ensure_ascii=False, indent=2)
            f.write("\n")
        if any(digest_file(ROOT / name) != sha for name, sha in receipt["audit_source_sha256"].items()):
            raise ValueError("audit_source_changed_during_analysis")
        receipt["runs"].sort(key=lambda x: x["id"])
        receipt["completed"] = len(receipt["runs"]) == len(selected)
        receipt["passed"] = receipt["completed"] and all(x["passed"] for x in receipt["runs"])
    finally:
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with (output / "receipt.json").open("x") as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
