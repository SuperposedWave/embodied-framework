"""Compare per-step native OpenWAM debug states/actions to framework output."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("comparison output already exists")
    fw_files = sorted((args.framework / "native-policy-debug" / "episode_000").glob("step_*/meta.json"))
    native_files = sorted((args.native / "policy-debug" / "episode_000").glob("step_*/meta.json"))
    framework = json.loads((args.framework / "summary.json").read_text())
    framework_result = json.loads((args.framework / "episodes/spatial-task0-trial0/result.json").read_text())
    native = json.loads((args.native / "results/results.json").read_text())
    rows = []
    for step, (fw_path, native_path) in enumerate(zip(fw_files, native_files)):
        fw, reference = json.loads(fw_path.read_text()), json.loads(native_path.read_text())
        fields = {}
        for name in ("state", "native_eef10_action", "libero_action7"):
            left, right = fw[name], reference[name]
            fields[name] = {"same_length": len(left) == len(right),
                            "exact_equal": left == right,
                            "max_abs_diff": max((abs(a-b) for a, b in zip(left, right)), default=None)}
        rows.append({"step": step, "framework_meta_sha256": sha(fw_path),
                     "native_meta_sha256": sha(native_path), "fields": fields})
    field_names = ("state", "native_eef10_action", "libero_action7")
    result = {
        "framework_summary_sha256": sha(args.framework / "summary.json"),
        "native_results_sha256": sha(args.native / "results/results.json"),
        "framework_synthetic": framework["synthetic"],
        "framework_status": framework_result["run_status"],
        "framework_success": framework_result["metrics"]["success"]["value"],
        "framework_policy_steps": framework_result["execution_statistics"]["env_steps"],
        "native_success": native["trials"][0]["success"],
        "native_policy_steps": native["trials"][0]["policy_steps"],
        "framework_debug_steps": len(fw_files), "native_debug_steps": len(native_files),
        "compared_steps": len(rows),
        "field_max_abs_diff": {name: max((row["fields"][name]["max_abs_diff"] for row in rows), default=None)
                               for name in field_names},
        "field_exact_equal_steps": {name: sum(row["fields"][name]["exact_equal"] for row in rows)
                                    for name in field_names},
        "first_numerical_divergence": next((row["step"] for row in rows
            if not all(row["fields"][name]["exact_equal"] for name in field_names)), None),
        "image_pixel_comparison": "unavailable: official native debug metadata does not retain image bytes",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"framework={len(fw_files)} native={len(native_files)} compared={len(rows)} "
          f"first_divergence={result['first_numerical_divergence']}")


if __name__ == "__main__":
    main()
