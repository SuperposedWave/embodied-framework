"""Separate sampling changes from independent-process numerical differences."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from prepare_mn_native_environment import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = [json.loads((p / "model-result.json").read_text()) for p in (args.first, args.second)]
    for p in (args.first, args.second):
        if not json.loads((p / "receipt.json").read_text())["passed"]:
            raise ValueError("completed real model diagnostic required")
    checks = {key: a[key] == b[key] for key in ("native_input_fingerprint", "native_package_versions", "jax_devices", "source_sha256", "rng_keys_before_requests")}
    if not all(checks.values()):
        raise ValueError("independent process controls differ; cannot isolate fixed-input fixed-RNG comparison")
    first = a["first_native_actions"][0]
    second = b["first_native_actions"][0]
    exact = a["native_chunk_sha256"][0] == b["native_chunk_sha256"][0]
    result = {"kind": "controlled_pi05_sampling_and_process_context_investigation_not_historical_cause_proof",
        "at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "within_first_process_same_rng_exact": a["same_rng_repeated"]["exact_equal"],
        "within_second_process_same_rng_exact": b["same_rng_repeated"]["exact_equal"],
        "different_rng_changes_first_action": a["next_rng"]["first_action_maximum_absolute_difference"] > 0,
        "independent_process_same_rng_same_input_full_chunk_exact": exact,
        "independent_process_first_action_max_abs_difference": max(abs(x - y) for x, y in zip(first, second)),
        "raw_first_actions": [first, second], "official_sampling_or_backend_flags_changed": False,
        "historical_action_difference_uniquely_attributed": False, "root_nondeterministic_kernel_identified": False,
        "plain_language": "同一进程、同一个随机数状态可以逐值复现；更换随机数状态会改变动作。重启进程后，在输入、随机数、软件版本和设备相同的条件下仍观察到数值差异，不能把历史差异全部归因于换了一份随机噪声。" if not exact else
            "同一进程与本轮独立进程在固定随机数时均逐值复现；更换随机数会改变动作，但尚未证明历史运行使用了哪个随机数状态。",
        "evidence": [str(p / "model-result.json") for p in (args.first, args.second)], "formal_acceptance": False}
    write(args.output, result)
    print(json.dumps({key: result[key] for key in ("within_first_process_same_rng_exact",
        "within_second_process_same_rng_exact", "independent_process_same_rng_same_input_full_chunk_exact",
        "independent_process_first_action_max_abs_difference")}), flush=True)


if __name__ == "__main__":
    main()
