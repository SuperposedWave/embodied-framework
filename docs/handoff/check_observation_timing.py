"""Persist synthetic observation timing checks and read-only native declarations.

No new GPU inference, simulation, downloads, model server or platform jobs.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"kind": "observation_timing_software_and_synthetic_evidence", "passed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "user_facing_date_Asia_Shanghai": "2026-10-20",
        "new_GPU_inference": False, "new_real_simulation": False, "platform_jobs_submitted": 0,
        "action_resampling": False, "inference_schedule_changed": False,
        "formal_acceptance": False, "checks": [], "crosses": [], "native_declarations": []}
    def command(name, argv):
        result = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True, text=True)
        log = result.stdout + result.stderr
        (output / (name + ".txt")).write_text(log, encoding="utf-8")
        item = {"name": name, "exit_code": result.returncode, "log": name + ".txt",
                "log_sha256": hashlib.sha256(log.encode()).hexdigest()}
        match = re.search(r"Ran (\d+) tests?", log)
        if match:
            item.update(tests=int(match.group(1)), skips=0 if "skipped=" not in log else -1)
        receipt["checks"].append(item)
        return result, item
    try:
        result, item = command("temporal-tests", ["-m", "unittest", "tests.test_v03_observation_timing", "-v"])
        assert result.returncode == 0 and item["tests"] >= 54 and item["skips"] == 0
        result, item = command("bridge-tests", ["-m", "unittest", "discover", "-s", "integrations/massworld/tests", "-v"])
        assert result.returncode == 0 and item["tests"] == 34 and item["skips"] == 0
        for case, expected in (("exact", 0), ("approximate", 0), ("rejected", 2), ("startup_rejected", 2)):
            config = "examples/synthetic_history_" + case + "_run.json"
            result, item = command("check-" + case, ["-m", "embodied_eval", "check", "--config", config])
            assert result.returncode == expected, (case, result.stderr)
            item["expected_exit_code"] = expected
            target = output / ("cli-" + case)
            result, item = command("run-" + case, ["-m", "embodied_eval", "run", "--config", config, "--output", str(target)])
            assert result.returncode == expected, (case, result.stderr)
            assert target.exists() == (expected == 0)
            item["expected_exit_code"] = expected
        from embodied_eval.codec import to_data
        from embodied_eval.v03.config import load_config
        from embodied_eval.v03.registry import default_registry
        from embodied_eval.v03.runner import run
        from embodied_eval.v03 import contracts as c
        cfg = load_config(ROOT / "examples/synthetic_history_exact_run.json")
        for i, model in enumerate(("synthetic_policy_a", "synthetic_policy_b")):
            for j, benchmark in enumerate(("synthetic_simulator_a", "synthetic_simulator_b")):
                options = dict(cfg.model.options)
                options["history"] = {**options["history"], "offsets_s": [-.1, 0.] if i else [-.2, -.1, 0.]}
                current = replace(cfg, model=replace(cfg.model, adapter=model, options=options),
                    benchmark=replace(cfg.benchmark, adapter=benchmark,
                        options={"period_s": .05 if not j else .1, "declare_sample_timing": True}))
                outcome = run(current, output / f"cross-{i}-{j}")
                assert outcome.exit_code == 0 and outcome.summary["synthetic"]
                diag = json.loads((outcome.output / "episodes/ep-0/diagnostics.internal.json").read_text())
                receipt["crosses"].append({"model": model, "benchmark": benchmark, "synthetic": True,
                    "summary": str((outcome.output / "summary.json").relative_to(ROOT)),
                    "statistics": to_data(outcome.results[0]["execution_statistics"]), "temporal": diag["temporal"]})
        # Source declarations, not verified training rates or new closed-loop runs.
        from tests.v03_helpers import cases
        registry = default_registry()
        for name, native_config in cases().items():
            cap = registry.model(native_config.model.adapter).describe(native_config.model.options)
            space = registry.benchmark(native_config.benchmark.adapter).describe(native_config.benchmark.options)
            receipt["native_declarations"].append({"name": name,
                "model": native_config.model.adapter, "profile": cap.io_profile_id,
                "source_revision": cap.source_revision, "checkpoint_ref": cap.checkpoint_ref,
                "model_timing": to_data(cap.temporal),
                "inputs": [{"path": req.public_path, "history": to_data(req.history)} for req in cap.inputs],
                "benchmark": native_config.benchmark.adapter, "protocol": space.protocol_version,
                "timing_mode": space.timing_mode, "sample_timing": to_data(space.sample_timing),
                "new_runtime_validation": False, "training_FPS_inferred": False})
        loaded = [name for name in ("torch", "jax", "sapien", "mujoco", "isaacsim") if name in sys.modules]
        assert not loaded, loaded
        receipt["GPU_or_simulator_libraries_loaded"] = loaded
        receipt["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "embodied_eval").rglob("*.py"))}
        receipt["passed"] = True
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "crosses": len(receipt["crosses"]),
        "native_declarations": len(receipt["native_declarations"]), "output": str(output)}))


if __name__ == "__main__":
    main()
