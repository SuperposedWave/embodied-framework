"""Native dependency/SAPIEN scene gate; no task/model/score claim."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    from embodied_eval.adapters.native_common import native_module
    helper = native_module(WORKSPACE / "upstream/OpenWAM", "benchmarks.robotwin.eval_policy_wrapper")
    helper._prewarm_cuda_for_curobo()
    helper._patch_warp_torch_namespace()
    from curobo.wrap.reacher.motion_gen import MotionGen  # Explicitly fail if optional helper swallowed import errors.
    import sapien
    from embodied_eval.adapters.native_robotwin_executor import _SceneMeter
    scene = sapien.Scene()
    scene.set_timestep(0.004)
    meter = _SceneMeter(scene, float(scene.get_timestep()))
    for _ in range(3):
        meter.step()
    versions = {name: metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "sapien", "mplib", "warp-lang", "nvidia-curobo", "toppra")}
    package = Path(sys.prefix) / "lib/python3.10/site-packages"
    original = {"sapien/wrapper/urdf_loader.py": "eabba86fe7d244ce8d469010f96914010a8883ea5835c86a8bc58be310510683",
                "mplib/planner.py": "785f77233baf4159f3327b664fae2507572c189223a6075f42546c1133ddfa0e"}
    patches = [{"file": name, "before_sha256": digest, "after_sha256": sha(package / name),
                "source": "RoboTwin fixed script/_install.sh; UTF-8 read and native mplib screw condition only"}
               for name, digest in original.items()]
    embodiment = WORKSPACE / "upstream/RoboTwin-openwam/assets/embodiments/aloha-agilex"
    derived = []
    for side in ("left", "right"):
        template = embodiment / ("curobo_" + side + "_tmp.yml")
        destination = embodiment / ("curobo_" + side + ".yml")
        expected = template.read_text().replace("${ASSETS_PATH}", str(WORKSPACE / "upstream/RoboTwin-openwam")).rstrip() + "\n"
        if destination.read_text() != expected:
            raise ValueError("native_embodiment_template_substitution_differs")
        derived.append({"template": str(template), "template_sha256": sha(template),
                        "generated": str(destination.resolve()), "generated_sha256": sha(destination),
                        "change": "native ASSETS_PATH placeholder substitution only; original verified archive files unchanged"})
    report = {"kind": "real_native_dependency_and_simple_sapien_scene_not_robot_task_or_model",
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "passed": meter.completed_steps == 3 and not meter.ambiguous,
              "versions": versions, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "actual_scene_steps": meter.completed_steps, "actual_scene_period_s": scene.get_timestep(),
              "scene_proxy_basic_gate_passed": True, "native_macro_coverage_verified": False,
              "task_reset_rendered": False, "model_called": False, "scoring_performed": False,
              "system_cuda_modified": False, "dependency_patches": patches, "derived_embodiment_configs": derived,
              "native_helper_sha256": sha(WORKSPACE / "upstream/OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py")}
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
