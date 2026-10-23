"""Model-free controlled check of constructor RNG versus declared reset seed.

Only diagnostic variants change constructor RNG. No production implementation,
source, protocol, controller or historical run is changed. Native settle10
and the declared reset/init state stay identical in all four variants.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=("framework", "native"), required=True)
    p.add_argument("--constructor-seed", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.resolve().relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from probe_mn_components import public_exception
    from embodied_eval.codec import to_data
    from embodied_eval.config import load_config
    from embodied_eval.contracts import Context
    from embodied_eval.v03.registry import default_registry
    cfg = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
    repo = Path(cfg.benchmark.options["repo"])
    package = repo / "libero/libero"
    temporary = tempfile.TemporaryDirectory(prefix="libero-constructor-audit-")
    paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
             "init_states": str(package / "init_files"), "datasets": str(repo / "datasets"),
             "assets": str(package / "assets")}
    (Path(temporary.name) / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = temporary.name
    sys.path.insert(0, str(repo))
    random.seed(0)
    np.random.seed(0)
    report = {"kind": "controlled_constructor_rng_diagnostic_not_protocol_change_or_model_inference",
              "side": args.side, "requested_constructor_seed": args.constructor_seed,
              "reset_seed": cfg.episodes[0].seed, "completed": False, "weights_loaded": False,
              "policy_actions": 0, "native_settle_steps": 10,
              "at_utc": datetime.now(timezone.utc).isoformat()}
    env = benchmark_adapter = None
    try:
        from libero.libero import benchmark
        import libero.libero.envs as envs
        original = envs.OffScreenRenderEnv
        def constructor(*values, **keywords):
            nonlocal env
            if args.constructor_seed is not None:
                np.random.seed(args.constructor_seed)
            state = np.random.get_state()
            report["constructor_rng_sha256"] = hashlib.sha256(state[1].tobytes() + str(state[2:]).encode()).hexdigest()
            env = original(*values, **keywords)
            return env
        envs.OffScreenRenderEnv = constructor
        if args.side == "framework":
            benchmark_adapter = default_registry().benchmark("libero").factory(cfg.benchmark.options, args.output)
            tasks = benchmark_adapter.enumerate_tasks(cfg.episodes)
            observation = benchmark_adapter.reset(tasks[0], Context(run_id="constructor-rng-audit", env_id="env0", episode_id=tasks[0].episode_id, message_id="start"))
            images = {key: np.asarray(sample.payload.data, np.uint8) for key, sample in observation.sensors.items()}
            report["public_states"] = to_data(observation.states)
        else:
            # Same fixed-source reference initialization as the comparison tool.
            np.random.seed(cfg.episodes[0].seed)
            task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
            initial = torch.load(package / "init_files" / task.problem_folder / task.init_states_file, weights_only=False)
            env = constructor(bddl_file_name=str(package / "bddl_files" / task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
            env.seed(cfg.episodes[0].seed)
            env.reset()
            raw = env.set_init_state(initial[0])
            for _ in range(10):
                raw, _, _, _ = env.step([0.] * 6 + [-1.])
            images = {"head": raw["agentview_image"], "wrist": raw["robot0_eye_in_hand_image"]}
        report["pixels"] = {}
        for key, pixels in images.items():
            Image.fromarray(pixels).save(args.output / (key + ".png"))
            report["pixels"][key] = {"sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "shape": list(pixels.shape)}
        sim = env.sim
        report["native_arrays"] = {}
        for group, obj, names in (("model", sim.model, ("body_pos", "body_quat", "geom_pos", "geom_rgba", "cam_pos", "cam_quat", "light_pos", "light_dir", "tex_rgb")),
                                  ("data", sim.data, ("qpos", "qvel", "mocap_pos", "mocap_quat"))):
            for name in names:
                if hasattr(obj, name):
                    value = np.asarray(getattr(obj, name)).copy()
                    report["native_arrays"][group + "." + name] = {"sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                        "shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist() if value.size < 3000 else None}
        report["completed"] = True
    except Exception as error:
        report["exception_chain"] = public_exception(error)
    finally:
        if benchmark_adapter is not None:
            benchmark_adapter.close()
        elif env is not None:
            env.close()
        temporary.cleanup()
        (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in ("side", "requested_constructor_seed", "completed", "constructor_rng_sha256")}))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
