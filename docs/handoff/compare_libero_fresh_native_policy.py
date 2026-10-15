"""Fresh native pi05 LIBERO rollout against an immutable framework trial.

Direct native environment, official WebsocketClientPolicy and native action
formula; never call ModelAdapter, BenchmarkAdapter or Runner execution methods.
Own localhost model server and child process groups only. Not formal acceptance.
"""
import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def recorded_snapshots(path):
    action = None
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "prediction":
                action = event["data"]["decision"]["steps"][0]["commands"][0]["parameters"]["osc_action7"]
            elif event["event"] == "reset":
                yield event["data"], None
            elif event["event"] == "step_result":
                if action is None:
                    raise ValueError("missing_recorded_action")
                yield event["data"]["observation"], action
                action = None


def child(args):
    from embodied_eval.config import load_config
    from embodied_eval.codec import to_data
    from embodied_eval.stage4_pairs import verify
    from replay_libero_native_actions import compare_snapshot, digest
    config = load_config(ROOT / "examples/stage4-pi05-libero-spatial.json")
    verify(config)
    manifest_path = args.framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["synthetic"] or manifest["config"] != to_data(config):
        raise ValueError("verified_real_fixed_framework_trial_required")
    episode = config.episodes[0]
    episode_dir = args.framework / "episodes" / episode.episode_id
    result_path = episode_dir / "result.json"
    source_result = json.loads(result_path.read_text())
    if source_result["run_status"] != "completed":
        raise ValueError("completed_framework_trial_required")
    trace = episode_dir / "trace.internal.jsonl"
    source = Path(config.model.openpi_repo) / "examples/libero/main.py"
    node = next(x for x in ast.parse(source.read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == "_quat2axisangle")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    formula = namespace["_quat2axisangle"]
    package = Path(config.benchmark.libero_repo) / "libero/libero"
    rows, requests = [], 0
    with tempfile.TemporaryDirectory(prefix="embodied-eval-fresh-native-") as setup:
        paths = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                 "init_states": str(package / "init_files"), "assets": str(package / "assets"),
                 "datasets": str(package.parent.parent / "datasets")}
        (Path(setup) / "config.yaml").write_text(yaml.safe_dump(paths))
        os.environ.update(LIBERO_CONFIG_PATH=setup, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        sys.path.insert(0, config.benchmark.libero_repo)
        sys.path.insert(0, str(Path(config.model.openpi_repo) / "packages/openpi-client/src"))
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        import torch
        np.random.seed(episode.seed)
        task = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
        initial = torch.load(package / "init_files" / task.problem_folder / task.init_states_file,
                             weights_only=False)  # Fixed blob verified before legacy native loader.
        env = OffScreenRenderEnv(bddl_file_name=str(package / "bddl_files" / task.problem_folder / task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        client = None
        try:
            env.seed(episode.seed)
            env.reset()
            raw = env.set_init_state(initial[0])
            for _ in range(10):
                raw, _, _, _ = env.step([0.] * 6 + [-1.])
            client = WebsocketClientPolicy("127.0.0.1", config.model.port)
            old = iter(recorded_snapshots(trace))
            first, _ = next(old)
            rows.append({"env_step": 0, "action_exact": True,
                         "fields": compare_snapshot(first, raw)})
            cache = deque()
            done, steps = False, 0
            while steps < 220 and not done:
                if not cache:
                    images = {}
                    for key, native_key in (("agentview_image", "observation/image"),
                                            ("robot0_eye_in_hand_image", "observation/wrist_image")):
                        image = np.ascontiguousarray(raw[key][::-1, ::-1])
                        images[native_key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
                    payload = {**images, "observation/state": np.concatenate((raw["robot0_eef_pos"],
                               formula(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])),
                               "prompt": str(task.language)}
                    actions = np.asarray(client.infer(payload)["actions"])
                    if actions.ndim != 2 or actions.shape[0] < 5 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                        raise ValueError("invalid_native_policy_response")
                    cache.extend(actions[:5])
                    requests += 1
                action = cache.popleft()
                raw, _, done, _ = env.step(action.tolist())
                steps += 1
                recorded = next(old, None)
                rows.append({"env_step": steps,
                    "action_exact": bool(recorded is not None and np.array_equal(action, recorded[1])),
                    "native_action": action.tolist(),
                    "framework_action": recorded[1] if recorded else None,
                    "action_max_abs_diff": float(np.max(np.abs(action - recorded[1]))) if recorded else None,
                    "fields": compare_snapshot(recorded[0], raw) if recorded else {},
                    "recorded_snapshot_available": recorded is not None})
                if steps % 25 == 0:
                    print(f"fresh native policy compared {steps} actions", flush=True)
            def same(row):
                return row["action_exact"] and all(x["exact_equal"] for x in row["fields"].values())
            passed = (steps == source_result["execution_statistics"]["env_steps"]
                      and bool(done) == source_result["metrics"]["success"]["value"]
                      and next(old, None) is None and all(same(row) for row in rows))
            receipt = {"kind": "fresh_independent_native_policy_environment_rollout_comparison",
                "checked_at_utc": datetime.now(timezone.utc).isoformat(), "pair": "openpi_libero",
                "passed": passed, "synthetic": False, "fresh_model_inference": True,
                "recorded_actions_used_to_control_native_environment": False,
                "formal_acceptance": False, "seed": episode.seed, "native_settle_steps": 10,
                "native_policy_actions": steps, "native_policy_requests": requests,
                "compared_snapshots": len(rows), "native_success": bool(done),
                "first_divergence": next((r["env_step"] for r in rows if not same(r)), None),
                "native_client": "openpi_client.websocket_client_policy.WebsocketClientPolicy",
                "model_rng": "fresh fixed JAX policy default key(0), same original GPU2; not a general cross-device determinism guarantee",
                "native_main_sha256": digest(source), "framework_manifest_sha256": digest(manifest_path),
                "framework_trace_sha256": digest(trace), "framework_result_sha256": digest(result_path),
                "server_identity_verified_by_handshake": False, "rows": rows}
            (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}), flush=True)
            return 0 if passed else 1
        finally:
            if client is not None:
                client._ws.close()
            env.close()


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    args.framework, args.output = args.framework.resolve(), args.output.resolve()
    if args.child:
        return child(args)
    from embodied_eval.config import load_config
    from embodied_eval.stage4_pairs import verify
    from probe_native_models import server_spec
    verify(load_config(ROOT / "examples/stage4-pi05-libero-spatial.json"))
    args.output.mkdir(parents=True, exist_ok=False)
    command, repo, port, gpu, _ = server_spec("openpi_libero")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", port))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_OFFLINE="1",
                       XLA_PYTHON_CLIENT_MEM_FRACTION="0.6", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                       PYTHONUNBUFFERED="1")
    server = worker = None
    try:
        with (args.output / "server.txt").open("x") as log:
            server = subprocess.Popen(command, cwd=repo, env=environment, stdout=log, stderr=log,
                                      start_new_session=True)
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError("native_server_exit")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.5):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("external_native_start_timeout")
                    time.sleep(1)
            with (args.output / "client.txt").open("x") as log:
                worker = subprocess.Popen([sys.executable, __file__, "--child", "--framework", str(args.framework),
                    "--output", str(args.output)], cwd=ROOT, env=environment, stdout=log, stderr=log,
                    start_new_session=True)
                code = worker.wait(timeout=1800)
            if not (args.output / "receipt.json").is_file():
                raise RuntimeError("native_child_missing_receipt")
            print(f"fresh native LIBERO comparison exit={code} receipt={args.output / 'receipt.json'}")
            return code
    except Exception as error:
        if not (args.output / "receipt.json").exists():
            (args.output / "receipt.json").write_text(json.dumps({"kind": "fresh_native_policy_comparison_error",
                "passed": False, "error_type": type(error).__name__, "formal_acceptance": False}) + "\n")
        raise
    finally:
        stop(worker)
        stop(server)


if __name__ == "__main__":
    raise SystemExit(main())
