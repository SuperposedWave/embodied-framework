"""Fresh direct-native reference of one completed current framework run.

The completed framework is reused read-only, not rerun or resumed. A correction
to a diagnostic comparison must not cause prior framework actions to be resent.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import socket
import sys

from probe_mn_components import ROOT, WORKSPACE, write
from run_mn_trials import server_spec
from run_v03_bounded import supervised_side


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-gpu", type=int, required=True)
    p.add_argument("--simulator-gpu", type=int, required=True)
    args = p.parse_args()
    from embodied_eval.codec import to_data, decode
    from embodied_eval.config import RunConfig, validate_config
    from check_v03_saved_boundaries import digest
    args.framework = args.framework.resolve()
    args.framework.relative_to(ROOT / "outputs")
    args.output = args.output.resolve()
    args.output.relative_to(ROOT / "outputs")
    manifest_path = args.framework / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((args.framework / "summary.json").read_text())
    if manifest["synthetic"] or manifest["config"]["contract_version"] != "0.3" or summary["errors"] or summary["completed"] != 1:
        raise ValueError("completed_real_current_framework_required")
    cfg = decode(RunConfig, manifest["config"])
    validate_config(cfg)
    args.port = cfg.model.options["port"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "framework").symlink_to(args.framework, target_is_directory=True)
    config_path = args.output / "config.json"
    write(config_path, to_data(cfg))
    protected = {str(manifest_path): digest(manifest_path)}
    for path in (args.framework / "episodes").glob("*/trace.internal.jsonl"):
        protected[str(path)] = digest(path)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    command, cwd, client = server_spec(cfg)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               QWEN3VL_PATH=str(WORKSPACE.parent.parent / "Model/Qwen3-VL-4B-Instruct-processor"))
    env.pop("LD_LIBRARY_PATH", None)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    if cfg.model.adapter == "openpi":
        sys.path.insert(0, str(ROOT / "integrations/massworld"))
        from platform_model_cache import environment as cache_environment
        env = cache_environment(env, ROOT / "outputs/worldeval-openpi-public-cache-20261019-run1/manifest.json")
    server_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.model_gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.6")
    client_env = dict(env, CUDA_VISIBLE_DEVICES=str(args.simulator_gpu), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
        CUDA_HOME=str(WORKSPACE / "upstream/toolchains/cuda-12.1-stage4/toolkit"))
    if cfg.benchmark.adapter == "robotwin":
        client_env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(WORKSPACE / "upstream/envs/robotwin-stage4/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json")
    receipt = {"kind": "fresh_native_only_current_framework_read_only_reuse", "passed": False,
               "at_utc": datetime.now(timezone.utc).isoformat(), "framework": str(args.framework),
               "framework_rerun_or_resumed": False, "formal_acceptance": False,
               "protected_sha256": protected, "model_command": command}
    try:
        row = supervised_side(command, cwd, client, config_path, args.output, args, "native", server_env, client_env)
        receipt.update(passed=row["passed"], native=row)
        assert all(digest(Path(path)) == value for path, value in protected.items())
        receipt["framework_evidence_unchanged"] = True
    finally:
        write(args.output / "receipt.json", receipt)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
