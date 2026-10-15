"""Bounded caller's native imports, offline, without models or simulators."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {
    "cosmos3": ("cosmos-framework-mn", "cf5d68c00d97ccd2480a2320ed652b92dec63102", "", "cosmos_framework.inference.inference", ("OmniInference",)),
    "openwam_arx": ("xpolicy-reference-mn", "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2",
        "policy/OpenWAM/OpenWAM", "openwam.deploy.server", ("build_server_from_config", "_load_deploy_yaml")),
    "dreamzero": ("dreamzero-mn", "ab790c198fbce33503358efbbd4187ce9a89adf3", "", "groot.vla.model.n1_5.sim_policy", ("GrootSimPolicy",)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-startup-cpu-only", action="store_true",
        help="Cosmos only: run its required init_script first, with current subprocess GPU visibility disabled")
    args = parser.parse_args()
    if args.native_startup_cpu_only and args.component != "cosmos3":
        raise ValueError("reviewed startup path is Cosmos-only")
    name, revision, relative, module_name, symbols = COMPONENTS[args.component]
    repo = ROOT.parent / "upstream" / name
    def check_source():
        actual = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        if actual != revision or dirty:
            raise ValueError("pinned source revision or clean state changed")
    check_source()
    sys.path.insert(0, str(repo / relative))
    # Current subprocess only, fail instead of implicitly fetching extra assets.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    record = {"kind": "offline_native_component_import_probe_not_model_validation",
        "component": args.component, "at_utc": datetime.now(timezone.utc).isoformat(), "interpreter": sys.executable,
        "source_revision": revision, "module": module_name, "passed": False,
        "offline": True, "model_loaded": False, "simulator_started": False,
        "global_proxy_modified": False, "raw_import_logs_persisted": False}
    record["native_startup_cpu_only"] = args.native_startup_cpu_only
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            if args.native_startup_cpu_only:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["WORLD_SIZE"] = "1"
                os.environ["RANK"] = os.environ["LOCAL_RANK"] = "0"
                importlib.import_module("cosmos_framework.inference.common.init").init_script()
            module = importlib.import_module(module_name)
            record["symbols_present"] = {symbol: hasattr(module, symbol) for symbol in symbols}
            record["passed"] = all(record["symbols_present"].values())
    except Exception as error:
        record.update(error_type=type(error).__name__, missing_module=error.name if isinstance(error, ModuleNotFoundError) else None)
        message = str(error)
        record["public_failure_categories"] = {"native_startup_order": "init_script" in message,
            "gpu_context_or_library": any(word in message for word in ("CUDA", "libcuda", "libcudnn", "NVIDIA driver")),
            "distributed_initialization": any(word in message for word in ("process group", "NCCL", "distributed")),
            "torchvision_operator": "torchvision::" in message}
        record["source_trace_locations"] = [{"file": str(Path(frame.filename).relative_to(repo)), "line": frame.lineno,
            "function": frame.name} for frame in traceback.extract_tb(error.__traceback__) if Path(frame.filename).is_relative_to(repo)]
    check_source()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: record.get(key) for key in ("component", "passed", "error_type", "missing_module")}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
