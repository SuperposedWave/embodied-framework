"""Install native-fork dependencies in a NEW Linux environment, not old venvs.

Uses the pinned project's declared dependencies and its optional LeRobot group.
No source edits, global driver/proxy changes or credential-bearing logs. A
solver/install failure is evidence of missing runtime readiness, not a pass.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds-per-route", type=int, default=600)
    parser.add_argument("--compatible-inference-fallback", action="store_true",
        help="Explicit dependency deviation: source-supported legacy dataset import, not training readiness")
    args = parser.parse_args()
    if args.max_seconds_per_route < 1:
        raise ValueError("positive bounded install timeout required")
    args.output.mkdir(parents=True, exist_ok=False)
    target = WORKSPACE / "upstream/envs" / ("pi05-arx-mn-inference" if args.compatible_inference_fallback else "pi05-arx-mn")
    if target.exists():
        raise ValueError("refuse to alter an existing environment")
    sys.path.insert(0, str(ROOT))
    from embodied_eval.native.assets_common import _revision
    repo = WORKSPACE / "upstream/xpolicy-reference-mn"
    _revision(repo, "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2")
    source = repo / "policy/Pi_05/openpi"
    project = tomllib.loads((source / "pyproject.toml").read_text())
    client = tomllib.loads((source / "packages/openpi-client/pyproject.toml").read_text())
    requirements = [dep for dep in project["project"]["dependencies"] if dep != "openpi-client"]
    requirements += client["project"]["dependencies"] + ["pytest>=8.3.4"]
    if args.compatible_inference_fallback:
        # The reviewed loader explicitly supports BOTH new lerobot.datasets
        # and legacy lerobot.common.datasets. Match the pinned legacy revision
        # already used in official-openpi's original Linux inference stack.
        requirements += ["lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5", "rerun-sdk==0.21.0"]
        # FSQ tokenizer imports chex eagerly. Newer optax no longer supplies
        # that implicit dependency. Keep these two at the native lock versions.
        native_lock = tomllib.loads((source / "uv.lock").read_text())
        requirements += [package["name"] + "==" + package["version"] for package in native_lock["package"] if package["name"] in ("optax", "chex")]
        requirements += ["jsonschema>=4.26,<5"]
    else:
        requirements += project["dependency-groups"]["lerobot"]
    constraints = project["tool"]["uv"]["override-dependencies"]
    # Generated installation inputs are evidence, not edits to the native repo.
    req_path, constraint_path = args.output / "requirements.txt", args.output / "constraints.txt"
    req_path.write_text("\n".join(requirements) + "\n")
    constraint_path.write_text("\n".join(constraints) + "\n")
    uv = WORKSPACE / "upstream/envs/downloader/bin/uv"
    python = "/root/.local/share/uv/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
    create = subprocess.run([str(uv), "venv", "--python", python, str(target)], capture_output=True, timeout=60)
    receipt = {"kind": "new_linux_dependency_environment_not_model_validation", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": str(target), "source_revision": "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2", "venv_exit_code": create.returncode,
        "attempts": [], "requirements": requirements, "constraints": constraints, "passed": False,
        "historical_environments_changed": False, "global_proxy_modified": False, "global_driver_changed": False,
        "mac_venv_used": False, "model_loaded": False, "real_evaluation": False}
    receipt["dependency_deviation"] = "legacy LeRobot pinned 0cf8648 + rerun-sdk 0.21.0; explicit native import fallback; inference only, training NOT verified" if args.compatible_inference_fallback else None
    if create.returncode == 0:
        for route in ("temporary_direct", "original_proxy_environment"):
            env = os.environ.copy()
            if route == "temporary_direct":
                for key in PROXY_KEYS:
                    env.pop(key, None)
            command = [str(uv), "pip", "install", "--python", str(target / "bin/python"), "--index-url", "https://pypi.org/simple",
                "--requirements", str(req_path.resolve()), "--constraints", str(constraint_path.resolve())]
            try:
                result = subprocess.run(command, env=env, capture_output=True, timeout=args.max_seconds_per_route)
                attempt = {"route": route, "exit_code": result.returncode}
                # Classify only public dependency conflicts. Do not persist raw installer logs.
                if b"No solution found" in result.stderr:
                    attempt["failure_class"] = "declared_dependency_constraints_unsatisfiable"
                    # Persist only known public package/version facts, no URLs,
                    # installer environment dumps or arbitrary error messages.
                    attempt["public_constraint_lines"] = [line.strip() for line in result.stderr.decode("utf-8", errors="replace").splitlines()
                        if any(word in line for word in ("rerun-sdk", "numpy>=", "lerobot=="))
                        and not any(word in line.lower() for word in ("http", "token", "password", "proxy", "key="))]
                elif result.returncode:
                    attempt["failure_class"] = "dependency_install_failed_no_raw_log_saved"
                receipt["attempts"].append(attempt)
                if result.returncode == 0:
                    receipt["passed"] = True
                    break
            except subprocess.TimeoutExpired:
                receipt["attempts"].append({"route": route, "failure_class": "bounded_dependency_install_timeout"})
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("passed", "environment", "attempts")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
