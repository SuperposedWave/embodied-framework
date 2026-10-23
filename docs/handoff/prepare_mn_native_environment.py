"""Prepare NEW component-local Linux environments from pinned source metadata.

Never executes upstream installation shell scripts, accepts runtime EULAs,
changes drivers/proxies, edits a reference checkout or alters an existing venv.
Installation is not inference, rendering, scoring or dependency certification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
COMPONENTS = {
    "cosmos3": ("cosmos-framework-mn", "cf5d68c00d97ccd2480a2320ed652b92dec63102", "", "3.13"),
    "dreamzero": ("dreamzero-mn", "ab790c198fbce33503358efbbd4187ce9a89adf3", "", "3.11"),
    "droid_sim": ("sim-evals-mn", "3a6b0e8177f696947f49c082728f10ef5133ede1", "", "3.11"),
    "openwam_arx": ("xpolicy-reference-mn", "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2", "policy/OpenWAM/OpenWAM", "3.11"),
}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def public_failure(stderr):
    """Only permit dependency/version facts; never persist arbitrary logs."""
    if b"No solution found" in stderr:
        return "dependency_constraints_unsatisfiable"
    if b"Failed to download" in stderr or b"error sending request" in stderr:
        return "dependency_network_or_object_download_failed"
    if b"Failed to build" in stderr:
        return "dependency_build_failed"
    return "installer_failed_raw_log_not_saved"


def public_details(stderr):
    output = stderr.decode("utf-8", errors="replace")
    packages = re.findall(r"Failed to (?:build|download) [`']([A-Za-z0-9_.+<>=-]+)[`']", output)
    return {"failed_public_packages": packages,
        "missing_public_modules": re.findall(r"ModuleNotFoundError: No module named ['\"]([A-Za-z0-9_.]+)['\"]", output),
        "backend_exception_types": sorted(set(re.findall(r"(?:^|\n)\s*([A-Za-z][A-Za-z0-9_]*(?:Error|Exception)):", output))),
        "no_solution_found": "No solution found" in output,
        "wheel_bootstrap_failure": "wheel-stub" in output or "wheel_stub" in output,
        "no_compatible_wheel": "No compatible wheel" in output,
        "compiler_cuda_missing": "CUDA_HOME" in output or "nvcc not found" in output,
        "runtime_eula_prompt_seen": "EULA" in output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-timeout", type=int, default=900)
    parser.add_argument("--complete-framework-dependency-from", type=Path,
        help="Only add framework jsonschema to this helper's own successfully installed new environment")
    parser.add_argument("--retry-dependency-install-from", type=Path,
        help="Retry this helper's own incomplete new environment, retaining the original failure receipt")
    parser.add_argument("--legacy-build-backend", action="store_true",
        help="Explicit build-only setuptools80.9 constraint for source-pinned flatdict4.0.1; not a native runtime version change")
    parser.add_argument("--bootstrap-native-flatdict", action="store_true",
        help="Build exact source-lock flatdict artifact with hash checking and isolated setuptools80.9 before frozen sync")
    args = parser.parse_args()
    if args.route_timeout < 1:
        raise ValueError("positive timeout required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    component = args.component
    if (args.legacy_build_backend or args.bootstrap_native_flatdict) and component != "droid_sim":
        raise ValueError("reviewed legacy build constraint is DROID-only")
    name, revision, relative, python = COMPONENTS[component]
    repo = WORKSPACE / "upstream" / name
    source = repo / relative
    sys.path.insert(0, str(ROOT))
    from embodied_eval.native.assets_common import _revision
    _revision(repo, revision)
    target = WORKSPACE / "upstream/envs" / (component + "-mn-native")
    complete_framework = args.complete_framework_dependency_from is not None
    retry_native = args.retry_dependency_install_from is not None
    if complete_framework and retry_native:
        raise ValueError("select at most one completion mode")
    if complete_framework:
        previous = json.loads((args.complete_framework_dependency_from / "receipt.json").read_text())
        if previous["component"] != component or not previous["passed"] or Path(previous["environment"]).resolve() != target.resolve():
            raise ValueError("require this helper's verified successful new environment")
    elif retry_native:
        previous = json.loads((args.retry_dependency_install_from / "receipt.json").read_text())
        if previous["component"] != component or previous["passed"] or Path(previous["environment"]).resolve() != target.resolve() or previous["source_revision"] != revision:
            raise ValueError("require this helper's failed new environment with identical source identity")
    elif target.exists():
        raise ValueError("refuse to alter an existing environment")
    project = tomllib.loads((source / "pyproject.toml").read_text())
    uv = WORKSPACE / "upstream/envs/downloader/bin/uv"
    receipt = {"kind": "new_linux_native_dependency_install_not_runtime_validation",
        "component": component, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision, "environment": str(target), "python_requested": python,
        "source_pyproject_sha256": hashlib.sha256((source / "pyproject.toml").read_bytes()).hexdigest(),
        "attempts": [], "passed": False, "model_loaded": False, "simulator_started": False,
        "runtime_eula_accepted": False, "global_proxy_modified": False, "global_driver_modified": False,
        "historical_environments_changed": False, "source_modified": False, "mac_venv_used": False}
    if retry_native:
        receipt["retry_of_incomplete_own_environment"] = str(args.retry_dependency_install_from / "receipt.json")
    if complete_framework:
        command = [str(uv), "pip", "install", "--python", str(target / "bin/python"), "jsonschema>=4.26,<5", "--index-url", "https://pypi.org/simple"]
        receipt["dependency_source"] = "framework_pyproject_jsonschema_requirement_only"
        receipt["own_new_environment_changed"] = True
        receipt["native_requirement_completion_receipt"] = str(args.complete_framework_dependency_from / "receipt.json")
    elif component in ("cosmos3", "droid_sim"):
        # A copied, byte-identical source project/lock permits uv sync to create
        # local state without adding .venv or changing locks inside source repos.
        runtime = output / "runtime-project"
        runtime.mkdir()
        for filename in ("pyproject.toml", "uv.lock"):
            (runtime / filename).write_bytes((source / filename).read_bytes())
        receipt["source_lock_sha256"] = hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest()
        receipt["dependency_source"] = "byte_identical_native_project_and_frozen_lock"
        if args.legacy_build_backend:
            # flatdict4.0.1 setup.py imports pkg_resources, removed in setuptools82.
            # Only the GENERATED installation project changes; native source and
            # lock/runtime versions remain pinned and byte-identical.
            copied = (runtime / "pyproject.toml").read_text()
            copied = copied.replace("[tool.uv]\n", '[tool.uv]\nbuild-constraint-dependencies = ["setuptools==80.9.0"]\n', 1)
            (runtime / "pyproject.toml").write_text(copied)
            receipt["dependency_source"] = "native_frozen_lock_with_explicit_build_only_constraint"
            receipt["build_only_deviation"] = "setuptools80.9.0 for flatdict4.0.1 pkg_resources setup; no runtime package versions changed"
        command = [str(uv), "sync", "--project", str(runtime), "--frozen", "--no-install-project", "--no-dev", "--python", python]
        if component == "cosmos3":
            command += ["--all-extras", "--group", "cu128-train", "--group", "policy-server"]
            receipt["selected_native_groups"] = ["all-extras", "cu128-train", "policy-server"]
            receipt["deviation_from_documented_default"] = "source-defined CUDA12.8/Torch2.10 group instead of documented CUDA13/Torch2.13; GPU compatibility unverified"
    else:
        requirements = project["project"]["dependencies"]
        req = output / "requirements.txt"
        req.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        receipt["dependency_source"] = "native_declared_requirements_without_source_package_install"
        receipt["declared_requirements"] = requirements
        command = [str(uv), "pip", "install", "--python", str(target / "bin/python"), "--requirements", str(req),
            "--index-url", "https://pypi.org/simple"]
        if component == "dreamzero":
            command += ["--extra-index-url", "https://download.pytorch.org/whl/cu129", "--index-strategy", "unsafe-best-match"]
            receipt["native_extra_index"] = "https://download.pytorch.org/whl/cu129"
    for route in ("temporary_direct", "original_proxy_environment"):
        env = os.environ.copy()
        if route == "temporary_direct":
            for key in PROXIES:
                env.pop(key, None)
        env["UV_PROJECT_ENVIRONMENT"] = str(target)
        env["UV_HTTP_TIMEOUT"] = "120"
        env["UV_NO_PROGRESS"] = "1"
        # Avoid a build-time CUDA extension compile unless the native dependency
        # resolver actually requires it; no system compiler/driver is installed.
        env["MAX_JOBS"] = "8"
        try:
            if args.bootstrap_native_flatdict:
                if not retry_native:
                    raise ValueError("bootstrap requires own failed native installation")
                lock = tomllib.loads((source / "uv.lock").read_text())
                flatdict = next(p for p in lock["package"] if p["name"] == "flatdict")
                requirement = output / "flatdict-hashed-requirement.txt"
                constraint = output / "flatdict-build-constraint.txt"
                if not requirement.exists():
                    requirement.write_text("flatdict==" + flatdict["version"] + " --hash=" + flatdict["sdist"]["hash"] + "\n")
                    constraint.write_text("setuptools==80.9.0\n")
                bootstrap = subprocess.run([str(uv), "pip", "install", "--python", str(target / "bin/python"),
                    "--require-hashes", "--no-deps", "--requirements", str(requirement), "--build-constraints", str(constraint),
                    "--index-url", "https://pypi.org/simple"], env=env, capture_output=True, timeout=args.route_timeout)
                receipt["attempts"].append({"route": route, "operation": "hashed_native_flatdict_bootstrap",
                    "exit_code": bootstrap.returncode, "public_details": public_details(bootstrap.stderr)})
                if bootstrap.returncode:
                    continue
                receipt["build_only_deviation"] = "exact native-lock flatdict4.0.1 sdist hash verified, build isolation setuptools80.9.0; runtime versions unchanged"
            if not complete_framework and component not in ("cosmos3", "droid_sim") and not target.exists():
                create = subprocess.run([str(uv), "venv", "--python", python, str(target)], env=env, capture_output=True, timeout=120)
                if create.returncode:
                    receipt["attempts"].append({"route": route, "operation": "create_venv", "exit_code": create.returncode,
                        "failure_class": public_failure(create.stderr)})
                    continue
            result = subprocess.run(command, env=env, capture_output=True, timeout=args.route_timeout)
            attempt = {"route": route, "exit_code": result.returncode}
            if result.returncode:
                attempt["failure_class"] = public_failure(result.stderr)
                attempt["public_details"] = public_details(result.stderr)
            receipt["attempts"].append(attempt)
            if result.returncode == 0:
                receipt["passed"] = True
                break
        except subprocess.TimeoutExpired:
            receipt["attempts"].append({"route": route, "failure_class": "bounded_install_timeout"})
        print(component + ": " + route + " finished, incomplete", flush=True)
    if receipt["passed"]:
        inventory = subprocess.run([str(target / "bin/python"), "-c",
            "import importlib.metadata as m,json; print(json.dumps(sorted([(d.metadata['Name'],d.version) for d in m.distributions()])))"],
            capture_output=True, timeout=60)
        if inventory.returncode == 0:
            write(output / "installed-package-versions.json", json.loads(inventory.stdout))
        check = subprocess.run([str(uv), "pip", "check", "--python", str(target / "bin/python")], capture_output=True, timeout=60)
        receipt["installed_dependency_check_exit_code"] = check.returncode
        receipt["runtime_validated"] = False
    _revision(repo, revision)
    receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("component", "passed", "attempts", "environment")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
