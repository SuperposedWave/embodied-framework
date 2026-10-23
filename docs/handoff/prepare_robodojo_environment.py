"""Isolated RoboDojo dependencies from the fixed native install declarations.

Does NOT run install.sh, change system packages/drivers, accept new terms, or
install into a historical environment. Native fork installation is separate.
Only a completed receipt proves installation, never rendering or task success.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from download_stage4_assets import PROXY_KEYS
from prepare_mn_native_environment import write, public_failure, public_details
from run_cosmos_droid_bounded import stop_owned

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
TARGET = WORKSPACE / "upstream/envs/robodojo-mn-native"
PINS = ["numpy==1.26.0", "packaging==23.0", "typing_extensions==4.12.2", "filelock==3.13.1",
    "websockets==12.0", "scipy==1.15.3", "warp-lang==1.11.0", "click==8.1.7", "psutil==5.9.8",
    "wheel==0.45.1", "starlette==0.45.3", "virtualenv==20.30.0", "ipython<9", "onnx>=1.18,<1.22"]


def isaac_requirements(path):
    """Read only static native requirement lists; never execute setup.py."""
    values = {}
    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.List):
            return [literal(item) for item in node.elts]
        if isinstance(node, ast.JoinedStr):
            return "".join(literal(part.value) if isinstance(part, ast.FormattedValue) else literal(part) for part in node.values)
        raise ValueError("native static requirement expression changed")
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("INSTALL_REQUIRES", "SUPPORTED_ARCHS", "SUPPORTED_ARCHS_ARM"):
                values[name] = literal(node.value)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "INSTALL_REQUIRES":
            if not isinstance(node.op, ast.Add):
                raise ValueError("native requirement operation changed")
            values["INSTALL_REQUIRES"] += literal(node.value)
    return values.get("INSTALL_REQUIRES", [])


def bounded(command, env, seconds):
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=seconds)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        stop_owned(process)
        stdout, stderr = process.communicate()
        return "timeout", stdout, stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("base", "native"), default="base")
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--route-timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / "outputs")
    output.mkdir(parents=True, exist_ok=False)
    if args.route_timeout <= 0:
        raise ValueError("positive per-command route bound required")
    from embodied_eval.native.assets_common import _revision
    source = WORKSPACE / "upstream/RoboDojo"
    _revision(source, "266130a3ec41ba9b20b0e5648d2f3542f219c06c")
    declaration = (source / "scripts/install.sh").read_text()
    for pin in PINS:
        if pin not in declaration:
            raise ValueError("reviewed native dependency declaration changed")
    for exact in ('"isaacsim[all,extscache]==5.1.0"', "torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0"):
        if exact not in declaration:
            raise ValueError("reviewed native simulator/torch declaration changed")
    if args.continue_from:
        previous_path = args.continue_from.resolve()
        previous_path.relative_to(ROOT / "outputs")
        previous = json.loads((previous_path / "receipt.json").read_text())
        if previous["kind"] != "isolated_robodojo_dependency_preparation_not_runtime" or previous["environment"] != str(TARGET):
            raise ValueError("continuation must be this helper's own environment")
    elif TARGET.exists():
        raise ValueError("refuse to alter a pre-existing environment")
    elif args.phase != "base":
        raise ValueError("native phase requires helper-owned base preparation")
    uv = WORKSPACE / "upstream/envs/downloader/bin/uv"
    receipt = {"kind": "isolated_robodojo_dependency_preparation_not_runtime", "phase": args.phase,
        "environment": str(TARGET), "started_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
        "source_revision": "266130a3ec41ba9b20b0e5648d2f3542f219c06c",
        "native_install_declaration_sha256": hashlib.sha256(declaration.encode()).hexdigest(),
        "native_runtime_pins": PINS, "attempts": [], "simulator_started": False,
        "shared_driver_modified": False, "global_proxy_modified": False, "upstream_installer_executed": False,
        "historical_environments_modified": False, "raw_logs_saved": False, "runtime_eula_accepted": False}
    constraints = output / "constraints.txt"
    constraints.write_text("\n".join(PINS + ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]) + "\n")
    build_constraint = output / "build-constraints.txt"
    build_constraint.write_text("setuptools==80.9.0\n")
    base = [str(uv), "pip", "install", "--python", str(TARGET / "bin/python"),
        "--constraint", str(constraints), "--build-constraint", str(build_constraint)]
    commands = []
    if not (TARGET / "bin/python").is_file():
        commands.append(("create_python311_environment", [str(uv), "venv", str(TARGET),
            "--python", str(WORKSPACE / "upstream/envs/droid_sim-mn-native/bin/python")]))
    if args.phase == "base":
        commands += [("native_cuda128_torch_stack", base + ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0",
            "--index-url", "https://download.pytorch.org/whl/cu128"]),
            ("native_isaacsim51_and_extension_cache", base + ["isaacsim[all,extscache]==5.1.0", *PINS,
                "--index-url", "https://pypi.org/simple", "--extra-index-url", "https://pypi.nvidia.com", "--index-strategy", "unsafe-best-match"]),
            ("native_base_and_framework_requirements", base + ["--requirements", str(source / "scripts/requirements.txt"),
                "opencv-python-headless==4.11.0.86", "pillow", "matplotlib", "scikit-learn", "omegaconf",
                "jsonschema>=4.26,<5", "setuptools==80.9.0", "setuptools_scm>=6.2", "flatdict==4.0.1",
                "--index-url", "https://pypi.org/simple"])]
    else:
        if not args.sources:
            raise ValueError("native phase requires exact source preparation receipt")
        preparation = json.loads((args.sources.resolve() / "receipt.json").read_text())
        if not preparation["passed"]:
            raise ValueError("fixed native fork preparation incomplete")
        runtime = output / "native-install-source"
        runtime.mkdir()
        copied = {}
        for reference in preparation["records"]:
            original = Path(reference["path"])
            if reference.get("source_kind", "").startswith("tls_fixed_commit_archive"):
                from prepare_robodojo_isaac_archive import verify_archive_tree
                verify_archive_tree(reference)
            else:
                _revision(original, reference["revision"])
            dest = runtime / reference["name"]
            # Editable builds may write metadata; only these local install copies
            # are mutable. Original exact gitlink reference trees remain clean.
            shutil.copytree(original, dest, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            copied[reference["name"]] = dest
        receipt["mutable_install_copies_not_reference_edits"] = True
        receipt["curobo_package_metadata_version_override"] = "0.0.0.dev0 (no tag metadata in install copy; code pinned to d17b54ce, NOT a claimed official release number)"
        lab_requires = sorted(set(requirement for package in ("isaaclab", "isaaclab_tasks")
            for requirement in isaac_requirements(copied["IsaacLab"] / "source" / package / "setup.py")))
        if "starlette==0.49.1" not in lab_requires:
            raise ValueError("reviewed native final-pin conflict changed")
        lab_requires.remove("starlette==0.49.1")
        receipt["native_script_final_pin_overrides_lab_metadata"] = "starlette0.45.3 as native pin_runtime_deps, Lab metadata requests0.49.1; livestream unvalidated; disclose dependency check mismatch"
        commands += [("native_isaaclab_declared_dependencies_with_script_final_pin", base + lab_requires + ["--index-url", "https://pypi.org/simple"]),
            ("native_isaaclab23_fork_editable_install_copies", base + [argument for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks")
                for argument in ("--editable", str(copied["IsaacLab"] / "source" / package))]
                + ["--no-deps", "--index-url", "https://pypi.org/simple"]),
            ("native_curobo_v2_cuda12_fork", base + ["--editable", str(copied["curobo"]) + "[cu12]", "--no-build-isolation",
                "--index-url", "https://pypi.org/simple"])]
    passed = True
    for label, command in commands:
        success = False
        for route in ("temporary_direct", "original_proxy_environment"):
            env = os.environ.copy()
            if route == "temporary_direct":
                for key in PROXY_KEYS:
                    env.pop(key, None)
            env.update(UV_NO_PROGRESS="1", UV_HTTP_TIMEOUT="120", MAX_JOBS="8", PYTHONDONTWRITEBYTECODE="1")
            if args.phase == "native":
                env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO"] = "0.0.0.dev0"
            code, _, stderr = bounded(command, env, args.route_timeout)
            attempt = {"command": label, "route": route, "exit_code": code}
            if code != 0:
                attempt.update(failure_class=public_failure(stderr), public_details=public_details(stderr))
            receipt["attempts"].append(attempt)
            if code == 0:
                success = True
                break
        if not success:
            passed = False
            break
    receipt.update(passed=passed, finished_at_utc=datetime.now(timezone.utc).isoformat())
    write(output / "receipt.json", receipt)
    print(json.dumps({"phase": args.phase, "passed": passed, "environment": str(TARGET)}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
