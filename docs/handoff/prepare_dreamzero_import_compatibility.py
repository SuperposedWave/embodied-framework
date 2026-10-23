"""Pin a source-unpinned dependency in our OWN new DreamZero runtime only.

DeepSpeed 0.19.7 and Transformers 4.51.3 re-enter the latter's unfinished
modeling_utils import. The released 0.17.6 handles that optional OPT import.
No upstream source, native sampler, historical environment or driver changes.
Installer and import logs remain in memory; only public classifications escape.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib
import urllib.request
from probe_mn_components import ROOT, WORKSPACE, write, public_exception, categories
from prepare_mn_native_environment import PROXIES, public_failure, public_details
from prepare_robodojo_environment import bounded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-from", type=Path, required=True)
    parser.add_argument("--cuda-toolchain", type=Path)
    args = parser.parse_args()
    output, previous = args.output.resolve(), args.environment_from.resolve()
    for path in (output, previous):
        path.relative_to(ROOT / "outputs")
    target = WORKSPACE / "upstream/envs/dreamzero-mn-native"
    original = json.loads((previous / "receipt.json").read_text())
    if (not original["passed"] or original["component"] != "dreamzero"
            or Path(original["environment"]).resolve() != target.resolve()):
        raise ValueError("this_helpers_own_installed_new_dreamzero_environment_required")
    repo = WORKSPACE / "upstream/dreamzero-mn"
    from embodied_eval.native.assets_common import _revision
    revision = "ab790c198fbce33503358efbbd4187ce9a89adf3"
    _revision(repo, revision)
    project = repo / "pyproject.toml"
    if "deepspeed" not in tomllib.loads(project.read_text())["project"]["dependencies"]:
        raise ValueError("fixed_source_unpinned_deepspeed_declaration_required")
    output.mkdir(parents=True, exist_ok=False)
    proof = {"kind": "own_dreamzero_source_allowed_dependency_compatibility_not_model_validation",
        "passed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(), "environment": str(target),
        "previous_environment_receipt": str(previous / "receipt.json"), "source_revision": revision,
        "source_pyproject_sha256": hashlib.sha256(project.read_bytes()).hexdigest(),
        "dependency": "deepspeed==0.17.6", "declared_dependency_unpinned": True,
        "release_source": "https://github.com/deepspeedai/DeepSpeed/blob/v0.17.6/deepspeed/runtime/hybrid_engine.py",
        "reason": "0.19.7 narrow optional OPT import exception handling propagates Transformers lazy circular import",
        "attempts": [], "raw_logs_saved": False, "model_loaded": False, "native_sampler_modified": False,
        "global_proxy_or_driver_modified": False, "reference_source_modified": False,
        "historical_environment_modified": False, "own_new_environment_modified": False}
    try:
        toolkit = None
        if args.cuda_toolchain is not None:
            from prepare_cuda129_toolchain import private_toolchain
            toolkit = private_toolchain(args.cuda_toolchain)
            proof["private_cuda_toolchain_receipt"] = str(args.cuda_toolchain.resolve().relative_to(ROOT) / "receipt.json")
        inventory = [str(target / "bin/python"), "-c", "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ('torch','transformers','deepspeed')}))"]
        before = subprocess.run(inventory, capture_output=True, timeout=30, check=True)
        proof["versions_before"] = json.loads(before.stdout)
        if proof["versions_before"]["transformers"] != "4.51.3" or proof["versions_before"]["torch"].split("+")[0] != "2.8.0":
            raise ValueError("native_pinned_core_versions_required")
        for route in ("temporary_direct", "original_proxy_environment"):
            environment = dict(os.environ, UV_NO_PROGRESS="1", DS_BUILD_OPS="0", MAX_JOBS="8")
            if toolkit is not None:
                environment.update(CUDA_HOME=str(toolkit), PATH=str(toolkit / "bin") + os.pathsep + environment.get("PATH", ""))
            if route == "temporary_direct":
                for key in PROXIES:
                    environment.pop(key, None)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            else:
                opener = urllib.request.build_opener()
            try:
                with opener.open("https://pypi.org/pypi/deepspeed/0.17.6/json", timeout=30) as response:
                    metadata = json.load(response)
                artifacts = [item for item in metadata["urls"] if item["packagetype"] == "sdist"]
                if len(artifacts) != 1 or metadata["info"]["version"] != "0.17.6":
                    raise ValueError("one_fixed_public_release_source_distribution_required")
                artifact = artifacts[0]
                digest = artifact["digests"]["sha256"]
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError("public_release_hash_required")
                requirement = output / "hashed-requirement.txt"
                content = "deepspeed==0.17.6 --hash=sha256:" + digest + "\n"
                if requirement.exists() and requirement.read_text() != content:
                    raise ValueError("release_metadata_changed_between_routes")
                if not requirement.exists():
                    requirement.write_text(content)
                proof["publisher_source_distribution"] = {"filename": artifact["filename"], "sha256": digest, "bytes": artifact["size"]}
                command = [str(WORKSPACE / "upstream/envs/downloader/bin/uv"), "pip", "install", "--python", str(target / "bin/python"),
                    "--no-deps", "--require-hashes", "--no-build-isolation", "--requirements", str(requirement),
                    "--index-url", "https://pypi.org/simple"]
                code, _, error = bounded(command, environment, 300)
                attempt = {"route": route, "exit_code": code}
                if code:
                    attempt.update(failure_class=public_failure(error), public_details=public_details(error))
                proof["attempts"].append(attempt)
                if code == 0:
                    proof["own_new_environment_modified"] = True
                    break
            except Exception as error:
                proof["attempts"].append({"route": route, "exception_chain": public_exception(error)})
        if proof["own_new_environment_modified"]:
            after = subprocess.run(inventory, capture_output=True, timeout=30, check=True)
            proof["versions_after"] = json.loads(after.stdout)
            command = [str(target / "bin/python"), "-c",
                "from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel; import json; print(json.dumps({'native_base_vla_transformers_import':True}))"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
            if toolkit is not None:
                env.update(CUDA_HOME=str(toolkit), PATH=str(toolkit / "bin") + os.pathsep + env.get("PATH", ""))
            code, out, error = bounded(command, env, 120)
            proof["native_transformers_import_exit_code"] = code
            proof["import_failure_categories"] = categories((out + error).decode(errors="replace"))
            proof["passed"] = code == 0 and proof["versions_after"] == dict(proof["versions_before"], deepspeed="0.17.6")
    except Exception as error:
        proof["exception_chain"] = public_exception(error)
    _revision(repo, revision)
    proof["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "receipt.json", proof)
    print(json.dumps({key: proof.get(key) for key in ("passed", "versions_before", "versions_after", "native_transformers_import_exit_code")}))
    return 0 if proof["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
