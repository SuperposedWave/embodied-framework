"""Reviewed stage-4 registry metadata and preflight; simulator imports remain lazy."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from . import contracts as c
from .validation import require
from .adapters import openpi_libero_spec as pi, vlabench_spec as vla
from .adapters import openpi_vlabench_spec as pivla
from . import robotwin_pairs

PAIRS = {("openpi_libero", "libero_openpi"), ("openwam_vlabench", "vlabench"),
         ("openpi_vlabench", "vlabench_openpi")} | robotwin_pairs.PAIRS
_VERIFIED = {}


def validate(config):
    if (config.model.adapter, config.benchmark.adapter) in robotwin_pairs.PAIRS:
        return robotwin_pairs.validate(config)
    require(len(config.episodes) == 1, "pilot_scope", "episodes", "stage-4 initial profiles select one fixed episode")
    require(config.model.checkpoint_dir is not None, "missing_source", "checkpoint_dir", "explicit matching checkpoint required")
    require(config.model.host in ("127.0.0.1", "localhost") and 1 <= config.model.port <= 65535
            and config.model.request_timeout_s > 0, "invalid_endpoint", "model", "local policy endpoint required")
    require(config.profile.chunk_mode == "adapter_owned" and config.profile.execution_horizon_steps == 1,
            "pilot_profile_mismatch", "profile", "native adapter owns the action cache")
    ep = config.episodes[0]
    if config.model.adapter == "openpi_libero":
        require(config.model.openpi_repo is not None and config.benchmark.libero_repo is not None,
                "missing_source", "adapters", "explicit openpi and its LIBERO submodule paths required")
        require(ep.seed == 7 and ep.budgets.env_steps == 220 and ep.budgets.sim_time_s == 11
                and ep.budgets.model_calls == 220, "pilot_scope", "episode", "openpi fixes seed 7 and 220 spatial ticks")
    else:
        model_repo = config.model.openpi_repo if config.model.adapter == "openpi_vlabench" else config.model.openwam_repo
        require(model_repo is not None and config.benchmark.vlabench_repo is not None,
                "missing_source", "adapters", "explicit matching model and VLABench paths required")
        require(config.benchmark.task_name == "select_fruit" and config.benchmark.track == "track_1_in_distribution"
                and config.benchmark.trial_index == 0, "pilot_scope", "benchmark", "reviewed VLABench profile is frozen track1/select_fruit/episode0")
        require(ep.budgets.env_steps == 300 and ep.budgets.model_calls == 300 and ep.budgets.sim_time_s is None,
                "pilot_scope", "episode", "VLABench allows 300 native decisions; period is measured from the environment")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(root, expected):
    root = Path(root).resolve()
    value = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    require(value.returncode == 0 and value.stdout.strip() == expected,
            "source_revision_mismatch", str(root), f"expected {expected}")
    clean = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], capture_output=True)
    require(clean.returncode == 0, "source_tree_modified", str(root), "stage-4 native source must match its pinned commit")


def _file(path, expected):
    require(path.is_file(), "asset_missing", str(path), "required native asset unavailable")
    actual = _sha(path)
    require(actual == expected, "asset_hash_mismatch", str(path), "native asset differs from reviewed SHA-256")
    return actual


def _key(config):
    return (config.model.adapter, config.model.checkpoint_dir, config.model.openpi_repo,
            config.model.openwam_repo, config.benchmark.libero_repo, config.benchmark.vlabench_repo)


def _hf_objects(checkpoint, manifest):
    digests = {}
    for item in json.loads(manifest.read_text())["objects"]:
        path = checkpoint / item["relative"]
        require(path.resolve().is_relative_to(checkpoint) and path.is_file()
                and path.stat().st_size == item["size"], "asset_missing", item["relative"],
                "pinned inference checkpoint object missing or size mismatch")
        if "sha256" in item:
            digests[item["relative"]] = _file(path, item["sha256"])
        else:
            blob = hashlib.sha1(f"blob {item['size']}\0".encode())
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    blob.update(block)
                    sha.update(block)
            require(blob.hexdigest() == item["git_blob"], "asset_hash_mismatch", item["relative"],
                    "checkpoint metadata differs from the pinned HF git blob")
            digests[item["relative"]] = sha.hexdigest()
    digests["pinned_hf_object_manifest"] = _sha(manifest)
    return digests


def verify(config):
    if (config.model.adapter, config.benchmark.adapter) in robotwin_pairs.PAIRS:
        return robotwin_pairs.verify(config)
    # A failed recheck must not leave a successful provenance marker behind.
    _VERIFIED.pop(_key(config), None)
    checkpoint = Path(config.model.checkpoint_dir).resolve()
    digests = {}
    if config.model.adapter == "openpi_libero":
        _revision(config.model.openpi_repo, pi.MODEL_REVISION)
        _revision(config.benchmark.libero_repo, pi.BENCHMARK_REVISION)
        metadata_file = Path(__file__).parent / "adapters/pi05_libero_objects.json"
        objects = json.loads(metadata_file.read_text())["objects"]
        try:
            import google_crc32c
        except ImportError as error:
            raise ValueError("openpi checkpoint verification needs the real-client extra (google-crc32c)") from error
        for item in objects:
            relative = item["name"].removeprefix("checkpoints/pi05_libero/")
            path = checkpoint / relative
            require(path.is_file() and path.stat().st_size == int(item["size"]),
                    "asset_missing", str(path), "checkpoint object missing or size mismatch")
            checksum = google_crc32c.Checksum()
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
                    sha.update(block)
            require(base64.b64encode(checksum.digest()).decode() == item["crc32c"],
                    "asset_hash_mismatch", str(path), "checkpoint differs from pinned official GCS CRC32C")
            digests[relative] = sha.hexdigest()
        require(digests[pi.NORMALIZATION_FILE] == pi.NORMALIZATION_SHA256,
                "asset_hash_mismatch", pi.NORMALIZATION_FILE, "openpi normalization mismatch")
        from .adapters.openwam_libero_spec import TASK_STEM
        root = Path(config.benchmark.libero_repo).resolve()
        for directory, extension in (("bddl_files", ".bddl"), ("init_files", ".pruned_init")):
            relative = f"libero/libero/{directory}/libero_spatial/{TASK_STEM}{extension}"
            blob = subprocess.run(["git", "-C", str(root), "show", f"{pi.BENCHMARK_REVISION}:{relative}"],
                                  capture_output=True)
            require(blob.returncode == 0, "asset_missing", relative, "pinned LIBERO task asset unavailable")
            digests[relative] = _file(root / relative, hashlib.sha256(blob.stdout).hexdigest())
        digests["official_gcs_object_metadata"] = _sha(metadata_file)
    else:
        if config.model.adapter == "openpi_vlabench":
            _revision(config.model.openpi_repo, pivla.MODEL_REVISION)
            digests.update(_hf_objects(checkpoint, Path(__file__).parent / "adapters/pi05_vlabench_objects.json"))
            require(digests[pivla.NORMALIZATION_FILE] == pivla.NORMALIZATION_SHA256,
                    "asset_hash_mismatch", pivla.NORMALIZATION_FILE, "openpi VLABench normalization mismatch")
        else:
            _revision(config.model.openwam_repo, vla.MODEL_REVISION)
            for name, expected in (("checkpoint_step_6000.safetensors", vla.CHECKPOINT_SHA256),
                                   ("normalization_stats.npy", vla.NORMALIZATION_SHA256),
                                   ("tokenizer/google/umt5-xxl/spiece.model", "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
                                   ("tokenizer/google/umt5-xxl/tokenizer.json", "6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b")):
                digests[name] = _file(checkpoint / name, expected)
            digests["config.yaml"] = _file(checkpoint / "config.yaml", vla.CONFIG_SHA256)
        _revision(config.benchmark.vlabench_repo, vla.BENCHMARK_REVISION)
        package = Path(config.benchmark.vlabench_repo).resolve() / "VLABench"
        track = package / "configs/evaluation/tracks/track_1_in_distribution.json"
        digests["track_config"] = _file(track, vla.TRACK_SHA256)
        for required in ("assets/base/default.xml", "assets/obj", "assets/scenes"):
            require((package / required).exists(), "asset_missing", required, "VLABench scene/object assets required")
        digests["base_scene_xml"] = _sha(package / "assets/base/default.xml")
    _VERIFIED[_key(config)] = digests
    return digests


def make_adapters(config, output):
    if (config.model.adapter, config.benchmark.adapter) in robotwin_pairs.PAIRS:
        return robotwin_pairs.make_adapters(config, output)
    verify(config)
    if config.model.adapter == "openpi_libero":
        from .adapters.openpi_libero_model import OpenpiLiberoModel
        from .adapters.libero_openpi_benchmark import LiberoOpenpiBenchmark
        model = OpenpiLiberoModel(config.model, output)
        try:
            return model, LiberoOpenpiBenchmark(config.benchmark)
        except Exception:
            model.close()
            raise
    if config.model.adapter == "openpi_vlabench":
        from .adapters.openpi_vlabench_model import OpenpiVLABenchModel
        from .adapters.vlabench_openpi_benchmark import VLABenchOpenpiBenchmark
        return (OpenpiVLABenchModel(config.model, output, config.benchmark.vlabench_repo),
                VLABenchOpenpiBenchmark(config.benchmark))
    from .adapters.openwam_vlabench_model import OpenWAMVLABenchModel
    from .adapters.vlabench_benchmark import VLABenchBenchmark
    return OpenWAMVLABenchModel(config.model, output), VLABenchBenchmark(config.benchmark)


def make_profile(config, space, cap):
    if (config.model.adapter, config.benchmark.adapter) in robotwin_pairs.PAIRS:
        return robotwin_pairs.make_profile(config, space, cap)
    return _spec(config).profile(space, cap)


def _spec(config):
    return {"openpi_libero": pi, "openpi_vlabench": pivla, "openwam_vlabench": vla}[config.model.adapter]


def fallback_metrics(config):
    if (config.model.adapter, config.benchmark.adapter) in robotwin_pairs.PAIRS:
        return robotwin_pairs.fallback_metrics(config)
    if config.model.adapter == "openpi_libero":
        source = "LIBERO official environment done@f78abd6"
        names = {"success": "boolean"}
    else:
        source = "VLABench official native evaluator@cf588fe"
        names = {"success": "boolean", "progress_score": "score", "intention_score": "score"}
    return {name: c.Metric(value=None, unit=unit, source=source, valid=False, unavailable_reason="not_scored")
            for name, unit in names.items()}


def provenance(config):
    if (config.model.adapter, config.benchmark.adapter) in robotwin_pairs.PAIRS:
        return robotwin_pairs.provenance(config)
    native = _spec(config)
    return {"synthetic": False, "model_adapter": config.model.adapter,
            "benchmark_adapter": config.benchmark.adapter,
            "model_source_revision": native.MODEL_REVISION, "benchmark_source_revision": native.BENCHMARK_REVISION,
            "model_asset_revision": getattr(native, "HF_REVISION", None),
            "native_train_config": getattr(native, "TRAIN_CONFIG", None),
            "normalization_sha256": native.NORMALIZATION_SHA256,
            "server_identity_verified": False, "asset_checks_completed": _key(config) in _VERIFIED,
            "verified_asset_sha256": _VERIFIED.get(_key(config), {}),
            "scene_asset_origin_verified": False if config.benchmark.adapter in ("vlabench", "vlabench_openpi") else True}
