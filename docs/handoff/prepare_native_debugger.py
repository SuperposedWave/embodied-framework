"""Private small PyPI py-spy executable; no global installs or proxy changes."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

from probe_mn_components import ROOT, write


def fetch(url):
    if not url.startswith(("https://pypi.org/", "https://files.pythonhosted.org/")):
        raise ValueError("official_PyPI_endpoint_required")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=60) as response:
            return response.read(10 << 20), "direct_request_only"
    except Exception:
        with urllib.request.build_opener().open(url, timeout=60) as response:
            return response.read(10 << 20), "existing_proxy_fallback_request_only"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.output.relative_to(ROOT / "outputs")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata, route = fetch("https://pypi.org/pypi/py-spy/json")
    release = json.loads(metadata)
    candidates = [row for row in release["urls"] if row["filename"].endswith(".whl") and "manylinux" in row["filename"]
        and "x86_64" in row["filename"] and row["size"] < 10 << 20]
    if len(candidates) != 1:
        raise ValueError("one_small_official_Linux_x86_64_wheel_required")
    selected = candidates[0]
    payload, download_route = fetch(selected["url"])
    if len(payload) != selected["size"] or hashlib.sha256(payload).hexdigest() != selected["digests"]["sha256"]:
        raise ValueError("official_wheel_size_and_SHA256_required")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".data/scripts/py-spy")]
        if len(names) != 1:
            raise ValueError("one_executable_required")
        binary = archive.read(names[0])
    if not binary.startswith(b"\x7fELF"):
        raise ValueError("Linux_ELF_executable_required")
    path = args.output / "py-spy"
    with path.open("xb") as stream:
        stream.write(binary)
    path.chmod(0o755)
    write(args.output / "receipt.json", {"kind": "private_read_only_stack_tool_not_evaluation", "passed": True,
        "at_utc": datetime.now(timezone.utc).isoformat(), "version": release["info"]["version"], "wheel": selected["filename"],
        "wheel_bytes": len(payload), "wheel_sha256": selected["digests"]["sha256"], "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(), "metadata_route": route, "download_route": download_route,
        "global_proxy_or_environment_modified": False, "model_weights_downloaded": False, "credentials_saved": False})
    print(json.dumps({"version": release["info"]["version"], "wheel_sha256_verified": True, "bytes": len(payload)}), flush=True)


if __name__ == "__main__":
    main()
