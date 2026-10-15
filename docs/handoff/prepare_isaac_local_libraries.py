"""Extract hash-pinned Ubuntu GLU/OpenGL libraries privately; never apt install.

Uses the host's existing package metadata, direct-first verified downloads and
dpkg-deb extraction (no maintainer scripts). Shared system/driver stay unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from download_stage4_assets import transfer
from prepare_mn_native_environment import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata, records = [], []
    for package in ("libglu1-mesa", "libopengl0"):
        result = subprocess.run(["apt-cache", "show", package], capture_output=True, text=True, check=True)
        stanza = result.stdout.split("\n\n", 1)[0]
        fields = dict(line.split(": ", 1) for line in stanza.splitlines() if ": " in line and not line.startswith(" "))
        if fields["Architecture"] != "amd64" or fields["Origin"] != "Ubuntu":
            raise ValueError("only host-native official Ubuntu amd64 libraries allowed")
        item = {"relative": Path(fields["Filename"]).name, "size": int(fields["Size"]),
                "sha256": fields["SHA256"], "url": "https://archive.ubuntu.com/ubuntu/" + fields["Filename"]}
        metadata.append({"package": package, "version": fields["Version"], "filename": fields["Filename"],
                         "bytes": item["size"], "sha256": item["sha256"]})
        record = transfer(args.output / "packages", item, 60)
        records.append(record)
        if record["status"] == "failed":
            break
        subprocess.run(["dpkg-deb", "--extract", str(args.output / "packages" / item["relative"]),
                        str(args.output / "library-root")], capture_output=True, check=True, timeout=60)
    passed = len(records) == 2 and all(row["status"] != "failed" for row in records)
    library_dir = args.output / "library-root/usr/lib/x86_64-linux-gnu"
    libraries = []
    if passed:
        for path in sorted(library_dir.glob("*.so*")):
            libraries.append({"file": str(path.relative_to(args.output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        if not (library_dir / "libGLU.so.1").is_file() or not (library_dir / "libOpenGL.so.0").is_file():
            raise ValueError("required native shared libraries absent after extraction")
    receipt = {"kind": "private_native_shared_libraries_not_system_install_or_runtime_validation",
        "at_utc": datetime.now(timezone.utc).isoformat(), "passed": passed, "packages": metadata,
        "download_records": records, "libraries": libraries, "library_directory": str(library_dir),
        "system_packages_changed": False, "shared_driver_modified": False,
        "global_proxy_modified": False, "maintainer_scripts_executed": False}
    write(args.output / "receipt.json", receipt)
    print(json.dumps({key:receipt[key] for key in ("passed", "packages", "library_directory")}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
