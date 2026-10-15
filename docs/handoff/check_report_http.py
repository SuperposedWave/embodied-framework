"""Check only an owned loopback preview's explicit payloads and denied paths.

No browser/remote-forwarding claim, uploads, incoming headers or request logs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
from urllib.parse import quote
from prepare_mn_extension import ROOT, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.preview, args.output = args.preview.resolve(), args.output.resolve()
    args.preview.relative_to(ROOT / "outputs")
    args.output.relative_to(ROOT / "outputs")
    preview = read(args.preview)
    if preview["bind"] != "127.0.0.1" or preview["status"] != "listening" or preview["external_upload_performed"]:
        raise ValueError("owned_local_listening_preview_required")
    def get(route):
        client = HTTPConnection("127.0.0.1", preview["port"], timeout=30)
        try:
            client.request("GET", route)
            response = client.getresponse()
            return response.status, response.read(), response.getheader("Location")
        finally:
            client.close()
    checks = []
    for route, expected in preview["exposed_file_sha256"].items():
        status, payload, _ = get(quote(route, safe="/"))
        matched = hashlib.sha256(payload).hexdigest() == expected
        checks.append({"route": route, "status": status, "sha256_matches": matched, "passed": status == 200 and matched})
    status, _, location = get("/")
    root_passed = status == 302 and location == "/" + quote(str(Path(preview["report"]).relative_to(ROOT)), safe="/")
    denied = []
    for route in ("/AGENTS.md", "/.venv/pyvenv.cfg", "/%2e%2e/%2e%2e/etc/passwd", "/outputs/"):
        code, _, _ = get(route)
        denied.append({"route": route, "status": code, "passed": code == 404})
    result = {"kind": "instance_loopback_http_verification_not_remote_browser_access", "at_utc": datetime.now(timezone.utc).isoformat(),
        "preview_receipt": str(args.preview.relative_to(ROOT)), "exposed_file_count": len(checks),
        "selected_payload_checks": checks, "root_redirect_passed": root_passed, "denied_paths": denied,
        "passed": root_passed and all(row["passed"] for row in checks + denied),
        "remote_browser_port_forwarding_verified": False, "external_upload_performed": False}
    save(args.output, result)
    print(json.dumps({key: result[key] for key in ("passed", "exposed_file_count", "remote_browser_port_forwarding_verified")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
