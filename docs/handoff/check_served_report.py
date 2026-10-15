"""Verify loopback preview bytes and non-exposure, no browser-access claim."""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
from pathlib import Path
from urllib.parse import quote, urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preview = json.loads(args.preview_receipt.read_text())
    if preview["bind"] != "127.0.0.1" or preview["status"] != "listening":
        raise ValueError("require an existing loopback preview")
    checks = []
    def request(path, method="GET"):
        connection = http.client.HTTPConnection("127.0.0.1", preview["port"], timeout=10)
        try:
            connection.request(method, path)
            result = connection.getresponse()
            return result.status, dict(result.getheaders()), result.read()
        finally:
            connection.close()
    for route, expected in preview["exposed_file_sha256"].items():
        status, headers, body = request(quote(route, safe="/"))
        actual = hashlib.sha256(body).hexdigest()
        checks.append({"route": route, "status": status, "sha256_matches": actual == expected,
            "passed": status == 200 and actual == expected})
    status, headers, body = request("/")
    root_passed = status == 302 and headers.get("Location") == urlsplit(preview["instance_local_url"]).path
    private = []
    for route in ("/docs/", "/outputs/", "/AGENTS.md", "/.env", "/.venv/pyvenv.cfg", "/docs/handoff/serve_report.py",
        "/outputs/../../AGENTS.md", "/outputs/%2e%2e/%2e%2e/AGENTS.md"):
        status, _, _ = request(route)
        private.append({"route": route, "status": status, "passed": status == 404})
    receipt = {"kind": "instance_loopback_http_verification_not_remote_browser_access", "at_utc": datetime.now(timezone.utc).isoformat(),
        "preview_receipt": str(args.preview_receipt), "exposed_file_count": len(checks), "selected_payload_checks": checks,
        "root_redirect_passed": root_passed, "non_exposed_route_checks": private,
        "passed": root_passed and all(row["passed"] for row in checks + private),
        "visual_browser_check_performed": False, "remote_port_forwarding_verified": False,
        "global_proxy_modified": False, "external_upload_performed": False}
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("passed", "exposed_file_count", "remote_port_forwarding_verified")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
