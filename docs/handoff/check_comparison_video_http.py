"""Read-only actual MP4 seeking checks against an owned loopback preview."""
import argparse
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import quote
from prepare_mn_extension import ROOT, read, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preview = read(args.preview)
    args.output.resolve().relative_to(ROOT / "outputs")
    if preview["bind"] != "127.0.0.1" or preview["status"] != "listening" or preview["external_upload_performed"]:
        raise ValueError("owned_loopback_preview_required")
    def request(method, route, value=None):
        connection = HTTPConnection("127.0.0.1", preview["port"], timeout=30)
        try:
            connection.request(method, quote(route, safe="/"), headers={} if value is None else {"Range": value})
            response = connection.getresponse()
            return response.status, response.read(), response.getheader("Content-Range"), response.getheader("Content-Length"), response.getheader("Content-Type")
        finally:
            connection.close()
    rows = []
    for route, expected in preview["exposed_file_sha256"].items():
        if not route.endswith(".mp4"):
            continue
        path = (ROOT / route.lstrip("/")).resolve(); path.relative_to(ROOT)
        payload = path.read_bytes(); size = len(payload)
        if size < 64 or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("verified_nonempty_actual_video_required")
        checks = []
        ranges = [("bytes=0-63", 0, 63), (f"bytes={size//2}-{size//2+63}", size//2, size//2+63), ("bytes=-64", size-64, size-1)]
        for value, start, end in ranges:
            status, actual, content_range, length, media = request("GET", route, value)
            checks.append({"range": value, "status": status, "passed": status == 206 and actual == payload[start:end+1]
                and content_range == f"bytes {start}-{end}/{size}" and length == str(end-start+1) and media == "video/mp4"})
        status, body, _, length, media = request("HEAD", route)
        checks.append({"kind": "HEAD", "status": status, "passed": status == 200 and body == b"" and length == str(size) and media == "video/mp4"})
        status, body, content_range, _, _ = request("GET", route, f"bytes={size}-")
        checks.append({"kind": "outside_range", "status": status, "passed": status == 416 and body == b"" and content_range == f"bytes */{size}"})
        rows.append({"route": route, "sha256": expected, "checks": checks, "passed": all(c["passed"] for c in checks)})
    result = {"kind": "actual_video_loopback_seek_verification_not_remote_browser_claim", "at_utc": datetime.now(timezone.utc).isoformat(),
        "preview_receipt": str(args.preview.resolve().relative_to(ROOT)), "video_count": len(rows), "videos": rows,
        "passed": bool(rows) and all(row["passed"] for row in rows), "raw_headers_or_logs_saved": False,
        "external_upload_performed": False, "remote_port_forwarding_verified": False, "model_or_simulator_calls": 0}
    save(args.output, result)
    print({"passed": result["passed"], "actual_videos_checked": len(rows)})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
