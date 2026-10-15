"""Selected loopback report/video snapshot with safe single byte ranges.

Keep the original report server behavior; only explicitly linked MP4 files gain
video MIME and range responses. No workspace listing, secrets, or public bind.
"""
import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

import serve_report


def byte_range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()) or size < 1:
        raise ValueError("single_byte_range_required")
    left, right = match.groups()
    if left:
        start = int(left)
        end = min(int(right) if right else size - 1, size - 1)
        if start >= size or end < start:
            raise ValueError("unsatisfiable_range")
    else:
        count = int(right)
        if count < 1:
            raise ValueError("positive_suffix_required")
        start, end = max(0, size - count), size - 1
    return start, end


def handler_for(report_route, routes):
    base = serve_report.handler_for(report_route, routes)
    class VideoHandler(base):
        def respond(self, head):
            path = unquote(urlsplit(self.path).path)
            if path not in routes or routes[path][1] != "video/mp4":
                return super().respond(head)
            payload, media = routes[path]
            size = len(payload)
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range") if not head else None
            if requested:
                try:
                    start, end = byte_range(requested, size)
                    status = 206
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(status)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head:
                self.wfile.write(payload[start:end + 1])
    return VideoHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("valid_local_port_required")
    # Scope is local to this process; the historical server source stays intact.
    serve_report.SUFFIXES = serve_report.SUFFIXES | {".mp4"}
    route, routes, hashes = serve_report.catalog(args.report)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream, ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(route, routes)) as server:
        receipt = {"kind": "explicit_report_and_video_loopback_snapshot_not_public_hosting",
            "at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "bind": "127.0.0.1", "port": args.port,
            "status": "listening", "report": str(args.report.resolve()),
            "instance_local_url": f"http://127.0.0.1:{args.port}{quote(route, safe='/')}",
            "remote_browser_requires_port_forwarding": True, "external_upload_performed": False,
            "directory_listing": False, "exposed_file_sha256": hashes, "single_video_byte_ranges": True,
            "request_headers_queries_and_logs_saved": False}
        json.dump(receipt, stream, ensure_ascii=False, indent=2); stream.write("\n"); stream.flush()
        print(json.dumps({key: value for key, value in receipt.items() if key != "exposed_file_sha256"}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
