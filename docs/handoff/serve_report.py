"""Loopback-only preview of one report and its explicitly linked local files.

No directory listing, arbitrary workspace access, external upload or report
rewrite. A remote instance needs platform forwarding or an SSH tunnel before
the user's browser can reach this service. Query strings/headers are not logged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SUFFIXES = {".html", ".json", ".md", ".txt", ".css", ".svg", ".png", ".jpg", ".jpeg", ".webp"}


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if value and ((tag == "a" and name == "href") or name in ("src", "poster") or (tag == "link" and name == "href")):
                parsed = urlsplit(value)
                if not parsed.scheme and not parsed.netloc and parsed.path:
                    self.paths.append(value)


def catalog(report, root=ROOT):
    root, report = root.resolve(), report.resolve()
    relative = report.relative_to(root)
    if report.suffix != ".html":
        raise ValueError("select an HTML report inside the framework")
    report_route = "/" + relative.as_posix()
    parser = References()
    parser.feed(report.read_text(encoding="utf-8"))
    routes, hashes, missing = {}, {}, []
    for route in [report_route, *[unquote(urlsplit(urljoin(quote(report_route, safe="/"), value)).path) for value in parser.paths]]:
        candidate = (root / route.lstrip("/")).resolve()
        try:
            child = candidate.relative_to(root)
        except ValueError:
            missing.append(route)
            continue
        if (any(part.startswith(".") for part in child.parts) or candidate.suffix not in SUFFIXES or not candidate.is_file()
                or not (candidate.parent == report.parent or child.parts[0] in ("docs", "outputs"))):
            missing.append(route)
            continue
        # Snapshot selected artifacts, not a filesystem server with traversal.
        payload = candidate.read_bytes()
        media = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix in (".md", ".txt"):
            media = "text/plain"
        if media.startswith("text/") or media == "application/json":
            media += "; charset=utf-8"
        routes[route] = (payload, media)
        hashes[route] = hashlib.sha256(payload).hexdigest()
    if report_route not in routes or missing:
        raise ValueError("missing, outside-root or unsupported report links: " + repr(missing))
    return report_route, routes, hashes


def handler_for(report_route, routes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(False)

        def do_HEAD(self):
            self.respond(True)

        def respond(self, head):
            path = unquote(urlsplit(self.path).path)
            if path == "/":
                self.send_response(302)
                self.send_header("Location", quote(report_route, safe="/"))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if path not in routes:
                self.send_error(404, "Not exposed by this report preview")
                return
            payload, media = routes[path]
            self.send_response(200)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head:
                self.wfile.write(payload)

        def log_message(self, *args):
            # Never save incoming URLs, query strings or request headers.
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--also-report", type=Path, action="append", default=[],
                        help="Preserve an explicitly selected prior report and its linked attachments")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--receipt", type=Path, required=True, help="New preview lifecycle receipt")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be 1..65535")
    route, routes, hashes = catalog(args.report)
    for additional in args.also_report:
        _, selected, selected_hashes = catalog(additional)
        for path in routes.keys() & selected.keys():
            if routes[path] != selected[path]:
                raise ValueError("inconsistent explicitly selected report attachment: " + path)
        routes.update(selected)
        hashes.update(selected_hashes)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as record:
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(route, routes)) as server:
            receipt = {"kind": "loopback_report_preview_not_public_hosting", "at_utc": datetime.now(timezone.utc).isoformat(),
                "pid": os.getpid(), "bind": "127.0.0.1", "port": args.port, "status": "listening",
                "report": str(args.report.resolve()), "instance_local_url": f"http://127.0.0.1:{args.port}{quote(route, safe='/')}",
                "additional_reports": [str(path.resolve()) for path in args.also_report],
                "remote_browser_requires_port_forwarding": True, "external_upload_performed": False,
                "historical_report_rewritten": False, "exposed_file_sha256": hashes, "directory_listing": False}
            record.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            record.flush()
            print(json.dumps({key: value for key, value in receipt.items() if key != "exposed_file_sha256"}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                receipt.update(status="stopped", stopped_at_utc=datetime.now(timezone.utc).isoformat())
                record.seek(0)
                record.truncate()
                record.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
