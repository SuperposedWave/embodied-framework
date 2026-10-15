"""Local synthetic HTTP safety tests, not real comparison video evidence."""
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from serve_comparison_report import byte_range, handler_for


class ComparisonVideoPreview(unittest.TestCase):
    def test_ranges_clamped_and_suffix_supported_invalid_multiple_and_outside_rejected(self):
        self.assertEqual(byte_range("bytes=2-99", 10), (2, 9))
        self.assertEqual(byte_range("bytes=2-", 10), (2, 9))
        self.assertEqual(byte_range("bytes=-3", 10), (7, 9))
        for value in ("bytes=", "bytes=-0", "bytes=10-", "bytes=6-2", "bytes=0-1,4-6", "items=1-2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                byte_range(value, 10)

    def test_video_range_head_and_private_paths(self):
        payload = b"0123456789"
        routes = {"/outputs/report.html": (b"report", "text/html; charset=utf-8"), "/outputs/clip.mp4": (payload, "video/mp4")}
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for("/outputs/report.html", routes))
        worker = threading.Thread(target=server.serve_forever)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        try:
            connection.request("GET", "/outputs/clip.mp4", headers={"Range": "bytes=2-5"})
            response = connection.getresponse()
            self.assertEqual(response.status, 206)
            self.assertEqual(response.getheader("Content-Range"), "bytes 2-5/10")
            self.assertEqual(response.read(), b"2345")
            connection.request("HEAD", "/outputs/clip.mp4", headers={"Range": "bytes=2-5"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Length"), "10")
            self.assertEqual(response.read(), b"")
            connection.request("GET", "/outputs/clip.mp4", headers={"Range": "bytes=10-"})
            response = connection.getresponse()
            self.assertEqual(response.status, 416)
            self.assertEqual(response.read(), b"")
            for path in ("/", "/outputs/", "/.env", "/outputs/../docs/private.json", "/%2e%2e/.env"):
                connection.request("GET", path)
                response = connection.getresponse()
                self.assertEqual(response.status, 302 if path == "/" else 404)
                response.read()
        finally:
            connection.close(); server.shutdown(); server.server_close(); worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
