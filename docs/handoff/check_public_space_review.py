"""Check the field proposal, actual baseline, immutable code and HTTP preview."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import ProxyHandler, build_opener

from serve_report import ROOT, catalog


class Examples(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.buffer = []
        self.examples = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre" and "data-example" in dict(attrs):
            self.active, self.buffer = True, []

    def handle_endtag(self, tag):
        if tag == "pre" and self.active:
            self.examples.append(json.loads("".join(self.buffer)))
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.buffer.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-port", type=int, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    base = inventory["base_commit"]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    immutable = {}
    for raw in filter(None, tracked):
        name = raw.decode()
        if name.startswith(("embodied_eval/", "examples/", "tests/")) or name in (
                "docs/contract-v0.1.md", "docs/contract-v0.2.md"):
            actual = (ROOT / name).read_bytes()
            old = subprocess.check_output(["git", "show", base + ":" + name], cwd=ROOT)
            assert actual == old, name
            immutable[name] = hashlib.sha256(actual).hexdigest()
    assert len(immutable) > 80
    baseline = json.loads(args.baseline.read_text())
    assert baseline["passed"] and baseline["checks"][0]["tests_run"] == 443
    codes = [case["actual_exit_code"] for case in baseline["checks"] if "actual_exit_code" in case]
    assert codes == [0, 0, 0, 1, 2], codes
    test_text = (args.baseline.parent / "unittest.txt").read_text()
    assert not re.search(r"skipped=\d+", test_text)
    examples = Examples()
    examples.feed(args.report.read_text())
    assert len(examples.examples) == 2
    space, action = examples.examples
    assert space["contract_version"] == "0.3"
    assert set(space["sensors"]) == {"head", "left_wrist", "right_wrist"}
    assert space["view_aliases"] == {"main": "head"}
    assert {command["target_id"] for command in action["commands"]} == {
        "robot0/left_eef", "robot0/right_eef", "robot0/left_grip", "robot0/right_grip"}
    for command in action["commands"]:
        if command["kind"] == "pose_target":
            assert command["pose"]["quaternion_xyzw"] == [0, 0, 0, 1]
    route, routes, hashes = catalog(args.report)
    # Local-only HTTP must not use ambient HTTP(S) proxies.
    opener = build_opener(ProxyHandler({}))
    preview = "http://127.0.0.1:" + str(args.preview_port)
    checks = []
    for path, (payload, _) in routes.items():
        with opener.open(preview + quote(path, safe="/"), timeout=10) as response:
            actual = response.read()
            assert response.status == 200 and actual == payload, path
            checks.append({"path": path, "status": response.status, "sha256": hashes[path]})
    with opener.open(preview + "/", timeout=10) as response:
        assert response.read() == routes[route][0]
    receipt = {"kind": "proposal_document_checks_not_0_3_implementation_or_acceptance",
               "at_utc": datetime.now(timezone.utc).isoformat(), "source_base_commit": base,
               "proposal_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
               "preserved_production_example_test_and_contract_files": immutable,
               "actual_software_tests": 443, "test_skips": 0, "five_CLI_exit_codes": codes,
               "synthetic_proposal_JSON_excerpts_parse": True,
               "proposal_excerpts_validated_against_implemented_0_3_schema": False,
               "HTTP_checks": checks, "preview_loopback_only": True,
               "user_desktop_port_forwarding_verified": False,
               "real_inference_or_simulation_performed": False,
               "public_contract_or_production_Adapter_or_Runner_changed": False,
               "stage4_accepted_or_contract_frozen": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"checks_passed": True, "immutable_files": len(immutable),
                      "HTTP_routes": len(checks), "tests": 443, "CLI": codes,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
