"""Use the fixed source's factory unchanged; bind transport to loopback only."""
import argparse
import logging
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.repo / "src"), str(args.repo / "packages/openpi-client/src")]
    source = runpy.run_path(str(args.repo / "scripts/serve_policy.py"), run_name="native_policy_entry")
    options = source["Args"](policy=source["Checkpoint"](config=args.config_name, dir=str(args.checkpoint)), port=args.port)
    policy = source["create_policy"](options)
    server = source["websocket_policy_server"].WebsocketPolicyServer(
        policy=policy, host="127.0.0.1", port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
