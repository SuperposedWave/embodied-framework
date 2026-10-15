"""Call the unchanged native policy/server API, with localhost-only binding."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent / "upstream/lingbot-vla-v2"
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer
    model = LingbotVLAv2Server(str(args.model_path), use_length=50, chunk_ret=True,
                               use_bf16=False, use_fp32=True, use_compile=True)
    WebsocketPolicyServer(model, host="127.0.0.1", port=args.port).serve_forever()


if __name__ == "__main__":
    main()
