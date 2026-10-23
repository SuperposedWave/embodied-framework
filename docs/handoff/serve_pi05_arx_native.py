"""Source-pinned native ARX policy, loopback only; no simulator configuration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("valid local port required")
    source = WORKSPACE / "upstream/xpolicy-reference-mn"
    from embodied_eval.native.model_assets import verify_model
    verify_model("openpi", {"io_profile": "arx_joint14", "repo": str(source), "checkpoint_dir": str(args.checkpoint)})
    native = source / "policy/Pi_05/openpi"
    sys.path[:0] = [str(native / "src"), str(native / "packages/openpi-client/src")]
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    selected = config.get_config("pi05_base_aloha_full_sim_arx-x5_seed_0")
    policy = policy_config.create_trained_policy(selected, args.checkpoint)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
