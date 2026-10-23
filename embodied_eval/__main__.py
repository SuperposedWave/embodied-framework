"""Single current 0.3 run/check entry. No historical runtime or converter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4
from .codec import loads, to_data
from .config import load_config
from .errors import ContractError
from .runner import run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="embodied_eval")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Run independent 0.3 adapters.")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path)
    preflight = commands.add_parser("check", help="GPU-free 0.3 semantic and asset preflight.")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--verify-assets", action="store_true", help="Hash local pinned assets; no downloads or inference.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (ValueError, OSError) as error:
        print(f"Configuration rejected: {error}", file=sys.stderr)
        return 2
    if args.command == "check":
        from .v03.preflight import check
        try:
            result = check(config, verify_assets=args.verify_assets)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
            return 0 if result["structurally_compatible"] and not result["asset_errors"] else 2
        except (ContractError, ValueError, OSError) as error:
            print(f"Preflight rejected: {error}", file=sys.stderr)
            return 2
    output = args.output or Path("outputs") / (
        config.run_id + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    try:
        outcome = run(config, output)
    except ContractError as error:
        print(f"Run rejected: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Run infrastructure error: {error}", file=sys.stderr)
        return 1
    print(f"{config.run_id}: completed={outcome.summary['completed']}, errors={outcome.summary['errors']}")
    print(outcome.output)
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
