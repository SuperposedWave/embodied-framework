"""Start exactly the independently working native prefix before policy imports.

The original recorded-action loop is excluded; only its reset prefix runs.
All fresh inference, recording, comparison and cleanup follow initialization.
"""
import replay_robodojo_native_actions as reference


def main():
    parser = reference.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=reference.Path, required=True)
    parser.add_argument("--output", type=reference.Path, required=True)
    parser.add_argument("--isaac-libraries", type=reference.Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--capture-native-images", action="store_true")
    args = parser.parse_args()
    args.trial, args.output = args.trial.resolve(), args.output.resolve()
    args.output.relative_to(reference.ROOT / "outputs")
    args.trial.relative_to(reference.ROOT / "outputs")
    launch = reference.json.loads((args.output / "native-launch-config.json").read_text())
    args.model_gpu, args.port = launch["model_gpu"], launch["port"]
    args.video_runtime = reference.ROOT / launch["video_runtime"]
    with (args.output / "native-lifecycle.jsonl").open("a") as stream:
        stream.write(reference.json.dumps({"phase": "before_minimal_native_reference_startup",
            "at_utc": reference.datetime.now(reference.timezone.utc).isoformat()}) + "\n")
    from native_robodojo_reference_startup import initialize
    initialized = initialize(args)
    from native_robodojo_reference_rollout import child
    child(args, initialized=initialized)


if __name__ == "__main__":
    main()
