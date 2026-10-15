"""Actual loopback socket with synthetic payloads; NOT real policy inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from threading import Thread
from importlib import metadata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from websockets.sync.server import serve
    from embodied_eval.adapters.native_policy_transport import NativePolicyPeer
    from embodied_eval.adapters.native_common import native_module
    from embodied_eval.mn.assets_common import _revision
    repo = ROOT.parent / "upstream/openpi"
    revision = "215abfb217dbac7d5f1273282331b9b1866c0479"
    _revision(repo, revision)
    codec = native_module(str(repo), "openpi_client.msgpack_numpy", "packages/openpi-client/src")
    array = np.arange(16, dtype=np.float32).reshape(2, 8)
    seen = []
    def handler(socket):
        socket.send(codec.packb({"synthetic_test_server": True}))
        payload = codec.unpackb(socket.recv())
        seen.append(payload)
        socket.send(codec.packb({"actions": payload["array"]}))
    with serve(handler, "127.0.0.1", 0, compression=None) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            peer = NativePolicyPeer(str(repo), "openpi_client.msgpack_numpy", "packages/openpi-client/src",
                "127.0.0.1", server.socket.getsockname()[1], 5)
            try:
                response = peer.infer({"array": array})
                exact = response["actions"].dtype == array.dtype and np.array_equal(response["actions"], array)
            finally:
                peer.close()
        finally:
            server.shutdown()
            thread.join()
    result = {"kind": "real_loopback_transport_synthetic_payload_not_model_or_simulator",
        "passed": bool(exact and len(seen) == 1), "synthetic": True,
        "native_codec_revision": revision, "websockets_version": metadata.version("websockets"),
        "request_count": len(seen), "array_dtype_shape_values_exact": bool(exact),
        "native_codec_sha256": hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
        "global_proxy_modified": False, "model_loaded": False, "simulator_started": False,
        "formal_acceptance": False}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
