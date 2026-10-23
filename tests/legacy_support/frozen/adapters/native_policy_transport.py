"""Bounded synchronous transport using the upstream NumPy/msgpack wire codec."""
from inspect import signature
from tests.legacy_support.frozen.adapters.native_common import native_module


def connection_options(connect, timeout_s):
    """Preserve direct transport on native websockets12 and framework16.

    Version12 has neither automatic proxy discovery nor synchronous keepalive.
    Only pass the newer options when the installed API explicitly declares them;
    never retry an inference or mutate environment/package versions.
    """
    options = dict(compression=None, max_size=None, open_timeout=timeout_s, close_timeout=5)
    parameters = signature(connect).parameters
    for name in ("proxy", "ping_interval"):
        if name in parameters:
            options[name] = None
    return options


class NativePolicyPeer:
    def __init__(self, repo, codec_module, codec_root, host, port, timeout_s):
        from websockets.sync.client import connect
        self.codec = native_module(repo, codec_module, codec_root)
        self.timeout_s = timeout_s
        self.socket = connect(f"ws://{host}:{port}", **connection_options(connect, timeout_s))
        try:
            self.metadata = self.codec.unpackb(self.socket.recv(timeout=timeout_s))
        except Exception:
            self.socket.close()
            raise

    def infer(self, value):
        self.socket.send(self.codec.packb(value))
        raw = self.socket.recv(timeout=self.timeout_s)
        if isinstance(raw, str):
            # Do not persist arbitrary server error text, which may contain credentials.
            raise RuntimeError("native policy server returned an error instead of binary output")
        return self.codec.unpackb(raw)

    def close(self):
        self.socket.close()
