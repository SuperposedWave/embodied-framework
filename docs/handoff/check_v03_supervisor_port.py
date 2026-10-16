"""Reproduce owned loopback TCP port recovery; no policy or simulator calls."""
import argparse
from datetime import datetime, timezone
import errno
import json
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        server.listen(1)
        address = server.getsockname()
        with socket.create_connection(address) as client:
            connection, _ = server.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b''
                client.shutdown(socket.SHUT_WR)
                assert connection.recv(1) == b''
    plain_errno = None
    with socket.socket() as plain:
        try:
            plain.bind(address)
        except OSError as error:
            plain_errno = error.errno
    with socket.socket() as reviewed:
        reviewed.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reviewed.bind(address)
        reviewed.listen(1)
    result = dict(kind='owned_loopback_port_recovery_diagnostic_not_adapter_execution', at_utc=datetime.now(timezone.utc).isoformat(),
        ordinary_bind_errno=plain_errno, expected_errno=errno.EADDRINUSE, reviewed_reuse_bind_and_listen_passed=True,
        passed=plain_errno == errno.EADDRINUSE, original_runtime_port_failure_unique_cause_retroactively_proven=False,
        no_external_process_or_socket_stopped=True, new_GPU_or_simulation=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
