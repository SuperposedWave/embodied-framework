"""Model-local native interpreter isolation; no simulator or Runner branch.

The private inherited pipe carries trusted local Python contract objects only.
It is not a remote/public protocol. Native preprocessing, RNG, action decoding
and chunk cache stay together inside the unchanged model implementation.
Synchronous calls add no timeout, retry, restart or concurrent trials.
"""
import os
from pathlib import Path
import subprocess
from ..errors import AdapterError
from ..validation import require
from .native_catalog import model_options

METHODS = ("begin_episode", "observe", "predict", "end_episode", "diagnostics", "close")


class NativeModelProcess:
    def __init__(self, family, options, output=None):
        self.cfg = model_options(family, options)
        require(self.cfg.runtime_python is not None, "missing_model_runtime", "model", "native interpreter path required")
        self.family, self.options, self.output = family, dict(options), output
        self.connection = self.worker = None
        self.closed = self.failed = False

    def _start(self):
        if self.worker is not None:
            return
        import multiprocessing
        require(Path(self.cfg.runtime_python).is_file() and os.access(self.cfg.runtime_python, os.X_OK),
                "asset_missing", "model.runtime_python", "native interpreter unavailable")
        self.connection, child = multiprocessing.Pipe()
        root = Path(__file__).resolve().parents[2]
        # Keep the venv's bin/python path, not its resolved system interpreter;
        # resolving the symlink would silently lose the selected environment.
        environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        if self.cfg.runtime_gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(self.cfg.runtime_gpu)
        try:
            self.worker = subprocess.Popen([self.cfg.runtime_python, "-m", "embodied_eval.mn.native_model_worker",
                "--pipe-fd", str(child.fileno())], cwd=root, env=environment, pass_fds=(child.fileno(),),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except BaseException:
            self.connection.close()
            self.connection = None
            raise
        finally:
            child.close()
        # Output directory is deliberately not passed to the native model.
        # It does not receive a Benchmark, private task or pair configuration.
        self._send((self.family, self.options), "initialize")
        self._receive("initialize")

    def _lost(self):
        self.failed = True
        return AdapterError("model_service", "native_model_worker_unavailable",
            "owned native model ended without a reply; no retry or restart")

    def _send(self, message, method):
        try:
            self.connection.send(message)
        except (BrokenPipeError, EOFError, OSError) as error:
            raise self._lost() from error

    def _receive(self, method):
        try:
            response = self.connection.recv()
        except (EOFError, OSError) as error:
            raise self._lost() from error
        if (not isinstance(response, dict) or type(response.get("ok")) is not bool or
                response.get("method") != method or (response["ok"] and "result" not in response)):
            self.failed = True
            raise AdapterError("model_service", "native_model_invalid_reply", "invalid private native model reply")
        if not response["ok"]:
            self.failed = True
            raise AdapterError("model_service", "native_model_operation_failed",
                "native model operation failed; no retry or restart")
        return response["result"]

    def _call(self, method, *args):
        require(not self.closed, "adapter_closed", "model", "native model already closed")
        require(method == "close" or not self.failed, "native_model_failed", "model", "only cleanup allowed after a failed operation")
        self._start()
        self._send((method, args), method)
        return self._receive(method)

    def begin_episode(self, context, task, profile, space):
        return self._call("begin_episode", context, task, profile, space)

    def observe(self, model_input):
        return self._call("observe", model_input)

    def predict(self, context):
        return self._call("predict", context)

    def end_episode(self, context):
        return self._call("end_episode", context)

    def diagnostics(self):
        if self.worker is None:
            return {"native_model_isolated": True, "native_worker_started": False, "retries_or_restarts": 0}
        return dict(self._call("diagnostics"), native_model_isolated=True, native_worker_started=True,
            native_worker_pid=self.worker.pid, retries_or_restarts=0, in_process_native_engine=False,
            engine_in_native_worker=True, native_runtime_python=self.cfg.runtime_python,
            native_runtime_gpu=self.cfg.runtime_gpu)

    def close(self):
        if self.closed:
            return
        acknowledged = self.worker is None
        try:
            if self.worker is not None and self.worker.poll() is None:
                self._call("close")
                acknowledged = True
            if self.worker is not None:
                code = self.worker.wait()
                if code != 0 or not acknowledged:
                    raise AdapterError("model_service", "native_model_close_failed",
                        "native model close acknowledgement and clean exit both required")
        finally:
            if self.connection is not None:
                self.connection.close()
            if self.worker is not None:
                self.worker.wait()
            self.closed = True
