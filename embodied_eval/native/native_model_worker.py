"""Private child entry point for model-only interpreter isolation."""
import argparse
from multiprocessing.connection import Connection
import resource
from .native_model_process import METHODS


def failure(method, error):
    # Never transfer arbitrary native exception text, logs, tokens or settings.
    return {"ok": False, "method": method, "error_type": type(error).__name__}


def worker_loop(connection, model):
    failed = False
    while True:
        method, args = connection.recv()
        if method not in METHODS or (failed and method != "close"):
            connection.send(failure(method, ValueError()))
            failed = True
            continue
        try:
            result = getattr(model, method)(*args)
        except Exception as error:
            connection.send(failure(method, error))
            failed = True
            if method == "close":
                return 1
        else:
            connection.send({"ok": True, "method": method, "result": result})
            if method == "close":
                return 0


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe-fd", type=int, required=True)
    args = parser.parse_args()
    connection = Connection(args.pipe_fd)
    model = None
    try:
        family, options = connection.recv()
        from .registry import default_registry
        # Component-specific runtime selection is transport configuration, not
        # a model input. Strip it to prevent recursive child creation.
        native_options = {key: value for key, value in options.items() if key not in ("runtime_python", "runtime_gpu")}
        model = default_registry().model(family).factory(native_options, None)
        connection.send({"ok": True, "method": "initialize", "result": None})
        return worker_loop(connection, model)
    except BaseException as error:
        try:
            connection.send(failure("initialize", error))
        except (BrokenPipeError, EOFError, OSError):
            pass
        if model is not None:
            try:
                model.close()
            except BaseException:
                pass
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
