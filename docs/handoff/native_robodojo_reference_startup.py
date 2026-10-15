"""Reuse the independently working native-factory startup, not an Adapter.

Only the prefix through native reset is reused from the recorded-action probe.
Its recorded-action loop is excluded. No extra render/warm-up/physics calls.
"""
import ast
import hashlib
from pathlib import Path


def initialize(args):
    import replay_robodojo_native_actions as reference
    path = Path(reference.__file__)
    source = path.read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "child")
    attempt = next(node for node in function.body if isinstance(node, ast.Try))
    prefix = []
    for statement in attempt.body:
        prefix.append(statement)
        if ast.unparse(statement) == "env.reset(seed=[options['layout_index']])":
            break
    else:
        raise ValueError("reviewed_native_reset_boundary_missing")
    function.body = function.body[:function.body.index(attempt)] + prefix + [
        ast.parse("return app, env, proof, client_calls, robot_evidence").body[0]]
    scope = dict(vars(reference))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), scope)
    app, env, proof, client_calls, robot_evidence = scope["child"](args)
    proof["startup_helper_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    proof["startup_prefix_excludes_recorded_action_loop"] = True
    return app, env, proof, client_calls, robot_evidence
