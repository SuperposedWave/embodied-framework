"""Private native-codec context conversion, independent of both adapters."""
from ..mn import contracts as native
from . import contracts as c


def native_context(context):
    return native.Context(**{key: getattr(context, key) for key in ("run_id", "env_id", "episode_id", "message_id")})


def public_context(context):
    return c.Context(**{key: getattr(context, key) for key in ("run_id", "env_id", "episode_id", "message_id")})
