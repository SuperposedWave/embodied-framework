"""Weight-owned execution capabilities. No environment identity or imports."""
from ..v03.execution_types import ModelExecution
from .timing_profiles import timing_facts


def model_execution(family, options):
    profile = options["io_profile"]
    owner, evidence = timing_facts(family, profile,
        execution_horizon=options.get("open_loop_horizon"))
    preferred, generated, low, high, adjustable = None, None, None, None, False
    if family == "openpi":
        generated = 50
        preferred = 50 if profile == "arx_joint14" else 5
        # These released local clients already implement prefix consumption.
        if profile != "arx_joint14":
            low, high, adjustable = 1, 50, True
    elif family == "lingbot_vla":
        generated = 50
        # use_length lives at the service boundary. Until it can be negotiated
        # and acknowledged, consume the whole returned chunk, not a fabricated
        # promise of 50 executed rows. The reviewed service DEFAULT is 50.
    elif family in ("cosmos3", "dreamzero"):
        generated = 32 if family == "cosmos3" else 8
        preferred = options.get("open_loop_horizon") or generated
        low, high, adjustable = 1, generated, True
        if family == "dreamzero":
            low = high = None
            adjustable = False
    # OpenWAM server horizon (or loaded ARX deploy YAML) must not be guessed
    # from a 32-row training tensor. Its native reference is explicitly unknown.
    return ModelExecution(generated_actions=generated, preferred_consume_actions=preferred,
        reference_consume_actions=generated if family in ("cosmos3", "dreamzero") else preferred,
        minimum_consume_actions=low, maximum_consume_actions=high, adjustable=adjustable,
        cache_owner=owner, action_period_s=None,
        provenance={"io_profile": profile, "source_audit": evidence,
            "verification": "source-reviewed reference, not a live server configuration acknowledgement"})
