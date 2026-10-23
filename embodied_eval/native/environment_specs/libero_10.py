"""Environment-only pinned protocol metadata; no selected model or weights."""
from dataclasses import replace
from .. import environment_records as c
from . import libero_30 as base

BENCHMARK_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
EXTENSION = base.EXTENSION
ACTION = "openpi-libero-osc7@1"
CONTROLLER = "openpi-libero-panda-osc-pose@1"
SPACE_ID = "libero-spatial-openpi-panda@f78abd6"


def space():
    original = base.space()
    return replace(original, spec_id=SPACE_ID, revision=BENCHMARK_REVISION,
        protocol_version="openpi-native-1",
        # Native LIBERO applies controller limits; do not alter the model's output.
        controllers={CONTROLLER: c.ControllerSpec(frame_id="world",
            parameter_schema={"type": "object", "properties": {"osc_action7": {
                "type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7}},
                "required": ["osc_action7"], "additionalProperties": False},
            parameter_units={"osc_action7": "LIBERO native controller input; environment applies its own limits"})},
        action_specs={ACTION: c.ActionSpec(command_kinds=("controller_command",),
            target_ids=("robot0/eef",), controller_refs=(CONTROLLER,))})
