"""Official pi05_libero native evaluation, separate from the OpenWAM protocol."""
from dataclasses import replace
from ..native import environment_records as c
from . import openwam_libero_spec as base

MODEL_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
BENCHMARK_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
CHECKPOINT_REF = "gs://openpi-assets/checkpoints/pi05_libero/"
NORMALIZATION_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
NORMALIZATION_FILE = "assets/physical-intelligence/libero/norm_stats.json"
EXTENSION = base.EXTENSION
ACTION = "openpi-libero-osc7@1"
CONTROLLER = "openpi-libero-panda-osc-pose@1"
MODEL_ID = "pi05-libero@openpi-215abfb"
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


def capability(config):
    return replace(base.capability(config), model_id=MODEL_ID, source_revision=MODEL_REVISION,
                   checkpoint_ref=CHECKPOINT_REF, normalization_ref="sha256:" + NORMALIZATION_SHA256,
                   produced_action_specs=(ACTION,))


def profile(space_spec, model_cap):
    original = base.profile(space_spec, model_cap)
    return replace(original, profile_id="pi05-libero-spatial@1", revision="openpi-native-1",
                   action_spec_refs=(ACTION,))
