"""Dedicated pi05 primitive-10task checkpoint and fixed native VLABench profile."""
from dataclasses import replace

from . import vlabench_spec as vla

MODEL_REVISION = "788ef1f20e655de5bb40869fb7b7fcdd49732387"
BENCHMARK_REVISION = vla.BENCHMARK_REVISION
HF_REVISION = "f658a8bec801aab960fe0ceab370a123b31c1837"
NORMALIZATION_FILE = "assets/vlabench/vlabench_ft_primitive/norm_stats.json"
NORMALIZATION_SHA256 = "bce33fc1831830d953ef9bc57e520c02ee44e02b7ad07f27ce92b931a879384f"
CHECKPOINT_REF = "hf:VLABench/pi05-primitive-10task@" + HF_REVISION
TRAIN_CONFIG = "pi05_ft_vlabench_primitive"
REPLAN_STEPS = 5
EXTENSION, ACTION, CONTROLLER, SENSORS = vla.EXTENSION, vla.ACTION, vla.CONTROLLER, vla.SENSORS


def space(period):
    return replace(vla.space(period), protocol_version="openpi-native-1")


def capability(config):
    return replace(vla.capability(config), model_id="pi05-vlabench-primitive10@f658a8b",
        source_revision=MODEL_REVISION, checkpoint_ref=CHECKPOINT_REF,
        normalization_ref="sha256:" + NORMALIZATION_SHA256)


def profile(space, cap):
    return replace(vla.profile(space, cap), profile_id="pi05-vlabench-track1-select-fruit@1")
