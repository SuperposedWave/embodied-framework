"""Source-reviewed timing facts, NOT new execution instructions.

Selected solely by the weight I/O profile. No environment identity, file I/O,
GPU import, inferred action duration, or takeover of native service histories.
The offline audit records exact source/metadata hashes separately; describe()
does not claim it just checked files or contacted the actual policy endpoint.
"""
from copy import deepcopy

PROFILES = {
    "openwam": ("eef10_osc", "eef10_absolute", "eef20_dual_legacy",
                "eef20_dual_wxyz_encoded_absolute", "eef20_arx_absolute"),
    "openpi": ("state8_osc", "state7_absolute", "arx_joint14"),
    "lingbot_vla": ("joint14",),
    "cosmos3": ("nano-policy-droid-joint8",),
    "dreamzero": ("droid-joint8-causal",),
}


def timing_facts(family, profile, *, execution_horizon=None):
    """Return isolated documentary provenance and the outer chunk owner.

    Nominal dataset FPS, future-video stride, teacher FPS and online action
    duration intentionally have DIFFERENT keys. None authorizes resampling.
    """
    if profile not in PROFILES.get(family, ()):
        raise ValueError("no source-reviewed native timing profile: " + str((family, profile)))
    facts = {
        "audit": "native-timing-source-audit@1",
        "scope": "fixed-source interpretation; not live server identity or dataset verification",
        "wire_observation": "current_snapshot_per_camera",
        "public_history_requested": False,
        "native_action_period_s": None,
        "physical_action_period_status": "unknown; no action resampling authorized",
    }
    if family == "openwam":
        arx = profile == "eef20_arx_absolute"
        owner = "adapter" if arx else "server"
        facts.update(
            training_raw_window_rows=33, training_action_rows=32,
            training_video_stride_rows=4,
            training_video_stride_scope="future video supervision only; action rows remain unstrided",
            generation_condition="one current composite image plus current proprioception",
            chunk_owner=owner,
            execution_rows_rule="native inference_horizon or full returned chunk; not seconds",
            native_input_history="none outside native engine; not a public fixed-rate history",
            sources=["openwam/deploy/policy.py:_build_conditions",
                     "openwam/dataloader/bases/lerobot_v3_reader.py:_finalize_action"
                     if profile in ("eef10_osc", "eef10_absolute") else
                     "openwam/dataloader/robodojo.py" if arx else "openwam/dataloader/robotwin.py:__getitem__",
                     "checkpoint:config.yaml:dataloader",
                     "framework:native/openwam_arx.py" if arx else "openwam/deploy/executors/sync_executor.py"],
        )
    elif family == "openpi":
        owner = "adapter"
        rows = 50 if profile == "arx_joint14" else 5
        facts.update(chunk_owner=owner, execution_rows_per_request=rows,
            native_input_history="one image per view and current state; batch axis is not history",
            future_action_offsets="t / dataset_meta.fps; t is an action row index, not image history",
            sources=["src/openpi/training/data_loader.py:create_torch_dataset",
                     "framework:native/native_models.py:predict"])
        if profile == "state8_osc":
            facts.update(example_converter_fps=10,
                converter_scope="minimal public example; released weight training manifest not established",
                sources=facts["sources"] + ["examples/libero/convert_libero_data_to_lerobot.py:main"])
        elif profile == "state7_absolute":
            facts.update(example_converter_fps=10,
                converter_scope="converter names pretrain dataset; matched training config names finetune dataset",
                sources=facts["sources"] + ["examples/vlabench/convert_to_lerobot.py:main",
                                            "src/openpi/training/config.py:pi05_ft_vlabench_primitive"])
        else:
            facts.update(converter_scope="ARX checkpoint has no locally verified physical sampling manifest")
    elif family == "lingbot_vla":
        owner = "adapter"
        facts.update(chunk_owner=owner, execution_rows_per_request=50,
            native_input_history="three current camera views, not three time points",
            training_action_rows=50, teacher_video_fps_default=1.0,
            teacher_video_scope="training future-video teacher; dataset-derived effective FPS can override it",
            future_video_offsets="[0, (chunk_size - 1) / dataset_meta.fps]; current and future, not past",
            sources=["lingbotvla/data/vla_data/base_dataset.py:get_delta_timestamps",
                     "lingbotvla/data/vla_data/base_dataset.py:get_video_delta_timestamps",
                     "tasks/vla/train_lingbotvla.py:get_video_target call",
                     "checkpoint:lingbotvla_cli.yaml:train.align_params.video",
                     "framework:docs/handoff/serve_lingbot_native.py"])
    elif family == "cosmos3":
        owner = "adapter"
        facts.update(chunk_owner=owner, execution_rows_per_request=execution_horizon or 32,
            server_default_generated_rows=32, server_default_conditioning_fps=15.0,
            recipe_dataset_fps=15.0,
            recipe_scope="pinned recipe and server defaults; empty checkpoint.json is not a training manifest",
            native_input_history="one current composite image; default history_length=1 is the current state row",
            conditioning_scope="FPS tensor conditions generation; not a controller frequency command",
            sources=["cosmos_framework/scripts/action_policy_server_robolab.py:_build_sample",
                     "cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_droid_nano.py",
                     "cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py",
                     "checkpoint:checkpoint.json"])
    elif family == "dreamzero":
        owner = "adapter"
        facts.update(chunk_owner=owner, execution_rows_per_request=execution_horizon or 8,
            training_action_rows=24, training_video_rows=25, configured_eval_video_indices=[0],
            native_input_history="server owns accumulated camera frames and causal hidden state",
            server_first_call_video_rows=1, server_later_video_rows=4,
            server_short_history="repeat earliest buffered frame to four; NOT four distinct timed captures",
            server_history_timing="RPC arrivals are accumulated without public capture timestamps",
            sources=["socket_test_optimized_AR.py:ARDroidRoboarenaPolicy._convert_observation",
                     "socket_test_optimized_AR.py:ARDroidRoboarenaPolicy.infer",
                     "checkpoint:experiment_cfg/conf.yaml:all_modality_configs.oxe_droid",
                     "framework:native/additional_models.py:predict"])
    else:
        raise ValueError("no source-reviewed native timing family: " + str(family))
    return owner, deepcopy(facts)
