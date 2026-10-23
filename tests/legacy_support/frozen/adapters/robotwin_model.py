"""Registered RoboTwin policy boundary: public snapshots in, native row out."""
import numpy as np

import tests.legacy_support.frozen._legacy_contracts as c
from embodied_eval.errors import AdapterError
from tests.legacy_support.frozen._legacy_validation import same_episode
from tests.legacy_support.frozen.adapters.native_common import finite_array, rgb_image
from tests.legacy_support.frozen.adapters.native_robotwin_models import NativeEndpoint, OpenWAMRobotwinNativePolicy, LingbotRobotwinNativePolicy
from tests.legacy_support.frozen.adapters.native_robotwin_executor import LingbotNativeActionCache
import tests.legacy_support.frozen.adapters.robotwin_spec as spec


class RobotwinModel:
    def __init__(self, config, output=None):
        self.config, self.output = config, output
        self.variant = config.adapter
        endpoint = NativeEndpoint(repo=config.lingbot_repo if self.variant == "lingbot_robotwin" else config.openwam_repo,
                                  host=config.host, port=config.port, timeout_s=config.request_timeout_s)
        cls = LingbotRobotwinNativePolicy if self.variant == "lingbot_robotwin" else OpenWAMRobotwinNativePolicy
        self.bridge = cls(endpoint)
        self.cache = LingbotNativeActionCache(self.bridge) if self.variant == "lingbot_robotwin" else None
        self.latest = self.context = self.profile = None
        self.calls = 0

    def describe(self):
        return spec.capability(self.config)

    def begin_episode(self, context, task, space, profile):
        if space.timing_mode != "measured_native" or space.spec_id != spec.space(self.variant).spec_id:
            raise AdapterError("compatibility", "native_profile_mismatch", "reviewed RoboTwin native space required")
        self.latest = None
        self.context, self.profile, self.instruction = context, profile, task.instruction
        self.calls = 0
        if self.cache:
            self.cache.begin_episode()
        else:
            self.bridge.reset(task.instruction)

    def observe(self, model_input):
        same_episode(model_input.context, self.context)
        if model_input.task_public.instruction != self.instruction:
            raise AdapterError("model_input", "native_instruction_changed", "native instruction is frozen per episode")
        self.latest = model_input

    def native_input(self, model_input):
        view = model_input.observation_view
        raw = view.extensions[spec.EXTENSION]
        return {"observation": {key: {"rgb": rgb_image(view, key)} for key in spec.SENSORS},
            "joint_action": {"vector": finite_array(raw["joint_drive_targets_and_grippers"], (14,), "joint drive targets")},
            "endpose": {**{side + "_endpose": finite_array(raw[side + "_endpose_world_wxyz"], (7,), side + " native endpose")
                           for side in ("left", "right")},
                        **{side + "_gripper": float(finite_array(np.asarray(raw[side + "_gripper_open_fraction"]).reshape(-1),
                                                               (1,), "gripper")[0]) for side in ("left", "right")}}}

    def predict(self, request_context):
        same_episode(request_context, self.context)
        if self.latest is None:
            raise AdapterError("model_input", "missing_observation", "public native snapshot required")
        native = self.native_input(self.latest)
        if self.cache:
            row = self.cache.next_action(native, self.instruction)
            acquire = self.cache.needs_observation
        else:
            row = self.bridge.predict(native, self.instruction)
            acquire = True
        dtype = str(row.dtype)
        if dtype not in ("float32", "float64", "int32", "int64"):
            raise AdapterError("model_output", "unsupported_native_dtype", "native row dtype cannot be faithfully transported")
        action, controller = spec.refs(self.variant)
        self.calls += 1
        return c.Prediction(context=request_context, request_id=request_context.message_id,
            based_on_observation_id=self.latest.observation_view.observation_id, profile_ref=self.profile.profile_id,
            decision=c.NativeAction(action_spec_ref=action, commands=(c.ControllerCommand(
                command_id=request_context.message_id + ":native", target_id="robot0/base", controller_profile_ref=controller,
                parameters={"native_row": row.tolist(), "native_dtype": dtype, "acquire_observation_after": acquire}),)),
            diagnostics=c.PredictionDiagnostics(transformations=(
                "native full joint14 chunk cache, no resampling" if self.cache else
                "official EEF20/EE16 helper unchanged; native wxyz versus helper xyzw convention conflict retained",)))

    def end_episode(self, context):
        if self.cache:
            self.cache.end_episode()
        self.latest = self.context = None

    def close(self):
        if self.cache:
            self.cache.end_episode()
        self.bridge.close()
        self.latest = self.context = None

    def diagnostics(self):
        return {**self.bridge.diagnostics(), "framework_registered": True, "adapter": self.variant,
                "predict_calls": self.calls, "cache_rows_remaining": len(self.cache.rows) if self.cache else None,
                "public_snapshot_only": True}
