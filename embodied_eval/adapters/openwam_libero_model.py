"""OpenWAM's official LIBERO client behind the public model contract."""
import importlib
from pathlib import Path
import sys

import numpy as np

from .. import _legacy_contracts as c
from ..errors import AdapterError
from .openwam_libero_spec import ACTION, CONTROLLER, EXTENSION, capability


class OpenWAMLiberoModel:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.policy = None
        self.latest = None
        self.calls = 0

    def describe(self):
        return capability(self.config)

    def begin_episode(self, context, task, space, profile):
        repo = Path(self.config.openwam_repo).resolve()
        if not (repo / "benchmarks/libero/openwam2libero_interface.py").is_file():
            raise AdapterError("model_service", "source_missing", f"OpenWAM source unavailable: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        native = importlib.import_module("benchmarks.libero.openwam2libero_interface")
        try:
            if self.policy is None:
                self.policy = native.OpenWAMLiberoPolicy(
                    host=self.config.host, port=self.config.port,
                    request_timeout=self.config.request_timeout_s,
                    action_mode="eef", image_transform="rotate_180", send_state=True,
                    debug=True, debug_dir=str(self.output / "native-policy-debug"))
            self.policy.reset()
        except Exception as error:
            raise AdapterError("model_service", "openwam_begin_failed", str(error)) from error
        self.context, self.task, self.profile = context, task, profile
        self.latest = None
        self.calls = 0

    def observe(self, model_input):
        self.latest = model_input

    def predict(self, request_context):
        if self.latest is None:
            raise AdapterError("model_input", "missing_observation", "OpenWAM has no public observation")
        obs = self.latest.observation_view
        ext = obs.extensions[EXTENSION]
        values = {}
        for public_name, native_name, shape in (
            ("eef_position_m", "robot0_eef_pos", (3,)),
            ("eef_quaternion_xyzw", "robot0_eef_quat", (4,)),
            ("gripper_qpos_m", "robot0_gripper_qpos", (2,)),
        ):
            value = np.asarray(ext[public_name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise AdapterError("model_input", "invalid_native_proprio",
                                   f"{public_name} must be finite with shape {shape}")
            values[native_name] = value
        native_obs = {
            "agentview_image": np.asarray(obs.sensors["agentview_image"].payload.data, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.asarray(obs.sensors["robot0_eye_in_hand_image"].payload.data, dtype=np.uint8),
            **values,
        }
        try:
            action = self.policy.act(native_obs, self.latest.task_public.instruction)
        except Exception as error:
            raise AdapterError("model_service", "openwam_predict_failed", str(error)) from error
        command = c.ControllerCommand(command_id=request_context.message_id + ":osc7", target_id="robot0/eef",
            controller_profile_ref=CONTROLLER, parameters={"osc_action7": [float(x) for x in action]})
        decision = c.ActionSequence(action_spec_ref=ACTION, execution_horizon_steps=1,
            steps=(c.ActionStep(index=0, duration_s=.05, commands=(command,)),))
        ref = self.output / "native-policy-debug" / f"episode_{self.policy._episode:03d}" / f"step_{self.policy._step-1:04d}" / "meta.json"
        self.calls += 1
        return c.Prediction(context=request_context, request_id=request_context.message_id,
            based_on_observation_id=obs.observation_id, profile_ref=self.profile.profile_id,
            decision=decision, diagnostics=c.PredictionDiagnostics(raw_output_ref=str(ref),
                transformations=("OpenWAM native EEF10 to LIBERO OSC7 (official client)",),
                model_forward_count=None))

    def end_episode(self, context):
        self.latest = None

    def close(self):
        if self.policy is not None:
            self.policy.close()
            self.policy = None

    def diagnostics(self):
        return {"adapter": "openwam_libero", "policy_calls": self.calls,
                "model_forward_count": None, "server": f"{self.config.host}:{self.config.port}"}
