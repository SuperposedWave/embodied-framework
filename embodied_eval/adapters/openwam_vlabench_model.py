"""Native OpenWAM VLABench view mapping and action conversion."""
from pathlib import Path
import numpy as np

from ..errors import AdapterError
from .native_common import controller_prediction, finite_array, native_module, rgb_image
from . import vlabench_spec as spec


class OpenWAMVLABenchModel:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.policy = self.latest = None
        self.calls = 0

    def describe(self):
        return spec.capability(self.config)

    def begin_episode(self, context, task, space, profile):
        self.profile, self.period = profile, space.control_period_s
        self.latest = None
        self.calls = 0
        try:
            native = native_module(self.config.openwam_repo, "benchmarks.vlabench.openwam2vlabench_interface")
            if self.policy is None:
                # Public sensors are reordered to forward / Franka wrist / right.
                self.policy = native.OpenWAMVLABenchPolicy(host=self.config.host, port=self.config.port,
                    request_timeout=self.config.request_timeout_s, head_camera_index=0,
                    left_wrist_camera_index=1, right_wrist_camera_index=2,
                    send_state=True, state_dim=10, action_dim=10, debug=True,
                    debug_dir=str(self.output / "native-policy-debug"))
                # A lost response may already have advanced the server's chunk cache.
                # Fail instead of silently resending that ambiguous observation.
                self.policy._client.predict = self.policy._client.predict_once
            self.policy.reset()
        except Exception as error:
            raise AdapterError("model_service", "openwam_vlabench_begin_failed", type(error).__name__) from error

    def observe(self, model_input):
        self.latest = model_input

    def native_input(self, model_input):
        obs = model_input.observation_view
        ext = obs.extensions[spec.EXTENSION]
        instruction = ext["instruction"]
        if not isinstance(instruction, str) or not instruction:
            raise AdapterError("model_input", "missing_instruction", "VLABench requires its public native instruction")
        return {"rgb": np.stack([rgb_image(obs, key) for key in spec.SENSORS]),
                "ee_state": finite_array(ext["ee_state_world_wxyz"], (8,), "ee_state_world_wxyz"),
                "robot_frame": finite_array(ext["robot_base_world_m"], (3,), "robot_base_world_m"),
                "instruction": instruction}

    def predict(self, request_context):
        if self.latest is None:
            raise AdapterError("model_input", "missing_observation", "VLABench needs a public observation")
        native_obs = self.native_input(self.latest)
        try:
            pos, euler, fingers = self.policy.predict(native_obs)
            parameters = {"position_world_m": finite_array(pos, (3,), "position_world_m").tolist(),
                          "euler_xyz_rad": finite_array(euler, (3,), "euler_xyz_rad").tolist(),
                          "finger_targets_m": finite_array(fingers, (2,), "finger_targets_m").tolist()}
        except Exception as error:
            raise AdapterError("model_service", "openwam_vlabench_predict_failed", type(error).__name__) from error
        self.calls += 1
        return controller_prediction(request_context, self.latest.observation_view, self.profile,
            spec.ACTION, spec.CONTROLLER, parameters, self.period,
            ("native OpenWAM VLABench camera resize and EEF10 conversion",))

    def end_episode(self, context):
        self.latest = None

    def close(self):
        if self.policy is not None:
            self.policy.close()
            self.policy = None
        self.latest = None

    def diagnostics(self):
        return {"adapter": "openwam_vlabench", "policy_calls": self.calls,
                "model_forward_count": None, "server_identity_verified": False}
