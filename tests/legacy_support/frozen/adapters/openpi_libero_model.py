"""pi05_libero preprocessing and five-action replan cache at the model boundary."""
from collections import deque
import math
from pathlib import Path

import numpy as np

from embodied_eval.errors import AdapterError
from tests.legacy_support.frozen.adapters.native_common import controller_prediction, finite_array, native_module, rgb_image
from tests.legacy_support.frozen.adapters.native_policy_transport import NativePolicyPeer
import tests.legacy_support.frozen.adapters.openpi_libero_spec as spec


def quat_to_axisangle(quat):
    """Exact native openpi LIBERO formula; input order is xyzw."""
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


class OpenpiLiberoModel:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.peer = None
        self.cache = deque()
        self.latest = None
        self.calls = self.native_requests = 0

    def describe(self):
        return spec.capability(self.config)

    def begin_episode(self, context, task, space, profile):
        self.profile, self.period = profile, space.control_period_s
        self.cache.clear()
        self.latest = None
        self.calls = self.native_requests = 0
        try:
            self.images = native_module(self.config.openpi_repo, "openpi_client.image_tools",
                                        "packages/openpi-client/src")
            if self.peer is None:
                self.peer = NativePolicyPeer(self.config.openpi_repo, "openpi_client.msgpack_numpy",
                    "packages/openpi-client/src", self.config.host, self.config.port,
                    self.config.request_timeout_s)
            # openpi's official LIBERO reset is a no-op; only its client deque is cleared.
        except Exception as error:
            raise AdapterError("model_service", "openpi_begin_failed", type(error).__name__) from error

    def observe(self, model_input):
        self.latest = model_input

    def native_input(self, model_input):
        obs = model_input.observation_view
        ext = obs.extensions[spec.EXTENSION]
        position = finite_array(ext["eef_position_m"], (3,), "eef_position_m")
        quat = finite_array(ext["eef_quaternion_xyzw"], (4,), "eef_quaternion_xyzw")
        fingers = finite_array(ext["gripper_qpos_m"], (2,), "gripper_qpos_m")
        images = {}
        for key, native_key in (("agentview_image", "observation/image"),
                                ("robot0_eye_in_hand_image", "observation/wrist_image")):
            image = np.ascontiguousarray(rgb_image(obs, key)[::-1, ::-1])
            images[native_key] = self.images.convert_to_uint8(self.images.resize_with_pad(image, 224, 224))
        return {**images, "observation/state": np.concatenate((position, quat_to_axisangle(quat), fingers)),
                "prompt": str(model_input.task_public.instruction)}

    def predict(self, request_context):
        if self.latest is None:
            raise AdapterError("model_input", "missing_observation", "openpi needs a public observation")
        if not self.cache:
            payload = self.native_input(self.latest)
            try:
                reply = self.peer.infer(payload)
                actions = np.asarray(reply["actions"])
                if (actions.ndim != 2 or actions.shape[0] < 5 or actions.shape[1] != 7
                        or actions.dtype.kind not in "iuf" or not np.isfinite(actions).all()):
                    raise ValueError("expected at least five finite seven-dimensional actions")
                self.cache.extend(action.copy() for action in actions[:5])
                self.native_requests += 1
            except Exception as error:
                raise AdapterError("model_service", "openpi_predict_failed", type(error).__name__) from error
        action = self.cache.popleft()
        self.calls += 1
        return controller_prediction(request_context, self.latest.observation_view, self.profile,
            spec.ACTION, spec.CONTROLLER, {"osc_action7": action.tolist()}, self.period,
            ("openpi 180-degree RGB rotation and native 224x224 padding", "native state8; replan every five actions"))

    def end_episode(self, context):
        self.cache.clear()
        self.latest = None

    def close(self):
        self.cache.clear()
        self.latest = None
        if self.peer is not None:
            self.peer.close()
            self.peer = None

    def diagnostics(self):
        return {"adapter": "openpi_libero", "policy_calls": self.calls,
                "native_requests": self.native_requests, "replan_steps": 5,
                "model_forward_count": None, "server_identity_verified": False}
