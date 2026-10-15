"""Source-defined RoboTwin model-side bridges, not full pair registrations.

These bridges only return native action arrays. They never execute a simulator,
invent a fixed action duration, or attest checkpoint identity. Higher-level
robotwin_model/robotwin_pairs integrate them using the approved variable-time
contract; real validation receipts remain separate from bridge diagnostics.
"""
from dataclasses import dataclass
import math

import numpy as np

from ..errors import AdapterError
from .native_common import finite_array, native_module
from .native_policy_transport import NativePolicyPeer

from .robotwin_identity import (OPENWAM_REVISION, OPENWAM_ROBOTWIN_REVISION,
                               LINGBOT_REVISION, LINGBOT_ROBOTWIN_REVISION)


@dataclass(frozen=True, kw_only=True)
class NativeEndpoint:
    repo: str
    host: str = "127.0.0.1"
    port: int
    timeout_s: float = 300

    def __post_init__(self):
        if (self.host not in ("127.0.0.1", "localhost") or isinstance(self.port, bool)
                or not 1 <= self.port <= 65535 or not math.isfinite(self.timeout_s)
                or self.timeout_s <= 0):
            raise ValueError("native bridge requires a bounded localhost endpoint")


def _rgb(value, label):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or 0 in image.shape:
        raise AdapterError("model_input", "invalid_native_image", f"{label} requires nonempty uint8 HWC RGB")
    return np.ascontiguousarray(image)


def _instruction(value):
    if not isinstance(value, str) or not value:
        raise AdapterError("model_input", "missing_instruction", "native RoboTwin task text is required")
    return value


class OpenWAMRobotwinNativePolicy:
    """Native three-camera/EEF20 input to RoboTwin EE16 output, server-owned cache."""
    source_revision = OPENWAM_REVISION
    benchmark_revision = OPENWAM_ROBOTWIN_REVISION
    action_type = "ee"

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.peer = None
        self.last_instruction = ""
        self.calls = 0

    def connect(self):
        from ..stage4_pairs import _revision
        _revision(self.endpoint.repo, self.source_revision)
        self.client = native_module(self.endpoint.repo, "benchmarks.utils.client")
        self.conversion = native_module(self.endpoint.repo, "benchmarks.utils.action_conversion")
        self.prompt = native_module(self.endpoint.repo, "benchmarks.robotwin.prompt_template")
        transport = native_module(self.endpoint.repo, "benchmarks.utils.transport")
        self.peer = transport.WSPolicyClient(f"ws://{self.endpoint.host}:{self.endpoint.port}",
                                            timeout=self.endpoint.timeout_s)
        try:
            if self.peer.ping().get("type") != "pong":
                raise ValueError("unexpected native policy liveness response")
        except Exception:
            self.close()
            raise

    def reset(self, instruction=""):
        if self.peer is None:
            self.connect()
        if self.peer.reset().get("type") != "reset_ack":
            raise AdapterError("model_service", "native_reset_failed", "OpenWAM RoboTwin reset was not acknowledged")
        self.last_instruction = instruction
        self.calls = 0

    def native_input(self, observation, instruction):
        # Project explicitly: segmentation, expert plans and scores never enter payload.
        instruction = _instruction(instruction)
        cameras = observation["observation"]
        endpose = observation["endpose"]
        left = finite_array(endpose["left_endpose"], (7,), "left_endpose")
        right = finite_array(endpose["right_endpose"], (7,), "right_endpose")
        grippers = [finite_array(np.asarray(endpose[key]).reshape(-1), (1,), key)[0]
                    for key in ("left_gripper", "right_gripper")]
        state = self.conversion.robotwin_endpose_to_eef20d(left, right, *grippers)
        encoded = {name: self.client.encode_numpy_b64(_rgb(cameras[key]["rgb"], key))
                   for name, key in (("head", "head_camera"), ("left_wrist", "left_camera"),
                                     ("right_wrist", "right_camera"))}
        return self.client.build_payload(**encoded, prompt=self.prompt.format_prompt_for_inference(instruction),
                                         state=finite_array(state, (20,), "native EEF20", np.float32).tolist())

    def predict(self, observation, instruction):
        if self.peer is None:
            self.connect()
        payload = self.native_input(observation, instruction)
        if instruction != self.last_instruction:
            # The official ModelClient also clears the server cache on a text change.
            self.reset(instruction)
        try:
            reply = self.peer.predict_once(payload)
            raw = finite_array(reply["action"], (20,), "OpenWAM EEF20 output", np.float32)
            action = finite_array(self.conversion.eef20d_to_ee16d(raw), (16,), "RoboTwin EE16 output", np.float32)
        except Exception as error:
            raise AdapterError("model_service", "native_predict_failed", type(error).__name__) from error
        self.calls += 1
        return action.copy()

    def close(self):
        if self.peer is not None:
            self.peer.close()
            self.peer = None
        self.last_instruction = ""

    def diagnostics(self):
        return {"bridge": "openwam_robotwin", "framework_registered": False,
                "policy_requests": self.calls, "server_identity_verified": False,
                "action_type": self.action_type, "model_forward_count": None}


class LingbotRobotwinNativePolicy:
    """Exact native observation names/reset and returned joint14 chunk, no execution."""
    source_revision = LINGBOT_REVISION
    benchmark_revision = LINGBOT_ROBOTWIN_REVISION
    action_type = "qpos"

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.peer = None
        self.calls = 0

    def connect(self):
        from ..stage4_pairs import _revision
        _revision(self.endpoint.repo, self.source_revision)
        self.peer = NativePolicyPeer(self.endpoint.repo, "deploy.msgpack_numpy", "",
                                    self.endpoint.host, self.endpoint.port, self.endpoint.timeout_s)

    def reset(self):
        if self.peer is None:
            self.connect()
        # No checkpoint path is sent: this must not load or replace server weights.
        self.peer.infer({"reset": True, "robo_name": "robotwin", "path_to_pi_model": None})
        self.calls = 0

    def native_input(self, observation, instruction):
        cameras = observation["observation"]
        return {**{key: _rgb(cameras[source]["rgb"], source) for key, source in (
                    ("observation.images.cam_high", "head_camera"),
                    ("observation.images.cam_left_wrist", "left_camera"),
                    ("observation.images.cam_right_wrist", "right_camera"))},
                "observation.state": finite_array(observation["joint_action"]["vector"], (14,), "joint14"),
                "task": _instruction(instruction)}

    def predict(self, observation, instruction):
        if self.peer is None:
            self.connect()
        payload = self.native_input(observation, instruction)
        try:
            reply = self.peer.infer(payload)
            actions = np.asarray(reply["action"])
            if actions.ndim == 1:
                actions = actions[None, :]
            if (actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 14
                    or actions.dtype.kind not in "iuf" or not np.isfinite(actions).all()):
                raise ValueError("expected a nonempty native joint14 chunk")
        except Exception as error:
            raise AdapterError("model_service", "native_predict_failed", type(error).__name__) from error
        self.calls += 1
        # Preserve the entire native chunk, dtype and order. Benchmark success and
        # action-limit checks must occur between its rows in the future executor.
        return actions.copy()

    def close(self):
        if self.peer is not None:
            self.peer.close()
            self.peer = None

    def diagnostics(self):
        return {"bridge": "lingbot_vla_v2_robotwin", "framework_registered": False,
                "policy_requests": self.calls, "server_identity_verified": False,
                "action_type": self.action_type, "model_forward_count": None}
