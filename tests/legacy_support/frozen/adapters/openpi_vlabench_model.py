"""Mapping from openpi-vlabench/examples/vlabench/eval.py Pi0, fixed revision.

Three raw cameras, state7, five-row cache, world EE targets. Normalization,
delta/absolute actions, padding, and model execution remain in the native server.
"""
from collections import deque
from pathlib import Path

import numpy as np

from embodied_eval.errors import AdapterError
import tests.legacy_support.frozen.adapters.openpi_vlabench_spec as spec
from tests.legacy_support.frozen.adapters.native_common import controller_prediction, finite_array, native_module, rgb_image
from tests.legacy_support.frozen.adapters.native_policy_transport import NativePolicyPeer


class OpenpiVLABenchModel:
    def __init__(self, config, output, benchmark_repo):
        self.config, self.output = config, Path(output)
        self.benchmark_repo = benchmark_repo
        self.peer = self.latest = None
        self.cache = deque()
        self.calls = self.native_requests = 0

    def describe(self):
        return spec.capability(self.config)

    def begin_episode(self, context, task, space, profile):
        self.profile, self.period = profile, space.control_period_s
        self.latest = None
        self.cache.clear()
        self.calls = self.native_requests = 0
        try:
            # VLABench's pure quaternion helper, not an invented rotation rule.
            self.utils = native_module(self.benchmark_repo, "VLABench.utils.utils")
            if self.peer is None:
                self.peer = NativePolicyPeer(self.config.openpi_repo, "openpi_client.msgpack_numpy",
                    "packages/openpi-client/src", self.config.host, self.config.port,
                    self.config.request_timeout_s)
            # Native reset clears the client deque, with no server reset RPC.
        except Exception as error:
            raise AdapterError("model_service", "openpi_vlabench_begin_failed", type(error).__name__) from error

    def native_input(self, model_input):
        obs = model_input.observation_view
        ext = obs.extensions[spec.EXTENSION]
        instruction = ext["instruction"]
        if not isinstance(instruction, str) or not instruction:
            raise AdapterError("model_input", "missing_instruction", "native VLABench instruction required")
        state = finite_array(ext["ee_state_world_wxyz"], (8,), "ee_state_world_wxyz")
        # This source revision uses a fixed offset on INPUT, but the public live
        # robot base on OUTPUT. Do not silently replace either with the other.
        position = state[:3] - np.array([0, -0.4, 0.78])
        euler = finite_array(self.utils.quaternion_to_euler(state[3:7]), (3,), "native_euler")
        return {"observation/image": rgb_image(obs, "forward_rgb"),
                "observation/second_image": rgb_image(obs, "right_rgb"),
                "observation/wrist_image": rgb_image(obs, "franka_wrist_rgb"),
                "observation/state": np.concatenate([position, euler, state[-1:]]),
                "prompt": instruction}

    def observe(self, model_input):
        self.latest = model_input

    def predict(self, request_context):
        if self.latest is None:
            raise AdapterError("model_input", "missing_observation", "openpi VLABench needs a public observation")
        if not self.cache:
            payload = self.native_input(self.latest)
            try:
                actions = np.asarray(self.peer.infer(payload)["actions"])
                if (actions.ndim != 2 or actions.shape[0] < spec.REPLAN_STEPS or actions.shape[1] != 7
                        or actions.dtype.kind not in "iuf" or not np.isfinite(actions).all()):
                    raise ValueError("expected at least five finite seven-dimensional actions")
                self.cache.extend(row.copy() for row in actions[:spec.REPLAN_STEPS])
                self.native_requests += 1
            except Exception as error:
                raise AdapterError("model_service", "openpi_vlabench_predict_failed", type(error).__name__) from error
        action = self.cache.popleft()
        base = finite_array(self.latest.observation_view.extensions[spec.EXTENSION]["robot_base_world_m"],
                            (3,), "robot_base_world_m")
        # Fixed native Pi0.predict copies the action slice then uses +=. With
        # float32 actions and a float64 live base that rounds in float32; an
        # out-of-place addition silently promotes precision and changes targets.
        position = action[:3].copy()
        position += base
        position = finite_array(position, (3,), "native_world_target")
        parameters = {"position_world_m": position.tolist(), "euler_xyz_rad": action[3:6].tolist(),
                      "finger_targets_m": [0.04, 0.04] if action[-1] >= 0.1 else [0., 0.]}
        self.calls += 1
        return controller_prediction(request_context, self.latest.observation_view, self.profile,
            spec.ACTION, spec.CONTROLLER, parameters, self.period,
            ("native three-camera state7 mapping; five-action replan cache",))

    def end_episode(self, context):
        self.latest = None
        self.cache.clear()

    def close(self):
        self.latest = None
        self.cache.clear()
        if self.peer is not None:
            self.peer.close()
            self.peer = None

    def diagnostics(self):
        return {"adapter": "openpi_vlabench", "policy_calls": self.calls,
                "native_requests": self.native_requests, "replan_steps": spec.REPLAN_STEPS,
                "model_forward_count": None, "server_identity_verified": False}
