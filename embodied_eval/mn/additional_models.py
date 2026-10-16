"""Independent robot-action policy clients. No simulator objects or names.

The policy server owns checkpoint normalization/inference. This boundary owns
public-to-wire packing, native chunk cache, session boundaries and action decode.
"""
from collections import deque
from uuid import uuid4
import numpy as np
from .. import _legacy_contracts as leaf
from .._legacy_validation import require
from ..adapters.native_common import native_module, finite_array
from ..adapters.native_policy_transport import NativePolicyPeer
from . import contracts as c
from .additional_catalog import policy_options, policy_capability, POLICIES
from .transforms import prepare_input


class JointPolicyModel:
    def __init__(self, family, options, output=None):
        self.family, self.cfg = family, policy_options(family, options)
        self.cap = policy_capability(family, options)
        self.peer = self.latest = None
        self.cache = deque()
        self.requests = self.calls = 0
        self.session_id = None
        self.horizon = self.cfg.open_loop_horizon or POLICIES[family]["horizon"]

    def begin_episode(self, context, task, profile, space):
        self.profile, self.space = profile, space
        self.cache.clear()
        self.latest = None
        self.requests = self.calls = 0
        self.session_id = str(uuid4())
        self.last_instruction = task.instruction
        if self.peer is None:
            self.peer = NativePolicyPeer(self.cfg.codec_repo, "openpi_client.msgpack_numpy",
                "packages/openpi-client/src", self.cfg.host, self.cfg.port, self.cfg.request_timeout_s)
        if self.family == "dreamzero":
            self.images = native_module(self.cfg.codec_repo, "openpi_client.image_tools", "packages/openpi-client/src")
        else:
            self.images = native_module(self.cfg.client_repo, "robolab.core.utils.image_utils")
        # DreamZero's native session_id change resets causal server buffers.
        # Cosmos's reviewed server is stateless per observation. Neither uses
        # a fabricated reset endpoint or consumes an inference during reset.

    def observe(self, model_input):
        self.latest = prepare_input(model_input, self.cap, self.profile, self.space)

    @staticmethod
    def _array(view, key, image=False):
        tensor = (view.sensors if image else view.states)[key].payload
        require(tensor is not None and tensor.data is not None, "payload_unavailable", key,
                "inline data required; DataRef loading is not implemented")
        return np.asarray(tensor.data, dtype=np.uint8 if image else np.float64)

    def native_input(self, view):
        joints = finite_array(self._array(view, "joints"), (7,), "measured joints", np.float64)
        opening = finite_array(self._array(view, "grip"), (1,), "measured gripper opening", np.float64)
        require(0 <= opening[0] <= 1, "state_out_of_bounds", "grip", "measured opening fraction must be 0..1")
        rgb = {key: self._array(view, key, True) for key in ("exterior_right", "exterior_left", "wrist")}
        require(all(img.ndim == 3 and img.shape[-1] == 3 for img in rgb.values()),
                "invalid_native_image", "images", "HWC RGB required")
        data = {"observation/joint_position": joints,
                "observation/gripper_position": 1. - opening,
                "prompt": view.task_public.instruction}
        if self.family == "dreamzero":
            data.update({name: self.images.resize_with_pad(rgb[slot], 180, 320) for slot, name in (
                ("exterior_right", "observation/exterior_image_0_left"),
                ("exterior_left", "observation/exterior_image_1_left"),
                ("wrist", "observation/wrist_image_left"))})
            # Source client sends this compatibility field, but the reviewed
            # DROID server consumes joints+gripper, not Cartesian measurements.
            data.update({"observation/cartesian_position": np.zeros((6,), np.float64),
                         "session_id": self.session_id, "endpoint": "infer"})
        else:
            # Preserve the reference RoboLab client: PIL pad to 360x640,
            # torch bilinear half-size exterior views, wrist above left/right.
            # Its measured articulation tensors arrive as float32; the public
            # format remains float64, model-side packing restores native dtype.
            data["observation/joint_position"] = joints.astype(np.float32)
            data["observation/gripper_position"] = (1. - opening).astype(np.float32)
            import torch
            import torch.nn.functional as functional
            views = {key: self.images.resize_with_pad(img, 360, 640) for key, img in rgb.items()}
            def half(img):
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                return functional.interpolate(tensor, size=(180, 320), mode="bilinear").squeeze(0).permute(1, 2, 0).numpy().astype(views["wrist"].dtype)
            data.update({"observation/image": np.concatenate((views["wrist"],
                np.concatenate((half(views["exterior_left"]), half(views["exterior_right"])), axis=1))),
                "session_id": self.session_id})
        return data

    def decode_action(self, row, context):
        row = finite_array(row, (8,), "native joint8")
        closed = float(row[-1])
        # Both reviewed clients binarize with strict >.5, including .5 and
        # finite out-of-range diffusion predictions. This is source-defined
        # decoding, not a generic runner clip or an invented action meaning.
        opening = 0. if closed > .5 else 1.
        # DreamZero's native client concatenates np.ones/np.zeros (float64)
        # onto the joint row, which promotes a float32 output to float64.
        # Cosmos's client instead mutates a same-dtype chunk copy.
        native_dtype = "float64" if self.family == "dreamzero" else str(row.dtype)
        targets = self.profile.binding.targets
        commands = tuple(leaf.JointTarget(command_id=context.message_id + f":j{i}",
            target_id=targets[f"j{i}"], mode="position", value=float(row[i]), unit="rad") for i in range(7))
        commands += (leaf.GripperCommand(command_id=context.message_id + ":grip",
            target_id=targets["grip"], mode="open_fraction", value=opening),)
        return c.Action(commands=commands, native_dtype=native_dtype)

    def predict(self, context):
        require(self.latest is not None, "missing_observation", "model", "observe before predict")
        self.calls += 1
        # Native policies execute their pending open-loop chunk even if a live
        # instruction changes; the next request consumes the newest instruction.
        if not self.cache:
            result = self.peer.infer(self.native_input(self.latest))
            self.requests += 1
            if self.family == "cosmos3":
                require(isinstance(result, dict) and "action" in result,
                        "invalid_native_output", "model", "Cosmos action chunk required")
                chunk = result["action"]
            else:
                chunk = result.get("actions") if isinstance(result, dict) else result
            chunk = np.asarray(chunk)
            require(chunk.ndim == 2 and chunk.shape[1] == 8 and chunk.shape[0] >= self.horizon
                    and chunk.dtype in (np.dtype("float32"), np.dtype("float64")) and np.isfinite(chunk).all(),
                    "invalid_native_output", "model", "finite floating joint8 chunk must cover the execution horizon")
            self.cache.extend(row.copy() for row in chunk[:self.horizon])
        action = self.decode_action(self.cache.popleft(), context)
        return c.Prediction(context=context, based_on_observation_id=self.latest.observation_id,
            action=action, model_forward_count=None,
            transformations=("source-defined-joint8-wire-decode", "closed-fraction-to-public-opening"))

    def end_episode(self, context):
        self.cache.clear()
        self.latest = None

    def close(self):
        if self.peer is not None:
            self.peer.close()
            self.peer = None
        self.cache.clear()

    def diagnostics(self):
        return {"synthetic": False, "policy_requests": self.requests, "predict_calls": self.calls,
                "cache_rows_remaining": len(self.cache), "execution_horizon": self.horizon,
                "model_forward_count_verified": False, "server_identity_verified": False,
                "normalization_and_inference_owner": "pinned-policy-server"}
