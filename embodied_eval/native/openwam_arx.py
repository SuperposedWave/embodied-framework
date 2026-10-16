"""ARX-trained absolute dual-pose OpenWAM profile, no environment object.

Native model engine/normalizer are used directly; no XPolicyLab model class,
pair factory, policy server dispatcher or simulator dependency is imported.
"""
from collections import deque
import numpy as np
from ..core import types as leaf
from ..core.validation import require
from ..adapters.native_common import native_module, finite_array
from . import contracts as c
from .native_catalog import model_options, capability
from .transforms import prepare_input

SOURCE_ROOT = "policy/OpenWAM/OpenWAM"


class OpenWAMArxModel:
    def __init__(self, options, output=None):
        self.cfg = model_options("openwam", options)
        require(self.cfg.io_profile == "eef20_arx_absolute", "unknown_io_profile", "model", "named ARX absolute-pose weights required")
        self.cap = capability("openwam", options)
        self.cache = deque()
        self.latest = self.server = None
        self.requests = self.calls = 0

    def _connect(self):
        if self.server is not None:
            return
        from omegaconf import OmegaConf
        from pathlib import Path
        repo = self.cfg.repo
        self.poses = native_module(repo, "openwam.dataloader.utils.poses", SOURCE_ROOT)
        self.calibration = native_module(repo, "openwam.dataloader.robodojo_contract", SOURCE_ROOT).arx_x5_calibration()
        self.format_prompt = native_module(repo, "openwam.dataloader.transforms.multiview", SOURCE_ROOT).format_prompt_for_inference
        deploy = native_module(repo, "openwam.deploy.server", SOURCE_ROOT)
        cfg = deploy._load_deploy_yaml(str(Path(repo) / SOURCE_ROOT / "configs/deploy.yaml"))
        for key, value in (("optimization.dit_cache.enabled", False), ("optimization.compile.enabled", False),
            ("optimization.decode_video", False), ("inference.inference_mode", "sync")):
            OmegaConf.update(cfg, key, value, merge=False)
        self.server = deploy.build_server_from_config(cfg, self.cfg.checkpoint_dir, device="cuda")
        self.server._init_policy()
        self.horizon = OmegaConf.select(self.server.cfg, "inference.inference_horizon", default=None)
        require(self.horizon is None or isinstance(self.horizon, int) and self.horizon > 0,
                "invalid_horizon", "model", "native execution horizon must be positive or full chunk")

    def begin_episode(self, context, task, profile, space):
        self.profile, self.space = profile, space
        self.cache.clear()
        self.latest = None
        self.requests = self.calls = 0
        self._connect()
        # Native Model.reset clears observation buffers, not engine RNG/state.

    def observe(self, model_input):
        self.latest = prepare_input(model_input, self.cap, self.profile, self.space)

    @staticmethod
    def _state(view, key, width):
        tensor = view.states[key].payload
        require(tensor.data is not None, "payload_unavailable", key, "DataRef reading is not implemented")
        return finite_array(tensor.data, (width,), key, np.float64)

    def native_input(self, view):
        from PIL import Image
        poses, grippers = {}, {}
        for side in ("left", "right"):
            position = self._state(view, side + "_position", 3)
            q = self._state(view, side + "_quaternion", 4)
            # Native client normalizes the measured quaternion once, before
            # the pinned frame helper performs its own validation/normalization.
            native_q = q[[3, 0, 1, 2]].copy()
            norm = np.linalg.norm(native_q)
            require(norm > 1e-8, "invalid_orientation", side, "nonzero measured quaternion required")
            native_q /= norm
            arm = self.calibration["arms"][side]
            poses[side] = self.poses.env_relative_world_to_robot_base(np.concatenate((position, native_q)),
                arm["base_pos_relative_to_env_origin"], arm["base_quat_wxyz"])
            grip = self._state(view, side + "_grip", 1)
            require(np.all((grip >= -1e-8) & (grip <= 1 + 1e-8)), "state_out_of_bounds", side, "native gripper tolerance required")
            grippers[side] = np.clip(grip, 0., 1.)
        state = self.poses.arms_to_eef20(poses["left"], grippers["left"], poses["right"], grippers["right"])
        images = {}
        for slot, name in (("head", "head_camera"), ("left_wrist", "left_wrist_camera"), ("right_wrist", "right_wrist_camera")):
            tensor = view.sensors[slot].payload
            require(tensor is not None and tensor.data is not None, "payload_unavailable", slot, "inline RGB required")
            images[name] = Image.fromarray(np.ascontiguousarray(tensor.data, dtype=np.uint8), mode="RGB")
        return {"images": images, "prompt": self.format_prompt(view.task_public.instruction), "state": state.astype(np.float32).tolist()}

    def native_rows(self, chunk):
        chunk = np.asarray(chunk)
        require(chunk.ndim == 2 and chunk.shape[1] == 20 and chunk.shape[0] > 0 and np.isfinite(chunk).all(),
                "invalid_model_output", "actions", "finite dual EEF20 chunk required")
        lp, lg, rp, rg = self.poses.eef20_to_arms(chunk.astype(np.float64))
        values = {}
        for side, pose, grip in (("left", lp, lg), ("right", rp, rg)):
            arm = self.calibration["arms"][side]
            values[side + "_ee_pose"] = self.poses.robot_base_to_env_relative_world(pose,
                arm["base_pos_relative_to_env_origin"], arm["base_quat_wxyz"]).astype(np.float32)
            values[side + "_ee_joint_state"] = np.clip(grip, 0., 1.).astype(np.float32)
        return [{key: array[i].copy() for key, array in values.items()} for i in range(len(chunk))]

    def decode_action(self, row, context):
        commands = []
        targets = self.profile.binding.targets
        for side in ("left", "right"):
            pose = finite_array(row[side + "_ee_pose"], (7,), "native pose", np.float32)
            grip = finite_array(row[side + "_ee_joint_state"], (1,), "native grip", np.float32)
            commands.extend((leaf.PoseTarget(command_id=context.message_id + ":" + side + "p", target_id=targets[side + "_eef"],
                pose=leaf.Pose(position_m=tuple(float(v) for v in pose[:3]), quaternion_xyzw=tuple(float(v) for v in pose[[4, 5, 6, 3]]), frame_id="world")),
                leaf.GripperCommand(command_id=context.message_id + ":" + side + "g", target_id=targets[side + "_grip"], mode="open_fraction", value=float(grip[0]))))
        return c.Action(commands=tuple(commands), native_dtype="float32")

    def predict(self, context):
        require(self.latest is not None, "missing_observation", "model", "public observation required")
        if not self.cache:
            obs = self.server._obs_preprocessor.preprocess(dict(self.native_input(self.latest)))
            conditions = {"observation": obs, "first_frame_image": [obs["image"]], "prompt": obs["prompt"]}
            if obs.get("state") is not None:
                conditions["proprio"] = obs["state"]
            actions = self.server.engine.generate_batch([conditions])["actions"]
            if hasattr(actions, "detach"):
                actions = actions.detach().cpu().numpy()
            actions = np.asarray(actions)
            require(actions.ndim == 3 and actions.shape[0] == 1 and actions.shape[2] == 20,
                    "invalid_model_output", "actions", "one stream of native EEF20 actions required")
            actions = self.server._policy._project_binary_dims(actions)
            count = len(actions[0]) if self.horizon is None else min(self.horizon, len(actions[0]))
            self.cache.extend(self.native_rows(actions[0, :count]))
            self.requests += 1
        action = self.decode_action(self.cache.popleft(), context)
        self.calls += 1
        return c.Prediction(context=context, based_on_observation_id=self.latest.observation_id, action=action, model_forward_count=None,
            transformations=("source-pinned-arx-base-calibration", "native-EEF20-to-public-absolute-dual-pose"))

    def end_episode(self, context):
        self.cache.clear()
        self.latest = None

    def close(self):
        self.cache.clear()
        self.latest = self.server = None

    def diagnostics(self):
        return {"synthetic": False, "native_generation_requests": self.requests, "predict_calls": self.calls,
            "in_process_native_engine": True, "model_forward_count_verified": False, "server_identity_verified": False,
            "checkpoint_calibration": "source-pinned-ARX-X5", "legacy_pair_factory_used": False}
