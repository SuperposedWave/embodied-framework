"""Model-family boundary, selected by checkpoint I/O profile, not environment.

No simulator package, simulator configuration, legacy pair factory, or private
environment object is accepted. Native model-side pure helpers are source-pinned.
"""
from collections import deque
import math
import numpy as np
from .. import contracts as leaf
from ..errors import AdapterError
from ..validation import require
from ..adapters.native_common import finite_array, native_module
from ..adapters.native_policy_transport import NativePolicyPeer
from . import contracts as c
from .native_catalog import model_options, capability, OSC_PROTOCOL, DUAL_PROTOCOL, GRIPPER_PROTOCOL, GRIPPER_CONTROLLER
from .rotations import euler_quaternion_xyzw, quat_axisangle
from .joint_native import RAW_GRIPPER_PROTOCOL


class NativeActionModel:
    def __init__(self,family,options,output=None):
        self.family,self.cfg=family,model_options(family,options)
        self.cap=capability(family,options)
        self.output=output
        self.peer=self.latest=None
        self.cache=deque()
        self.calls=self.requests=0
        self.last_instruction=None

    def _connect(self):
        if self.peer is not None:
            return
        repo=self.cfg.repo
        if self.family=="openwam":
            self.utils=native_module(repo,"benchmarks.utils")
            self.conversion=native_module(repo,"benchmarks.utils.action_conversion")
            transport=native_module(repo,"benchmarks.utils.transport")
            self.peer=transport.WSPolicyClient(f"ws://{self.cfg.host}:{self.cfg.port}",timeout=self.cfg.request_timeout_s)
            pong=self.peer.ping()
            require(pong.get("type")=="pong","native_ping_failed","endpoint","unexpected policy liveness response")
            if self.cfg.io_profile=="eef10_osc":
                require(pong.get("representation")=="eef","model_representation_mismatch","endpoint","native EEF output required")
        elif self.family=="openpi":
            codec_root = "policy/Pi_05/openpi/packages/openpi-client/src" if self.cfg.io_profile == "arx_joint14" else "packages/openpi-client/src"
            self.peer=NativePolicyPeer(repo,"openpi_client.msgpack_numpy",codec_root,self.cfg.host,self.cfg.port,self.cfg.request_timeout_s)
            if self.cfg.io_profile=="state8_osc":
                self.images=native_module(repo,"openpi_client.image_tools","packages/openpi-client/src")
        else:
            self.peer=NativePolicyPeer(repo,"deploy.msgpack_numpy","",self.cfg.host,self.cfg.port,self.cfg.request_timeout_s)

    def begin_episode(self,context,task,profile,space):
        self.context,self.profile,self.space=context,profile,space
        self.cache.clear()
        self.latest=None
        self.calls=self.requests=0
        self.last_instruction=task.instruction
        self._connect()
        if self.family=="openwam":
            require(self.peer.reset().get("type")=="reset_ack","native_reset_failed","model","reset acknowledgement required")
        elif self.family=="lingbot_vla":
            self.peer.infer({"reset":True,"robo_name":"robotwin","path_to_pi_model":None})

    def observe(self,model_input):
        from .transforms import prepare_input
        self.latest=prepare_input(model_input,self.cap,self.profile,self.space)

    @staticmethod
    def _state(view,key):
        require(view.states[key].payload.data is not None,"payload_unavailable",key,"DataRef reading is not implemented")
        return np.asarray(view.states[key].payload.data,dtype=np.float64)

    @staticmethod
    def _image(view,key):
        require(view.sensors[key].payload.data is not None,"payload_unavailable",key,"DataRef reading is not implemented")
        image=np.asarray(view.sensors[key].payload.data,dtype=np.uint8)
        require(image.ndim==3 and image.shape[-1]==3,"invalid_native_image",key,"HWC RGB required")
        return image

    def native_input(self,view):
        p=self.cfg.io_profile
        state=lambda key:self._state(view,key)
        image=lambda key:self._image(view,key)
        instruction=view.task_public.instruction
        if self.family=="openpi":
            if p == "arx_joint14":
                packed = np.concatenate(tuple(np.concatenate((state(side+"_joints"), state(side+"_grip"))) for side in ("left", "right"))).astype(np.float32)
                return {"state": packed, "images": {name: np.transpose(image(slot), (2, 0, 1))
                    for slot, name in (("head", "cam_high"), ("left_wrist", "cam_left_wrist"), ("right_wrist", "cam_right_wrist"))},
                    "prompt": instruction}
            if p=="state8_osc":
                rgb={name:self.images.convert_to_uint8(self.images.resize_with_pad(np.ascontiguousarray(image(slot)[::-1,::-1]),224,224))
                     for slot,name in (("head","observation/image"),("left_wrist","observation/wrist_image"))}
                return {**rgb,"observation/state":np.concatenate((state("position"),quat_axisangle(state("quaternion")),state("fingers"))),"prompt":instruction}
            from scipy.spatial.transform import Rotation
            return {"observation/image":image("head"),"observation/second_image":image("right_wrist"),
                    "observation/wrist_image":image("left_wrist"),"observation/state":np.concatenate((
                    state("position")-np.array([0.,-.4,.78]),Rotation.from_quat(state("quaternion")).as_euler("xyz"),1.-state("grip"))),"prompt":instruction}
        if self.family=="lingbot_vla":
            return {"observation.images.cam_high":image("head"),"observation.images.cam_left_wrist":image("left_wrist"),
                    "observation.images.cam_right_wrist":image("right_wrist"),"observation.state":state("drive_targets"),"task":instruction}
        if p=="eef10_osc":
            native_state=self.conversion.libero_obs_to_eef10(state("position"),state("quaternion"),state("fingers"))
        elif p=="eef10_absolute":
            quat=state("quaternion")
            native_state=self.conversion.vlabench_obs_to_eef10(np.concatenate((state("position"),quat[[3,0,1,2]],1.-state("grip"))),state("base"))
        elif p=="eef20_dual_wxyz_encoded_absolute":
            from .openwam_dual_pose import encode_state
            native_state=encode_state(state,self.conversion)
            prompt=native_module(self.cfg.repo,"benchmarks.robotwin.prompt_template")
            instruction=prompt.format_prompt_for_inference(instruction)
        else:
            native_poses=[]
            for side in ("left","right"):
                quat=state(side+"_quaternion")
                # Retain the source-defined convention conflict ONLY in this
                # explicitly legacy I/O profile; never call it physical xyzw.
                native_poses.append(np.concatenate((state(side+"_position"),quat[[3,0,1,2]])))
            native_state=self.conversion.robotwin_endpose_to_eef20d(*native_poses,state("left_grip")[0],state("right_grip")[0])
            prompt=native_module(self.cfg.repo,"benchmarks.robotwin.prompt_template")
            instruction=prompt.format_prompt_for_inference(instruction)
        encoded={}
        for key in ("head","left_wrist","right_wrist"):
            if key not in view.sensors:
                continue
            frame=image(key)
            if p=="eef10_osc":
                frame=frame[::-1,::-1]
            if p not in ("eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute"):
                frame=self.utils.resize_for_lshape_slot(frame,{"head":"head_camera","left_wrist":"left_wrist_camera","right_wrist":"right_wrist_camera"}[key])
            encoded[key]=self.utils.encode_numpy_b64(np.ascontiguousarray(frame))
        return self.utils.build_payload(**encoded,prompt=instruction,state=np.asarray(native_state,np.float32).tolist())

    def decode_action(self,row,context,view,acquire=True):
        p=self.cfg.io_profile
        targets=self.profile.binding.targets
        command_id=context.message_id
        if p=="eef20_dual_wxyz_encoded_absolute":
            from .openwam_dual_pose import decode_action
            return decode_action(row,context,targets,self.conversion,acquire=acquire)
        if p in ("eef10_osc","state8_osc"):
            if p=="eef10_osc":
                raw=finite_array(row,(10,),"native EEF10",np.float32)
                osc=np.concatenate((np.clip(raw[:3],-1,1),np.clip(self.conversion.rot6d_to_axis_angle(raw[3:9]),-1,1),
                                    np.array([self.conversion.libero_open_scale_to_gripper_cmd(raw[9])],np.float32))).astype(np.float32)
            else:
                osc=finite_array(row,(7,),"native OSC7")
            cmd=leaf.ControllerCommand(command_id=command_id+":osc",target_id=targets["eef"],controller_profile_ref="osc-pose-normalized7@1",parameters={"osc_action7":osc.tolist()})
            return c.Action(commands=(cmd,),native_protocol=OSC_PROTOCOL,native_dtype=str(osc.dtype))
        if p=="eef20_dual_legacy":
            raw=finite_array(row,(20,),"native EEF20",np.float32)
            native=finite_array(self.conversion.eef20d_to_ee16d(raw),(16,),"native legacy EE16",np.float32)
            cmd=leaf.ControllerCommand(command_id=command_id+":native",target_id=targets["dual"],controller_profile_ref="dual-eef16-legacy@1",
                                       parameters={"native_row":native.tolist(),"native_dtype":str(native.dtype)})
            return c.Action(commands=(cmd,),native_protocol=DUAL_PROTOCOL,native_dtype=str(native.dtype))
        if p in ("joint14", "arx_joint14"):
            native=finite_array(row,(14,),"native joint14")
            commands=[]
            for side,offset in (("left",0),("right",7)):
                for i in range(6):
                    slot=side+f"_j{i}"
                    commands.append(leaf.JointTarget(command_id=command_id+":"+slot,target_id=targets[slot],mode="position",value=float(native[offset+i]),unit="rad"))
                slot=side+"_grip"
                commands.append(leaf.ControllerCommand(command_id=command_id+":"+slot,target_id=targets[slot],
                    controller_profile_ref=GRIPPER_CONTROLLER,parameters={"requested_open_fraction":float(native[offset+6])}))
            return c.Action(commands=tuple(commands),acquire_observation_after=True if p == "arx_joint14" else acquire,
                native_dtype=str(native.dtype),native_protocol=RAW_GRIPPER_PROTOCOL if p == "arx_joint14" else GRIPPER_PROTOCOL)
        if p=="eef10_absolute":
            raw=finite_array(row,(10,),"native EEF10",np.float32)
            position,euler,fingers=self.conversion.eef10_to_vlabench_ee(raw,self._state(view,"base"))
        else:
            raw=finite_array(row,(7,),"native absolute action7")
            position=raw[:3].copy()
            position+=self._state(view,"base") # preserve native in-place rounding
            euler=raw[3:6]
            fingers=np.array([.04,.04]) if raw[-1]>=.1 else np.zeros(2)
        pose=leaf.PoseTarget(command_id=command_id+":pose",target_id=targets["eef"],pose=leaf.Pose(position_m=tuple(float(x) for x in position),
                            quaternion_xyzw=euler_quaternion_xyzw(euler),frame_id="world"))
        grip=leaf.GripperCommand(command_id=command_id+":grip",target_id=targets["grip"],mode="width_m",value=float(np.sum(fingers)))
        return c.Action(commands=(pose,grip),source_euler_xyz_rad={targets["eef"]:tuple(float(x) for x in euler)})

    def predict(self,context):
        require(self.latest is not None,"missing_observation","model","public observation required")
        forwarded=None
        if not self.cache:
            instruction=self.latest.task_public.instruction
            if self.family=="openwam" and self.cfg.io_profile in ("eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute") and instruction!=self.last_instruction:
                require(self.peer.reset().get("type")=="reset_ack","native_reset_failed","model","text-change reset failed")
                self.last_instruction=instruction
            payload=self.native_input(self.latest)
            if self.family=="openwam":
                reply=self.peer.predict_once(payload) # ambiguous requests are never resent
                if self.cfg.io_profile=="eef20_dual_wxyz_encoded_absolute":
                    original=finite_array(reply["action"],(20,),"policy output")
                    require(np.array_equal(original,original.astype(np.float32)),"native_dtype_loss","policy output",
                            "new profile must not silently narrow a different native representation")
                rows=finite_array(reply["action"],(20 if self.cfg.io_profile in ("eef20_dual_legacy", "eef20_dual_wxyz_encoded_absolute") else 10,),"policy output",np.float32)[None,:]
            else:
                reply=self.peer.infer(payload)
                rows=np.asarray(reply["action" if self.family=="lingbot_vla" else "actions"])
                if rows.ndim==1:
                    rows=rows[None,:]
                width=14 if self.family=="lingbot_vla" or self.cfg.io_profile == "arx_joint14" else 7
                require(rows.ndim==2 and rows.shape[1]==width and rows.dtype.kind in "iuf" and np.isfinite(rows).all(),"invalid_model_output","actions","finite native action chunk required")
                if self.family=="openpi" and self.cfg.io_profile != "arx_joint14":
                    require(len(rows)>=5,"invalid_model_output","actions","native five-row replan requires five actions")
                    rows=rows[:5]
                if self.cfg.io_profile == "arx_joint14":
                    require(len(rows) == 50, "invalid_model_output", "actions", "pinned pi05 ARX configuration returns 50-row chunks")
            self.cache.extend(row.copy() for row in rows)
            self.requests+=1
        row=self.cache.popleft()
        action=self.decode_action(row,context,self.latest,acquire=not self.cache)
        from .transforms import convert_pose_action
        action=convert_pose_action(action,self.profile,self.space,self.latest)
        self.calls+=1
        return c.Prediction(context=context,based_on_observation_id=self.latest.observation_id,action=action,
                            model_forward_count=forwarded,transformations=("checkpoint I/O profile:"+self.cfg.io_profile,))

    def end_episode(self,context):
        self.cache.clear(); self.latest=None

    def close(self):
        self.cache.clear(); self.latest=None
        if self.peer is not None:
            self.peer.close(); self.peer=None

    def diagnostics(self):
        return {"family":self.family,"io_profile":self.cfg.io_profile,"policy_calls":self.calls,"native_requests":self.requests,
                "cache_rows_remaining":len(self.cache),"server_identity_verified":False,"public_samples_only":True}
