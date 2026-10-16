"""Simulator boundary: no model config, model endpoint or model source needed.

Existing environment-only loaders/scorers are reused, NOT pair factories or
model adapters. Protocol selection is explicit simulator configuration.
"""
from dataclasses import replace
from contextlib import contextmanager, nullcontext
import copy
import numpy as np
from .. import _legacy_contracts as leaf
from ..adapters.runtime_options import BenchmarkConfig, EpisodeConfig
from ..codec import to_data
from ..errors import AdapterError
from .._legacy_validation import require
from ..geometry import normalize_quaternion
from ..adapters.robotwin_benchmark import RobotwinBenchmark, _RngDomain
from ..adapters.native_robotwin_runtime import prewarm_native_cuda
from ..adapters.native_common import finite_array, native_module
from . import contracts as c
from .native_catalog import benchmark_options, space as public_space, DUAL_PROTOCOL


@contextmanager
def observe_native_motion_failures(robot, mode):
    """Observe native failures WITHOUT changing native planning or execution.

    The source catches TOPP exceptions/empty paths and continues the gripper,
    or ignores an unsuccessful pose plan. Those are partial executions, not
    successful atomic arm commands. Restore every observed entry after the call.
    TOPP means the native joint-trajectory timing routine, not an IK solver.
    """
    failed, restored = set(), []
    def observer(original, side):
        def call(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
            except Exception:
                failed.add(side)
                raise
            if mode == "ee":
                if not isinstance(result, dict) or result.get("status") != "Success" or len(result.get("position", ())) == 0:
                    failed.add(side)
            elif len(result[1]) == 0:
                failed.add(side)
            return result
        return call
    class PlannerObserver:
        def __init__(self, original, side):
            self.original = original
            self.TOPP = observer(original.TOPP, side)
        def __getattr__(self, name):
            return getattr(self.original, name)
    try:
        for side in ("left", "right"):
            name = side + ("_plan_path" if mode == "ee" else "_mplib_planner")
            original = getattr(robot, name)
            replacement = observer(original, side) if mode == "ee" else PlannerObserver(original, side)
            setattr(robot, name, replacement)
            restored.append((name, original))
        yield failed
    finally:
        for name, original in restored:
            setattr(robot, name, original)


def legacy_context(context):
    return leaf.Context(run_id=context.run_id,env_id=context.env_id,episode_id=context.episode_id,message_id=context.message_id)


class _IndependentRobotwinRuntime(RobotwinBenchmark):
    """Simulator's historical protocol labels do not select the current model."""
    def _prepare_cuda_runtime(self):
        prewarm_native_cuda()

    def _choose_instruction(self,episode_info):
        if self._instruction_protocol=="sealed-instruction":
            # Port of fixed eval_manifest.choose_instructions: a fresh isolated
            # seed domain, description limit100, identical selection state for
            # seen/unseen. No import of a model repository.
            with _RngDomain(self.selected_seed).activate():
                descriptions=self.generator(self.config.task_name,[episode_info],100)
                selection=np.random.get_state()
                selected={}
                for key in ("seen","unseen"):
                    np.random.set_state(selection)
                    selected[key]=str(np.random.choice(descriptions[0][key]))
            return selected[self.config.instruction_type]
        descriptions=self.generator(self.config.task_name,[episode_info],100)
        return str(np.random.choice(descriptions[0][self.config.instruction_type]))


class NativeSimulator:
    def __init__(self,family,options,output):
        self.family,self.cfg=family,benchmark_options(family,options)
        self.space=public_space(family,options)
        self.tasks={}
        cfg=self.cfg
        if family=="libero":
            from ..adapters.libero_spatial_benchmark import LiberoSpatialBenchmark
            from ..adapters import openpi_libero_spec
            class PinnedLiberoRuntime(LiberoSpatialBenchmark):
                def _load_init_states(self):
                    # Task bytes were verified BEFORE construction. Keep the
                    # trusted historical NumPy loader in our adapter, not in a
                    # modified reference checkout (PyTorch2.6 default changed).
                    import torch
                    from pathlib import Path
                    path=Path(self.config.libero_repo)/"libero/libero/init_files"/self.native_task.problem_folder/self.native_task.init_states_file
                    return torch.load(path,weights_only=False)
            if cfg.protocol=="spatial-10settle":
                PinnedLiberoRuntime.space_factory=staticmethod(openpi_libero_spec.space)
                PinnedLiberoRuntime.controller_ref=openpi_libero_spec.CONTROLLER
                PinnedLiberoRuntime.settle_steps=10
                PinnedLiberoRuntime.native_state_dtype=np.float64
                PinnedLiberoRuntime.action_dtype=None
                PinnedLiberoRuntime.score_revision="f78abd6"
            self.runtime=PinnedLiberoRuntime(BenchmarkConfig(adapter="libero_spatial",libero_repo=cfg.repo))
            try:
                native_module(cfg.repo,"libero.libero")
            except Exception:
                self.runtime.close()
                raise
        elif family=="vlabench":
            from ..adapters.vlabench_benchmark import VLABenchBenchmark
            self.runtime=VLABenchBenchmark(BenchmarkConfig(adapter="vlabench",vlabench_repo=cfg.repo,
                task_name=cfg.task_name,track=cfg.track,trial_index=cfg.trial_index))
        else:
            # These names retain the historical simulator revision/initialization
            # defaults only. No selected model/config/repo is passed to runtime.
            legacy_protocol="openwam_robotwin" if cfg.protocol=="sealed-instruction" else "lingbot_robotwin"
            self.runtime=_IndependentRobotwinRuntime(BenchmarkConfig(adapter="robotwin_lingbot",robotwin_repo=cfg.repo,
                task_name=cfg.task_name,task_config=cfg.task_config,instruction_type=cfg.instruction_type,
                max_seed_candidates=cfg.max_seed_candidates),legacy_protocol,output)
            self.runtime._instruction_protocol=cfg.protocol

    def describe(self):
        actual=self.runtime.describe()
        if self.family=="vlabench":
            self.space=replace(self.space,control_period_s=actual.control_period_s)
        if self.family=="robotwin":
            self.space=replace(self.space,physics_period_s=actual.physics_period_s)
        return self.space

    def enumerate_tasks(self,episodes):
        require(len(episodes)==1,"pilot_scope","episodes","rebuild native scene between trials; one current trial per process")
        old_eps=[to_data(EpisodeConfig(episode_id=e.episode_id,seed=e.seed,budgets=e.budgets)) for e in episodes]
        old_tasks=self.runtime.enumerate_tasks({"episodes":old_eps})
        self.tasks={t.episode_id:t for t in old_tasks}
        return [c.TaskSpec(task_id=t.task_id,episode_id=t.episode_id,public=t.public,budgets=t.budgets,
            initialization_ref=t.initialization_ref,evaluation_config_ref=t.evaluation_config_ref,
            evaluation_private=t.evaluation_private) for t in old_tasks]

    def _project(self,obs):
        old=obs.context
        context=c.Context(run_id=old.run_id,env_id=old.env_id,episode_id=old.episode_id,message_id=old.message_id)
        states={}
        instruction=None
        def add(key,value,captured_at=None,commanded=False):
            array=np.asarray(value,np.float64).reshape(-1)
            states[key]=c.StateSample(captured_at_s=obs.sim_time_s if captured_at is None else captured_at,
                provenance="controller_target" if commanded else "measurement",
                payload=leaf.Tensor(dtype="float64",shape=array.shape,layout="vector",data=array.tolist()))
        if self.family=="libero":
            ext=obs.extensions["libero.native_proprio.v1"]
            add("eef_position",ext["eef_position_m"])
            add("eef_quaternion",ext["eef_quaternion_xyzw"])
            add("gripper_positions",ext["gripper_qpos_m"])
        elif self.family=="vlabench":
            ext=obs.extensions["vlabench.native_proprio.v1"]
            ee=np.asarray(ext["ee_state_world_wxyz"],np.float64)
            add("eef_position",ee[:3])
            add("eef_quaternion",ee[[4,5,6,3]])
            add("gripper_open_fraction",[1.-ee[-1]]) # native accessor is an inverted closed flag
            add("robot_base_position",ext["robot_base_world_m"])
            instruction=ext["instruction"]
        else:
            ext=obs.extensions["robotwin.native_proprio.v1"]
            captured=ext["captured_at_sim_time_s"]
            add("joint_drive_targets",ext["joint_drive_targets_and_grippers"],captured,True)
            for side in ("left","right"):
                ee=np.asarray(ext[side+"_endpose_world_wxyz"],np.float64)
                add(side+"_eef_position",ee[:3],captured)
                add(side+"_eef_quaternion",ee[[4,5,6,3]],captured)
                add(side+"_gripper_open_fraction",[ext[side+"_gripper_open_fraction"]],captured,True)
                # Query articulation qpos directly, without capturing another
                # camera frame or consuming the environment's sampling RNG.
                # The native getter's last scalar is a gripper COMMAND; only
                # its six arm qpos values are measurements.
                with self.runtime._scope():
                    measured=finite_array(getattr(self.runtime.env.robot,
                        "get_"+side+"_arm_real_jointState")(), (7,), "measured arm qpos")
                add(side+"_joint_positions",measured[:6],obs.sim_time_s)
        return c.Observation(context=context,observation_id=obs.observation_id,spec_ref=self.space.spec_id,
            env_step=obs.env_step,sim_time_s=obs.sim_time_s,sensors=obs.sensors,states=states,public_instruction=instruction,
            frame_transforms=obs.frame_transforms)

    def reset(self,task,context):
        self.context=context
        return self._project(self.runtime.reset(self.tasks[task.episode_id],legacy_context(context)))

    def _cartesian_parameters(self,action):
        pose=next((cmd for cmd in action.commands if isinstance(cmd,leaf.PoseTarget)),None)
        grip=next((cmd for cmd in action.commands if isinstance(cmd,leaf.GripperCommand)),None)
        require(pose is not None and grip is not None and pose.pose.frame_id=="world" and grip.mode=="width_m",
                "unsupported_action","cartesian","absolute world pose and physical gripper width required")
        euler=action.source_euler_xyz_rad.get(pose.target_id)
        if euler is None:
            from scipy.spatial.transform import Rotation
            euler=Rotation.from_quat(pose.pose.quaternion_xyzw).as_euler("xyz")
        else:
            from .rotations import euler_quaternion_xyzw
            expected=euler_quaternion_xyzw(euler)
            require(abs(abs(float(np.dot(expected,pose.pose.quaternion_xyzw)))-1.)<1e-12,
                    "rotation_hint_mismatch","source_euler_xyz_rad","Euler hint does not represent public quaternion")
        return {"position_world_m":list(pose.pose.position_m),"euler_xyz_rad":list(euler),
                "finger_targets_m":[float(grip.value)/2.,float(grip.value)/2.]}

    def advance(self,action,context,remaining):
        request=legacy_context(context)
        if self.family=="libero":
            cmd=action.commands[0]
            require(isinstance(cmd,leaf.ControllerCommand),"unsupported_action","libero","native normalized OSC command required")
            parameters=dict(cmd.parameters)
            values=np.asarray(parameters["osc_action7"],dtype=action.native_dtype)
            parameters["osc_action7"]=values.tolist()
            native=replace(cmd,controller_profile_ref=self.runtime.controller_ref,parameters=parameters)
            control=leaf.ActionStep(index=0,duration_s=self.space.control_period_s,commands=(native,))
            reply=self.runtime.advance(control,request,remaining)
        elif self.family=="vlabench":
            parameters=self._cartesian_parameters(action)
            from ..adapters.vlabench_spec import CONTROLLER
            cmd=leaf.ControllerCommand(command_id=context.message_id+":ik",target_id="robot0/eef",controller_profile_ref=CONTROLLER,parameters=parameters)
            reply=self.runtime.advance(leaf.ActionStep(index=0,duration_s=self.space.control_period_s,commands=(cmd,)),request,remaining)
            reply=replace(reply,execution_feedback=replace(reply.execution_feedback,command_ids=tuple(cmd.command_id for cmd in action.commands)))
        else:
            return self._advance_robotwin(action,context,remaining)
        return c.StepResult(observation=self._project(reply.observation),execution_feedback=reply.execution_feedback,
                            terminated=reply.terminated,truncated=reply.truncated)

    @staticmethod
    def robotwin_native_action(action):
        """Pure public->native conversion, also usable in offline evidence checks."""
        cmds={cmd.target_id.split("/",1)[1]:cmd for cmd in action.commands}
        if action.native_protocol==DUAL_PROTOCOL:
            cmd=action.commands[0]
            row=finite_array(cmd.parameters["native_row"],(16,),"native EE16")
            dtype=cmd.parameters["native_dtype"]
            action_type="ee"
        elif any(isinstance(cmd,leaf.JointTarget) for cmd in action.commands):
            row=np.asarray([value for side in ("left","right") for value in
                [*[cmds[side+f"_j{i}"].value for i in range(6)],
                    cmds[side+"_grip"].parameters["requested_open_fraction"] if isinstance(cmds[side+"_grip"],leaf.ControllerCommand)
                    else cmds[side+"_grip"].value]])
            dtype=action.native_dtype
            action_type="qpos"
        else:
            values=[]
            for side in ("left","right"):
                pose=cmds[side+"_eef"].pose
                require(pose.frame_id=="world","frame_mismatch","robotwin pose","world coordinates required")
                q=np.asarray(pose.quaternion_xyzw)
                grip=cmds[side+"_grip"]
                value=grip.parameters["requested_open_fraction"] if isinstance(grip,leaf.ControllerCommand) else grip.value
                values.extend((*pose.position_m,*q[[3,0,1,2]],float(value)))
            row=np.asarray(values)
            dtype=action.native_dtype
            action_type="ee"
        cast=row.astype(dtype or row.dtype)
        require(np.isfinite(cast).all() and np.array_equal(row,cast),"native_dtype_loss","native row","lossless declared dtype required")
        return cast,action_type

    def _advance_robotwin(self,action,context,remaining):
        runtime=self.runtime
        cast,action_type=self.robotwin_native_action(action)
        # Action mode is selected by semantic commands, not by model identity.
        runtime.executor.action_type=action_type
        with runtime._scope():
            # Old ambiguous native EE16 keeps its historical behavior. All new
            # physically typed pose/joint actions must report a dropped arm.
            observation = nullcontext(set()) if action.native_protocol == DUAL_PROTOCOL else observe_native_motion_failures(runtime.env.robot, action_type)
            with observation as failed_arms:
                receipt=runtime.executor.execute(cast,remaining)
        final=receipt.official_success or receipt.native_action_limit_reached
        can_continue=(remaining.env_steps is None or remaining.env_steps>1) and (remaining.model_calls is None or remaining.model_calls>0)
        if action.acquire_observation_after and not final and not receipt.error_type and not failed_arms and can_continue:
            try:
                with runtime._scope():
                    runtime.raw=copy.deepcopy(runtime.env.get_obs())
                runtime.captured_at=receipt.cumulative_sim_time_s
            except Exception as error:
                raise AdapterError("execution","native_observation_failed",type(error).__name__,execution_started=True,state_known=False) from error
        obs=self._project(runtime._snapshot())
        def confirmed(cmd):
            target = cmd.target_id.split("/", 1)[1]
            return not any(target.startswith(side + "_j") or target == side + "_eef" for side in failed_arms)
        feedback=leaf.ExecutionFeedback(request_id=context.message_id,command_ids=tuple(cmd.command_id for cmd in action.commands if confirmed(cmd)),
            status="execution_failed" if receipt.error_type or failed_arms else "executed",
            code="execution_failed" if receipt.error_type else "planning_failed" if failed_arms else "ok",
            consumed_env_steps=1,consumed_physics_steps=receipt.consumed_physics_steps,elapsed_sim_time_s=receipt.elapsed_sim_time_s)
        return c.StepResult(observation=obs,execution_feedback=feedback,terminated=receipt.official_success,
                            truncated=receipt.native_action_limit_reached and not receipt.official_success)

    def score(self,context):
        return self.runtime.score(legacy_context(context))

    def close(self):
        self.runtime.close()

    def diagnostics(self):
        native = dict(self.runtime.diagnostics())
        legacy_labels = {}
        if self.family == "robotwin":
            # Historical environment-loader labels do NOT identify the current
            # policy or its observation cadence. Keep their provenance apart.
            legacy_labels = {"loader_variant_label": native.pop("variant", None),
                "camera_cadence_label": native.get("camera_cadence")}
            native["camera_cadence"] = "action-requested observation, independent of model family; measured qpos at each public snapshot"
        return {"family":self.family,"protocol":self.cfg.protocol,"native":native,
                "native_environment_loader_legacy_labels":legacy_labels,
                "model_configuration_received":False,"model_repository_required":False}
