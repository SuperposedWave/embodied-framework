"""0.2 structural evidence. All closed-loop fixtures here are SYNTHETIC."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests.legacy_support.frozen import _legacy_contracts as leaf
from embodied_eval.codec import decode, dumps, to_data
from embodied_eval.errors import ContractError
from embodied_eval.native import contracts as c
from tests.legacy_support.frozen.mn.config import RunConfig, AdapterConfig, EpisodeConfig
from tests.legacy_support.frozen.mn.registry import AdapterRegistry, default_registry
from tests.legacy_support.frozen.mn.synthetic import register, space, capability, SyntheticModel, SyntheticBenchmark
from tests.legacy_support.frozen.mn.binding import check_compatibility, resolve, model_input
from tests.legacy_support.frozen.mn.preflight import check
from tests.legacy_support.frozen.mn.runner import run
from tests.legacy_support.frozen.mn.validation import validate_observation, validate_prediction, validate_space
from embodied_eval.native.transforms import prepare_input, convert_pose_action


def config(model="synthetic_policy_a",benchmark="synthetic_simulator_a",options=None):
    options=options or {}
    camera=options.get("camera_name","camera")
    robot=options.get("robot_id","robot0")
    return RunConfig(run_id="mn-test",model=AdapterConfig(adapter=model),benchmark=AdapterConfig(adapter=benchmark,options=options),
        binding=c.BindingConfig(inputs={"image":camera,"instruction":"instruction"},targets={"joint":robot+"/j0"},action_spec_ref="joint-position@1"),
        episodes=(EpisodeConfig(episode_id="episode0",budgets=leaf.Budgets(env_steps=3,model_calls=3)),))


class IndependentArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.registry=AdapterRegistry(); register(self.registry)
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def execute(self,cfg,name="run"):
        return run(cfg,self.root/name,self.registry)

    def test_two_models_two_simulators_all_four_combinations(self):
        self.assertIsNot(self.registry.model("synthetic_policy_a").factory,self.registry.model("synthetic_policy_b").factory)
        self.assertIsNot(self.registry.benchmark("synthetic_simulator_a").factory,self.registry.benchmark("synthetic_simulator_b").factory)
        for i,m in enumerate(self.registry.models):
            for j,b in enumerate(self.registry.benchmarks):
                with self.subTest(model=m,benchmark=b):
                    result=self.execute(config(m,b),f"run-{i}-{j}")
                    self.assertEqual(result.exit_code,0)
                    self.assertTrue(result.summary["synthetic"])
                    self.assertFalse(result.results[0]["metrics"]["success"].value)
                    expected_time=.3 if b=="synthetic_simulator_b" else .15
                    self.assertAlmostEqual(result.results[0]["execution_statistics"]["sim_time_s"],expected_time)

    def test_third_model_added_without_runner_changes(self):
        class Third(SyntheticModel):
            def predict(self,context):
                prediction=super().predict(context)
                return replace(prediction,action=replace(prediction.action,commands=(replace(prediction.action.commands[0],value=-.3),)))
        entry=self.registry.model("synthetic_policy_a")
        self.registry.register_model("third",replace(entry,factory=Third,describe=lambda o:replace(entry.describe(o),model_id="third-policy")))
        self.assertEqual(self.execute(config(model="third")).exit_code,0)

    def test_third_benchmark_added_without_model_changes(self):
        class Third(SyntheticBenchmark):
            pass
        self.registry.register_benchmark("third",replace(self.registry.benchmark("synthetic_simulator_a"),factory=Third))
        self.assertEqual(self.execute(config(benchmark="third",options={"camera_name":"third_camera","robot_id":"third_arm"})).exit_code,0)

    def test_registry_duplicate_rejected(self):
        with self.assertRaises(ContractError):
            self.registry.register_model("synthetic_policy_a",self.registry.model("synthetic_policy_a"))

    def test_unknown_adapter_rejected(self):
        with self.assertRaises(ContractError):
            check(config(model="missing"),self.registry)

    def test_camera_and_robot_renaming_needs_only_bindings(self):
        self.assertEqual(self.execute(config(options={"camera_name":"different_image","robot_id":"armA"})).exit_code,0)

    def test_training_coverage_is_warning_not_gate(self):
        result=check(config(),self.registry)
        self.assertTrue(result["structurally_compatible"])
        self.assertTrue(result["warnings"][0].startswith("training_coverage_unverified"))

    def test_missing_camera_is_structural_failure(self):
        cfg=config(); cfg=replace(cfg,binding=replace(cfg.binding,inputs={"image":"missing","instruction":"instruction"}))
        self.assertFalse(check(cfg,self.registry)["structurally_compatible"])

    def test_missing_joint_is_structural_failure(self):
        cfg=config(); cfg=replace(cfg,binding=replace(cfg.binding,targets={"joint":"robot0/missing"}))
        self.assertFalse(check(cfg,self.registry)["structurally_compatible"])

    def test_unknown_binding_key_not_silently_ignored(self):
        cfg=config(); cfg=replace(cfg,binding=replace(cfg.binding,inputs=cfg.binding.inputs|{"extra":"camera"}))
        self.assertFalse(check(cfg,self.registry)["structurally_compatible"])

    def test_wrong_action_kind_rejected(self):
        s=space({}); cap=replace(capability({}),output=c.OutputSpec(command_kinds=("pose_target",),target_slots={"joint":"joint"}))
        self.assertFalse(check_compatibility(s,cap,config().binding).compatible)

    def test_dual_arm_cannot_drop_second_output(self):
        cap=replace(capability({}),output=c.OutputSpec(command_kinds=("joint_target",),target_slots={"left":"joint","right":"joint"}))
        binding=replace(config().binding,targets={"left":"robot0/j0"})
        self.assertIn("target_binding_mismatch",{i.code for i in check_compatibility(space({}),cap,binding).issues})

    def test_two_outputs_cannot_overwrite_one_target(self):
        cap=replace(capability({}),output=c.OutputSpec(command_kinds=("joint_target",),target_slots={"left":"joint","right":"joint"}))
        binding=replace(config().binding,targets={"left":"robot0/j0","right":"robot0/j0"})
        self.assertIn("duplicate_target",{i.code for i in check_compatibility(space({}),cap,binding).issues})

    def test_variable_native_action_counts_actual_physics(self):
        result=self.execute(config(options={"timing_mode":"measured_native"}))
        stats=result.results[0]["execution_statistics"]
        self.assertEqual(stats["env_steps"],3)
        self.assertEqual(stats["physics_steps"],3+4+5)
        self.assertAlmostEqual(stats["sim_time_s"],12*.004)

    def test_native_sim_time_budget_even_zero_is_rejected(self):
        cfg=config(options={"timing_mode":"measured_native"})
        cfg=replace(cfg,episodes=(replace(cfg.episodes[0],budgets=leaf.Budgets(env_steps=3,sim_time_s=0)),))
        result=check(cfg,self.registry)
        self.assertFalse(result["structurally_compatible"])
        with self.assertRaises(ContractError):
            self.execute(cfg)
        self.assertFalse((self.root/"run").exists())

    def test_model_owned_cache_preserves_forward_counts(self):
        cfg=replace(config(),model=AdapterConfig(adapter="synthetic_policy_a",options={"cache_rows":2}))
        result=self.execute(cfg)
        self.assertEqual(result.results[0]["execution_statistics"]["model_forward_count"],2)
        # Exercise the real boundary classes with a SYNTHETIC peer: Pi keeps
        # exactly five rows; Ling keeps its complete seven-row fixture chunk.
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional native client NumPy unavailable")
        from embodied_eval.native.native_models import NativeActionModel
        request=c.Context(run_id="r",env_id="e",episode_id="ep",message_id="start")
        for family,io,width,kept in (("openpi","state8_osc",7,5),("openpi","state7_absolute",7,5),("lingbot_vla","joint14",14,7)):
            with self.subTest(io_profile=io):
                class Peer:
                    requests=0
                    def infer(self,payload):
                        if payload.get("reset"):
                            return {}
                        self.requests+=1
                        rows=np.arange(7*width,dtype=np.float32).reshape(7,width)*.001
                        return {"action" if family=="lingbot_vla" else "actions":rows}
                native=NativeActionModel(family,{"repo":"/not-loaded","checkpoint_dir":"/not-loaded","io_profile":io})
                native.peer=Peer()
                targets={key:"robot0/"+key for key in native.cap.output.target_slots}
                profile=c.BoundProfile(profile_id="synthetic-peer",model_id=native.cap.model_id,space_ref="synthetic",
                    binding=c.BindingConfig(inputs={},targets=targets,action_spec_ref="not-executed"))
                task=leaf.PublicTask(instruction="synthetic peer only")
                native.begin_episode(request,task,profile,space({}))
                native.native_input=lambda _: {"fixture":True}
                view=c.ModelInput(context=request,observation_id="obs",sim_time_s=0.,task_public=task,sensors={},states={
                    "base":c.StateSample(captured_at_s=0.,payload=leaf.Tensor(dtype="float64",shape=(3,),layout="vector",data=[0.,0.,0.]))})
                native.latest=view
                for index in range(kept):
                    prediction=native.predict(replace(request,message_id=f"request:{index}"))
                    self.assertIsNone(prediction.model_forward_count)
                    self.assertEqual(native.requests,1)
                    if family=="lingbot_vla":
                        self.assertEqual(prediction.action.acquire_observation_after,index==kept-1)
                self.assertEqual(len(native.cache),0)
                native.predict(replace(request,message_id="next-chunk"))
                self.assertEqual(native.peer.requests,2)
                native.end_episode(request)
                self.assertEqual(len(native.cache),0)
                native.begin_episode(request,task,profile,space({}))
                self.assertEqual(native.requests,0)
                self.assertIsNone(native.latest)

    def test_episode_cache_resets_between_episodes(self):
        cfg=replace(config(),model=AdapterConfig(adapter="synthetic_policy_a",options={"cache_rows":4}),
                    episodes=(config().episodes[0],replace(config().episodes[0],episode_id="episode1")))
        result=self.execute(cfg)
        self.assertEqual([r["execution_statistics"]["model_forward_count"] for r in result.results],[1,1])

    def test_existing_output_refused(self):
        self.execute(config())
        with self.assertRaises(ContractError):
            self.execute(config())

    def test_preflight_does_not_construct_adapters(self):
        with patch.object(SyntheticModel,"__init__",side_effect=AssertionError("model constructed")),patch.object(SyntheticBenchmark,"__init__",side_effect=AssertionError("env constructed")):
            self.assertTrue(check(config(),self.registry)["structurally_compatible"])

    def test_lightweight_registry_imports_no_gpu_or_numpy(self):
        result=subprocess.run([sys.executable,"-c","import sys; from tests.legacy_support.frozen.mn.registry import default_registry; default_registry(); assert not any(n in sys.modules for n in ('torch','jax','numpy','sapien','mujoco')); print('lightweight')"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_v02_context_is_not_v01_wire(self):
        context=c.Context(run_id="r",env_id="e",episode_id="ep",message_id="m")
        with self.assertRaises(ContractError):
            decode(leaf.Context,to_data(context))

    def test_unknown_wire_fields_rejected(self):
        with self.assertRaises(ContractError):
            decode(c.Context,{"run_id":"r","env_id":"e","episode_id":"ep","message_id":"m","secret":True})

    def test_non_finite_configuration_rejected(self):
        with self.assertRaises(ContractError):
            decode(RunConfig,to_data(config())|{"binding":{"inputs":{},"targets":{},"action_spec_ref":float('nan')}})

    def test_options_are_adapter_owned_and_strict(self):
        cfg=replace(config(),model=AdapterConfig(adapter="synthetic_policy_a",options={"typo":1}))
        with self.assertRaises(ContractError):
            check(cfg,self.registry)

    def test_public_projection_never_contains_private_task(self):
        cfg=config(); env=SyntheticBenchmark({}); cap=capability({}); s=space({}); profile=resolve(s,cap,cfg.binding)
        task=replace(env.enumerate_tasks(cfg.episodes)[0],evaluation_private={"success":True,"expert_action":[99]})
        obs=env.reset(task,c.Context(run_id="r",env_id="e",episode_id="episode0",message_id="start"))
        view=model_input(obs,task,cap,profile)
        self.assertNotIn("expert_action",dumps(view)); self.assertNotIn("evaluation_private",dumps(view))
        view.sensors["image"].payload.data[0][0][0]=255
        self.assertEqual(obs.sensors["camera"].payload.data[0][0][0],0)

    def test_dynamic_public_instruction_preferred(self):
        cfg=config(); env=SyntheticBenchmark({}); task=env.enumerate_tasks(cfg.episodes)[0]
        goal=leaf.Tensor(shape=(1,1,3),dtype="uint8",layout="HWC",data=[[[1,2,3]]])
        task=replace(task,public=replace(task.public,goal_image=goal))
        obs=env.reset(task,c.Context(run_id="r",env_id="e",episode_id="episode0",message_id="start"))
        view=model_input(replace(obs,public_instruction="current instruction"),task,capability({}),resolve(space({}),capability({}),cfg.binding))
        self.assertEqual(view.task_public.instruction,"current instruction")
        self.assertEqual(view.task_public.goal_image,goal)
        view.task_public.goal_image.data[0][0][0]=99
        self.assertEqual(task.public.goal_image.data[0][0][0],1)

    def test_stale_sensor_rejected(self):
        cfg=config(); env=SyntheticBenchmark({}); task=env.enumerate_tasks(cfg.episodes)[0]
        obs=env.reset(task,c.Context(run_id="r",env_id="e",episode_id="episode0",message_id="start"))
        cap=capability({}); cap=replace(cap,inputs=(replace(cap.inputs[0],max_age_s=.1),cap.inputs[1]))
        with self.assertRaises(ContractError):
            model_input(replace(obs,sim_time_s=1.),task,cap,resolve(space({}),cap,cfg.binding))

    def test_readiness_is_not_real_validation(self):
        result=check(config(),self.registry)
        self.assertEqual(result["real_validation"],"not_performed_by_preflight")
        self.assertFalse(result["assets_verified"])

    def test_architecture_sources_do_not_dispatch_pairs(self):
        folder=Path(__file__).resolve().parents[1]/"embodied_eval/native"
        for path in folder.glob("*.py"):
            content=path.read_text()
            self.assertNotIn("stage4_pairs",content,path.name)
            self.assertNotIn("robotwin_pairs",content,path.name)
            self.assertNotIn("config.model.adapter, config.benchmark.adapter",content,path.name)
        models=(folder/"native_models.py").read_text()
        self.assertNotIn("benchmark_repo",models)
        self.assertNotIn("VLABench.utils",models)
        self.assertNotIn("libero.libero",models)

    def test_all_native_descriptors_are_gpu_free(self):
        from embodied_eval.native.model_catalog import PROFILES
        from embodied_eval.native.model_catalog import capability as native_cap
        from embodied_eval.native.benchmark_catalog import space as native_space
        for p,(f,*_) in PROFILES.items():
            decode(c.CapabilitySpec,native_cap(f,{"io_profile":p,"repo":"/nonexistent","checkpoint_dir":"/nonexistent"}))
        for f,p in (("libero","spatial-30settle"),("libero","spatial-10settle"),("vlabench","frozen-track1"),("robotwin","sealed-instruction"),("robotwin","seeded-instruction")):
            options={"repo":"/nonexistent","protocol":p,"task_name":"adjust_bottle" if f=="robotwin" else "select_fruit"}
            validate_space(native_space(f,options))


class SemanticStateTests(unittest.TestCase):
    def setup_case(self,semantic="eef_position",unit="m",frame="local"):
        s=space({})
        s=replace(s,frames={"world":leaf.FrameSpec(parent_id=None),"local":leaf.FrameSpec(parent_id="world",static_transform=leaf.Transform(translation_m=(1.,2.,3.),quaternion_xyzw=(0.,0.,0.,1.)))},
            state_fields={"position":c.StateSpec(semantic=semantic,unit=unit,frame_id=frame,tensor=leaf.TensorSpec(dtype="float64",shape=(3,),layout="vector"),origin="measured TCP")})
        cap=replace(capability({}),inputs=(c.InputSpec(key="position",kind="state",semantic=semantic,unit=unit,frame_id="world",tensor=leaf.TensorSpec(dtype="float64",shape=(3,),layout="vector")),))
        binding=c.BindingConfig(inputs={"position":"position"},targets={"joint":"robot0/j0"},action_spec_ref="joint-position@1",frame_conversions=(("local","world"),))
        ctx=c.Context(run_id="r",env_id="e",episode_id="ep",message_id="obs")
        sample=c.StateSample(captured_at_s=0.,payload=leaf.Tensor(dtype="float64",shape=(3,),layout="vector",data=[.1,.2,.3]))
        obs=c.Observation(context=ctx,observation_id="obs",spec_ref=s.spec_id,env_step=0,sim_time_s=0.,sensors={},states={"position":sample})
        task=c.TaskSpec(task_id="t",episode_id="ep",public=leaf.PublicTask(instruction="task"),budgets=leaf.Budgets(env_steps=1),initialization_ref="i",evaluation_config_ref="s")
        return s,cap,binding,obs,task

    def test_explicit_position_frame_conversion(self):
        s,cap,binding,obs,task=self.setup_case()
        profile=resolve(s,cap,binding)
        view=prepare_input(model_input(obs,task,cap,profile),cap,profile,s)
        self.assertEqual(view.states["position"].payload.data,[1.1,2.2,3.3])
        self.assertEqual(view.state_specs["position"].frame_id,"world")

    def test_missing_frame_conversion_rejected(self):
        s,cap,binding,obs,task=self.setup_case()
        self.assertIn("frame_mismatch",{i.code for i in check_compatibility(s,cap,replace(binding,frame_conversions=())).issues})

    def test_disconnected_frames_rejected(self):
        s,cap,binding,obs,task=self.setup_case()
        s=replace(s,frames=s.frames|{"other":leaf.FrameSpec(parent_id=None)})
        binding=replace(binding,frame_conversions=(("local","other"),))
        self.assertIn("disconnected_frames",{i.code for i in check_compatibility(s,cap,binding).issues})

    def test_missing_dynamic_transform_rejected(self):
        s,cap,binding,obs,task=self.setup_case()
        s=replace(s,frames=s.frames|{"local":leaf.FrameSpec(parent_id="world",dynamic=True)})
        profile=resolve(s,cap,binding)
        with self.assertRaises(ContractError):
            prepare_input(model_input(obs,task,cap,profile),cap,profile,s)

    def test_control_targets_cannot_claim_measurement(self):
        s,cap,binding,obs,task=self.setup_case("joint_drive_target","rad","world")
        with self.assertRaises(ContractError):
            validate_observation(obs,s)

    def test_missing_state_sample_rejected(self):
        s,cap,binding,obs,task=self.setup_case()
        with self.assertRaises(ContractError):
            model_input(replace(obs,states={}),task,cap,resolve(s,cap,binding))

    def test_payload_shape_is_checked(self):
        s,cap,binding,obs,task=self.setup_case()
        sample=replace(obs.states["position"],payload=leaf.Tensor(dtype="float64",shape=(2,),layout="vector",data=[0.,0.]))
        with self.assertRaises(ContractError):
            validate_observation(replace(obs,states={"position":sample}),s)
        env=SyntheticBenchmark({})
        dynamic=env.reset(None,c.Context(run_id="r",env_id="e",episode_id="ep",message_id="reset"))
        sensor=replace(env.space.sensors["camera"],tensor=leaf.TensorSpec(dtype="uint8",shape=(-1,-1,3),layout="HWC"))
        declared=replace(env.space,sensors={"camera":sensor})
        validate_space(declared)
        validate_observation(dynamic,declared)

    def test_native_arrays_require_matching_signature(self):
        s,cap,binding,obs,task=self.setup_case()
        cap=replace(cap,output=replace(cap.output,native_protocol="opaque@1",native_signature="row14-qpos"))
        self.assertIn("native_protocol_mismatch",{i.code for i in check_compatibility(s,cap,binding).issues})

    def test_native_arrays_same_name_wrong_meaning_rejected(self):
        s,cap,binding,obs,task=self.setup_case()
        s=replace(s,native_protocols={"opaque@1":"row14-velocity"})
        cap=replace(cap,output=replace(cap.output,native_protocol="opaque@1",native_signature="row14-qpos"))
        self.assertFalse(check_compatibility(s,cap,binding).compatible)

    def test_pose_conversion_is_not_implicit_ik(self):
        s,cap,binding,obs,task=self.setup_case()
        profile=resolve(s,cap,binding)
        action=c.Action(commands=(leaf.PoseTarget(command_id="cmd",target_id="robot0/eef",pose=leaf.Pose(position_m=(.1,.2,.3),quaternion_xyzw=(0.,0.,0.,1.),frame_id="local")),))
        converted=convert_pose_action(action,profile,s,model_input(obs,task,cap,profile))
        self.assertEqual(converted.commands[0].pose.position_m,(1.1,2.2,3.3))
        self.assertEqual(converted.commands[0].pose.frame_id,"world")
