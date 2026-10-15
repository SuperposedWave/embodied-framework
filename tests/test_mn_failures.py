"""Failure and evidence semantics for the independent 0.2 path."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from embodied_eval import contracts as leaf
from embodied_eval.errors import AdapterError,ContractError
from embodied_eval.mn import contracts as c
from embodied_eval.mn.registry import AdapterRegistry
from embodied_eval.mn.synthetic import register,SyntheticModel,SyntheticBenchmark,capability,space
from embodied_eval.mn.config import RunConfig,AdapterConfig,EpisodeConfig
from embodied_eval.mn.binding import resolve
from embodied_eval.mn.validation import validate_prediction,validate_observation
from embodied_eval.mn.runner import run


def cfg():
    return RunConfig(run_id="mn-failure",model=AdapterConfig(adapter="synthetic_policy_a"),benchmark=AdapterConfig(adapter="synthetic_simulator_a"),
        binding=c.BindingConfig(inputs={"image":"camera","instruction":"instruction"},targets={"joint":"robot0/j0"},action_spec_ref="joint-position@1"),
        episodes=(EpisodeConfig(episode_id="episode0",budgets=leaf.Budgets(env_steps=2,model_calls=2)),))


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.registry=AdapterRegistry(); register(self.registry)
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.output=Path(self.temporary.name)/"run"

    def model(self,cls):
        self.registry.models["synthetic_policy_a"]=replace(self.registry.model("synthetic_policy_a"),factory=cls)

    def bench(self,cls):
        self.registry.benchmarks["synthetic_simulator_a"]=replace(self.registry.benchmark("synthetic_simulator_a"),factory=cls)

    def test_service_exception_does_not_claim_zero_forwards(self):
        class Bad(SyntheticModel):
            def predict(self,context): raise RuntimeError("test service failure")
        self.model(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertEqual(result.exit_code,1)
        self.assertIsNone(result.results[0]["execution_statistics"]["model_forward_count"])
        self.assertEqual(result.results[0]["execution_statistics"]["env_steps"],0)

    def test_reset_failure_has_no_valid_score(self):
        class Bad(SyntheticBenchmark):
            def reset(self,*args): raise RuntimeError("reset")
        self.bench(Bad)
        result=run(cfg(),self.output,self.registry)
        metric=result.results[0]["metrics"]["success"]
        self.assertFalse(metric.valid); self.assertIsNone(metric.value)

    def test_score_failure_is_null_not_false_success(self):
        class Bad(SyntheticBenchmark):
            def score(self,context): raise RuntimeError("score")
        self.bench(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertEqual(result.exit_code,1)
        metric=result.results[0]["metrics"]["success"]
        self.assertFalse(metric.valid); self.assertIsNone(metric.value)

    def test_unknown_execution_state_disables_scoring(self):
        class Bad(SyntheticBenchmark):
            def advance(self,*args): raise AdapterError("execution","unknown","test",execution_started=True,state_known=False)
            def score(self,*args): raise AssertionError("must not score unknown state")
        self.bench(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertEqual(result.results[0]["termination_reason"],"execution_state_unknown")
        self.assertFalse(result.summary["episodes"][0]["execution_statistics_complete"])

    def test_pre_execution_rejection_keeps_last_confirmed_state(self):
        class Bad(SyntheticBenchmark):
            def advance(self,*args): raise AdapterError("execution","planning_failed","test",execution_started=False,state_known=True)
        self.bench(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertTrue(result.summary["episodes"][0]["execution_statistics_complete"])
        self.assertTrue(result.results[0]["metrics"]["success"].valid)
        class Rejected(SyntheticBenchmark):
            def advance(self,action,context,remaining):
                obs=self._observation()
                ident="rejection-observation"
                obs=replace(obs,observation_id=ident,context=replace(obs.context,message_id=ident))
                return c.StepResult(observation=obs,terminated=False,truncated=False,
                    execution_feedback=leaf.ExecutionFeedback(request_id=context.message_id,status="rejected",
                        command_ids=tuple(cmd.command_id for cmd in action.commands),consumed_env_steps=0,elapsed_sim_time_s=0.))
        self.bench(Rejected)
        env_only=replace(cfg(),episodes=(replace(cfg().episodes[0],budgets=leaf.Budgets(env_steps=2)),))
        result=run(env_only,self.output.with_name("rejected-reply"),self.registry)
        self.assertEqual(result.exit_code,1)
        self.assertEqual(result.results[0]["execution_statistics"]["model_calls"],1)
        self.assertEqual(result.results[0]["execution_statistics"]["env_steps"],0)
        self.assertTrue(result.results[0]["metrics"]["success"].valid)

    def test_close_failure_marks_episode_error(self):
        class Bad(SyntheticModel):
            def close(self): raise RuntimeError("close")
        self.model(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertEqual(result.exit_code,1)
        self.assertEqual(result.results[0]["run_status"],"error")
        self.assertEqual(result.summary["completed"],0)

    def test_partial_planning_failure_keeps_actual_count_and_known_state_without_retry(self):
        class Partial(SyntheticBenchmark):
            def advance(self, *args):
                reply = super().advance(*args)
                return replace(reply, execution_feedback=replace(reply.execution_feedback,
                    status="execution_failed", code="planning_failed", command_ids=()))
        self.bench(Partial)
        result = run(cfg(), self.output, self.registry)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.results[0]["execution_statistics"]["env_steps"], 1)
        self.assertEqual(result.results[0]["execution_statistics"]["model_calls"], 1)
        self.assertTrue(result.summary["episodes"][0]["execution_statistics_complete"])
        self.assertTrue(result.results[0]["metrics"]["success"].valid)
        self.assertEqual(result.summary["episodes"][0]["errors"][0]["code"], "action_not_executed")

    def test_diagnostics_failure_has_recorded_summary(self):
        class Bad(SyntheticModel):
            def diagnostics(self): raise RuntimeError("diagnostics")
        self.model(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertEqual(result.exit_code,1)
        self.assertTrue((self.output/"summary.json").is_file())
        self.assertEqual(result.results[0]["termination_reason"],"run_error")
        self.assertEqual(result.summary["episodes"][0]["termination_reason"],"run_error")
        class BadEnd(SyntheticModel):
            def end_episode(self,context): raise RuntimeError("end")
        self.model(BadEnd)
        result=run(cfg(),self.output.with_name("end-error"),self.registry)
        self.assertEqual(result.exit_code,1)
        self.assertEqual(result.results[0]["termination_reason"],"run_error")
        self.assertEqual(result.summary["episodes"][0]["termination_reason"],"run_error")

    def test_model_cannot_mutate_benchmark_descriptor(self):
        class Mutating(SyntheticModel):
            def begin_episode(self,context,task,profile,public_space):
                public_space.sensors.clear()
                super().begin_episode(context,task,profile,public_space)
        self.model(Mutating)
        self.assertEqual(run(cfg(),self.output,self.registry).exit_code,0)

    def test_invalid_reply_clock_not_counted_as_success(self):
        class Bad(SyntheticBenchmark):
            def advance(self,*args):
                result=super().advance(*args)
                return replace(result,observation=replace(result.observation,env_step=99))
        self.bench(Bad)
        result=run(cfg(),self.output,self.registry)
        self.assertEqual(result.exit_code,1)
        self.assertFalse(result.summary["episodes"][0]["execution_statistics_complete"])

    def prediction_case(self):
        config=cfg(); s=space({}); cap=capability({}); profile=resolve(s,cap,config.binding)
        env=SyntheticBenchmark({}); task=env.enumerate_tasks(config.episodes)[0]
        context=c.Context(run_id="r",env_id="e",episode_id="episode0",message_id="request")
        obs=env.reset(task,context)
        pred=c.Prediction(context=context,based_on_observation_id=obs.observation_id,
            action=c.Action(commands=(leaf.JointTarget(command_id="cmd",target_id="robot0/j0",mode="position",value=.1,unit="rad"),)))
        return s,cap,profile,context,obs,pred

    def test_prediction_cannot_cross_episode(self):
        s,cap,profile,request,obs,pred=self.prediction_case()
        with self.assertRaises(ContractError):
            validate_prediction(replace(pred,context=replace(request,episode_id="other")),request,obs,s,cap,profile)

    def test_prediction_cannot_use_stale_observation(self):
        s,cap,profile,request,obs,pred=self.prediction_case()
        with self.assertRaises(ContractError):
            validate_prediction(replace(pred,based_on_observation_id="old"),request,obs,s,cap,profile)

    def test_joint_degrees_not_accepted_as_radians(self):
        s,cap,profile,request,obs,pred=self.prediction_case()
        bad=replace(pred,action=replace(pred.action,commands=(replace(pred.action.commands[0],unit="deg"),)))
        with self.assertRaises(ContractError):
            validate_prediction(bad,request,obs,s,cap,profile)

    def test_pose_hint_cannot_address_a_joint(self):
        s,cap,profile,request,obs,pred=self.prediction_case()
        bad=replace(pred,action=replace(pred.action,source_euler_xyz_rad={"robot0/j0":(0.,0.,0.)}))
        with self.assertRaises(ContractError):
            validate_prediction(bad,request,obs,s,cap,profile)

    def test_future_acquisition_time_rejected(self):
        s,cap,profile,request,obs,pred=self.prediction_case()
        sensor=replace(obs.sensors["camera"],captured_at_s=1.)
        with self.assertRaises(ContractError):
            validate_observation(replace(obs,sensors={"camera":sensor}),s)

    def test_failed_asset_check_starts_no_adapters(self):
        entry=self.registry.model("synthetic_policy_a")
        def reject(options): raise ContractError("asset_hash_mismatch","checkpoint","test")
        def forbidden(*args): raise AssertionError("factory must not run")
        self.registry.models["synthetic_policy_a"]=replace(entry,verify_assets=reject,factory=forbidden)
        with self.assertRaises(ContractError):
            run(cfg(),self.output,self.registry)
        self.assertFalse(self.output.exists())
