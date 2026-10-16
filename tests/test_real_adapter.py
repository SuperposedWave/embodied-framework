"""The real pilot registration must not weaken the mock-only baseline."""
from dataclasses import replace
from pathlib import Path
import unittest

from embodied_eval import _legacy_contracts as c
from embodied_eval.adapters.openwam_libero_spec import EXTENSION, capability, profile, space
from embodied_eval.compatibility import check_compatibility
from tests.legacy_support.config import BenchmarkConfig, EpisodeConfig, ModelConfig, ProfileConfig, RunConfig, run_provenance, validate_config
from embodied_eval.errors import ContractError
from embodied_eval.visibility import build_model_input


class RealPilotRegistrationTests(unittest.TestCase):
    def config(self):
        return RunConfig(run_id="pilot", episodes=(EpisodeConfig(episode_id="trial-0", seed=42,
            budgets=c.Budgets(env_steps=600, sim_time_s=30, model_calls=600)),),
            model=ModelConfig(adapter="openwam_libero", openwam_repo="/explicit/OpenWAM",
                              checkpoint_dir="/explicit/checkpoint"),
            benchmark=BenchmarkConfig(adapter="libero_spatial", libero_repo="/explicit/LIBERO"),
            profile=ProfileConfig(chunk_mode="adapter_owned", execution_horizon_steps=1))

    def test_real_pair_is_registered_and_not_synthetic(self):
        cfg = validate_config(self.config())
        self.assertFalse(run_provenance(cfg)["synthetic"])
        self.assertFalse(check_compatibility(space(), capability(cfg.model), profile(space(), capability(cfg.model))).issues)

    def test_mixed_pairs_are_rejected(self):
        cfg = self.config()
        with self.assertRaises(ContractError):
            validate_config(replace(cfg, benchmark=BenchmarkConfig()))

    def test_pilot_scope_is_fixed(self):
        cfg = self.config()
        with self.assertRaises(ContractError):
            validate_config(replace(cfg, episodes=(replace(cfg.episodes[0], seed=43),)))

    def test_native_proprio_is_named_and_not_a_fabricated_pose(self):
        declared = space()
        self.assertEqual(declared.required_extensions, (EXTENSION,))
        self.assertEqual(declared.robots["robot0"].end_effectors["eef"].tcp_id, "grip_site")
        self.assertNotIn("pose", declared.extensions[EXTENSION])

    def observation(self, *, eef_position=(0.1, 0.2, 0.3)):
        context = c.Context(run_id="pilot", env_id="env-0", episode_id="trial-0", message_id="obs-0")
        sample = c.SensorSample(captured_at_s=0, frame_id="agentview_camera", validity="valid",
            provenance="sensor", payload=c.Tensor(dtype="uint8", shape=(1, 1, 3), layout="HWC",
                                             data=[[[1, 2, 3]]]))
        wrist = replace(sample, frame_id="eye_in_hand_camera")
        return c.Observation(context=context, observation_id="obs-0", spec_ref=space().spec_id,
            env_step=0, sim_time_s=0, sensors={"agentview_image": sample,
                                               "robot0_eye_in_hand_image": wrist},
            robots={"robot0": c.RobotState()}, extensions={EXTENSION: {
                "eef_position_m": list(eef_position), "eef_quaternion_xyzw": [0, 0, 0, 1],
                "gripper_qpos_m": [0.02, 0.02]}, "private.scorer": {"success": True}})

    def test_public_view_whitelists_native_proprio_but_not_private_score(self):
        spec, cap = space(), capability(None)
        task = c.TaskSpec(task_id="task0", protocol_id=spec.protocol_id, protocol_version=spec.protocol_version,
            episode_id="trial-0", public=c.PublicTask(instruction="pick up bowl"),
            initialization_ref="private-init", observation_policy_ref="two-rgb", budgets=c.Budgets(env_steps=600),
            evaluation_config_ref="private-scorer", evaluation_private={"success": True})
        public = build_model_input(task, self.observation(), profile(spec, cap))
        self.assertEqual(set(public.observation_view.extensions), {EXTENSION})
        self.assertFalse(hasattr(public, "evaluation_private"))

    def test_model_maps_official_native_client_action_to_controller_command(self):
        try:
            import numpy as np
            from embodied_eval.adapters.openwam_libero_model import OpenWAMLiberoModel
        except ImportError:
            self.skipTest("real adapter optional dependencies unavailable")
        class Policy:
            _episode = 0
            _step = 0
            def act(self, native_obs, instruction):
                self._step += 1
                assert native_obs["robot0_eef_pos"].shape == (3,)
                assert instruction == "pick up bowl"
                return np.asarray([.1, .2, .3, .4, .5, .6, -1], dtype=np.float32)
        model = OpenWAMLiberoModel(self.config().model, Path("/tmp/unused-pilot-debug"))
        model.policy = Policy()
        model.profile = profile(space(), capability(None))
        model.latest = c.ModelInput(context=self.observation().context,
            task_public=c.PublicTask(instruction="pick up bowl"), observation_view=self.observation())
        request = replace(self.observation().context, message_id="request-1")
        prediction = model.predict(request)
        command = prediction.decision.steps[0].commands[0]
        self.assertIsInstance(command, c.ControllerCommand)
        self.assertEqual(command.target_id, "robot0/eef")
        self.assertEqual(command.parameters["osc_action7"][-1], -1.0)
        self.assertIsNone(prediction.diagnostics.model_forward_count)

    def test_bad_native_proprio_fails_before_model_call(self):
        try:
            from embodied_eval.adapters.openwam_libero_model import OpenWAMLiberoModel
        except ImportError:
            self.skipTest("real adapter optional dependencies unavailable")
        from embodied_eval.errors import AdapterError
        model = OpenWAMLiberoModel(self.config().model, Path("/tmp/unused-pilot-debug"))
        model.profile = profile(space(), capability(None))
        model.latest = c.ModelInput(context=self.observation().context,
            task_public=c.PublicTask(instruction="pick up bowl"),
            observation_view=self.observation(eef_position=(0.1, 0.2)))
        with self.assertRaises(AdapterError):
            model.predict(replace(self.observation().context, message_id="request-1"))

    def test_benchmark_single_step_uses_official_done_for_score(self):
        root = Path(__file__).resolve().parents[2] / "upstream" / "LIBERO"
        if not root.is_dir():
            self.skipTest("LIBERO source not available in this checkout")
        try:
            import numpy as np
            from embodied_eval.adapters.libero_spatial_benchmark import LiberoSpatialBenchmark
        except ImportError:
            self.skipTest("real adapter optional dependencies unavailable")
        class Environment:
            def step(self, action):
                assert len(action) == 7
                return {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
                        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
                        "robot0_eef_pos": [0, 0, 1], "robot0_eef_quat": [0, 0, 0, 1],
                        "robot0_gripper_qpos": [0.02, 0.02]}, 1.0, True, {}
            def close(self):
                pass
        adapter = LiberoSpatialBenchmark(BenchmarkConfig(adapter="libero_spatial", libero_repo=str(root)))
        try:
            adapter.context = self.observation().context
            adapter.env = Environment()
            command = c.ControllerCommand(command_id="cmd-1", target_id="robot0/eef",
                controller_profile_ref="libero-panda-osc-pose@1", parameters={"osc_action7": [0] * 7})
            result = adapter.advance(c.ActionStep(index=0, duration_s=.05, commands=(command,)),
                                     replace(adapter.context, message_id="request-1"), c.Budgets(env_steps=600))
            self.assertEqual(result.observation.env_step, 1)
            self.assertTrue(result.terminated)
            self.assertTrue(adapter.score(adapter.context)["success"].value)
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
