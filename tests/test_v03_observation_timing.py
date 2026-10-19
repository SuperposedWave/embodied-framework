"""Synthetic timing conformance. No model weights, GPU or task-success evidence."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from embodied_eval import contracts as public
from embodied_eval.codec import decode, to_data
from embodied_eval.core import types as leaf
from embodied_eval.errors import ContractError
from embodied_eval.v03 import contracts as c
from embodied_eval.v03.config import AdapterConfig, EpisodeConfig, RunConfig
from embodied_eval.v03.matching import compatibility, model_input, resolve
from embodied_eval.v03.preflight import check
from embodied_eval.v03.registry import AdapterRegistry, default_registry
from embodied_eval.v03.runner import run
from embodied_eval.v03.synthetic import register, space, capability, SyntheticModel, SyntheticBenchmark
from embodied_eval.v03.temporal import ObservationHistory, assess, validate_policy
from embodied_eval.v03.transforms import inputs_for_model
from embodied_eval.v03.validation import validate_space

ROOT = Path(__file__).resolve().parents[1]
POLICY = c.TemporalPolicy(mode="observation_aligned")


def requirement(**changes):
    return c.HistoryRequirement(**{"offsets_s": (-.2, -.1, 0.), "startup": "available_prefix",
        "provenance": {"synthetic": True, "rule": "fixture explicitly accepts actual partial history at reset"}, **changes})


def fixture(h=None, period=.05, policy=POLICY):
    options = {"declare_sample_timing": True, "period_s": period}
    s = space(options)
    cap = capability({"history": to_data(h or requirement())})
    profile = resolve(s, cap, temporal=policy)
    bench = SyntheticBenchmark(options)
    cfg = config(h or requirement(), period=period)
    task = bench.enumerate_tasks(cfg.episodes)[0]
    context = c.Context(run_id=cfg.run_id, env_id="env-0", episode_id="ep-0", message_id="start")
    bench.reset(task, context)
    return s, cap, profile, bench, task, ObservationHistory(cap, profile, s, policy)


def snapshot(bench, index, time=None):
    bench.steps = index
    bench.sim_time = index * bench.config.period_s if time is None else time
    return bench.snapshot()


def config(h=None, period=.05, model="synthetic_policy_a", benchmark="synthetic_simulator_a", policy=POLICY):
    return RunConfig(run_id="synthetic-history", model=AdapterConfig(adapter=model,
        options={"history": to_data(h or requirement())}),
        benchmark=AdapterConfig(adapter=benchmark, options={"declare_sample_timing": True, "period_s": period}),
        episodes=(EpisodeConfig(episode_id="ep-0", budgets=leaf.Budgets(env_steps=8, model_calls=8)),), temporal=policy)


class TemporalContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = AdapterRegistry()
        register(self.registry)

    def issues(self, cfg):
        return {i["code"] for i in check(cfg, self.registry)["issues"]}

    def test_public_types_and_roundtrip(self):
        self.assertIs(public.HistoryRequirement, c.HistoryRequirement)
        cfg = config()
        self.assertEqual(decode(RunConfig, to_data(cfg)), cfg)
        s, cap, *_ = fixture()
        self.assertEqual(decode(c.SpaceSpec, to_data(s)), s)
        self.assertEqual(decode(c.CapabilitySpec, to_data(cap)), cap)

    def test_old_config_remains_native_with_unknown_action_timing(self):
        cfg = replace(config(), model=AdapterConfig(adapter="synthetic_policy_a"), temporal=c.TemporalPolicy())
        result = check(cfg, self.registry)
        self.assertTrue(result["structurally_compatible"])
        self.assertEqual(result["temporal"]["status"], "native_unverified")
        self.assertFalse(result["temporal"]["action_resampling"])
        self.assertFalse(result["temporal"]["runtime_samples_verified"])

    def test_history_cannot_be_ignored_in_native_mode(self):
        self.assertIn("history_requires_alignment", self.issues(replace(config(), temporal=c.TemporalPolicy())))

    def test_alignment_label_without_requirements_is_rejected(self):
        self.assertIn("missing_history_requirement", self.issues(replace(config(), model=AdapterConfig(adapter="synthetic_policy_a"))))

    def test_strict_initial_history_is_rejected_without_moving_robot(self):
        self.assertIn("history_startup_unavailable", self.issues(config(requirement(startup="reject"))))

    def test_exact_noninteger_grid_is_rejected(self):
        self.assertIn("history_grid_mismatch", self.issues(config(period=.04)))

    def test_noninteger_matching_needs_explicit_authorization(self):
        h = requirement(selection="nearest", tolerance_s=.021)
        self.assertIn("history_approximation_not_authorized", self.issues(config(h, period=.04)))
        result = check(config(h, period=.04, policy=replace(POLICY, allow_approximate=True)), self.registry)
        self.assertTrue(result["structurally_compatible"])
        self.assertEqual(result["temporal"]["status"], "approximate_authorized")

    def test_unknown_acquisition_period_is_not_control_period(self):
        cfg = config()
        cfg = replace(cfg, benchmark=replace(cfg.benchmark, options={"period_s": .05}))
        self.assertIn("unknown_history_sampling", self.issues(cfg))

    def test_macro_acquisitions_are_not_assumed_periodic(self):
        cfg = config()
        cfg = replace(cfg, benchmark=replace(cfg.benchmark, options={"declare_sample_timing": True, "timing_mode": "measured_native"}))
        self.assertIn("unknown_history_sampling", self.issues(cfg))

    def test_declared_capacity_covers_full_window(self):
        self.assertIn("history_capacity_insufficient", self.issues(replace(config(period=.01), temporal=replace(POLICY, max_samples_per_input=4))))

    def test_preflight_has_no_model_or_environment_factory_calls(self):
        with patch.object(SyntheticModel, "__init__", side_effect=AssertionError("model factory")), \
             patch.object(SyntheticBenchmark, "__init__", side_effect=AssertionError("environment factory")):
            result = check(config(), self.registry, verify_assets=True)
        self.assertTrue(result["structurally_compatible"])
        self.assertTrue(result["assets_verified"])
        self.assertFalse(result["temporal"]["runtime_samples_verified"])

    def test_invalid_offsets_and_unknown_fields_fail_closed(self):
        for offsets in ((), (0., -.1), (-.1, -.1, 0.), (-.1,), (.1, 0.)):
            with self.subTest(offsets=offsets), self.assertRaises(ContractError):
                check(config(requirement(offsets_s=offsets)), self.registry)
        raw = to_data(config())
        raw["temporal"]["auto_repeat_images"] = True
        with self.assertRaises(ContractError):
            decode(RunConfig, raw)

    def test_invalid_tolerance_and_missing_source_fail_closed(self):
        for changes in ({"tolerance_s": -.1}, {"tolerance_s": .01}, {"provenance": {}}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                check(config(requirement(**changes)), self.registry)

    def test_nonfinite_timing_is_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ContractError):
                check(config(requirement(offsets_s=(-value, 0.))), self.registry)

    def test_policy_capacity_and_unused_authorization(self):
        for policy in (c.TemporalPolicy(max_samples_per_input=0), c.TemporalPolicy(max_samples_per_input=4097),
                       c.TemporalPolicy(allow_approximate=True)):
            with self.subTest(policy=policy), self.assertRaises(ContractError):
                validate_policy(policy)

    def test_conflicting_histories_cannot_overwrite_one_public_channel(self):
        s, cap, *_ = fixture()
        cap = replace(cap, inputs=cap.inputs + (replace(cap.inputs[0], key="another",
            history=requirement(offsets_s=(-.1, 0.))),))
        self.assertIn("conflicting_history_requirement", {i.code for i in compatibility(s, cap, temporal=POLICY)[0].issues})

    def test_identical_references_to_one_channel_can_share_history(self):
        s, cap, *_ = fixture()
        cap = replace(cap, inputs=cap.inputs + (replace(cap.inputs[0], key="another"),))
        self.assertFalse(compatibility(s, cap, temporal=POLICY)[0].issues)

    def test_extreme_finite_times_are_rejected_without_numeric_overflow(self):
        result = check(config(requirement(offsets_s=(-1e308, 0.))), self.registry)
        self.assertFalse(result["structurally_compatible"])
        self.assertIn("history_capacity_insufficient", {i["code"] for i in result["issues"]})

    def test_sampling_declaration_is_strict(self):
        s = space({})
        for path, timing in (("sensors.missing", c.SampleTiming(kind="uniform", period_s=.05, provenance={"synthetic": True})),
                             ("sensors.head", c.SampleTiming(kind="event", period_s=.05, provenance={"synthetic": True})),
                             ("sensors.head", c.SampleTiming(kind="uniform", period_s=0, provenance={"synthetic": True})),
                             ("sensors.head", c.SampleTiming(kind="unknown"))):
            with self.subTest(path=path, timing=timing), self.assertRaises(ContractError):
                validate_space(replace(s, sample_timing={path: timing}))

    def test_action_duration_metadata_cannot_be_negative_or_unattributed(self):
        s, cap, *_ = fixture()
        for temporal in (c.ModelTiming(native_action_period_s=-.1, provenance={"synthetic": True}),
                         c.ModelTiming(native_action_period_s=.1)):
            report = compatibility(s, replace(cap, temporal=temporal), temporal=POLICY)[0]
            self.assertTrue(report.issues)


class ObservationBufferTests(unittest.TestCase):
    def view(self, obs, s, cap, profile, task, buffer):
        return model_input(obs, task, cap, profile, s, history=buffer)

    def test_exact_history_selects_times_not_last_three_frames(self):
        s, cap, profile, bench, task, buffer = fixture(period=.02)
        for i in range(11):
            obs = snapshot(bench, i)
            buffer.push(obs)
        view = self.view(obs, s, cap, profile, task, buffer)
        frames = view.history["sensors.head"]
        self.assertEqual([round(e.sample.captured_at_s, 8) for e in frames], [0., .1, .2])
        self.assertEqual([e.sample.payload.data[0][0][0] for e in frames], [0, 5, 10])
        self.assertEqual([e.source_observation_id for e in frames], ["obs:0", "obs:5", "obs:10"])
        self.assertEqual(decode(c.ModelInput, to_data(view)), view)

    def test_declared_partial_startup_uses_only_real_available_suffix(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        expected = [(0.,), (0., .1), (0., .1, .2)]
        for i in range(3):
            obs = snapshot(bench, i)
            buf.push(obs)
            view = self.view(obs, s, cap, p, task, buf)
            self.assertEqual(tuple(round(e.sample.captured_at_s, 8) for e in view.history["sensors.head"]), expected[i])
            self.assertEqual(len(view.history_missing_offsets_s.get("sensors.head", ())), 2 - i)

    def test_noninteger_nearest_tie_uses_earlier_actual_frame(self):
        h = requirement(selection="nearest", tolerance_s=.021)
        s, cap, p, bench, task, buf = fixture(h, .04, replace(POLICY, allow_approximate=True))
        for i in range(6):
            obs = snapshot(bench, i)
            buf.push(obs)
        view = self.view(obs, s, cap, p, task, buf)
        frames = view.history["sensors.head"]
        self.assertAlmostEqual(frames[1].requested_at_s, .1)
        self.assertAlmostEqual(frames[1].sample.captured_at_s, .08)
        self.assertAlmostEqual(buf.receipt(view)["selections"]["sensors.head"][1]["error_s"], -.02)

    def test_previous_never_selects_a_frame_after_target(self):
        h = requirement(selection="previous", tolerance_s=.031)
        s, cap, p, bench, task, buf = fixture(h, .04, replace(POLICY, allow_approximate=True))
        for i in range(7):
            obs = snapshot(bench, i)
            buf.push(obs)
        for frame in self.view(obs, s, cap, p, task, buf).history["sensors.head"]:
            self.assertLessEqual(frame.sample.captured_at_s, frame.requested_at_s + 1e-9)

    def test_missing_mid_episode_frame_is_not_startup_padding(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        for i in (0, 2):
            obs = snapshot(bench, i)
            buf.push(obs)
        with self.assertRaises(ContractError) as caught:
            self.view(obs, s, cap, p, task, buf)
        self.assertEqual(caught.exception.issue.code, "history_time_mismatch")

    def test_time_tolerance_is_enforced(self):
        h = requirement(selection="nearest", tolerance_s=.005)
        s, cap, p, bench, task, buf = fixture(h, .04, replace(POLICY, allow_approximate=True))
        for i in range(6):
            obs = snapshot(bench, i)
            buf.push(obs)
        with self.assertRaises(ContractError):
            self.view(obs, s, cap, p, task, buf)

    def test_repeated_capture_is_not_another_frame_and_keeps_first_availability(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        first = snapshot(bench, 0)
        buf.push(first)
        reused = replace(snapshot(bench, 1), sensors=first.sensors)
        buf.push(reused)
        self.assertEqual(len(buf.streams["sensors.head"]), 1)
        self.assertEqual(buf.streams["sensors.head"][0].available_at_s, 0.)
        self.assertEqual(buf.reused_captures, 1)
        with self.assertRaises(ContractError):
            self.view(reused, s, cap, p, task, buf)

    def test_identical_pixels_at_distinct_real_capture_times_are_valid(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        for i in range(3):
            obs = snapshot(bench, i)
            obs = replace(obs, sensors={"head": replace(obs.sensors["head"], payload=leaf.Tensor(
                dtype="uint8", shape=(2, 2, 3), layout="HWC", data=[[[0, 0, 0]] * 2] * 2))})
            buf.push(obs)
        self.assertEqual(len(self.view(obs, s, cap, p, task, buf).history["sensors.head"]), 3)

    def test_one_capture_cannot_satisfy_multiple_requested_times(self):
        h = requirement(selection="nearest", tolerance_s=.2)
        s, cap, p, bench, task, buf = fixture(h, .1, replace(POLICY, allow_approximate=True))
        obs = snapshot(bench, 2)
        buf.push(obs)
        with self.assertRaises(ContractError) as caught:
            self.view(obs, s, cap, p, task, buf)
        self.assertEqual(caught.exception.issue.code, "history_sample_reused")

    def test_same_capture_time_with_different_pixels_is_rejected(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        obs = snapshot(bench, 0)
        buf.push(obs)
        altered = snapshot(bench, 1)
        altered = replace(altered, sensors={"head": replace(altered.sensors["head"], captured_at_s=0.)})
        with self.assertRaises(ContractError) as caught:
            buf.push(altered)
        self.assertEqual(caught.exception.issue.code, "history_capture_conflict")

    def test_future_capture_is_rejected(self):
        *_, bench, task, buf = fixture()
        obs = snapshot(bench, 0)
        with self.assertRaises(ContractError):
            buf.push(replace(obs, sensors={"head": replace(obs.sensors["head"], captured_at_s=.1)}))

    def test_future_availability_cannot_be_selected(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        obs = snapshot(bench, 0)
        buf.push(obs)
        entry = buf.streams["sensors.head"][0]
        buf.streams["sensors.head"][0] = replace(entry, available_at_s=.1)
        with self.assertRaises(ContractError):
            self.view(obs, s, cap, p, task, buf)

    def test_late_available_frame_can_only_be_used_after_arrival(self):
        h = requirement(selection="previous", tolerance_s=.101)
        s, cap, p, bench, task, buf = fixture(h, .1, replace(POLICY, allow_approximate=True))
        obs = snapshot(bench, 2)
        obs = replace(obs, sensors={"head": replace(obs.sensors["head"], captured_at_s=.1)})
        buf.push(obs)
        self.assertEqual(buf.streams["sensors.head"][0].available_at_s, .2)

    def test_buffer_owns_samples_and_does_not_mutate_input(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        obs = snapshot(bench, 0)
        buf.push(obs)
        view = self.view(obs, s, cap, p, task, buf)
        view.history["sensors.head"][0].sample.payload.data[0][0][0] = 99
        self.assertEqual(obs.sensors["head"].payload.data[0][0][0], 0)
        self.assertEqual(buf.streams["sensors.head"][0].sample.payload.data[0][0][0], 0)

    def test_uniform_declaration_is_checked_against_actual_intervals(self):
        *_, bench, task, buf = fixture(period=.1)
        buf.push(snapshot(bench, 0))
        with self.assertRaises(ContractError) as caught:
            buf.push(snapshot(bench, 1, time=.06))
        self.assertEqual(caught.exception.issue.code, "sample_period_mismatch")

    def test_runtime_buffer_is_bounded_and_prunes_by_time(self):
        s, cap, p, bench, task, buf = fixture(period=.01)
        for i in range(100):
            obs = snapshot(bench, i)
            buf.push(obs)
        self.assertLessEqual(len(buf.streams["sensors.head"]), 21)
        self.assertEqual(len(self.view(obs, s, cap, p, task, buf).history["sensors.head"]), 3)

    def test_cross_episode_and_clock_regression_are_rejected(self):
        *_, bench, task, buf = fixture()
        obs = snapshot(bench, 1)
        buf.push(obs)
        with self.assertRaises(ContractError):
            buf.push(replace(obs, context=replace(obs.context, episode_id="other"), observation_id="different"))
        buf.clear()
        buf.push(obs)
        with self.assertRaises(ContractError):
            buf.push(snapshot(bench, 0))

    def test_clear_removes_old_episode_history(self):
        s, cap, p, bench, task, buf = fixture(period=.1)
        for i in range(3):
            buf.push(snapshot(bench, i))
        buf.clear()
        obs = snapshot(bench, 0)
        buf.push(obs)
        self.assertEqual(len(self.view(obs, s, cap, p, task, buf).history["sensors.head"]), 1)

    def test_latest_only_input_cannot_silently_bypass_history(self):
        s, cap, p, bench, task, _ = fixture()
        with self.assertRaises(ContractError) as caught:
            model_input(snapshot(bench, 0), task, cap, p, s)
        self.assertEqual(caught.exception.issue.code, "missing_history_buffer")

    def test_native_mode_keeps_no_image_history_in_memory(self):
        s, cap = space({}), capability({})
        buf = ObservationHistory(cap, resolve(s, cap), s)
        self.assertFalse(buf.requirements)
        self.assertFalse(buf.diagnostics()["active"])


class TemporalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.registry = AdapterRegistry()
        register(self.registry)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def test_two_models_two_environments_all_four_combinations(self):
        for i, model in enumerate(("synthetic_policy_a", "synthetic_policy_b")):
            for j, bench in enumerate(("synthetic_simulator_a", "synthetic_simulator_b")):
                cfg = config(requirement(offsets_s=(-.1, 0.) if i else (-.2, -.1, 0.)),
                             period=.05 if not j else .1, model=model, benchmark=bench)
                outcome = run(cfg, self.output / f"cross-{i}-{j}", self.registry)
                self.assertEqual(outcome.exit_code, 0, outcome.summary)
                stats = outcome.results[0]["execution_statistics"]
                self.assertEqual(stats["env_steps"], 8)
                self.assertEqual(stats["model_calls"], 8)
                self.assertAlmostEqual(stats["sim_time_s"], 8 * (.05 if not j else .1))
                diag = json.loads((outcome.output / "episodes/ep-0/diagnostics.internal.json").read_text())
                self.assertEqual(diag["temporal"]["checked_model_inputs"], 8)
                self.assertEqual(len(diag["model"]["history_inputs"]), 8)
                self.assertEqual(diag["model"]["history_inputs"][-1]["missing_offsets_s"], [])
                self.assertEqual(diag["temporal"]["extra_observe_calls"], 0)
                self.assertFalse(json.loads((outcome.output / "temporal-plan.json").read_text())["action_resampling"])

    def test_history_does_not_change_cache_forward_schedule(self):
        cfg = config()
        cfg = replace(cfg, model=replace(cfg.model, options={**cfg.model.options, "cache_rows": 3}))
        outcome = run(cfg, self.output / "cached", self.registry)
        self.assertEqual(outcome.exit_code, 0)
        self.assertEqual(outcome.results[0]["execution_statistics"]["model_forward_count"], 3)

    def test_alignment_trace_contains_actual_times_and_no_duplicate_payload(self):
        h = requirement(selection="nearest", tolerance_s=.021)
        cfg = config(h, period=.04, policy=replace(POLICY, allow_approximate=True))
        outcome = run(cfg, self.output / "nearest", self.registry)
        self.assertEqual(outcome.exit_code, 0)
        rows = [json.loads(line) for line in (outcome.output / "episodes/ep-0/observation_alignment.internal.jsonl").read_text().splitlines()]
        self.assertNotIn("payload", json.dumps(rows))
        at = rows[5]["selections"]["sensors.head"][1]
        self.assertAlmostEqual(at["requested_at_s"], .1)
        self.assertAlmostEqual(at["captured_at_s"], .08)

    def test_two_episodes_do_not_share_history(self):
        cfg = config()
        cfg = replace(cfg, episodes=cfg.episodes + (replace(cfg.episodes[0], episode_id="ep-1"),))
        outcome = run(cfg, self.output / "two", self.registry)
        self.assertEqual(outcome.exit_code, 0)
        for episode in ("ep-0", "ep-1"):
            path = outcome.output / f"episodes/{episode}/model_input.public.jsonl"
            first = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(len(first["history"]["sensors.head"]), 1)
            self.assertEqual(first["history"]["sensors.head"][0]["sample"]["captured_at_s"], 0.)

    def test_preflight_rejection_does_not_create_output_or_factories(self):
        with patch.object(SyntheticModel, "__init__", side_effect=AssertionError("GPU model")), \
             patch.object(SyntheticBenchmark, "__init__", side_effect=AssertionError("simulator")):
            with self.assertRaises(ContractError):
                run(config(requirement(startup="reject")), self.output / "rejected", self.registry)
        self.assertFalse((self.output / "rejected").exists())

    def test_runtime_frame_gap_is_error_not_warmup_or_retry(self):
        class DroppedImageEnvironment(SyntheticBenchmark):
            def snapshot(self):
                obs = super().snapshot()
                return replace(obs, sensors={}) if self.steps == 3 else obs
        entry = self.registry.benchmark("synthetic_simulator_a")
        self.registry.register_benchmark("dropped", replace(entry, factory=DroppedImageEnvironment))
        outcome = run(config(benchmark="dropped"), self.output / "dropped", self.registry)
        self.assertEqual(outcome.exit_code, 1)
        self.assertEqual(outcome.results[0]["execution_statistics"]["env_steps"], 3)
        self.assertEqual(outcome.summary["episodes"][0]["errors"][0]["code"], "missing_required_input")

    def test_third_environment_requires_no_runner_or_other_adapter_change(self):
        class ThirdEnvironment(SyntheticBenchmark):
            pass
        self.registry.register_benchmark("third", replace(self.registry.benchmark("synthetic_simulator_a"), factory=ThirdEnvironment))
        outcome = run(config(period=.025, benchmark="third"), self.output / "third", self.registry)
        self.assertEqual(outcome.exit_code, 0)
        self.assertAlmostEqual(outcome.results[0]["execution_statistics"]["sim_time_s"], .2)


class MultiChannelHistoryTests(unittest.TestCase):
    def arm_fixture(self, dynamic=False):
        s, cap, _, bench, task, _ = fixture(period=.1)
        cameras = {side + "_wrist": replace(s.sensors["head"], role=side + "_wrist",
                   native_source={"synthetic": True, "field": side + "_camera"}) for side in ("left", "right")}
        site = "robot0/position_site"
        state = c.StateSpec(semantic="eef_position", unit="m", frame_id="robot0/base",
            tensor=leaf.TensorSpec(dtype="float64", shape=(3,), layout="vector"), origin="synthetic state",
            site_id=site, native_source={"synthetic": True})
        base = leaf.FrameSpec(parent_id="world", dynamic=dynamic,
            static_transform=None if dynamic else leaf.Transform(translation_m=(1., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.)))
        timing = c.SampleTiming(kind="uniform", period_s=.1, provenance={"synthetic": True})
        s = replace(s, sensors={**s.sensors, **cameras}, state_fields={"eef_position": state},
            frames={**s.frames, "robot0/base": base}, reference_sites={site: c.ReferenceSite(robot_id="robot0",
                kind="measured_position", origin="synthetic fixture", native_source={"synthetic": True})},
            sample_timing={**s.sample_timing, "sensors.left_wrist": timing, "sensors.right_wrist": timing,
                           "states.eef_position": timing})
        inputs = tuple(c.InputSpec(key=side + "_rgb", public_path="sensors." + side + "_wrist", kind="sensor",
            semantic="rgb", history=requirement(offsets_s=(-.1, 0.))) for side in ("left", "right"))
        cap = replace(cap, inputs=cap.inputs + inputs + (c.InputSpec(key="position", public_path="states.eef_position",
            kind="state", semantic="eef_position", unit="m", frame_id="world", site_id=site, history=requirement()),))
        validate_space(s)
        p = resolve(s, cap, temporal=POLICY)
        return s, cap, p, bench, task, ObservationHistory(cap, p, s, POLICY)

    def obs(self, bench, i, dynamic=False):
        obs = snapshot(bench, i)
        sensors = dict(obs.sensors)
        for side, offset in (("left", 20), ("right", 40)):
            sensors[side + "_wrist"] = replace(obs.sensors["head"], payload=leaf.Tensor(dtype="uint8", shape=(2, 2, 3),
                layout="HWC", data=[[[i + offset, 0, 0]] * 2] * 2))
        states = {"eef_position": c.StateSample(captured_at_s=obs.sim_time_s,
            payload=leaf.Tensor(dtype="float64", shape=(3,), layout="vector", data=[float(i), 0., 0.]))}
        transforms = {"robot0/base": leaf.TimedTransform(captured_at_s=obs.sim_time_s,
            transform=leaf.Transform(translation_m=(float(i), 0., 0.), quaternion_xyzw=(0., 0., 0., 1.)))} if dynamic else {}
        return replace(obs, sensors=sensors, states=states, frame_transforms=transforms)

    def test_both_wrist_channels_and_state_history_are_preserved(self):
        s, cap, p, bench, task, buf = self.arm_fixture()
        for i in range(3):
            obs = self.obs(bench, i)
            buf.push(obs)
        view = model_input(obs, task, cap, p, s, history=buf)
        self.assertEqual(set(view.history), {"sensors.head", "sensors.left_wrist", "sensors.right_wrist", "states.eef_position"})
        self.assertEqual([e.sample.payload.data[0][0][0] for e in view.history["sensors.left_wrist"]], [21, 22])
        self.assertEqual([e.sample.payload.data[0][0][0] for e in view.history["sensors.right_wrist"]], [41, 42])
        self.assertTrue(all(e.sample.payload.dtype == "float64" for e in view.history["states.eef_position"]))

    def test_history_static_frame_conversion_uses_each_original_state(self):
        s, cap, p, bench, task, buf = self.arm_fixture()
        for i in range(3):
            obs = self.obs(bench, i)
            buf.push(obs)
        view = inputs_for_model(model_input(obs, task, cap, p, s, history=buf), cap, s)
        self.assertEqual([e.sample.payload.data[0] for e in view.history["states.eef_position"]], [1., 2., 3.])
        self.assertEqual(view.state_specs["eef_position"].frame_id, "world")

    def test_history_dynamic_transform_uses_capture_time_not_latest_pose(self):
        s, cap, p, bench, task, buf = self.arm_fixture(dynamic=True)
        for i in range(3):
            obs = self.obs(bench, i, dynamic=True)
            buf.push(obs)
        view = inputs_for_model(model_input(obs, task, cap, p, s, history=buf), cap, s)
        self.assertEqual([e.sample.payload.data[0] for e in view.history["states.eef_position"]], [0., 2., 4.])

    def test_old_state_must_not_use_newer_transform(self):
        s, cap, p, bench, task, buf = self.arm_fixture(dynamic=True)
        for i in range(3):
            obs = self.obs(bench, i, dynamic=True)
            if i == 1:
                # A stale transform is valid public data, but not proof of the
                # transform at this history acquisition time.
                obs = replace(obs, frame_transforms={"robot0/base": replace(obs.frame_transforms["robot0/base"], captured_at_s=0.)})
            buf.push(obs)
        view = model_input(obs, task, cap, p, s, history=buf)
        with self.assertRaises(ContractError) as caught:
            inputs_for_model(view, cap, s)
        self.assertEqual(caught.exception.issue.code, "missing_transform")

    def test_no_silent_drop_of_required_wrist(self):
        s, cap, p, bench, task, buf = self.arm_fixture()
        obs = self.obs(bench, 0)
        obs = replace(obs, sensors={k: v for k, v in obs.sensors.items() if k != "right_wrist"})
        buf.push(obs)
        with self.assertRaises(ContractError):
            model_input(obs, task, cap, p, s, history=buf)

    def test_sensor_fps_does_not_create_undelivered_interior_frames(self):
        s, cap, *_ = fixture(period=.1)
        s = replace(s, sample_timing={"sensors.head": c.SampleTiming(kind="uniform", period_s=.02, provenance={"synthetic": True})})
        cap = replace(cap, inputs=(replace(cap.inputs[0], history=requirement(offsets_s=(-.02, 0.))),))
        codes = {i.code for i in compatibility(s, cap, temporal=POLICY)[0].issues}
        self.assertIn("history_delivery_grid_mismatch", codes)

    def test_exact_decisions_cannot_be_between_sensor_acquisitions(self):
        s, cap, *_ = fixture(period=.05)
        s = replace(s, sample_timing={"sensors.head": c.SampleTiming(kind="uniform", period_s=.1, provenance={"synthetic": True})})
        self.assertIn("history_delivery_grid_mismatch", {i.code for i in compatibility(s, cap, temporal=POLICY)[0].issues})


class NativeTimingMetadataTests(unittest.TestCase):
    def test_current_native_profiles_keep_unverified_action_periods_explicit(self):
        from tests.v03_helpers import cases
        registry = default_registry()
        seen = set()
        for cfg in cases().values():
            cap = registry.model(cfg.model.adapter).describe(cfg.model.options)
            self.assertIsNone(cap.temporal.native_action_period_s)
            self.assertTrue(cap.temporal.provenance)
            self.assertTrue(all(req.history is None for req in cap.inputs))
            seen.add(cfg.model.adapter)
        self.assertGreaterEqual(len(seen), 5)

    def test_vlabench_static_fallback_is_not_declared_camera_frequency(self):
        from tests.v03_helpers import cases
        from embodied_eval.v03.benchmarks import canonical_space
        from embodied_eval.native.registry import default_registry as native_registry
        cfg = next(c for c in cases().values() if c.benchmark.adapter == "vlabench")
        native = native_registry().benchmark("vlabench").describe(cfg.benchmark.options)
        static = canonical_space("vlabench", native)
        self.assertEqual(static.sample_timing["sensors.head"].kind, "unknown")
        live = canonical_space("vlabench", replace(native, control_period_s=.1), runtime_verified=True)
        self.assertEqual(live.sample_timing["sensors.head"].period_s, .1)

    def test_current_snapshot_encoder_rejects_new_history_instead_of_dropping_it(self):
        from embodied_eval.v03.models import CanonicalModel
        encoder = CanonicalModel.__new__(CanonicalModel)
        encoder.engine = unittest.mock.Mock()
        with self.assertRaises(ContractError):
            encoder.observe(unittest.mock.Mock(history={"sensors.head": (object(),)}))
        encoder.engine.observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
