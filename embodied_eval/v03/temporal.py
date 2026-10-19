"""Causal observation history, independent of all model/environment identities.

This module never steps an environment, predicts actions, pads images, changes
timestamps or schedules model inference. Native execution remains untouched.
"""
from collections import deque
from dataclasses import replace
import copy
import math

from ..codec import decode, to_data
from ..core.validation import require
from ..errors import Issue
from . import contracts as c

EPS = 1e-9  # Seconds; floating-point clock tolerance, not a sampling allowance.
EXTENSION = "observation-timing@1"


def validate_policy(policy):
    decode(c.TemporalPolicy, policy)
    require(1 <= policy.max_samples_per_input <= 4096, "invalid_history_capacity",
            "temporal.max_samples_per_input", "explicit bounded history capacity in [1,4096] required")
    require(not policy.allow_approximate or policy.mode == "observation_aligned",
            "invalid_temporal_policy", "temporal", "approximation authorization applies only to observation alignment")
    return policy


def validate_requirement(req):
    h = req.history
    if h is None:
        return
    decode(c.HistoryRequirement, h)
    require(req.kind in ("sensor", "state"), "unsupported_history_input", req.key,
            "only actual sensor/state acquisition histories are supported")
    require(bool(h.offsets_s) and len(h.offsets_s) <= 4096
            and h.offsets_s[-1] == 0 and all(t <= 0 for t in h.offsets_s)
            and all(a < b for a, b in zip(h.offsets_s, h.offsets_s[1:])),
            "invalid_history_offsets", req.key, "unique increasing nonpositive offsets ending at zero required")
    require(h.tolerance_s >= 0 and (h.selection != "exact" or h.tolerance_s == 0),
            "invalid_history_tolerance", req.key, "exact selection has zero declared sampling tolerance")
    require(bool(h.provenance), "missing_temporal_provenance", req.key,
            "weight/profile source for history and startup requirements must be declared")


def validate_sample_timing(space):
    for path, timing in space.sample_timing.items():
        decode(c.SampleTiming, timing)
        domain, dot, key = path.partition(".")
        table = space.sensors if domain == "sensors" else space.state_fields if domain == "states" else {}
        require(bool(dot) and key in table, "unknown_timing_input", path,
                "timing must refer to an actual canonical sensor/state")
        require(bool(timing.provenance), "missing_temporal_provenance", path,
                "sampling declarations require their source, including an explanation of unknown timing")
        require((timing.period_s is not None and timing.period_s > 0) if timing.kind == "uniform"
                else timing.period_s is None, "invalid_sample_timing", path,
                "only verified uniform sampling has a positive period; event/unknown is not zero Hz")


def assess(space, cap, paths, policy=None):
    """Static checks only. Exact/approximate is interface matching, not a score."""
    policy = validate_policy(policy or c.TemporalPolicy())
    issues, warnings, requirements = [], [], []
    def check(ok, code, path, message):
        if not ok:
            issues.append(Issue(code, path, message))
    def on_grid(value, period):
        ratio = value / period
        return math.isfinite(ratio) and abs(value - round(ratio) * period) <= EPS
    timing = cap.temporal
    if timing is not None:
        decode(c.ModelTiming, timing)
        check(timing.native_action_period_s is None or timing.native_action_period_s > 0,
              "invalid_model_timing", "capability.temporal", "known native action period must be positive")
        check(bool(timing.provenance), "missing_temporal_provenance", "capability.temporal",
              "model timing/cache declaration needs source evidence")
    if timing is None or timing.native_action_period_s is None:
        warnings.append("native_action_period_unverified:no_action_resampling_or_cross_rate_equivalence_claim")
    declared_histories = {}
    for req in cap.inputs:
        validate_requirement(req)
        if req.history is None or req.key not in paths:
            continue
        path = paths[req.key]
        domain, key = path.split(".", 1)
        if key not in (space.sensors if domain == "sensors" else space.state_fields):
            continue  # The ordinary structural checker reports required absence.
        h = req.history
        check(path not in declared_histories or declared_histories[path] == h,
              "conflicting_history_requirement", path,
              "one public channel cannot silently overwrite another requirement with different history rules")
        declared_histories[path] = h
        requirements.append({"input": path, "required": req.required, **to_data(h)})
        check(policy.mode == "observation_aligned", "history_requires_alignment", path,
              "history must explicitly opt into observation_aligned; native mode cannot silently ignore it")
        check(h.selection == "exact" or policy.allow_approximate, "history_approximation_not_authorized", path,
              "nearest/previous frame matching requires explicit run-level authorization")
        check(not (req.required and h.offsets_s[0] < 0 and h.startup == "reject"),
              "history_startup_unavailable", path,
              "reset supplies no pre-episode history; this release will not pad or move the robot to manufacture it")
        check(len(h.offsets_s) <= policy.max_samples_per_input, "history_capacity_insufficient", path,
              "capacity is smaller than the requested number of distinct samples")
        if h.offsets_s[0] < 0:
            sample_timing = space.sample_timing.get(path)
            known = sample_timing is not None and sample_timing.kind == "uniform"
            check(known, "unknown_history_sampling", path,
                  "fixed-offset history requires source-declared uniform acquisition; event/unknown timing is not assumed uniform")
            if known:
                period = sample_timing.period_s
                check(period is not None and period > 0, "invalid_sample_timing", path, "positive sampling period required")
                if period is not None and period > 0:
                    window = (-h.offsets_s[0] + h.tolerance_s) / period
                    check(math.isfinite(window) and window <= policy.max_samples_per_input - 1 + EPS,
                          "history_capacity_insufficient", path,
                          "capacity must retain the entire acquisition window, not just the selected frames")
                    if h.selection == "exact":
                        check(all(on_grid(offset, period) for offset in h.offsets_s),
                              "history_grid_mismatch", path, "required offsets do not lie on the declared acquisition grid")
                        if space.timing_mode == "fixed_period":
                            control = space.control_period_s
                            check(on_grid(control, period) and all(on_grid(offset, control) for offset in h.offsets_s),
                                  "history_delivery_grid_mismatch", path,
                                  "one-snapshot-per-step interface cannot supply missing interior frames or exact off-grid decision captures")
        if h.startup == "available_prefix" and h.offsets_s[0] < 0:
            warnings.append("history_partial_startup_declared:" + path + ":no_padding_no_warmup")
    check(policy.mode != "observation_aligned" or bool(requirements), "missing_history_requirement", "temporal",
          "observation_aligned needs explicit model input requirements, not a label on native execution")
    approximate = any(r["selection"] != "exact" for r in requirements)
    report = {"extension": EXTENSION, "mode": policy.mode,
              "status": "rejected" if issues else "native_unverified" if policy.mode == "native"
                        else "approximate_authorized" if approximate else "exact_grid_declared",
              "runtime_samples_verified": False, "runtime_validation_required": bool(requirements),
              "requirements": requirements, "model_timing": to_data(timing),
              "scope": "observation_only", "action_alignment": "not_checked_in_this_extension",
              "action_resampling": False, "inference_schedule_changed": False,
              "controller_changed": False, "max_samples_per_input": policy.max_samples_per_input}
    return issues, warnings, report


class ObservationHistory:
    """Episode-local bounded streams; repeated captures never become new frames."""
    def __init__(self, cap, profile, space, policy=None):
        self.space = space
        self.policy = validate_policy(policy or c.TemporalPolicy())
        issues, _, _ = assess(space, cap, profile.input_paths, self.policy)
        if issues:
            issue = issues[0]
            require(False, issue.code, issue.path, issue.message)
        self.requirements = {profile.input_paths[r.key]: r.history for r in cap.inputs
                             if r.history is not None and r.key in profile.input_paths}
        self.clear()

    def clear(self):
        self.streams = {path: deque() for path in self.requirements}
        self.identity = None
        self.latest_id = None
        self.latest_time = None
        self.checks = self.reused_captures = self.selected = self.partial = 0

    def push(self, obs):
        if not self.requirements:
            return
        from .validation import validate_observation
        validate_observation(obs, self.space)
        identity = (obs.context.run_id, obs.context.env_id, obs.context.episode_id)
        require(self.identity in (None, identity), "cross_episode", "history", "clear history between episodes")
        require(self.latest_time is None or obs.sim_time_s >= self.latest_time,
                "history_clock_regression", "history", "acquisition availability clock cannot move backwards")
        require(obs.observation_id != self.latest_id, "history_observation_reused", "history", "snapshot identity must advance")
        self.identity, self.latest_id, self.latest_time = identity, obs.observation_id, obs.sim_time_s
        for path, h in self.requirements.items():
            domain, key = path.split(".", 1)
            sample = (obs.sensors if domain == "sensors" else obs.states).get(key)
            stream = self.streams[path]
            if sample is not None and sample.validity == "valid" and sample.payload is not None:
                if stream and sample.captured_at_s == stream[-1].sample.captured_at_s:
                    require(sample == stream[-1].sample, "history_capture_conflict", path,
                            "one acquisition timestamp cannot name different data")
                    self.reused_captures += 1
                else:
                    require(not stream or sample.captured_at_s > stream[-1].sample.captured_at_s,
                            "history_capture_regression", path, "sample acquisition times must not move backwards")
                    timing = self.space.sample_timing.get(path)
                    if stream and timing is not None and timing.kind == "uniform":
                        gap = sample.captured_at_s - stream[-1].sample.captured_at_s
                        ticks = round(gap / timing.period_s)
                        require(ticks >= 1 and abs(gap - ticks * timing.period_s) <= EPS,
                                "sample_period_mismatch", path,
                                "actual capture interval contradicts the declared acquisition grid")
                    # Do not attach a newer robot transform to an older image/state.
                    transforms = {k: v for k, v in obs.frame_transforms.items()
                                  if abs(v.captured_at_s - sample.captured_at_s) <= EPS}
                    stream.append(c.HistorySelection(requested_at_s=sample.captured_at_s,
                        source_observation_id=obs.observation_id, available_at_s=obs.sim_time_s,
                        sample=copy.deepcopy(sample), frame_transforms=copy.deepcopy(transforms)))
            cutoff = obs.sim_time_s + h.offsets_s[0] - h.tolerance_s - EPS
            while stream and stream[0].sample.captured_at_s < cutoff:
                stream.popleft()
            require(len(stream) <= self.policy.max_samples_per_input, "history_buffer_capacity", path,
                    "actual acquisition window exceeds declared bounded capacity; no silent eviction")

    def assemble(self, view):
        if not self.requirements:
            return view
        require(self.policy.mode == "observation_aligned", "history_requires_alignment", "temporal", "explicit opt-in required")
        require((view.context.run_id, view.context.env_id, view.context.episode_id) == self.identity
                and view.observation_id == self.latest_id and view.sim_time_s == self.latest_time,
                "history_decision_mismatch", "history", "selection must be based on the latest available public snapshot")
        history, missing = {}, {}
        for path, h in self.requirements.items():
            domain, key = path.split(".", 1)
            if key not in (view.sensors if domain == "sensors" else view.states):
                continue  # Ordinary input validation handles required fields.
            selected, skipped, used = [], [], set()
            for offset in h.offsets_s:
                target = view.sim_time_s + offset
                if target < -EPS and h.startup == "available_prefix":
                    skipped.append(offset)
                    continue
                require(target >= -EPS, "history_before_episode", path, "required history predates reset; padding is forbidden")
                target = max(0., target)
                candidates = [entry for entry in self.streams[path]
                              if entry.available_at_s <= view.sim_time_s + EPS
                              and entry.sample.captured_at_s <= view.sim_time_s + EPS
                              and (h.selection != "previous" or entry.sample.captured_at_s <= target + EPS)]
                require(bool(candidates), "history_sample_missing", path, "no causal acquired sample for the requested time")
                distance = min(abs(e.sample.captured_at_s - target) for e in candidates)
                entry = min((e for e in candidates if abs(e.sample.captured_at_s - target) <= distance + EPS),
                            key=lambda e: e.sample.captured_at_s)
                require(abs(entry.sample.captured_at_s - target) <= h.tolerance_s + EPS,
                        "history_time_mismatch", path, "no actual sample within the declared time tolerance")
                require(entry.sample.captured_at_s not in used, "history_sample_reused", path,
                        "one acquisition cannot satisfy two different history offsets")
                used.add(entry.sample.captured_at_s)
                selected.append(replace(entry, requested_at_s=target))
            history[path] = tuple(copy.deepcopy(selected))
            if skipped:
                missing[path] = tuple(skipped)
            self.selected += len(selected)
            self.partial += bool(skipped)
        self.checks += 1
        return replace(view, history=history, history_missing_offsets_s=missing)

    @staticmethod
    def receipt(view):
        # No duplicate images in the alignment log; payloads are in ModelInput.
        return {"extension": EXTENSION, "observation_id": view.observation_id,
                "decision_sim_time_s": view.sim_time_s,
                "missing_startup_offsets_s": view.history_missing_offsets_s,
                "selections": {path: [{"requested_at_s": e.requested_at_s,
                    "captured_at_s": e.sample.captured_at_s, "available_at_s": e.available_at_s,
                    "error_s": e.sample.captured_at_s - e.requested_at_s,
                    "source_observation_id": e.source_observation_id} for e in entries]
                    for path, entries in view.history.items()}}

    def diagnostics(self):
        return {"extension": EXTENSION, "active": bool(self.requirements),
                "checked_model_inputs": self.checks, "selected_samples": self.selected,
                "partial_startup_channels": self.partial, "reused_captures_not_new_frames": self.reused_captures,
                "buffered_samples": {path: len(entries) for path, entries in self.streams.items()},
                "padding": False, "extra_observe_calls": 0, "extra_environment_steps": 0,
                "inference_schedule_changed": False, "action_resampling": False}
