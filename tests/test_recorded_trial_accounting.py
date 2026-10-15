"""Synthetic tests of read-only evidence accounting, not real task results."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/handoff"))
from verify_recorded_trial_accounting import analyze


def events():
    scope = dict(run_id="synthetic", env_id="e", episode_id="ep", contract_version="0.2")
    return [
        {"event": "reset", "data": {"context": dict(scope, message_id="o0"),
            "observation_id": "o0", "env_step": 0, "sim_time_s": 0.0}},
        {"event": "prediction", "data": {"context": dict(scope, message_id="request:1"),
            "based_on_observation_id": "o0", "action": {"commands": [{"command_id": "left"}, {"command_id": "right"}]}}},
        {"event": "step_result", "data": {"execution_feedback": {"request_id": "request:1", "status": "executed",
            "consumed_env_steps": 1, "consumed_physics_steps": 50, "elapsed_sim_time_s": .2,
            "command_ids": ["left", "right"], "code": "ok"},
            "observation": {"context": dict(scope, message_id="o1"), "observation_id": "o1", "env_step": 1, "sim_time_s": .2},
            "terminated": True, "truncated": False}},
    ]


class RecordedTrialAccounting(unittest.TestCase):
    def test_success_accounts_both_commands_and_actual_physics_time(self):
        value = analyze(events())
        self.assertEqual(value["actual_env_steps"], 1)
        self.assertEqual(value["actual_physics_steps"], 50)
        self.assertEqual(value["actual_sim_time_s"], .2)
        self.assertEqual(value["prediction_count"], 1)
        self.assertEqual(value["failed_replies"], [])

    def test_partial_native_failure_counts_physics_but_does_not_confirm_left_arm(self):
        rows = events()
        rows[-1]["data"]["execution_feedback"].update(status="execution_failed", code="planning_failed", command_ids=["right"])
        rows[-1]["data"]["terminated"] = False
        value = analyze(rows)
        self.assertEqual(value["actual_physics_steps"], 50)
        self.assertEqual(value["failed_replies"][0]["unconfirmed_command_ids"], ["left"])
        self.assertEqual(value["failed_replies"][0]["confirmed_command_ids"], ["right"])

    def test_missing_duplicate_or_unrequested_confirmation_is_rejected(self):
        for commands in (["right"], ["left", "right", "right"], ["left", "other"]):
            with self.subTest(commands=commands):
                rows = events()
                rows[-1]["data"]["execution_feedback"]["command_ids"] = commands
                with self.assertRaises(Exception):
                    analyze(rows)

    def test_invalid_code_count_time_scope_or_unreplied_prediction_is_rejected(self):
        baseline = events()
        cases = [baseline[:-1], [*baseline, copy.deepcopy(baseline[0])]]
        for name, value in (("code", "native_motion_plan_failed"), ("consumed_env_steps", 2),
                ("consumed_physics_steps", None), ("elapsed_sim_time_s", -.2), ("request_id", "wrong")):
            rows = copy.deepcopy(baseline)
            rows[-1]["data"]["execution_feedback"][name] = value
            cases.append(rows)
        rows = copy.deepcopy(baseline)
        rows[-1]["data"]["observation"]["context"]["episode_id"] = "other"
        cases.append(rows)
        for index, rows in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(Exception):
                analyze(rows)


if __name__ == "__main__":
    unittest.main()
