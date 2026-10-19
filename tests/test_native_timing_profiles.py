"""Metadata/CPU fixtures only; source audit is a separate offline receipt."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from embodied_eval.native.timing_profiles import PROFILES, timing_facts
from embodied_eval.v03.preflight import check
from embodied_eval.v03.registry import default_registry
from docs.handoff.audit_native_timing import checked_source, metadata_check, replay_dreamzero_assembly, source_files
from tests.v03_helpers import cases


class NativeTimingProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = cases()

    def test_all_thirteen_configurations_have_reviewed_profile_metadata(self):
        self.assertEqual(len(self.configs), 13)
        seen = set()
        for cfg in self.configs.values():
            cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
            self.assertEqual(cap.temporal.provenance["source_audit"]["audit"], "native-timing-source-audit@1")
            seen.add((cfg.model.adapter, cfg.model.options["io_profile"]))
        self.assertEqual(len(seen), 11)

    def test_every_profile_keeps_physical_action_duration_unknown(self):
        for family, profiles in PROFILES.items():
            for profile in profiles:
                _, facts = timing_facts(family, profile)
                self.assertIsNone(facts["native_action_period_s"])
                self.assertIn("unknown", facts["physical_action_period_status"])

    def test_no_new_public_histories_enabled_on_current_weight_codecs(self):
        for cfg in self.configs.values():
            cap = default_registry().model(cfg.model.adapter).describe(cfg.model.options)
            self.assertTrue(all(req.history is None for req in cap.inputs))
            self.assertFalse(cap.temporal.provenance["source_audit"]["public_history_requested"])

    def test_metadata_is_not_an_asset_probe_or_server_connection(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("file I/O")), \
             patch("subprocess.run", side_effect=AssertionError("process")):
            for family, profiles in PROFILES.items():
                for profile in profiles:
                    timing_facts(family, profile)

    def test_model_metadata_interface_cannot_accept_environment_identity(self):
        self.assertEqual(tuple(inspect.signature(timing_facts).parameters), ("family", "profile", "execution_horizon"))
        with self.assertRaises(TypeError):
            timing_facts("openwam", "eef10_osc", benchmark="anything")

    def test_openwam_arx_outer_chunk_is_owned_by_adapter(self):
        owner, facts = timing_facts("openwam", "eef20_arx_absolute")
        self.assertEqual(owner, "adapter")
        self.assertEqual(facts["chunk_owner"], "adapter")

    def test_openwam_network_profiles_preserve_server_chunk_owner(self):
        for profile in PROFILES["openwam"]:
            if profile != "eef20_arx_absolute":
                self.assertEqual(timing_facts("openwam", profile)[0], "server")

    def test_openwam_video_stride_is_not_action_seconds(self):
        for profile in PROFILES["openwam"]:
            facts = timing_facts("openwam", profile)[1]
            self.assertEqual((facts["training_raw_window_rows"], facts["training_action_rows"], facts["training_video_stride_rows"]), (33, 32, 4))
            self.assertIn("unstrided", facts["training_video_stride_scope"])
            self.assertIsNone(facts["native_action_period_s"])

    def test_openpi_execution_horizons_are_profile_owned(self):
        for profile, expected in (("state8_osc", 5), ("state7_absolute", 5), ("arx_joint14", 50)):
            self.assertEqual(timing_facts("openpi", profile)[1]["execution_rows_per_request"], expected)

    def test_openpi_libero_converter_is_only_an_example(self):
        facts = timing_facts("openpi", "state8_osc")[1]
        self.assertEqual(facts["example_converter_fps"], 10)
        self.assertIn("not established", facts["converter_scope"])

    def test_openpi_finetune_dataset_not_mislabelled_as_pretrain_converter(self):
        self.assertIn("pretrain", timing_facts("openpi", "state7_absolute")[1]["converter_scope"])

    def test_lingbot_teacher_fps_is_future_supervision_not_camera_frequency(self):
        facts = timing_facts("lingbot_vla", "joint14")[1]
        self.assertEqual(facts["teacher_video_fps_default"], 1.0)
        self.assertIn("override", facts["teacher_video_scope"])
        self.assertIn("future", facts["future_video_offsets"])

    def test_cosmos_generation_conditioning_does_not_set_control_frequency(self):
        facts = timing_facts("cosmos3", "nano-policy-droid-joint8")[1]
        self.assertEqual(facts["server_default_conditioning_fps"], 15.)
        self.assertIn("not a controller", facts["conditioning_scope"])
        self.assertIn("empty checkpoint.json", facts["recipe_scope"])

    def test_dreamzero_current_wire_frame_is_not_server_generated_history(self):
        facts = timing_facts("dreamzero", "droid-joint8-causal")[1]
        self.assertEqual(facts["configured_eval_video_indices"], [0])
        self.assertEqual((facts["server_first_call_video_rows"], facts["server_later_video_rows"]), (1, 4))
        self.assertIn("repeat", facts["server_short_history"])
        self.assertIn("without public capture timestamps", facts["server_history_timing"])

    def test_explicit_cosmos_and_dreamzero_short_execution_horizons_are_reported(self):
        for family in ("cosmos3", "dreamzero"):
            cfg = next(c for c in self.configs.values() if c.model.adapter == family)
            options = {**cfg.model.options, "open_loop_horizon": 4}
            cap = default_registry().model(family).describe(options)
            self.assertEqual(cap.temporal.provenance["source_audit"]["execution_rows_per_request"], 4)

    def test_mutating_one_provenance_does_not_change_later_descriptions(self):
        facts = timing_facts("dreamzero", "droid-joint8-causal")[1]
        facts["configured_eval_video_indices"].append(9)
        facts["sources"].clear()
        later = timing_facts("dreamzero", "droid-joint8-causal")[1]
        self.assertEqual(later["configured_eval_video_indices"], [0])
        self.assertTrue(later["sources"])

    def test_unknown_family_or_profile_cannot_inherit_reviewed_facts(self):
        for family, profile in (("openwam", "new-weight"), ("unknown", "joint14"), ("openpi", "joint14")):
            with self.assertRaises(ValueError):
                timing_facts(family, profile)

    def test_preflight_displays_documentary_scope_without_claiming_live_verification(self):
        report = check(self.configs["openwam_arx_robotwin"])
        timing = report["temporal"]["model_timing"]
        self.assertEqual(timing["replanning_owner"], "adapter")
        self.assertIsNone(timing["native_action_period_s"])
        self.assertFalse(report["asset_checks_requested"])
        self.assertIn("not live server", timing["provenance"]["source_audit"]["scope"])

    def test_offline_citations_exist_for_every_supported_profile(self):
        for family, profiles in PROFILES.items():
            for profile in profiles:
                self.assertTrue(source_files(family, profile))


class TimingAuditIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "small.json"
        self.file.write_text('{"example": true}\n')

    def test_metadata_hash_alone_is_not_claimed_pinned_training_proof(self):
        result = metadata_check(self.file)
        self.assertIn("not_preexisting_training_proof", result["pin_status"])

    def test_metadata_existing_sha256_is_verified(self):
        digest = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.assertEqual(metadata_check(self.file, expected_sha=digest)["pin_status"], "existing_asset_pin_matches")

    def test_metadata_different_sha256_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
            metadata_check(self.file, expected_sha="0" * 64)

    def manifest(self, **extra):
        payload = self.file.read_bytes()
        data = {"relative": "small.json", "size": len(payload), **extra}
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"objects": [data]}))
        return path

    def test_metadata_existing_git_blob_is_verified(self):
        payload = self.file.read_bytes()
        digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        manifest = self.manifest(git_blob=digest)
        self.assertEqual(metadata_check(self.file, manifest)["pin_status"], "existing_asset_manifest_matches")

    def test_metadata_different_git_blob_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "git_blob_mismatch"):
            metadata_check(self.file, self.manifest(git_blob="0" * 40))

    def test_metadata_manifest_does_not_match_just_basename_in_another_folder(self):
        manifest = self.manifest(sha256=hashlib.sha256(self.file.read_bytes()).hexdigest(), relative="other/small.json")
        with self.assertRaisesRegex(ValueError, "path_missing_or_ambiguous"):
            metadata_check(self.file, manifest)

    def test_metadata_manifest_sha_cannot_be_ignored(self):
        with self.assertRaisesRegex(ValueError, "manifest_sha256_mismatch"):
            metadata_check(self.file, self.manifest(sha256="0" * 64))

    def git_fixture(self):
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.DEVNULL, text=True).strip()
        git("init", "--quiet")
        git("add", "small.json")
        git("-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid", "commit", "-m", "fixture", "--quiet")
        return git("rev-parse", "HEAD")

    def test_source_bytes_and_anchor_checked_against_fixed_git_blob(self):
        rev = self.git_fixture()
        result = checked_source(self.root, rev, "small.json", ('"example"',))
        self.assertTrue(result["pinned_git_blob_bytes_equal"])
        self.assertEqual(result["anchors"][0]["line"], 1)

    def test_source_same_head_but_modified_working_file_is_rejected(self):
        rev = self.git_fixture()
        self.file.write_text("changed")
        with self.assertRaisesRegex(ValueError, "selected_source_modified"):
            checked_source(self.root, rev, "small.json")

    def test_source_wrong_revision_is_rejected(self):
        self.git_fixture()
        with self.assertRaisesRegex(ValueError, "source_revision_mismatch"):
            checked_source(self.root, "0" * 40, "small.json")

    def test_source_missing_review_anchor_is_rejected(self):
        rev = self.git_fixture()
        with self.assertRaisesRegex(ValueError, "anchor_missing"):
            checked_source(self.root, rev, "small.json", ("absent",))

    def test_native_assembly_constant_change_requires_new_review(self):
        source = 'class ARDroidRoboarenaPolicy:\n FRAMES_PER_CHUNK = 5\n def _convert_observation(self, obs):\n  return {}\n'
        with self.assertRaisesRegex(ValueError, "requires_new_review"):
            replay_dreamzero_assembly(source)


if __name__ == "__main__":
    unittest.main()
