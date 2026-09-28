"""Analytical schema and information-boundary tests; no simulation imports."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nso.scene_contract_v40 import (SceneContractError, assert_formal_test_ready,
    canonical_json_bytes, content_sha256, load_json_strict, public_planner_spec,
    seed_commitment, validate_budget_declarations, validate_catalog, validate_protocol, validate_public_spec,
    validate_scene_spec)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/virtual3d/v40_scene_protocol_20260920.json"
_spec = importlib.util.spec_from_file_location("prepare_scene_catalog_v40", ROOT / "scripts/prepare_scene_catalog_v40.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def fixture_bundle(protocol):
    # Synthetic ephemeral unit-test material, never the actual reserved test seeds.
    counter = iter(range(36))
    return prepare.build_bundle(protocol,
        seed_factory=lambda: hashlib.sha256(f"unit-fixture-only-{next(counter)}".encode()).hexdigest())


class TestSceneContractV40(unittest.TestCase):
    def setUp(self):
        self.protocol = load_json_strict(PROTOCOL)
        self.catalog, self.development, self.escrow = fixture_bundle(self.protocol)
        self.scene = self.development["DEV_A_00"]

    def test_expected_parent_allocation_and_no_geometry(self):
        validate_protocol(self.protocol)
        validate_catalog(self.catalog, self.protocol)
        counts = {split: sum(entry["split"] == split for entry in self.catalog["entries"])
                  for split in ("development", "main_test", "boundary_test")}
        self.assertEqual(counts, {"development": 6, "main_test": 24, "boundary_test": 12})
        self.assertEqual(len(self.development), 6)
        self.assertFalse(self.catalog["geometry_generation_executed"])
        self.assertEqual(self.catalog["policy_or_metric_execution_count"], 0)

    def test_test_seeds_not_serialized_in_public_artifacts(self):
        public = canonical_json_bytes({"catalog": self.catalog, "development": self.development})
        for record in self.escrow["records"]:
            self.assertNotIn(record["seed_hex"].encode(), public)
            self.assertIn(record["commitment"].encode(), public)

    def test_commitment_binds_seed_parent_and_split(self):
        seed = hashlib.sha256(b"synthetic").hexdigest()
        original = seed_commitment("protocol", "parent", "main_test", seed)
        self.assertNotEqual(original, seed_commitment("protocol", "other", "main_test", seed))
        self.assertNotEqual(original, seed_commitment("protocol", "parent", "development", seed))
        self.assertNotEqual(original, seed_commitment("protocol2", "parent", "main_test", seed))
        for invalid in ("", "1" * 63, "Z" * 64, 3):
            with self.subTest(invalid=invalid), self.assertRaises(SceneContractError):
                seed_commitment("protocol", "parent", "main_test", invalid)

    def test_development_seed_tampering_rejected(self):
        self.scene["seed_record"]["seed_hex"] = hashlib.sha256(b"changed").hexdigest()
        with self.assertRaisesRegex(SceneContractError, "commitment mismatch"):
            validate_scene_spec(self.scene)

    def test_test_scene_cannot_expose_seed(self):
        self.scene["split"] = "main_test"
        self.scene["seed_record"]["visibility"] = "sealed"
        with self.assertRaisesRegex(SceneContractError, "test seed"):
            validate_scene_spec(self.scene)
        self.scene["seed_record"]["seed_hex"] = None
        validate_scene_spec(self.scene, expected_split="main_test")

    def test_public_extractor_excludes_identity_truth_roi_and_seed(self):
        result = public_planner_spec(self.scene)
        self.assertEqual(set(result), {"schema_version", "task", "navigation", "sensor", "motion", "pose_model", "structure_prior"})
        serialized = canonical_json_bytes(result)
        for forbidden in ("renderer_private", "evaluation_private", "parent_id", "family", "seed_hex", "gt_mesh_asset", "evaluation_roi_asset", "lineage"):
            self.assertNotIn(('"' + forbidden + '"').encode(), serialized)
        self.assertNotIn(self.scene["seed_record"]["seed_hex"].encode(), serialized)

    def test_public_copy_and_private_changes_do_not_affect_planner(self):
        original = public_planner_spec(self.scene)
        self.scene["status"] = "materialized_unfrozen"
        self.scene["renderer_private"]["instances_asset"] = {
            "status": "materialized", "relative_path": "private/synthetic-instances.json",
            "sha256": hashlib.sha256(b"synthetic fixture record, no geometry").hexdigest()}
        self.assertEqual(original, public_planner_spec(self.scene))
        original["task"]["max_actions"] = 1
        self.assertEqual(self.scene["public"]["task"]["max_actions"], 160)

    def test_unknown_fields_fail_closed_at_public_and_private_boundaries(self):
        for field in ("public", "renderer_private", "evaluation_private"):
            altered = deepcopy(self.scene)
            altered[field]["hidden_structure"] = 1
            with self.subTest(field=field), self.assertRaises(SceneContractError):
                public_planner_spec(altered)
        for field in ("task", "navigation", "sensor", "motion", "pose_model", "structure_prior"):
            altered = deepcopy(self.scene)
            altered["public"][field]["instance_roi"] = [1, 2, 3]
            with self.subTest(field=field), self.assertRaises(SceneContractError):
                public_planner_spec(altered)

    def test_actions_start_at_zero_and_budget_can_be_configured(self):
        self.scene["public"]["task"]["max_actions"] = 80
        self.scene["public"]["task"]["budget_tier"] = "short"
        validate_scene_spec(self.scene)
        for field, value in (("start_action", 18), ("forced_prefix_actions", ["forward"]), ("max_actions", True), ("max_actions", 0)):
            altered = deepcopy(self.scene)
            altered["public"]["task"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SceneContractError):
                validate_scene_spec(altered)

    def test_pose_and_sensor_numeric_validation(self):
        pose = self.scene["public"]["pose_model"]
        pose.update(kind="noisy_odometry", translation_std_m=0.02, yaw_std_deg=0.5)
        validate_scene_spec(self.scene)
        pose["kind"] = "exact"
        with self.assertRaises(SceneContractError):
            validate_scene_spec(self.scene)
        for invalid in (float("nan"), float("inf"), True, -0.1, 10 ** 1000):
            altered = deepcopy(self.development["DEV_B_00"])
            altered["public"]["sensor"]["depth_max_m"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(SceneContractError):
                validate_scene_spec(altered)

    def test_coarse_navigation_cannot_claim_private_geometry(self):
        self.scene["public"]["navigation"]["hidden_surface_geometry_allowed"] = True
        with self.assertRaises(SceneContractError):
            validate_scene_spec(self.scene)

    def test_many_to_many_prior_rejects_class_as_structure_code(self):
        table = self.scene["public"]["structure_prior"]["probability_by_category"]
        table["cabinet"] = [1.0, 0.0, 0.0, 0.0]
        with self.assertRaises(SceneContractError):
            validate_scene_spec(self.scene)

    def test_many_to_many_prior_requires_multiple_classes_per_structure(self):
        table = self.scene["public"]["structure_prior"]["probability_by_category"]
        for category in ("rack", "workstation", "machine"):
            table[category] = [0.0, 0.3, 0.3, 0.4]
        with self.assertRaises(SceneContractError):
            validate_scene_spec(self.scene)

    def test_prior_cannot_use_test_calibration_or_claim_unbuilt_training(self):
        prior = self.scene["public"]["structure_prior"]
        prior["source_split"] = "main_test"
        with self.assertRaises(SceneContractError):
            validate_scene_spec(self.scene)
        prior["source_split"] = "development"
        prior["calibration_status"] = "frozen_development"
        with self.assertRaisesRegex(SceneContractError, "real training manifest"):
            validate_scene_spec(self.scene)

    def test_cross_split_mesh_and_prototype_lineages_rejected(self):
        for key in ("meshes", "prototypes"):
            altered = deepcopy(self.catalog)
            altered["entries"][1]["lineage"][key] = altered["entries"][0]["lineage"][key]
            with self.subTest(key=key), self.assertRaisesRegex(SceneContractError, "crosses"):
                validate_catalog(altered, self.protocol)

    def test_layout_alias_and_transformed_parent_rejected(self):
        self.catalog["entries"][1]["lineage"]["layout"] = self.catalog["entries"][0]["lineage"]["layout"]
        with self.assertRaisesRegex(SceneContractError, "layout lineage"):
            validate_catalog(self.catalog, self.protocol)
        self.catalog, _, _ = fixture_bundle(self.protocol)
        self.catalog["entries"][1]["lineage"]["derived_from_parent"] = "DEV_A_00"
        with self.assertRaisesRegex(SceneContractError, "derived variants"):
            validate_catalog(self.catalog, self.protocol)

    def test_missing_parent_duplicate_parent_and_wrong_family_rejected(self):
        altered = deepcopy(self.catalog)
        altered["entries"].pop()
        with self.assertRaises(SceneContractError):
            validate_catalog(altered, self.protocol)
        altered = deepcopy(self.catalog)
        altered["entries"][1]["parent_id"] = altered["entries"][0]["parent_id"]
        with self.assertRaises(SceneContractError):
            validate_catalog(altered, self.protocol)
        self.catalog["entries"][1]["family"] = "F"
        with self.assertRaises(SceneContractError):
            validate_catalog(self.catalog, self.protocol)

    def test_pending_assets_cannot_advertise_hashes(self):
        self.catalog["entries"][0]["assets"]["mesh_bundle"]["sha256"] = hashlib.sha256(b"not a mesh").hexdigest()
        with self.assertRaisesRegex(SceneContractError, "pending assets"):
            validate_catalog(self.catalog, self.protocol)

    def test_asset_paths_cannot_escape(self):
        self.scene["status"] = "materialized_unfrozen"
        for path in ("/absolute.json", "../escape.json", "folder/../escape.json", "folder\\escape.json"):
            self.scene["renderer_private"]["world_asset"] = {"status": "materialized", "relative_path": path,
                "sha256": hashlib.sha256(b"synthetic").hexdigest()}
            with self.subTest(path=path), self.assertRaises(SceneContractError):
                validate_scene_spec(self.scene)

    def test_selection_by_outcome_and_structure_code_forbidden(self):
        for key in ("selection_uses_method_results", "category_is_hidden_structure_code", "geometry_generation_in_p0"):
            altered = deepcopy(self.protocol)
            altered["generation_policy"][key] = True
            with self.subTest(key=key), self.assertRaises(SceneContractError):
                validate_protocol(altered)

    def test_all_pending_specs_fail_formal_admission_even_with_ready_label(self):
        for status in ("pending_geometry", "materialized_unfrozen", "frozen_ready"):
            self.scene["status"] = status
            with self.subTest(status=status), self.assertRaises(SceneContractError):
                assert_formal_test_ready(self.scene, catalog=self.catalog, protocol=self.protocol)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(SceneContractError):
                    load_json_strict(path)

    def test_catalog_bound_to_protocol_and_repeated_seed_rejected(self):
        self.catalog["protocol_sha256"] = hashlib.sha256(b"wrong").hexdigest()
        with self.assertRaises(SceneContractError):
            validate_catalog(self.catalog, self.protocol)
        with self.assertRaisesRegex(ValueError, "repeated seed"):
            prepare.build_bundle(self.protocol, seed_factory=lambda: hashlib.sha256(b"repeated fixture").hexdigest())

    def test_filesystem_escrow_permissions_no_overwrite_no_private_validation_read(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            with patch.object(prepare, "build_bundle", return_value=fixture_bundle(self.protocol)):
                result = prepare.write_bundle(PROTOCOL, output)
            self.assertLess(result["bytes_written"], 5 * 1024 * 1024)
            self.assertFalse(result["formal_test_ready"])
            self.assertEqual((output / "private/test_seed_escrow.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((output / "private").stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                prepare.write_bundle(PROTOCOL, output)
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if "private" in path.parts:
                    raise AssertionError("public validation must not read escrow")
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                report = prepare.validate_public_bundle(output)
            self.assertEqual(report["public_files_verified"], 15)
            self.assertEqual(report["protocol_bound_scene_specs"], 6)
            self.assertFalse(report["private_seed_file_read"])

    def test_probability_floor_matches_instance_belief_support(self):
        for probability in (0.0, 1e-12, 0.999e-9):
            altered = deepcopy(self.scene)
            altered["public"]["structure_prior"]["probability_by_category"]["cabinet"] = [
                probability, 0.3, 0.3, 0.4 - probability]
            with self.subTest(probability=probability), self.assertRaises(SceneContractError):
                validate_scene_spec(altered)
        self.scene["public"]["structure_prior"]["probability_by_category"]["cabinet"] = [1e-9, 0.3, 0.3, 0.4-1e-9]
        validate_scene_spec(self.scene)

    def test_cross_module_floor_after_normalization_at_sum_tolerance(self):
        from nso.instance_belief_v40 import InstanceBeliefV40

        for first, accepted in ((1e-9, False), (1.000000001e-9, True)):
            scene = deepcopy(self.scene)
            prior = scene["public"]["structure_prior"]
            prior["probability_by_category"]["cabinet"] = [first, .3, .3, .3999999995]
            palette = {category: (16 + index * 32, 17, 23)
                       for index, category in enumerate(prior["categories"])}
            arguments = dict(palette=palette, structure_names=prior["abstract_structures"],
                             class_structure_prior=prior["probability_by_category"])
            with self.subTest(first=first, accepted=accepted):
                if accepted:
                    validate_scene_spec(scene)
                    ledger = InstanceBeliefV40(**arguments)
                    self.assertGreaterEqual(float(ledger.class_priors["cabinet"].min()), 1e-9)
                else:
                    with self.assertRaisesRegex(SceneContractError, "normalized probabilities"):
                        validate_scene_spec(scene)
                    with self.assertRaisesRegex(ValueError, "near-zero probability"):
                        InstanceBeliefV40(**arguments)

    def test_pending_navigation_cannot_forward_any_materialized_asset(self):
        for reference in ("private/gt.json", "public/graph.json", "arbitrary.json"):
            altered = deepcopy(self.scene)
            altered["public"]["navigation"]["status"] = "materialized"
            altered["public"]["navigation"]["graph_asset"] = {
                "status": "materialized", "relative_path": reference,
                "sha256": hashlib.sha256(b"synthetic asset claim").hexdigest()}
            with self.subTest(reference=reference), self.assertRaisesRegex(SceneContractError, "public navigation"):
                public_planner_spec(altered)

    def test_protocol_binding_rejects_all_undeclared_public_changes(self):
        mutations = [
            ("task", "max_actions", 80), ("task", "budget_tier", "short"),
            ("sensor", "depth_max_m", 3.0), ("sensor", "depth_noise_relative_std", 0.02),
            ("motion", "camera_height_m", 1.0), ("navigation", "resolution_m", 2.0),
            ("pose_model", "kind", "noisy_odometry")]
        for group, key, value in mutations:
            altered = deepcopy(self.scene)
            altered["public"][group][key] = value
            with self.subTest(group=group, key=key), self.assertRaisesRegex(SceneContractError, "protocol defaults"):
                validate_catalog(self.catalog, self.protocol, scene_specs=[altered])
        altered = deepcopy(self.scene)
        altered["public"]["structure_prior"]["probability_by_category"]["cabinet"] = [.25, .25, .25, .25]
        with self.assertRaisesRegex(SceneContractError, "protocol defaults"):
            public_planner_spec(altered, protocol=self.protocol)

    def test_protocol_changes_must_be_common_and_catalog_bound(self):
        self.protocol["public_defaults"]["task"]["max_actions"] = 80
        self.protocol["public_defaults"]["sensor"]["depth_max_m"] = 3.0
        catalog, development, _ = fixture_bundle(self.protocol)
        validate_catalog(catalog, self.protocol, scene_specs=list(development.values()))
        with self.assertRaises(SceneContractError):
            validate_catalog(self.catalog, self.protocol)

    def _boundary_fixture(self):
        entry = next(item for item in self.catalog["entries"] if item["parent_id"] == "TEST_E_00")
        scene = deepcopy(self.scene)
        scene.update(parent_id=entry["parent_id"], family=entry["family"], split=entry["split"],
                     lineage=deepcopy(entry["lineage"]), seed_record={"visibility": "sealed", "seed_hex": None,
                                                                   "commitment": entry["seed_commitment"]})
        declaration = {"schema_version": "v40.boundary_budget_declarations.v1",
                       "protocol_sha256": content_sha256(self.protocol),
                       "assignments": {"TEST_E_00": {"budget_tier": "short", "max_actions": 80}}}
        return scene, declaration

    def test_boundary_budget_requires_exact_predeclared_tier(self):
        scene, declaration = self._boundary_fixture()
        scene["public"]["task"].update(budget_tier="short", max_actions=80)
        with self.assertRaisesRegex(SceneContractError, "protocol defaults"):
            validate_catalog(self.catalog, self.protocol, scene_specs=[scene])
        validate_catalog(self.catalog, self.protocol, scene_specs=[scene], budget_declarations=declaration)
        public_planner_spec(scene, protocol=self.protocol, budget_declarations=declaration)
        scene["public"]["task"]["max_actions"] = 81
        with self.assertRaisesRegex(SceneContractError, "protocol defaults"):
            validate_catalog(self.catalog, self.protocol, scene_specs=[scene], budget_declarations=declaration)

    def test_boundary_declaration_cannot_override_sensor_or_prior(self):
        scene, declaration = self._boundary_fixture()
        scene["public"]["task"].update(budget_tier="short", max_actions=80)
        scene["public"]["sensor"]["depth_max_m"] = 3.0
        with self.assertRaisesRegex(SceneContractError, "protocol defaults"):
            validate_catalog(self.catalog, self.protocol, scene_specs=[scene], budget_declarations=declaration)
        declaration["assignments"]["TEST_E_00"]["sensor"] = {"depth_max_m": 3.0}
        with self.assertRaises(SceneContractError):
            validate_budget_declarations(declaration, self.protocol, self.catalog)

    def test_boundary_declaration_rejects_main_dev_and_unknown_parents(self):
        _, declaration = self._boundary_fixture()
        for parent in ("DEV_A_00", "TEST_A_00", "UNKNOWN_PARENT"):
            altered = deepcopy(declaration)
            altered["assignments"] = {parent: {"budget_tier": "short", "max_actions": 80}}
            with self.subTest(parent=parent), self.assertRaisesRegex(SceneContractError, "registered boundary"):
                validate_catalog(self.catalog, self.protocol, budget_declarations=altered)

    def test_boundary_declaration_hash_and_tier_consistency(self):
        _, declaration = self._boundary_fixture()
        declaration["protocol_sha256"] = hashlib.sha256(b"other protocol").hexdigest()
        with self.assertRaises(SceneContractError):
            validate_budget_declarations(declaration, self.protocol)
        _, declaration = self._boundary_fixture()
        for tier, actions in (("standard", 80), ("short", 160), ("short", 0)):
            declaration["assignments"]["TEST_E_00"] = {"budget_tier": tier, "max_actions": actions}
            with self.subTest(tier=tier, actions=actions), self.assertRaises(SceneContractError):
                validate_budget_declarations(declaration, self.protocol)

    def test_catalog_binds_scene_identity_commitment_and_lineage(self):
        altered = deepcopy(self.scene)
        altered["lineage"]["prototypes"] = ["unregistered.prototype"]
        with self.assertRaisesRegex(SceneContractError, "reservation"):
            validate_catalog(self.catalog, self.protocol, scene_specs=[altered])
        with self.assertRaisesRegex(SceneContractError, "repeated parent"):
            validate_catalog(self.catalog, self.protocol, scene_specs=[self.scene, self.scene])


if __name__ == "__main__":
    unittest.main()
