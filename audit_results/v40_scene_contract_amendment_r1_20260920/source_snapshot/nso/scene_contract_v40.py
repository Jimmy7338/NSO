"""V40 scene reservations and information boundaries; standard library only.

Validation never imports a renderer, reads a mesh, or evaluates a policy.
Seed commitments bind seed material, not nonexistent scene assets.  Public
specification extraction is an explicit allowlist and returns an isolated copy.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


PROTOCOL_VERSION = "v40.scene_protocol.v1"
SCENE_VERSION = "v40.scene_spec.v1"
CATALOG_VERSION = "v40.scene_catalog.v1"
PUBLIC_VERSION = "v40.public_planner_spec.v1"
COMMITMENT_DOMAIN = "NSO_V40_SCENE_SEED_V1"
MIN_STRUCTURE_PROBABILITY = 1e-9
SPLITS = ("development", "main_test", "boundary_test")
FAMILIES = {"A": "cabinet_rooms", "B": "shelves", "C": "mixed_facilities",
            "D": "geometry_sufficient_neutral", "E": "coverage_pressure",
            "F": "prior_or_recognition_mismatch"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SceneContractError(ValueError):
    """A strict schema, split or information boundary has been violated."""


def _fail(path, message):
    raise SceneContractError(f"{path}: {message}")


def _object(value, keys, path):
    if type(value) is not dict or set(value) != set(keys):
        _fail(path, f"expected exactly keys {sorted(keys)}")


def _literal(value, expected, path):
    if type(value) is not type(expected) or value != expected:
        _fail(path, f"expected {expected!r}")


def _choice(value, choices, path):
    if type(value) is not str or value not in choices:
        _fail(path, f"expected one of {tuple(choices)!r}")


def _identifier(value, path):
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        _fail(path, "expected nonempty identifier")


def _number(value, path, minimum=0.0, strictly_positive=False):
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, "expected finite number, not bool")
    if value < minimum or (strictly_positive and value <= 0):
        _fail(path, "number outside allowed range")


def _integer(value, path, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(path, "expected integer in allowed range")


def _sha(value, path):
    if type(value) is not str or not _HEX64.fullmatch(value) or len(set(value)) == 1:
        _fail(path, "expected SHA256 hex; repeated-character placeholders forbidden")


def _ids(value, path):
    if type(value) is not list or not value:
        _fail(path, "expected nonempty identifier list")
    for index, item in enumerate(value):
        _identifier(item, f"{path}[{index}]")
    if len(value) != len(set(value)):
        _fail(path, "duplicate identifier")


def canonical_json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def content_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json_strict(path):
    """Reject duplicate keys and nonfinite JSON constants before validation."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail(str(path), f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        _fail(str(path), f"nonfinite JSON constant {value}")

    with Path(path).open(encoding="utf-8") as stream:
        try:
            return json.load(stream, object_pairs_hook=pairs, parse_constant=constant)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise SceneContractError(f"{path}: malformed JSON") from exc


def pending_asset():
    return {"status": "pending", "relative_path": None, "sha256": None}


def _asset(value, path):
    _object(value, ("status", "relative_path", "sha256"), path)
    _choice(value["status"], ("pending", "materialized"), path + ".status")
    if value["status"] == "pending":
        if value["relative_path"] is not None or value["sha256"] is not None:
            _fail(path, "pending assets must have null path and hash")
        return
    name = value["relative_path"]
    if type(name) is not str or not name or "\\" in name:
        _fail(path, "expected relative POSIX asset path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in (".", "..") for part in name.split("/")):
        _fail(path, "asset path must remain inside its asset root")
    _sha(value["sha256"], path + ".sha256")


def validate_public_spec(value):
    _object(value, ("schema_version", "task", "navigation", "sensor", "motion",
                    "pose_model", "structure_prior"), "public")
    _literal(value["schema_version"], PUBLIC_VERSION, "public.schema_version")
    task = value["task"]
    _object(task, ("start_action", "forced_prefix_actions", "max_actions", "return_required",
                   "budget_basis", "budget_tier"), "public.task")
    _literal(task["start_action"], 0, "public.task.start_action")
    _literal(task["forced_prefix_actions"], [], "public.task.forced_prefix_actions")
    _integer(task["max_actions"], "public.task.max_actions", 1)
    _literal(task["return_required"], True, "public.task.return_required")
    _choice(task["budget_basis"], ("fixed_public_task_scale", "public_navigation_length"), "budget_basis")
    _choice(task["budget_tier"], ("short", "standard"), "budget_tier")
    nav = value["navigation"]
    _object(nav, ("mode", "shared_with_all_methods", "hidden_surface_geometry_allowed",
                  "resolution_m", "status", "graph_asset"), "public.navigation")
    _literal(nav["mode"], "provided_coarse_topology", "navigation.mode")
    _literal(nav["shared_with_all_methods"], True, "navigation.shared_with_all_methods")
    _literal(nav["hidden_surface_geometry_allowed"], False, "navigation.hidden_surface_geometry_allowed")
    _number(nav["resolution_m"], "navigation.resolution_m", strictly_positive=True)
    _asset(nav["graph_asset"], "navigation.graph_asset")
    _literal(nav["status"], nav["graph_asset"]["status"], "navigation.status")
    sensor = value["sensor"]
    _object(sensor, ("model", "width", "height", "intrinsic", "depth_min_m", "depth_max_m",
                     "lidar_max_range_m", "depth_noise_relative_std"), "public.sensor")
    _literal(sensor["model"], "rgbd_and_planar_lidar", "sensor.model")
    _integer(sensor["width"], "sensor.width", 1)
    _integer(sensor["height"], "sensor.height", 1)
    k = sensor["intrinsic"]
    if type(k) is not list or len(k) != 3 or any(type(row) is not list or len(row) != 3 for row in k):
        _fail("sensor.intrinsic", "expected 3x3 list")
    for row in k:
        for number in row:
            _number(number, "sensor.intrinsic", minimum=-1e12)
    if k[0][0] <= 0 or k[1][1] <= 0 or k[0][1] != 0 or k[1][0] != 0 or k[2] != [0, 0, 1]:
        _fail("sensor.intrinsic", "positive focal lengths and zero-skew pinhole matrix required")
    if not 0 <= k[0][2] < sensor["width"] or not 0 <= k[1][2] < sensor["height"]:
        _fail("sensor.intrinsic", "principal point outside image")
    for key in ("depth_min_m", "depth_max_m", "lidar_max_range_m"):
        _number(sensor[key], "sensor." + key, strictly_positive=True)
    if sensor["depth_min_m"] >= sensor["depth_max_m"]:
        _fail("sensor", "depth range must be increasing")
    _number(sensor["depth_noise_relative_std"], "sensor.depth_noise_relative_std")
    motion = value["motion"]
    _object(motion, ("translation_step_m", "rotation_step_deg", "camera_height_m"), "public.motion")
    for key, number in motion.items():
        _number(number, "motion." + key, strictly_positive=True)
    if motion["rotation_step_deg"] > 180:
        _fail("motion.rotation_step_deg", "must not exceed 180 degrees")
    pose = value["pose_model"]
    _object(pose, ("kind", "translation_std_m", "yaw_std_deg", "correlation"), "public.pose_model")
    _choice(pose["kind"], ("exact", "noisy_odometry"), "pose_model.kind")
    _literal(pose["correlation"], "per_trajectory", "pose_model.correlation")
    _number(pose["translation_std_m"], "pose_model.translation_std_m")
    _number(pose["yaw_std_deg"], "pose_model.yaw_std_deg")
    if pose["kind"] == "exact" and (pose["translation_std_m"] or pose["yaw_std_deg"]):
        _fail("pose_model", "exact poses cannot have nonzero noise")
    prior = value["structure_prior"]
    _object(prior, ("categories", "abstract_structures", "probability_by_category",
                    "source_split", "calibration_status", "training_manifest"), "public.structure_prior")
    _ids(prior["categories"], "prior.categories")
    _ids(prior["abstract_structures"], "prior.abstract_structures")
    if len(prior["categories"]) < 2 or len(prior["abstract_structures"]) < 2:
        _fail("prior", "many-to-many prior requires at least two categories and structures")
    _literal(prior["source_split"], "development", "prior.source_split")
    _choice(prior["calibration_status"], ("uncalibrated_design_prior", "frozen_development"), "prior.calibration_status")
    _asset(prior["training_manifest"], "prior.training_manifest")
    if prior["calibration_status"] == "frozen_development" and prior["training_manifest"]["status"] != "materialized":
        _fail("prior", "frozen calibration requires a real training manifest")
    table = prior["probability_by_category"]
    _object(table, prior["categories"], "prior.probability_by_category")
    support = [0] * len(prior["abstract_structures"])
    for category, row in table.items():
        if type(row) is not list or len(row) != len(support):
            _fail("prior." + category, "wrong structure probability count")
        for index, probability in enumerate(row):
            _number(probability, "prior." + category, minimum=MIN_STRUCTURE_PROBABILITY)
            if probability > 1:
                _fail("prior." + category, "probability exceeds one")
            support[index] += int(probability > 0)
        if not math.isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-9) or sum(p > 0 for p in row) < 2:
            _fail("prior." + category, "each category must support multiple structures and sum to one")
    if min(support) < 2:
        _fail("prior", "each abstract structure must be supported by multiple categories")


def _evaluation_definition(value):
    expected = {"target_set": "all_task_instances_accessible_external_surfaces",
                "include_horizontal": True, "include_vertical": True,
                "exclude_closed_internal": True, "route_independent": True,
                "undetected_instances_in_denominator": True,
                "primary_surface_aggregation": "instance_macro_f1", "threshold_m": 0.05}
    _object(value, expected, "evaluation_definition")
    for key, item in expected.items():
        _literal(value[key], item, "evaluation_definition." + key)


def validate_protocol(value):
    _object(value, ("schema_version", "protocol_id", "families", "independence", "generation_policy",
                    "commitment", "public_defaults", "evaluation_definition", "max_output_bytes"), "protocol")
    _literal(value["schema_version"], PROTOCOL_VERSION, "protocol.schema_version")
    _identifier(value["protocol_id"], "protocol.protocol_id")
    families = value["families"]
    _object(families, FAMILIES, "protocol.families")
    for key, label in FAMILIES.items():
        family = families[key]
        _object(family, ("label", "test_split", "test_parents", "development_parents"), "family." + key)
        for field, expected in {"label": label, "test_split": "main_test" if key in "ABCD" else "boundary_test",
                                "test_parents": 6, "development_parents": 1}.items():
            _literal(family[field], expected, "family." + key + "." + field)
    independence = {"unit": "base_parent_scene", "isolation_axes": ["layout", "prototype", "mesh"],
                    "nonindependent_variants": ["rotation", "reflection", "sensor_noise", "structural_condition"],
                    "cross_split_lineage_overlap_allowed": False}
    _object(value["independence"], independence, "independence")
    for key, expected in independence.items():
        _literal(value["independence"][key], expected, "independence." + key)
    policy = {"selection_uses_method_results": False, "category_is_hidden_structure_code": False,
              "geometry_generation_in_p0": False, "geometry_calibration_split": "development",
              "recognition_noise_separate_from_structure_shift": True}
    _object(value["generation_policy"], policy, "generation_policy")
    for key, expected in policy.items():
        _literal(value["generation_policy"][key], expected, "generation_policy." + key)
    commitment = {"algorithm": "sha256", "domain": COMMITMENT_DOMAIN, "seed_bits": 256, "private_escrow": True}
    _object(value["commitment"], commitment, "commitment")
    for key, expected in commitment.items():
        _literal(value["commitment"][key], expected, "commitment." + key)
    validate_public_spec(value["public_defaults"])
    _evaluation_definition(value["evaluation_definition"])
    _integer(value["max_output_bytes"], "max_output_bytes", 1)
    if value["max_output_bytes"] > 5 * 1024 * 1024:
        _fail("max_output_bytes", "P0 output cap exceeds 5 MiB")


def seed_commitment(protocol_id, parent_id, split, seed_hex):
    """Bind a 256-bit seed to its identity; never generate or expose geometry."""
    _identifier(protocol_id, "protocol_id")
    _identifier(parent_id, "parent_id")
    _choice(split, SPLITS, "split")
    if type(seed_hex) is not str or not _HEX64.fullmatch(seed_hex):
        _fail("seed", "expected exactly 256 bits in lowercase hexadecimal")
    payload = {"domain": COMMITMENT_DOMAIN, "protocol_id": protocol_id,
               "parent_id": parent_id, "split": split, "seed_hex": seed_hex}
    return content_sha256(payload)


def _lineage(value, path):
    _object(value, ("status", "layout", "prototypes", "meshes", "derived_from_parent"), path)
    _choice(value["status"], ("reserved_not_materialized", "content_audited"), path + ".status")
    _identifier(value["layout"], path + ".layout")
    _ids(value["prototypes"], path + ".prototypes")
    _ids(value["meshes"], path + ".meshes")
    if value["derived_from_parent"] is not None:
        _fail(path, "transformed/derived variants cannot be registered as independent parents")


def validate_budget_declarations(value, protocol, catalog=None):
    """Explicit boundary-parent budget exceptions; no other public overrides.

    This sidecar is separately frozen and binds the unmodified protocol hash.
    Its absence permits only the protocol defaults.  It does not generate a
    test specification, nor permit a dev/main parent to adopt a boundary tier.
    """
    validate_protocol(protocol)
    _object(value, ("schema_version", "protocol_sha256", "assignments"), "budget_declarations")
    _literal(value["schema_version"], "v40.boundary_budget_declarations.v1", "budget_declarations.schema_version")
    _literal(value["protocol_sha256"], content_sha256(protocol), "budget_declarations.protocol_sha256")
    assignments = value["assignments"]
    if type(assignments) is not dict:
        _fail("budget_declarations.assignments", "expected parent-keyed object")
    entries = {entry["parent_id"]: entry for entry in catalog["entries"]} if catalog is not None else None
    standard = protocol["public_defaults"]["task"]["max_actions"]
    for parent, declaration in assignments.items():
        _identifier(parent, "budget_declarations.parent_id")
        _object(declaration, ("budget_tier", "max_actions"), "budget_declarations." + parent)
        _choice(declaration["budget_tier"], ("short", "standard"), "budget_declarations.budget_tier")
        _integer(declaration["max_actions"], "budget_declarations.max_actions", 1)
        if declaration["budget_tier"] == "standard" and declaration["max_actions"] != standard:
            _fail("budget_declarations", "standard tier must equal the protocol default budget")
        if declaration["budget_tier"] == "short" and declaration["max_actions"] >= standard:
            _fail("budget_declarations", "short tier must be below the protocol default budget")
        if entries is not None and (parent not in entries or entries[parent]["split"] != "boundary_test"):
            _fail("budget_declarations", "budget exceptions require a registered boundary-test parent")


def _bind_public_to_protocol(scene_spec, protocol, budget_declarations=None):
    validate_protocol(protocol)
    _literal(scene_spec["protocol_id"], protocol["protocol_id"], "scene.protocol_id")
    expected = deepcopy(protocol["public_defaults"])
    if budget_declarations is not None:
        validate_budget_declarations(budget_declarations, protocol)
        declaration = budget_declarations["assignments"].get(scene_spec["parent_id"])
        if declaration is not None:
            if scene_spec["split"] != "boundary_test":
                _fail("scene.public", "boundary budget declaration cannot apply to development/main scenes")
            expected["task"].update(declaration)
    if scene_spec["public"] != expected:
        _fail("scene.public", "values differ from protocol defaults or explicit boundary budget declaration")


def validate_scene_spec(value, expected_split=None, *, protocol=None, budget_declarations=None):
    _object(value, ("schema_version", "protocol_id", "parent_id", "family", "split", "status", "lineage",
                    "seed_record", "public", "renderer_private", "evaluation_private"), "scene")
    _literal(value["schema_version"], SCENE_VERSION, "scene.schema_version")
    _identifier(value["protocol_id"], "scene.protocol_id")
    _identifier(value["parent_id"], "scene.parent_id")
    _choice(value["family"], FAMILIES, "scene.family")
    _choice(value["split"], SPLITS, "scene.split")
    if expected_split is not None:
        _literal(value["split"], expected_split, "scene.split")
    if value["split"] != "development" and value["split"] != ("main_test" if value["family"] in "ABCD" else "boundary_test"):
        _fail("scene", "family and test split disagree")
    _choice(value["status"], ("pending_geometry", "materialized_unfrozen", "frozen_ready"), "scene.status")
    _lineage(value["lineage"], "scene.lineage")
    seed = value["seed_record"]
    _object(seed, ("visibility", "seed_hex", "commitment"), "scene.seed_record")
    _sha(seed["commitment"], "scene.seed_record.commitment")
    if value["split"] == "development":
        _literal(seed["visibility"], "development", "seed_record.visibility")
        if seed_commitment(value["protocol_id"], value["parent_id"], value["split"], seed["seed_hex"]) != seed["commitment"]:
            _fail("seed_record", "development seed commitment mismatch")
    else:
        _literal(seed["visibility"], "sealed", "seed_record.visibility")
        if seed["seed_hex"] is not None:
            _fail("seed_record", "test seed must not appear in scene specification")
    validate_public_spec(value["public"])
    renderer = value["renderer_private"]
    _object(renderer, ("world_asset", "instances_asset", "asset_inventory"), "renderer_private")
    for key, item in renderer.items():
        _asset(item, "renderer_private." + key)
    evaluator = value["evaluation_private"]
    _object(evaluator, ("definition", "gt_mesh_asset", "evaluation_roi_asset", "surface_set_asset"), "evaluation_private")
    _evaluation_definition(evaluator["definition"])
    for key in ("gt_mesh_asset", "evaluation_roi_asset", "surface_set_asset"):
        _asset(evaluator[key], "evaluation_private." + key)
    if value["status"] == "pending_geometry":
        assets = list(renderer.values()) + [evaluator[key] for key in ("gt_mesh_asset", "evaluation_roi_asset", "surface_set_asset")]
        assets.append(value["public"]["navigation"]["graph_asset"])
        if any(asset["status"] != "pending" for asset in assets):
            _fail("scene", "pending geometry requires pending/null private assets and public navigation graph")
    if value["status"] == "frozen_ready":
        assets = list(renderer.values()) + [evaluator[key] for key in ("gt_mesh_asset", "evaluation_roi_asset", "surface_set_asset")]
        assets += [value["public"]["navigation"]["graph_asset"], value["public"]["structure_prior"]["training_manifest"]]
        if any(asset["status"] != "materialized" for asset in assets) or value["lineage"]["status"] != "content_audited":
            _fail("scene", "frozen_ready cannot have pending assets or unaudited content lineage")
        if value["public"]["structure_prior"]["calibration_status"] != "frozen_development":
            _fail("scene", "frozen_ready requires a frozen development prior")
    if protocol is not None:
        _bind_public_to_protocol(value, protocol, budget_declarations)
    elif budget_declarations is not None:
        _fail("scene", "budget declarations require the binding protocol")


def public_planner_spec(scene_spec, *, protocol=None, budget_declarations=None):
    """Whitelist only the shared task, calibration and abstract design prior.

    Does not return parent/family/split, seed, instance truth, evaluation ROI,
    GT, lineage, asset inventory, or private renderer/evaluator references.
    Validation rejects unknown nested keys rather than silently forwarding them.
    """
    validate_scene_spec(scene_spec, protocol=protocol, budget_declarations=budget_declarations)
    return deepcopy(scene_spec["public"])


def validate_catalog(catalog, protocol, *, scene_specs=None, budget_declarations=None):
    validate_protocol(protocol)
    _object(catalog, ("schema_version", "protocol_id", "protocol_sha256", "status", "entries",
                     "geometry_generation_executed", "policy_or_metric_execution_count"), "catalog")
    _literal(catalog["schema_version"], CATALOG_VERSION, "catalog.schema_version")
    _literal(catalog["protocol_id"], protocol["protocol_id"], "catalog.protocol_id")
    _literal(catalog["protocol_sha256"], content_sha256(protocol), "catalog.protocol_sha256")
    _literal(catalog["status"], "parent_and_seed_reservations_only", "catalog.status")
    _literal(catalog["geometry_generation_executed"], False, "catalog.geometry_generation_executed")
    _literal(catalog["policy_or_metric_execution_count"], 0, "catalog.policy_or_metric_execution_count")
    entries = catalog["entries"]
    if type(entries) is not list or len(entries) != 42:
        _fail("catalog.entries", "expected 6 development + 24 main + 12 boundary parents")
    counts, parents, commitments, layouts = {}, set(), set(), set()
    lineage_splits = {"prototype": {}, "mesh": {}}
    for entry in entries:
        _object(entry, ("parent_id", "family", "split", "lineage", "seed_commitment", "assets"), "catalog.entry")
        parent, family, split = entry["parent_id"], entry["family"], entry["split"]
        _identifier(parent, "entry.parent_id")
        _choice(family, FAMILIES, "entry.family")
        _choice(split, SPLITS, "entry.split")
        if split != "development" and split != protocol["families"][family]["test_split"]:
            _fail("catalog", "test family assigned to incorrect split")
        if parent in parents:
            _fail("catalog", "duplicate independent parent")
        parents.add(parent)
        counts[(family, split)] = counts.get((family, split), 0) + 1
        _sha(entry["seed_commitment"], "entry.seed_commitment")
        if entry["seed_commitment"] in commitments:
            _fail("catalog", "duplicate seed commitment")
        commitments.add(entry["seed_commitment"])
        lineage = entry["lineage"]
        _lineage(lineage, "entry.lineage")
        _literal(lineage["status"], "reserved_not_materialized", "entry.lineage.status")
        if lineage["layout"] in layouts:
            _fail("catalog", "same layout lineage cannot count as independent parents")
        layouts.add(lineage["layout"])
        for axis, names in (("prototype", lineage["prototypes"]), ("mesh", lineage["meshes"])):
            for name in names:
                previous = lineage_splits[axis].setdefault(name, split)
                if previous != split:
                    _fail("catalog", f"{axis} lineage crosses development/main/boundary split")
        _object(entry["assets"], ("layout", "prototype_bundle", "mesh_bundle"), "entry.assets")
        for name, record in entry["assets"].items():
            _asset(record, "entry.assets." + name)
            if record["status"] != "pending":
                _fail("catalog", "reservation catalog must not claim frozen/materialized scene assets")
    expected = {(family, "development"): 1 for family in FAMILIES}
    expected.update({(family, info["test_split"]): 6 for family, info in protocol["families"].items()})
    if counts != expected:
        _fail("catalog", "family/split parent allocation must match 6 + 24 + 12")
    if budget_declarations is not None:
        validate_budget_declarations(budget_declarations, protocol, catalog)
    if scene_specs is not None:
        if type(scene_specs) is not list:
            _fail("catalog.scene_specs", "expected explicit list of scene specifications")
        indexed = {entry["parent_id"]: entry for entry in entries}
        seen = set()
        for scene in scene_specs:
            validate_scene_spec(scene, protocol=protocol, budget_declarations=budget_declarations)
            parent = scene["parent_id"]
            if parent not in indexed or parent in seen:
                _fail("catalog.scene_specs", "unknown or repeated parent specification")
            seen.add(parent)
            entry = indexed[parent]
            if (scene["family"] != entry["family"] or scene["split"] != entry["split"]
                    or scene["lineage"] != entry["lineage"]
                    or scene["seed_record"]["commitment"] != entry["seed_commitment"]):
                _fail("catalog.scene_specs", "scene identity, lineage or seed commitment differs from reservation")


def assert_formal_test_ready(scene_spec, *, catalog, protocol):
    """Fail closed: this P0 reservation format cannot authorize formal tests.

    A later materialized-asset schema must verify content-level split isolation,
    actual meshes/layouts/prototypes, evaluation target set and protocol freeze.
    Merely replacing status strings or seed hashes cannot fulfill that audit.
    """
    validate_catalog(catalog, protocol, scene_specs=[scene_spec])
    _fail("formal_test_admission", "P0 reservation only: geometry, content lineage audit and evaluation freeze are pending")


def make_development_spec(protocol, entry, seed_hex):
    validate_protocol(protocol)
    if entry["split"] != "development":
        _fail("make_development_spec", "this constructor only creates development skeletons")
    spec = {"schema_version": SCENE_VERSION, "protocol_id": protocol["protocol_id"],
            "parent_id": entry["parent_id"], "family": entry["family"], "split": "development",
            "status": "pending_geometry", "lineage": deepcopy(entry["lineage"]),
            "seed_record": {"visibility": "development", "seed_hex": seed_hex,
                            "commitment": entry["seed_commitment"]},
            "public": deepcopy(protocol["public_defaults"]),
            "renderer_private": {key: pending_asset() for key in ("world_asset", "instances_asset", "asset_inventory")},
            "evaluation_private": {"definition": deepcopy(protocol["evaluation_definition"]),
                                   **{key: pending_asset() for key in ("gt_mesh_asset", "evaluation_roi_asset", "surface_set_asset")}}}
    validate_scene_spec(spec, expected_split="development", protocol=protocol)
    return spec
