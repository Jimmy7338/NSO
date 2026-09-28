"""Static shape/serialization tests; no World, sensing, planner or quality runs."""
import ast
from collections import deque
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nso.development_geometry_v40 import (MAX_ASSET_BYTES, PALETTE_V40, STRUCTURES,
    build_development_geometry, load_mesh_arrays, mesh_audit, mesh_npz_bytes, structure_boxes, union_box_mesh)
from nso.scene_contract_v40 import load_json_strict, public_planner_spec

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audit_results/v40_scene_contract_20260920"
_spec = importlib.util.spec_from_file_location("build_development_geometry_v40", ROOT / "scripts/build_development_geometry_v40.py")
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def inside(points, boxes):
    bounds = np.asarray(boxes).reshape(-1, 3, 2)
    return np.any(np.all((points[:, None, :] > bounds[None, :, :, 0]) &
                         (points[:, None, :] < bounds[None, :, :, 1]), axis=2), axis=1)


class DevelopmentGeometryP1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_json_strict(SOURCE / "protocol.json")
        cls.specs = {family: load_json_strict(SOURCE / "development" / f"DEV_{family}_00.json") for family in "ABCDEF"}
        cls.geometry = {family: build_development_geometry(spec, cls.protocol) for family, spec in cls.specs.items()}

    def test_box_union_removes_touching_and_overlapping_internal_faces(self):
        for shift, volume, area in ((1., 2., 10.), (.5, 1.5, 8.)):
            v, t = union_box_mesh([[0, 1, 0, 1, 0, 1], [shift, shift+1, 0, 1, 0, 1]])
            audit = mesh_audit(v, t, np.zeros(len(t), np.int32))["by_instance"]["0"]
            self.assertAlmostEqual(audit["signed_volume_m3"], volume)
            self.assertAlmostEqual(audit["surface_area_m2"], area)

    def test_shape_normals_point_from_solid_to_free_space(self):
        for shape in STRUCTURES:
            boxes, marker = structure_boxes(shape, [1.2, .8, 1.6])
            v, t = union_box_mesh(boxes)
            p = v[t]; centers = p.mean(axis=1)
            normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            with self.subTest(shape=shape):
                self.assertTrue(inside(centers - 1e-6*normals, boxes).all())
                self.assertFalse(inside(centers + 1e-6*normals, boxes).any())
                mesh_audit(v, t, np.zeros(len(t), np.int32))

    def test_open_structures_are_real_cavities_not_missing_triangles(self):
        point = np.asarray([[0., 0., .35*1.6]])
        for shape in STRUCTURES:
            boxes, _ = structure_boxes(shape, [1.2, .8, 1.6])
            self.assertEqual(bool(inside(point, boxes)[0]), shape == "planar")

    def test_marker_is_on_attached_face_with_outward_normal(self):
        for shape in STRUCTURES:
            boxes, marker = structure_boxes(shape, [1.2, .8, 1.6])
            center = np.asarray(marker["center_local_m"])[None, :]
            normal = np.asarray(marker["normal_local"])[None, :]
            self.assertTrue(inside(center - 1e-6*normal, boxes)[0])
            self.assertFalse(inside(center + 1e-6*normal, boxes)[0])

    def test_six_meshes_closed_finite_and_labeled_per_instance(self):
        for family, geometry in self.geometry.items():
            with self.subTest(family=family):
                self.assertEqual(set(np.unique(geometry.triangle_instance_id)), {-1, 0, 1, 2, 3})
                self.assertEqual(len(geometry.private_instances), 4)
                self.assertTrue(all(item["closed_edge_manifold"] for item in geometry.audit["by_instance"].values()))
                self.assertFalse(geometry.vertices.flags.writeable)
                self.assertFalse(geometry.triangles.flags.writeable)
                self.assertFalse(geometry.audit["world_or_sensor_instantiated"])

    def test_geometry_reproducible_from_only_development_inputs(self):
        repeated = build_development_geometry(self.specs["A"], self.protocol)
        self.assertEqual(repeated.logical_sha256(), self.geometry["A"].logical_sha256())
        for name, array in repeated.arrays().items():
            np.testing.assert_array_equal(array, self.geometry["A"].arrays()[name])

    def test_test_split_rejected_before_generation(self):
        scene = deepcopy(self.specs["A"])
        scene["split"] = "main_test"
        with self.assertRaises(ValueError):
            build_development_geometry(scene, self.protocol)

    def test_categories_and_structures_are_many_to_many(self):
        by_class, by_structure = {}, {}
        for geometry in self.geometry.values():
            for instance in geometry.private_instances:
                by_class.setdefault(instance["category"], set()).add(instance["structure"])
                by_structure.setdefault(instance["structure"], set()).add(instance["category"])
        self.assertEqual(set(by_class), set(PALETTE_V40))
        self.assertTrue(all(len(values) >= 2 for values in by_class.values()))
        self.assertTrue(all(len(values) >= 2 for values in by_structure.values()))

    def test_neutral_pressure_and_mismatch_are_distinct_design_conditions(self):
        self.assertEqual({item["structure"] for item in self.geometry["D"].private_instances}, {"planar"})
        area = lambda g: np.prod(np.asarray(g.public_workspace["bounds_xy_m"])[1])
        self.assertGreater(area(self.geometry["E"]), 2*area(self.geometry["A"]))
        self.assertTrue(all(item["relationship_shift"] and not item["recognition_corruption"]
                            for item in self.geometry["F"].private_instances))
        self.assertTrue(all(item["declared_class_structure_probability"] <= .15
                            for item in self.geometry["F"].private_instances))

    def test_static_inflated_free_space_reaches_neighborhood_of_each_facility(self):
        # Development feasibility only: no graph is exported or given to a planner.
        for family, geometry in self.geometry.items():
            width, height = geometry.public_workspace["bounds_xy_m"][1]
            xs, ys = np.arange(.25, width, .25), np.arange(.25, height, .25)
            x, y = np.meshgrid(xs, ys, indexing="ij")
            obstacles = [np.asarray(box).reshape(3, 2).T[:, :2] for box in geometry.background_boxes if box[5] > 0]
            obstacles += [np.asarray(item["world_aabb_m"])[:, :2] for item in geometry.private_instances]
            free = np.ones(x.shape, dtype=bool)
            for bounds in obstacles:
                free &= ~((x >= bounds[0, 0]-.2) & (x <= bounds[1, 0]+.2) &
                          (y >= bounds[0, 1]-.2) & (y <= bounds[1, 1]+.2))
            start = (int(np.argmin(abs(xs-.75))), int(np.argmin(abs(ys-.75))))
            self.assertTrue(free[start], family)
            queue, seen = deque([start]), {start}
            while queue:
                i, j = queue.popleft()
                for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if 0 <= ni < len(xs) and 0 <= nj < len(ys) and free[ni, nj] and (ni, nj) not in seen:
                        seen.add((ni, nj)); queue.append((ni, nj))
            reachable = np.asarray([[xs[i], ys[j]] for i, j in seen])
            for item in geometry.private_instances:
                center = np.asarray(item["center_world_m"])[:2]
                self.assertLess(float(np.linalg.norm(reachable-center, axis=1).min()), 1.6, (family, item["instance_id"]))

    def test_private_metadata_does_not_enter_public_planner_spec(self):
        for family, geometry in self.geometry.items():
            public = public_planner_spec(self.specs[family], protocol=self.protocol)
            self.assertEqual(public, self.protocol["public_defaults"])
            self.assertFalse(geometry.public_workspace["instance_truth_included"])
            for key in ("private_instances", "marker_patches", "structure", "world_aabb_m", "triangle_instance_id"):
                self.assertNotIn(key, geometry.public_workspace)

    def test_roundtrip_parser_keeps_arrays_and_rejects_invalid_indices(self):
        geometry = self.geometry["D"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.npz"
            path.write_bytes(mesh_npz_bytes(geometry))
            parsed = load_mesh_arrays(path)
            for name, array in geometry.arrays().items():
                np.testing.assert_array_equal(parsed[name], array)
            changed = {key: np.array(value) for key, value in parsed.items()}
            changed["triangles"][0, 0] = len(parsed["vertices"])
            np.savez_compressed(path, **changed)
            with self.assertRaises(ValueError):
                load_mesh_arrays(path)

    def test_invalid_geometry_and_archive_fields_rejected(self):
        for boxes in ([], [[0, 0, 0, 1, 0, 1]], [[0, 1, 0, 1, 0, float("nan")]]):
            with self.assertRaises(ValueError):
                union_box_mesh(boxes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unexpected.npz"
            np.savez_compressed(path, unexpected=np.ones(3))
            with self.assertRaises(ValueError):
                load_mesh_arrays(path)

    def test_builder_contract_artifacts_and_storage_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "assets"
            report = builder.build_assets(SOURCE, output)
            self.assertLess(report["bytes_written"], MAX_ASSET_BYTES)
            self.assertEqual(report["worlds_created"], 0)
            self.assertEqual(report["facilities"], 24)
            for family in "ABCDEF":
                parent = output / f"DEV_{family}_00"
                metadata = load_json_strict(parent / "evaluation_private/instances.json")
                self.assertTrue({"private_instances", "background_boxes", "public_workspace"}.issubset(metadata))
                parsed = load_mesh_arrays(parent / "renderer_private/geometry.npz")
                self.assertEqual(set(np.unique(parsed["triangle_instance_id"])), {-1, 0, 1, 2, 3})
            with self.assertRaises(FileExistsError):
                builder.build_assets(SOURCE, output)

    def test_geometry_implementation_does_not_import_world_or_renderer(self):
        tree = ast.parse((ROOT / "nso/development_geometry_v40.py").read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
        self.assertFalse(any(name.startswith(("env", "open3d")) for name in imports))


if __name__ == "__main__":
    unittest.main()
