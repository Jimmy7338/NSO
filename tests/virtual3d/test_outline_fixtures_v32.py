"""Independent geometry checks; no scoring, World, mapper or sensor creation."""
import hashlib
from pathlib import Path
import unittest
import numpy as np

from nso.outline_fixtures_v32 import (PROTOCOL_SHA256, GAP_MM, TRANSLATION_MM, ROTATIONS,
    NEIGHBOR_GAP_MM, noise_fixtures_v32, noise_fixture_names_v32)
from tests.virtual3d.test_outline_fixtures_v31 import (area, volume, bounds, edge_counts,
    ray_hits, xy_projected_edge_degrees)


class OutlineNoiseFixtureV32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = noise_fixtures_v32()

    def prediction(self, name, slot=0):
        return self.cases[name].predictions[slot]

    def test_frozen_catalogue_finite_readonly_and_deterministic(self):
        path = Path(__file__).resolve().parents[2]/'docs/research/V32_NOISE_FIXTURES_PROTOCOL_20260917.md'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), PROTOCOL_SHA256)
        self.assertEqual(tuple(self.cases), noise_fixture_names_v32())
        self.assertEqual(len(self.cases), 33)
        again = noise_fixtures_v32()
        for name, case in self.cases.items():
            self.assertEqual(len(case.references), len(case.predictions))
            self.assertEqual(len(case.references), len(case.facts['reference_seed_xyz']))
            self.assertFalse(case.labels['q_gate_declared'])
            for mesh in (*case.references, *case.predictions):
                self.assertFalse(mesh.vertices.flags.writeable)
                self.assertFalse(mesh.triangles.flags.writeable)
                triangles = mesh.vertices[mesh.triangles]
                sizes = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
                self.assertTrue(np.isfinite(triangles).all())
                self.assertTrue(np.all(sizes > 0.), name)
            for first, second in zip(case.predictions, again[name].predictions):
                self.assertEqual(first.vertices.tobytes(), second.vertices.tobytes())
                self.assertEqual(first.triangles.tobytes(), second.triangles.tobytes())

    def test_corner_gap_exact_width_area_and_no_silent_repair(self):
        for mm in GAP_MM:
            name = f'corner_gap_{mm:03d}mm'
            mesh = self.prediction(name)
            gap = mm/1000.
            self.assertAlmostEqual(area(mesh), 16.-1.6*gap, places=12)
            degrees = xy_projected_edge_degrees(mesh)
            ends = [np.array(p) for p, degree in degrees.items() if degree == 1]
            if mm == 0:
                self.assertEqual(ends, [])
                self.assertTrue(all(n == 2 for n in degrees.values()))
            else:
                self.assertEqual(len(ends), 2)
                self.assertAlmostEqual(float(np.linalg.norm(ends[0]-ends[1])), gap, places=12)
                # Viewing through the missing corner must remain unblocked.
                self.assertEqual(ray_hits(mesh, (-1., gap/2, .8), (1., 0., 0.)), [3.])

    def test_rigid_translation_including_49_50_51mm(self):
        base = self.prediction('corner_gap_000mm')
        for mm in TRANSLATION_MM:
            mesh = self.prediction(f'translate_x_{mm:03d}mm')
            expected = np.broadcast_to([mm/1000., 0., 0.], base.vertices.shape)
            np.testing.assert_allclose(mesh.vertices-base.vertices, expected, atol=1e-14, rtol=0)
            np.testing.assert_allclose(np.diff(bounds(mesh), axis=0)[0], [2., 3., 1.6], atol=1e-14)
            self.assertAlmostEqual(area(mesh), 16.)
            self.assertEqual(mesh.triangles.tobytes(), base.triangles.tobytes())
        self.assertTrue(all(self.cases[f'translate_x_{mm:03d}mm'].facts['explicitly_samples_5cm_boundary']
                            for mm in (49, 50, 51)))

    def test_rotations_preserve_distances_not_world_bounds(self):
        base = self.prediction('corner_gap_000mm')
        base_distances = np.linalg.norm(base.vertices[:, None]-base.vertices[None, :], axis=-1)
        for name, degrees in ROTATIONS:
            mesh = self.prediction('rotate_z_'+name)
            distances = np.linalg.norm(mesh.vertices[:, None]-mesh.vertices[None, :], axis=-1)
            np.testing.assert_allclose(distances, base_distances, atol=1e-12)
            np.testing.assert_allclose(mesh.vertices.mean(axis=0), [1., 1.5, .8], atol=1e-12)
            self.assertAlmostEqual(area(mesh), 16., places=12)
            self.assertFalse(np.allclose(bounds(mesh), bounds(base)))
            # Directly check one known off-centre vector, independent of stored matrix.
            v = base.vertices[0, :2]-[1., 1.5]
            w = mesh.vertices[0, :2]-[1., 1.5]
            angle = np.rad2deg(np.arctan2(v[0]*w[1]-v[1]*w[0], np.dot(v, w)))
            self.assertAlmostEqual(angle, degrees, places=11)

    def test_missing_side_and_real_l_are_distinct_geometric_risks(self):
        missing = self.prediction('one_vertical_side_missing')
        self.assertAlmostEqual(area(missing), 11.2)
        self.assertEqual(ray_hits(missing, (1., 1.5, .8), (1., 0., 0.)), [])
        case = self.cases['concave_l_vertical']
        self.assertAlmostEqual(volume(case.references[0]), 6.4)
        self.assertAlmostEqual(area(case.predictions[0]), 16.)
        self.assertEqual(ray_hits(case.references[0], (1.25, 2., -1), (0, 0, 1)), [])
        self.assertTrue(all(n == 2 for n in xy_projected_edge_degrees(case.predictions[0]).values()))

    def test_near_instances_stay_separate_and_bridge_is_real_false_geometry(self):
        for mm in NEIGHBOR_GAP_MM:
            name, gap = f'neighbors_gap_{mm:03d}mm', mm/1000.
            correct, wrong = self.cases[name], self.cases[name+'_wrong_bridge']
            self.assertEqual(len(correct.references), 2)
            first, second = correct.predictions
            self.assertAlmostEqual(bounds(second)[0, 0]-bounds(first)[1, 0], gap, places=12)
            for pred, ref in zip(correct.predictions, correct.references):
                self.assertEqual(pred.vertices.tobytes(), ref.vertices.tobytes())
                self.assertEqual(pred.triangles.tobytes(), ref.triangles.tobytes())
            self.assertAlmostEqual(volume(wrong.predictions[0])-volume(first), .09*gap, places=12)
            self.assertEqual(wrong.predictions[1].vertices.tobytes(), second.vertices.tobytes())
            self.assertFalse(correct.facts['cross_instance_union_allowed'])
            self.assertFalse(wrong.facts['cross_instance_union_allowed'])
            self.assertEqual(ray_hits(first, (1.5, 1.5, .8), (1., 0., 0.)), [.5])
            self.assertEqual(ray_hits(wrong.predictions[0], (1.5, 1.5, .8), (1., 0., 0.)), [round(.5+gap, 10)])

    def test_independent_face_translations_preserve_valid_planar_rectangles(self):
        normalized = []
        for mm in (1, 5, 10):
            case = self.cases[f'independent_face_jitter_{mm:03d}mm']
            mesh = case.predictions[0]
            self.assertEqual(len(mesh.triangles), 8)
            self.assertAlmostEqual(area(mesh), 16., places=12)
            offsets = np.asarray(case.facts['face_offsets_xyz_m'])
            self.assertTrue(np.all(np.abs(offsets) <= mm/1000.))
            self.assertTrue(np.all(np.linalg.norm(offsets, axis=1) > 0))
            self.assertEqual(len(set(map(tuple, offsets))), 4)
            normalized.append(offsets/(mm/1000.))
            for face in range(4):
                points = mesh.vertices[np.unique(mesh.triangles[2*face:2*face+2])]
                self.assertEqual(len(points), 4)
                axis, _ = case.facts['face_order_axis_sign'][face]
                self.assertAlmostEqual(float(np.ptp(points[:, axis])), 0., places=14)
                side_lengths = np.sort(np.ptp(points, axis=0))
                np.testing.assert_allclose(side_lengths, [0., 1.6, 3. if axis == 0 else 2.], atol=1e-12)
        np.testing.assert_allclose(normalized[0], normalized[1], atol=1e-15)
        np.testing.assert_allclose(normalized[0], normalized[2], atol=1e-15)

    def test_identical_observation_compatible_with_solid_and_true_open_enclosure(self):
        closed = self.cases['ambiguity_closed_solid_missing_corner']
        opened = self.cases['ambiguity_true_open_shell_same_observation']
        for field in ('vertices', 'triangles'):
            self.assertEqual(getattr(closed.predictions[0], field).tobytes(),
                             getattr(opened.predictions[0], field).tobytes())
        self.assertNotEqual(closed.references[0].vertices.tobytes(), opened.references[0].vertices.tobytes())
        self.assertTrue(all(n == 2 for n in edge_counts(opened.references[0]).values()))
        self.assertTrue(all(n == 2 for n in edge_counts(closed.references[0]).values()))
        self.assertFalse(closed.facts['enclosure_open'])
        self.assertTrue(opened.facts['enclosure_open'])
        # The extra free-ray query distinguishes truths, but is NOT in their
        # common input mesh. Material watertightness does not seal the enclosure.
        self.assertEqual(ray_hits(closed.references[0], (-1., .006, .8), (1., 0., 0.)), [1., 3.])
        self.assertEqual(ray_hits(opened.references[0], (-1., .006, .8), (1., 0., 0.)), [2.998, 3.])
        self.assertEqual(ray_hits(opened.references[0], (-1., .001, .8), (1., 0., 0.)), [1., 3.])
        self.assertAlmostEqual(volume(opened.references[0]), (6.-1.996*2.996-.002*.008)*1.6, places=11)
        observation = closed.predictions[0]
        # Every triangle centre is on an external surface present in both truths.
        for triangle in observation.vertices[observation.triangles]:
            center = triangle.mean(axis=0)
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            normal /= np.linalg.norm(normal)
            for reference in (closed.references[0], opened.references[0]):
                self.assertAlmostEqual(ray_hits(reference, center+.1*normal, -normal)[0], .1, places=10)

    def test_thin_material_loop_vs_one_mm_true_opening_keep_same_reference(self):
        closed = self.cases['thin_walls_continuous_002mm']
        opened = self.cases['thin_walls_001mm_opening']
        self.assertEqual(closed.references[0].vertices.tobytes(), opened.references[0].vertices.tobytes())
        self.assertEqual(closed.references[0].triangles.tobytes(), opened.references[0].triangles.tobytes())
        a, b = closed.predictions[0], opened.predictions[0]
        self.assertTrue(all(n == 2 for n in edge_counts(a).values()))
        self.assertTrue(all(n == 2 for n in edge_counts(b).values()))
        self.assertAlmostEqual(volume(a), .032, places=12)
        self.assertAlmostEqual(volume(a)-volume(b), .002*.001*1.6, places=12)
        self.assertEqual(ray_hits(a, (-1., .0015, .8), (1., 0., 0.)), [.999, 1.001, 2.999, 3.001])
        self.assertEqual(ray_hits(b, (-1., .0015, .8), (1., 0., 0.)), [2.999, 3.001])


if __name__ == '__main__':
    unittest.main()
