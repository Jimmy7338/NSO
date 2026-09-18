"""Analytic fixture facts only: no score, saved mesh, simulator or TSDF calls."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import unittest
import numpy as np

from nso.outline_fixtures_v31 import (PROTOCOL_SHA256, geometry_fixtures_v31,
    reference_meshes_v31, association_fixtures_v31)


def area(mesh):
    faces = mesh.vertices[mesh.triangles]
    return float(np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=1).sum()/2)


def volume(mesh):
    faces = mesh.vertices[mesh.triangles]
    return float(np.einsum('ij,ij->i', faces[:, 0], np.cross(faces[:, 1], faces[:, 2])).sum()/6)


def bounds(mesh):
    return np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])


def edge_counts(mesh):
    counts = Counter()
    for row in mesh.triangles:
        for i in range(3):
            counts[tuple(sorted((int(row[i]), int(row[(i+1) % 3]))))] += 1
    return counts


def components(mesh):
    graph = defaultdict(set)
    for a, b in edge_counts(mesh):
        graph[a].add(b); graph[b].add(a)
    unseen = set(graph)
    result = 0
    while unseen:
        todo = [next(iter(unseen))]
        while todo:
            point = todo.pop()
            if point in unseen:
                unseen.remove(point); todo.extend(graph[point])
        result += 1
    return result


def xy_projected_edge_degrees(mesh):
    # Used only on vertical, common-grid rectangle fixtures: all projected
    # triangle edges are zero-length or exactly one pre-segmented boundary edge.
    edges = set()
    for face in mesh.vertices[mesh.triangles]:
        for i in range(3):
            a, b = tuple(face[i, :2]), tuple(face[(i+1) % 3, :2])
            if a != b:
                edges.add(tuple(sorted((a, b))))
    degrees = Counter()
    for a, b in edges:
        degrees[a] += 1; degrees[b] += 1
    return degrees


def ray_hits(mesh, origin, direction):
    """Independent triangle intersections; count a physical point only once."""
    faces = mesh.vertices[mesh.triangles]
    d, o = np.asarray(direction, float), np.asarray(origin, float)
    e1, e2 = faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]
    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    determinant = np.einsum('ij,ij->i', e1, h)
    valid = np.abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1/determinant[valid]
    s = o-faces[:, 0]
    u = inverse*np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inverse*np.einsum('j,ij->i', d, q)
    t = inverse*np.einsum('ij,ij->i', e2, q)
    valid &= (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10) & (t >= 0)
    return sorted(set(np.round(t[valid], 10)))


class OutlineFixtureV31Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = geometry_fixtures_v31()

    def mesh(self, name):
        return self.fixtures[name].prediction

    def test_protocol_bound_arrays_readonly_and_fixed_catalogue(self):
        protocol = Path(__file__).resolve().parents[2]/'docs/research/V31_ANALYTIC_FIXTURES_PROTOCOL_20260917.md'
        self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), PROTOCOL_SHA256)
        self.assertEqual(len(self.fixtures), 17)
        for fixture in self.fixtures.values():
            self.assertFalse(fixture.prediction.vertices.flags.writeable)
            self.assertFalse(fixture.prediction.triangles.flags.writeable)
            self.assertIn(fixture.reference_name, ('box', 'l_shape'))

    def test_closed_box_and_vertical_sheets_are_distinct_representations(self):
        closed, vertical = self.mesh('box_closed'), self.mesh('box_vertical')
        self.assertAlmostEqual(area(closed), 28.)
        self.assertAlmostEqual(volume(closed), 9.6)
        self.assertTrue(all(n == 2 for n in edge_counts(closed).values()))
        self.assertAlmostEqual(area(vertical), 16.)
        self.assertTrue(any(n == 1 for n in edge_counts(vertical).values()))
        self.assertTrue(all(n == 2 for n in xy_projected_edge_degrees(vertical).values()))
        self.assertEqual(ray_hits(closed, (1, 1.5, -1), (0, 0, 1)), [1., 2.6])
        self.assertEqual(ray_hits(vertical, (1, 1.5, -1), (0, 0, 1)), [])

    def test_retriangulation_and_duplicate_faces_preserve_physical_hits(self):
        rays = [((x, y, -1), (0, 0, 1)) for x in (.25, 1., 1.75) for y in (.25, 1.5, 2.75)]
        rays += [((-1, y, z), (1, 0, 0)) for y in (.5, 1.5, 2.5) for z in (.2, .8, 1.4)]
        rays += [((x, -1, z), (0, 1, 0)) for x in (.25, 1., 1.75) for z in (.2, .8, 1.4)]
        for base in ('box', 'box_vertical'):
            original = self.mesh(base+'_closed' if base == 'box' else base)
            for name in (base+'_retriangulated', base+'_duplicate'):
                changed = self.mesh(name)
                self.assertGreater(len(changed.triangles), len(original.triangles))
                for origin, direction in rays:
                    self.assertEqual(ray_hits(original, origin, direction), ray_hits(changed, origin, direction),
                                     (name, origin, direction))
            self.assertAlmostEqual(area(self.mesh(base+'_retriangulated')), area(original))
            self.assertAlmostEqual(area(self.mesh(base+'_duplicate')), 2*area(original))

    def test_four_thin_walls_keep_measured_material_and_central_void(self):
        mesh = self.mesh('box_four_thin_walls')
        self.assertAlmostEqual(volume(mesh), .032, places=11)
        self.assertAlmostEqual(area(mesh), 32.04, places=10)
        np.testing.assert_allclose(np.diff(bounds(mesh), axis=0)[0], [2.002, 3.002, 1.6])
        self.assertTrue(all(n == 2 for n in edge_counts(mesh).values()))
        self.assertEqual(components(mesh), 1)
        self.assertEqual(ray_hits(mesh, (1, 1.5, -1), (0, 0, 1)), [])
        self.assertEqual(ray_hits(mesh, (-1, 1.5, .8), (1, 0, 0)), [.999, 1.001, 2.999, 3.001])

    def test_l_concavity_and_explicit_wrong_shortcuts(self):
        mesh, vertical = self.mesh('l_closed'), self.mesh('l_vertical')
        self.assertAlmostEqual(volume(mesh), 6.4)
        self.assertAlmostEqual(area(mesh), 24.)
        self.assertAlmostEqual(area(vertical), 16.)
        self.assertTrue(all(n == 2 for n in edge_counts(mesh).values()))
        self.assertTrue(all(n == 2 for n in xy_projected_edge_degrees(vertical).values()))
        self.assertEqual(ray_hits(mesh, (1.25, 2., -1), (0, 0, 1)), [])
        self.assertEqual(ray_hits(vertical, (-1, .5, .8), (1, 0, 0)), [1., 3.])
        shortcut, filled = self.mesh('l_convex_shortcut'), self.mesh('l_notch_filled')
        self.assertAlmostEqual(volume(shortcut), 8.)
        self.assertAlmostEqual(volume(filled), 9.6)
        self.assertTrue(all(n == 2 for n in edge_counts(shortcut).values()))
        self.assertEqual(ray_hits(shortcut, (1.25, 2., -1), (0, 0, 1)), [1., 2.6])
        self.assertEqual(ray_hits(shortcut, (1.75, 2.5, -1), (0, 0, 1)), [])
        self.assertEqual(ray_hits(filled, (1.75, 2.5, -1), (0, 0, 1)), [1., 2.6])

    def test_missing_wall_and_wrong_dimensions_cannot_be_hidden(self):
        missing = self.mesh('box_vertical_missing_side')
        self.assertAlmostEqual(area(missing), 11.2)
        degrees = xy_projected_edge_degrees(missing)
        self.assertEqual(sorted(degrees.values()), [1, 1, 2, 2])
        self.assertEqual(ray_hits(missing, (1, 1.5, .8), (1, 0, 0)), [])
        ref = bounds(reference_meshes_v31()['box'])
        expanded, shifted = bounds(self.mesh('box_expanded_030')), bounds(self.mesh('box_shifted_030'))
        np.testing.assert_allclose(expanded, ref + np.array([[-.3]*3, [.3]*3]))
        np.testing.assert_allclose(shifted-ref, [[.3, 0., 0.], [.3, 0., 0.]])

    def test_extra_components_open_strip_and_empty_are_retained(self):
        detached, strip = self.mesh('box_extra_detached_box'), self.mesh('box_extra_open_strip')
        self.assertEqual(components(detached), 2)
        self.assertEqual(components(strip), 2)
        self.assertAlmostEqual(volume(detached), 9.664)
        self.assertAlmostEqual(area(strip), 28.64)
        self.assertEqual(ray_hits(strip, (2.8, 0., .8), (0, 1, 0)), [1.])
        self.assertTrue(any(n == 1 for n in edge_counts(strip).values()))
        self.assertEqual(self.mesh('empty').vertices.shape, (0, 3))
        self.assertEqual(self.mesh('empty').triangles.shape, (0, 3))

    def test_missing_and_duplicate_seed_keep_two_reference_assets(self):
        for name, fixture in association_fixtures_v31().items():
            self.assertEqual(len(fixture['assets']), 2)
            self.assertEqual(len(fixture['observed_meshes']), len(fixture['seeds']))
            matches = []
            for seed in fixture['seeds']:
                point = None if seed is None else np.asarray(seed['observed_seed_xyz'])
                matches.append([] if point is None else [asset['id'] for asset in fixture['assets']
                    if np.all(point >= asset['bounds'][0]) and np.all(point <= asset['bounds'][1])])
            self.assertEqual(matches, fixture['expected']['seed_matches'], name)
        duplicate = association_fixtures_v31()['duplicate_seed']
        self.assertTrue(all(len(mesh.triangles) for mesh in duplicate['observed_meshes']))
        self.assertEqual(duplicate['seeds'][0], duplicate['seeds'][1])
        np.testing.assert_allclose(bounds(duplicate['observed_meshes'][1])[0], [5, 0, 0])


if __name__ == '__main__':
    unittest.main()
