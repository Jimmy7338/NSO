"""Independent analytic sensor counterexamples for the V24 geometry backend.

The renderer knows the fixture rectangles; the backend receives only paid depth,
calibration, pose and, where stated, a seed copied from an actual hit. No fixture
bounds, surface identity, class or hidden dimension enters the backend.
"""
import inspect
import unittest

import numpy as np

from nso.observed_shape_v24 import ObservedShapeBackendV24


def rectangle(center, half_u, half_v, identity):
    return tuple(np.asarray(v, float) for v in (center, half_u, half_v)) + (identity,)


def cuboid(lo=(-.5, 1.6, 0.), hi=(.5, 2.6, 1.2), identity=10):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    center, half = (lo + hi) / 2., (hi - lo) / 2.
    faces = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for sign in (-1., 1.):
            c = center.copy(); c[axis] += sign * half[axis]
            u = np.zeros(3); u[others[0]] = half[others[0]]
            v = np.zeros(3); v[others[1]] = half[others[1]]
            faces.append(rectangle(c, u, v, identity))
    return faces


def surroundings():
    # Positive finite background depths distinguish real silhouettes from
    # missing pixels. These scene rectangles are never passed to the backend.
    return [rectangle((0., 2., 0.), (5., 0., 0.), (0., 5., 0.), 1),
            rectangle((0., 3.8, 1.6), (5., 0., 0.), (0., 0., 1.6), 2),
            rectangle((2.5, 2., 1.6), (0., 5., 0.), (0., 0., 1.6), 2)]


def camera(origin=(-1.5, 0., 1.4), target=(0., 2., .6), focal=70.,
           height=72, width=96):
    forward = np.asarray(target, float) - np.asarray(origin, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = origin
    intrinsic = np.array([[focal, 0., (width-1)/2.],
                          [0., focal, (height-1)/2.], [0., 0., 1.]])
    return intrinsic, pose, height, width


def render(rectangles, view=None):
    """Analytic finite-plane intersections; optical depth, not ray distance."""
    intrinsic, pose, height, width = camera() if view is None else view
    v, u = np.mgrid[:height, :width]
    local = np.stack(((u-intrinsic[0, 2])/intrinsic[0, 0],
                      (v-intrinsic[1, 2])/intrinsic[1, 1], np.ones_like(u)), axis=-1)
    directions = local @ pose[:3, :3].T
    depth = np.full((height, width), np.inf)
    identities = np.full((height, width), -1, int)
    for center, half_u, half_v, identity in rectangles:
        normal = np.cross(half_u, half_v)
        denominator = directions @ normal
        with np.errstate(divide='ignore', invalid='ignore'):
            t = ((center-pose[:3, 3]) @ normal)/denominator
            delta = pose[:3, 3] + directions*t[..., None] - center
            # General plane basis, including deliberately tilted ground.
            gram = np.array([[half_u @ half_u, half_u @ half_v],
                             [half_u @ half_v, half_v @ half_v]])
            coords = np.stack((delta @ half_u, delta @ half_v), axis=-1) @ np.linalg.inv(gram)
        valid = (np.abs(denominator) > 1e-10) & (t > .08) & (t < 6.)
        valid &= np.all(np.abs(coords) <= 1.+1e-10, axis=-1) & (t < depth)
        depth[valid] = t[valid]; identities[valid] = identity
    depth[~np.isfinite(depth)] = 0.
    points = pose[:3, 3] + directions*depth[..., None]
    return dict(depth=depth.astype(np.float32), intrinsic=intrinsic, pose=pose,
                identities=identities, points=points)


def observed_seed(frame, target=(0., 1.6, .6), identity=10):
    points = frame['points'][frame['identities'] == identity]
    if len(points) == 0:
        raise AssertionError('Fixture has no measured seed surface')
    return points[np.argmin(np.linalg.norm(points-np.asarray(target), axis=1))].copy()


def feed(backend, frame, identifier, seed=True):
    return backend.observe(frame['depth'], frame['intrinsic'], frame['pose'],
                           observation_id=identifier,
                           observed_seed_xyz=observed_seed(frame) if seed else None)


def mesh_vertices(mesh):
    return np.asarray(mesh.vertices) if mesh is not None else np.empty((0, 3))


def nearest_distances(source, target):
    from scipy.spatial import cKDTree
    if len(target) == 0:
        return np.full(len(source), np.inf)
    return cKDTree(np.asarray(target)).query(np.asarray(source))[0]


class ObservedShapeV24Tests(unittest.TestCase):
    def supported_box(self):
        frames = [render(surroundings()+cuboid()),
                  render(surroundings()+cuboid(),
                         camera(origin=(-1.6, 1.9, 1.0), target=(-.5, 2.1, .6)))]
        backend = ObservedShapeBackendV24()
        for index, frame in enumerate(frames):
            feed(backend, frame, index)
        return backend, frames

    def test_ground_is_fitted_from_nonzero_tilt_and_offset(self):
        plane = rectangle((0., 2., .25), (5., 0., .5), (0., 5., -.3), 1)
        frame = render([plane], camera(origin=(0., -.6, 1.4), target=(0., 2., .25)))
        backend = ObservedShapeBackendV24()
        feed(backend, frame, 0, seed=False)
        result = backend.snapshot()
        estimated = result['ground_plane']
        self.assertIsNotNone(estimated)
        expected = np.cross(plane[1], plane[2]); expected /= np.linalg.norm(expected)
        normal = np.asarray(estimated['normal'])
        self.assertGreater(float(normal @ expected), np.cos(np.deg2rad(2.)))
        self.assertAlmostEqual(float(estimated['offset']), -float(expected @ plane[0]), delta=.02)
        points = np.asarray(result['ground_points_xyz'])
        self.assertGreater(len(points), 100)
        self.assertLess(float(np.quantile(np.abs(points @ expected-expected @ plane[0]), .95)), .02)
        self.assertFalse(result['completion']['accepted'])
        self.assertEqual(len(mesh_vertices(result['inferred_mesh'])), 0)

    def test_near_ground_real_attachment_survives_ground_separation(self):
        # A connected protrusion only 6--12 cm above the floor is real surface,
        # even though an absolute-z .15 m obstacle cutoff would discard it.
        scene = surroundings()+cuboid()+cuboid((-.85, 1.8, .06), (-.46, 2.3, .12), 20)
        frame = render(scene, camera(origin=(-1.7, .8, .75), target=(-.3, 2., .3)))
        attachment = frame['points'][frame['identities'] == 20]
        self.assertGreater(len(attachment), 20)
        backend = ObservedShapeBackendV24(); feed(backend, frame, 0)
        result = backend.snapshot()
        retained = nearest_distances(attachment, result['cleaned_points_xyz']) < .04
        self.assertGreater(float(np.mean(retained)), .65)
        ground = np.asarray(result['ground_points_xyz'])
        self.assertLess(float(np.quantile(np.abs(ground[:, 2]), .95)), .025)

    def test_large_platform_cannot_replace_a_supported_lower_ground_plane(self):
        # Pixel majority alone does not identify ground: the visible top of a
        # wide machine can have more support than the genuinely lower floor.
        platform = rectangle((0., 2., .65), (1., 0., 0.), (0., 1., 0.), 20)
        frame = render(surroundings()+[platform],
                       camera(origin=(0., .2, 1.5), target=(0., 2., .55)))
        floor_count = np.count_nonzero(frame['identities'] == 1)
        platform_count = np.count_nonzero(frame['identities'] == 20)
        self.assertGreater(floor_count, 500)
        self.assertGreater(platform_count, floor_count)
        backend = ObservedShapeBackendV24(); feed(backend, frame, 0, seed=False)
        result = backend.snapshot()
        self.assertIsNotNone(result['ground_plane'])
        self.assertAlmostEqual(float(result['ground_plane']['offset']), 0., delta=.025)
        ground = np.asarray(result['ground_points_xyz'])
        self.assertLess(float(np.quantile(np.abs(ground[:, 2]), .95)), .025)

    def test_single_front_panel_cannot_invent_hidden_depth(self):
        panel = rectangle((0., 1.6, .6), (.5, 0., 0.), (0., 0., .6), 10)
        frame = render(surroundings()+[panel], camera(origin=(0., 0., .8), target=(0., 1.6, .6)))
        backend = ObservedShapeBackendV24(); feed(backend, frame, 0)
        result = backend.snapshot()
        self.assertFalse(result['completion']['accepted'])
        self.assertEqual(len(mesh_vertices(result['inferred_mesh'])), 0)
        self.assertGreater(len(result['cleaned_points_xyz']), 100)

    def test_missing_border_pixels_do_not_certify_hidden_extent(self):
        # Preserve the same two visible planes, but remove rays around every
        # observed object boundary. Invalid depth is not free background.
        from scipy.ndimage import binary_dilation, binary_erosion
        frame = render(surroundings()+cuboid())
        object_mask = frame['identities'] == 10
        inner = binary_erosion(object_mask, iterations=3)
        border = binary_dilation(object_mask, iterations=3) & ~inner
        frame['depth'][border] = 0.
        frame['identities'][border] = -1
        self.assertGreater(np.count_nonzero(inner), 100)
        backend = ObservedShapeBackendV24(); feed(backend, frame, 0)
        result = backend.snapshot()
        self.assertFalse(result['completion']['accepted'])
        self.assertEqual(len(mesh_vertices(result['inferred_mesh'])), 0)

    def test_two_supported_faces_allow_explicit_regular_shell_inference(self):
        backend, frames = self.supported_box()
        result = backend.snapshot()
        self.assertTrue(result['completion']['accepted'], result['completion'])
        inferred = mesh_vertices(result['inferred_mesh'])
        self.assertGreater(len(inferred), 0)
        # Expected dimensions belong only to the assertion. They were not
        # supplied as bounds or read by the geometry backend.
        self.assertTrue(np.allclose(inferred.min(axis=0), [-.5, 1.6, 0.], atol=.09))
        self.assertTrue(np.allclose(inferred.max(axis=0), [.5, 2.6, 1.2], atol=.09))
        actual_hits = np.concatenate([f['points'][f['depth'] > 0] for f in frames])
        measured = np.asarray(result['measured_points_xyz'])
        self.assertLess(float(nearest_distances(measured, actual_hits).max()), .005)
        self.assertEqual(int(result['completion']['free_ray_conflicts']), 0)

    def test_new_free_rays_revoke_previously_supported_shell(self):
        backend, _ = self.supported_box()
        self.assertTrue(backend.snapshot()['completion']['accepted'])
        # The prior two views also fit an open L-shaped pair of panels. This
        # new paid side view sees through the inferred right face, providing
        # a real contradiction rather than merely another absent observation.
        front = rectangle((0., 1.6, .6), (.5, 0., 0.), (0., 0., .6), 10)
        left = rectangle((-.5, 2.1, .6), (0., .5, 0.), (0., 0., .6), 10)
        reveal = render(surroundings()+[front, left],
                        camera(origin=(1.7, 2.2, .8), target=(-.5, 2.2, .6)))
        feed(backend, reveal, 2)
        result = backend.snapshot()
        self.assertFalse(result['completion']['accepted'])
        self.assertGreater(int(result['completion']['free_ray_conflicts']), 0)
        self.assertEqual(len(mesh_vertices(result['inferred_mesh'])), 0)

    def test_same_paid_frame_is_idempotent_but_reused_id_cannot_change_depth(self):
        frame = render(surroundings()+cuboid())
        backend = ObservedShapeBackendV24(); feed(backend, frame, 0)
        before = backend.snapshot(); feed(backend, frame, 0); after = backend.snapshot()
        np.testing.assert_array_equal(before['measured_points_xyz'], after['measured_points_xyz'])
        self.assertEqual(before['completion'], after['completion'])
        changed = {**frame, 'depth': frame['depth'].copy()}
        valid = np.argwhere(changed['depth'] > 0)[0]
        changed['depth'][tuple(valid)] += .1
        with self.assertRaises(ValueError):
            feed(backend, changed, 0)

    def test_raw_triangle_centroid_cannot_certify_unsupported_vertices(self):
        import open3d as o3d
        frame = render(surroundings()+cuboid())
        backend = ObservedShapeBackendV24(); feed(backend, frame, 0)
        # Both triangle centroids lie on the observed front plane. Only the
        # first has supported vertices; the second invents metres of outline.
        vertices = np.asarray([[-.02, 1.6, .55], [.02, 1.6, .55], [0., 1.6, .59],
                               [-3., 1.6, -1.], [3., 1.6, -1.], [0., 1.6, 3.8]])
        raw = o3d.geometry.TriangleMesh()
        raw.vertices = o3d.utility.Vector3dVector(vertices.copy())
        raw.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
        result = backend.snapshot(raw_mesh=raw)
        kept = mesh_vertices(result['observed_mesh'])
        self.assertGreater(len(kept), 0, 'Cleaning must retain the supported surface')
        self.assertLess(float(nearest_distances(kept, result['cleaned_points_xyz']).max()), .15)
        np.testing.assert_array_equal(np.asarray(raw.vertices), vertices)
        np.testing.assert_array_equal(np.asarray(raw.triangles), [[0, 1, 2], [3, 4, 5]])

    def test_unobserved_seed_and_semantic_oracle_inputs_are_rejected(self):
        frame = render(surroundings()+cuboid())
        backend = ObservedShapeBackendV24()
        with self.assertRaises(ValueError):
            backend.observe(frame['depth'], frame['intrinsic'], frame['pose'],
                            observation_id=0, observed_seed_xyz=[30., 30., 30.])
        signature = inspect.signature(backend.observe)
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()))
        for forbidden in ('semantic', 'category', 'evaluation_bounds', 'truth_mesh', 'dimensions'):
            self.assertNotIn(forbidden, signature.parameters)
            with self.assertRaises(TypeError):
                backend.observe(frame['depth'], frame['intrinsic'], frame['pose'],
                                observation_id=1, **{forbidden: np.zeros((2, 2))})


if __name__ == '__main__':
    unittest.main()
