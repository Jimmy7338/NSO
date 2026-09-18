"""Independent shape fixtures for V23; no rollout, labels, or observation count.

The intentionally open corner has sufficient orthographic outlines. Its high
score is evidence for the declared outline task, never watertight reconstruction.
"""
import copy
import json
import unittest

import numpy as np
import open3d as o3d

from utils.facility_outline_v23 import OutlineEvaluatorV23


def box(size=(1., 1., 1.), origin=(0., 0., 0.)):
    return o3d.geometry.TriangleMesh.create_box(*size).translate(origin)


def rectangle(vertices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, float))
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    return mesh


def front_panel():
    return rectangle([(0., 0., 0.), (1., 0., 0.), (1., 0., 1.), (0., 0., 1.)])


def open_corner():
    """Three mutually perpendicular square faces, not a closed solid."""
    mesh = front_panel()
    mesh += rectangle([(0., 0., 0.), (0., 1., 0.), (0., 1., 1.), (0., 0., 1.)])
    mesh += rectangle([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)])
    mesh.remove_duplicated_vertices()
    return mesh


def edge_microcubes():
    """Disconnected 2 mm cubes every 25 mm along the twelve unit-cube edges.

    A distance-only boundary score confuses these sparse fragments with an
    enclosure, although almost none of each projected square is reconstructed.
    """
    centers = set()
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        for a in (0., 1.):
            for b in (0., 1.):
                for value in np.linspace(0., 1., 41):
                    p = [0., 0., 0.]
                    p[axis], p[other[0]], p[other[1]] = float(value), a, b
                    centers.add(tuple(p))
    mesh = o3d.geometry.TriangleMesh()
    for center in sorted(centers):
        mesh += box(size=(.002, .002, .002), origin=tuple(x-.001 for x in center))
    return mesh


class SixOutlineFixture:
    """World-like evaluator input, deliberately without sensor/class truth APIs."""
    def __init__(self, first=None):
        self.meshes = [box(origin=(3.*i, 0., 0.)) for i in range(6)]
        if first is not None:
            self.meshes[0] = copy.deepcopy(first)
        self.objects = [dict(id=i, evaluation_bounds=[[3.*i-.45, -.45, -.45],
                                                     [3.*i+2., 1.8, 1.8]]) for i in range(6)]
        self.mesh = self.prediction()

    def instance_mesh(self, identifier):
        return copy.deepcopy(self.meshes[identifier])

    def prediction(self, first=None, omitted=()):
        mesh = o3d.geometry.TriangleMesh()
        for identifier, original in enumerate(self.meshes):
            if identifier not in omitted:
                mesh += copy.deepcopy(first if identifier == 0 and first is not None else original)
        return mesh


def evaluator(world):
    return OutlineEvaluatorV23(assets=[dict(id=item['id'],
        mesh=world.instance_mesh(item['id']), bounds=item['evaluation_bounds'])
        for item in world.objects], boundary_spacing_m=.01,
        thresholds=(.02, .05, .10), completion_tolerance_m=.05)


class FacilityOutlineV23Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = SixOutlineFixture()
        cls.evaluator = evaluator(cls.world)

    def score_first(self, first, **kwargs):
        return self.evaluator.evaluate(self.world.prediction(first=first), .9,
            returned=True, **kwargs)

    def test_correct_shell_scores_high_and_three_projections_are_not_duplicated(self):
        vertices_before = np.asarray(self.world.mesh.vertices).copy()
        result = self.evaluator.evaluate(self.world.mesh, .9, returned=True,
            paid_actions=80, budget=100)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['missing_asset_count'], 0)
        self.assertEqual(result['completion_fraction'], 1.)
        self.assertAlmostEqual(result['05cm']['outline_macro_f1'], 1., places=10)
        self.assertAlmostEqual(result['05cm']['outline_macro_quality'], 1., places=10)
        self.assertAlmostEqual(result['05cm']['joint_outline'], .9, places=10)
        for instance in result['instances']:
            self.assertEqual(set(instance['projections']), {'xy', 'xz', 'yz'})
            self.assertTrue(instance['completed'])
            for projection in instance['projections'].values():
                self.assertAlmostEqual(projection['05cm']['f1'], 1., places=10)
                self.assertAlmostEqual(projection['iou'], 1., places=10)
                self.assertLessEqual(projection['hausdorff_upper_m'], .01)
        np.testing.assert_array_equal(vertices_before, np.asarray(self.world.mesh.vertices))
        json.dumps(result, allow_nan=False)

    def test_internal_geometry_and_duplicate_triangles_do_not_create_reward(self):
        plain = self.score_first(box())
        enriched = box() + box(size=(.6, .6, .6), origin=(.2, .2, .2))
        enriched += rectangle([(.1, .1, .5), (.9, .1, .5), (.9, .9, .5), (.1, .9, .5)])
        enriched += box()  # More geometry samples are not more evidence.
        actual = self.score_first(enriched)
        self.assertAlmostEqual(actual['05cm']['outline_macro_f1'], plain['05cm']['outline_macro_f1'], places=12)
        self.assertAlmostEqual(actual['05cm']['outline_macro_quality'], plain['05cm']['outline_macro_quality'], places=12)
        self.assertEqual(actual['completion_fraction'], plain['completion_fraction'])
        for name in ('xy', 'xz', 'yz'):
            before = plain['instances'][0]['projections'][name]
            after = actual['instances'][0]['projections'][name]
            self.assertAlmostEqual(after['iou'], before['iou'], places=12)
            self.assertAlmostEqual(after['05cm']['f1'], before['05cm']['f1'], places=12)

    def test_front_panel_cannot_supply_missing_thickness_projections(self):
        instance = self.score_first(front_panel())['instances'][0]
        self.assertFalse(instance['completed'])
        self.assertAlmostEqual(instance['projections']['xz']['05cm']['f1'], 1., places=10)
        for name in ('xy', 'yz'):
            self.assertEqual(instance['projections'][name]['05cm']['f1'], 0.)
            self.assertEqual(instance['projections'][name]['iou'], 0.)
        self.assertAlmostEqual(instance['05cm']['outline_f1'], 1/3, places=10)
        self.assertAlmostEqual(instance['dimensions']['absolute_size_error_xyz_m'][1], 1., places=10)

    def test_wrong_size_is_not_normalized_to_the_truth_window(self):
        instance = self.score_first(box(size=(1.35, 1., 1.)))['instances'][0]
        self.assertLess(instance['05cm']['outline_f1'], 1.)
        self.assertFalse(instance['completed'])
        self.assertAlmostEqual(instance['dimensions']['absolute_size_error_xyz_m'][0], .35, places=10)
        self.assertGreaterEqual(max(p['hausdorff_lower_m'] for p in instance['projections'].values()), .34)

    def test_translation_is_not_removed_by_per_object_recentering(self):
        instance = self.score_first(box(origin=(.25, .20, .15)))['instances'][0]
        self.assertLess(instance['05cm']['outline_f1'], 1.)
        self.assertFalse(instance['completed'])
        self.assertAlmostEqual(instance['dimensions']['center_error_inf_m'], .25, places=10)
        np.testing.assert_allclose(instance['dimensions']['absolute_size_error_xyz_m'], 0., atol=1e-12)

    def test_missing_external_protrusion_changes_outline_and_worst_error(self):
        protruding = box() + box(size=(.35, .32, .40), origin=(1., .34, .30))
        world = SixOutlineFixture(first=protruding)
        metric = evaluator(world)
        correct = metric.evaluate(world.mesh, .9, returned=True)['instances'][0]
        omitted = metric.evaluate(world.prediction(first=box()), .9, returned=True)['instances'][0]
        self.assertAlmostEqual(correct['05cm']['outline_f1'], 1., places=10)
        self.assertTrue(correct['completed'])
        self.assertLess(omitted['05cm']['outline_f1'], correct['05cm']['outline_f1'])
        self.assertFalse(omitted['completed'])
        for name in ('xy', 'xz'):
            self.assertLess(omitted['projections'][name]['05cm']['recall'], 1.)
            self.assertGreaterEqual(omitted['projections'][name]['hausdorff_lower_m'], .34)
            self.assertLessEqual(omitted['projections'][name]['hausdorff_lower_m'], .35+1e-9)
            self.assertGreaterEqual(omitted['projections'][name]['hausdorff_upper_m'], .35-1e-9)
            self.assertLessEqual(omitted['projections'][name]['hausdorff_upper_m']-
                                 omitted['projections'][name]['hausdorff_lower_m'], .01)
        self.assertAlmostEqual(omitted['projections']['yz']['05cm']['f1'], 1., places=10)
        self.assertAlmostEqual(omitted['dimensions']['absolute_size_error_xyz_m'][0], .35, places=10)

    def test_reference_window_cannot_silently_cut_off_a_real_protrusion(self):
        truth = box() + box(size=(.35, .32, .40), origin=(1., .34, .30))
        # The base fits; the real attachment extends beyond x=1.2. Cropping
        # the reference here would change the task and reward its omission.
        with self.assertRaisesRegex(ValueError, 'window.*truncates reference'):
            OutlineEvaluatorV23(assets=[dict(id=0, mesh=truth,
                bounds=[[-.1, -.1, -.1], [1.2, 1.1, 1.1]])])

    def test_external_disconnected_false_geometry_is_not_discarded_as_internal(self):
        extra = box() + box(size=(.2, .2, .2), origin=(1.5, .3, .3))
        instance = self.score_first(extra)['instances'][0]
        self.assertLess(instance['05cm']['outline_f1'], 1.)
        self.assertLess(instance['projections']['xy']['05cm']['precision'], 1.)
        self.assertLess(instance['projections']['xz']['05cm']['precision'], 1.)
        self.assertFalse(instance['completed'])

    def test_exterior_notch_is_not_convexified(self):
        # The XY outline is an L with area .4+.4-.16=.64 square metres.
        # A convex hull would silently replace the exposed notch with a chord.
        l_shape = box(size=(1., .4, 1.)) + box(size=(.4, 1., 1.))
        world = SixOutlineFixture(first=l_shape)
        instance = evaluator(world).evaluate(world.prediction(first=box()), .9,
            returned=True)['instances'][0]
        self.assertAlmostEqual(instance['projections']['xy']['iou'], .64, places=10)
        self.assertLess(instance['projections']['xy']['05cm']['f1'], 1.)
        self.assertFalse(instance['completed'])

    def test_enclosed_projected_hole_is_task_excluded(self):
        frame = box(size=(1., .2, 1.))
        frame += box(size=(1., .2, 1.), origin=(0., .8, 0.))
        frame += box(size=(.2, .6, 1.), origin=(0., .2, 0.))
        frame += box(size=(.2, .6, 1.), origin=(.8, .2, 0.))
        world = SixOutlineFixture(first=frame)
        instance = evaluator(world).evaluate(world.prediction(first=box()), .9,
            returned=True)['instances'][0]
        self.assertAlmostEqual(instance['05cm']['outline_f1'], 1., places=10)
        self.assertAlmostEqual(instance['projections']['xy']['iou'], 1., places=10)
        self.assertTrue(instance['completed'])

    def test_empty_and_missing_assets_are_zero_with_equal_instance_weights(self):
        empty = self.evaluator.evaluate(o3d.geometry.TriangleMesh(), .9, returned=True)
        self.assertEqual(empty['missing_asset_count'], 6)
        self.assertEqual(empty['05cm']['outline_macro_f1'], 0.)
        self.assertEqual(empty['05cm']['outline_macro_quality'], 0.)
        self.assertEqual(empty['05cm']['joint_outline'], 0.)
        self.assertEqual(empty['completion_fraction'], 0.)
        self.assertEqual(len(empty['instances']), 6)
        json.dumps(empty, allow_nan=False)
        # Omit a much larger object: it still contributes exactly one sixth.
        large_world = SixOutlineFixture(first=box(size=(1.7, 1.5, 1.5)))
        partial = evaluator(large_world).evaluate(large_world.prediction(omitted=(0,)), .9, returned=True)
        self.assertEqual(partial['missing_asset_count'], 1)
        self.assertAlmostEqual(partial['05cm']['outline_macro_f1'], 5/6, places=10)
        self.assertAlmostEqual(partial['05cm']['outline_macro_quality'], 5/6, places=10)
        self.assertAlmostEqual(partial['05cm']['joint_outline'], .9*5/6, places=10)
        self.assertEqual(partial['instances'][0]['05cm']['outline_f1'], 0.)

    def test_open_mesh_with_sufficient_outlines_is_valid_for_this_task(self):
        mesh = open_corner()
        self.assertFalse(mesh.is_watertight())
        instance = self.score_first(mesh)['instances'][0]
        self.assertTrue(instance['completed'])
        self.assertAlmostEqual(instance['05cm']['outline_f1'], 1., places=10)
        self.assertAlmostEqual(instance['05cm']['outline_quality'], 1., places=10)
        for projection in instance['projections'].values():
            self.assertAlmostEqual(projection['iou'], 1., places=10)
        # This fixture intentionally establishes the scope limit: full 3-D
        # surface completeness cannot be inferred from three outlines.

    def test_disconnected_edge_speckles_cannot_pass_as_an_enclosure(self):
        fragments = edge_microcubes()
        # Use a one-object unit task to isolate J; five correct background
        # objects must not obscure this specific evaluator failure.
        metric = OutlineEvaluatorV23(assets=[dict(id=0, mesh=box(),
            bounds=[[-.45, -.45, -.45], [2., 1.8, 1.8]])], boundary_spacing_m=.01)
        result = metric.evaluate(fragments, .9, returned=True, paid_actions=80, budget=100)
        instance = result['instances'][0]
        self.assertGreater(instance['05cm']['outline_f1'], .99)
        self.assertLess(instance['05cm']['outline_quality'], .05)
        self.assertLess(result['05cm']['outline_macro_quality'], .05)
        self.assertLess(result['05cm']['joint_outline'], .05)
        self.assertFalse(instance['completed'])
        self.assertEqual(result['completion_fraction'], 0.)
        for projection in instance['projections'].values():
            self.assertGreater(projection['05cm']['f1'], .99)
            self.assertLess(projection['iou'], .001)
            self.assertLess(projection['hausdorff_upper_m'], .02)
        self.assertLess(max(instance['dimensions']['absolute_size_error_xyz_m']), .003)
        json.dumps(result, allow_nan=False)
        # In the six-object task, the other five retain their normal weight.
        mixed = self.score_first(fragments, paid_actions=80, budget=100)
        self.assertAlmostEqual(mixed['05cm']['outline_macro_quality'],
            (5+instance['05cm']['outline_quality'])/6, places=10)
        self.assertAlmostEqual(mixed['05cm']['joint_outline'],
            .9*(5+instance['05cm']['outline_quality'])/6, places=10)

    def test_good_outline_does_not_override_task_failures_or_budget(self):
        unverified = self.score_first(box())
        self.assertAlmostEqual(unverified['05cm']['outline_macro_f1'], 1., places=10)
        self.assertFalse(unverified['budget_verified'])
        self.assertFalse(unverified['eligible'])
        for extra in (dict(failed=True), dict(collisions=1), dict(paid_actions=101, budget=100)):
            result = self.score_first(box(), **extra)
            self.assertAlmostEqual(result['05cm']['outline_macro_f1'], 1., places=10)
            self.assertFalse(result['eligible'])
        self.assertFalse(self.evaluator.evaluate(self.world.mesh, .79, returned=True)['eligible'])
        self.assertFalse(self.evaluator.evaluate(self.world.mesh, .9, returned=False)['eligible'])


if __name__ == '__main__':
    unittest.main()
