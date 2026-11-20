import unittest
import numpy as np
from model import components, posterior, neutral_nll, select_temperature


class StructuralTemperatureTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([[.0003, .002], [.001, .0005], [.004, .001]])
        self.q = np.array([.2, .5, .3])

    def test_temperature_one_exact_B8(self):
        for shape in ([.5, .5], [.1, .9], [.9, .1]):
            w = np.log(self.q[:, None]) + np.log(np.asarray(shape)[None, :])
            w = w - self.e / (2 * .01**2); w -= w.max()
            expected = np.exp(w); expected /= expected.sum()
            np.testing.assert_array_equal(posterior(self.e, self.q, shape)['joint_probability_mass'], expected)

    def test_conditionals_unchanged_and_shared(self):
        expected = components(self.e, self.q)[0]
        for t in (.25, 2., 128.):
            for shape in ([.5, .5], [.1, .9]):
                row = posterior(self.e, self.q, shape, t)
                np.testing.assert_array_equal(row['conditional_geometry_mass'], expected)
                np.testing.assert_allclose(row['joint_probability_mass'] / row['shape_probability_mass'], expected)

    def test_G_class_order_cannot_flip(self):
        for t in (.25, 1., 8., 128.):
            row = posterior(self.e, self.q, temperature=t)
            self.assertEqual(np.sign(row['shape_probability_mass'][1] - .5),
                             np.sign(row['neutral_log_evidence_difference']))

    def test_prior_odds_applied_after_scaling(self):
        row = posterior(self.e, self.q, (.1, .9), 16.)
        odds = row['shape_probability_mass'][1] / row['shape_probability_mass'][0]
        self.assertAlmostEqual(np.log(odds), row['neutral_log_evidence_difference']/16 + np.log(9))

    def test_extreme_evidence_stable(self):
        row = posterior([[0., 100.], [1., 101.]], [.5, .5], temperature=128.)
        self.assertTrue(np.isfinite(row['joint_probability_mass']).all())
        self.assertEqual(neutral_nll(100000., 0, 1.), 100000.)
        self.assertAlmostEqual(row['joint_probability_mass'].sum(), 1.)

    def test_zero_prior_particle_and_duplicate_splitting(self):
        original = posterior(self.e, self.q, temperature=8.)
        split = posterior(np.repeat(self.e, 2, axis=0), np.repeat(self.q/2, 2), temperature=8.)
        np.testing.assert_allclose(split['joint_probability_mass'].reshape(3,2,2).sum(axis=1), original['joint_probability_mass'])
        zero = posterior(np.vstack([self.e, [0., 0.]]), np.r_[self.q,0.], temperature=8.)
        np.testing.assert_allclose(zero['joint_probability_mass'][:-1], original['joint_probability_mass'])

    def test_calibration_isolation_and_equal_object_weighting(self):
        rows = [dict(split='calibration',object_id=str(o),condition=c,log_evidence_difference=4.,target=o)
                for o in (0,1) for c in ('front','front_side','around','occluded')]
        r=select_temperature(rows);self.assertEqual(r['temperature'],128.)
        with self.assertRaises(ValueError):select_temperature([dict(rows[0],split='test')])
        with self.assertRaises(ValueError):select_temperature(rows+[rows[0]])
        with self.assertRaises(ValueError):select_temperature(rows[:-1])

    def test_exact_tie_prefers_one(self):
        rows = [dict(split='calibration',object_id='one',condition=c,log_evidence_difference=0.,target=0)
                for c in ('front','front_side','around','occluded')]
        self.assertEqual(select_temperature(rows)['temperature'],1.)

    def test_invalid_input(self):
        for t in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):posterior(self.e,self.q,temperature=t)
        with self.assertRaises(ValueError):posterior(self.e,self.q,(0.,1.))
        with self.assertRaises(ValueError):posterior(self.e,[0.,0.,0.])

if __name__ == '__main__': unittest.main()
