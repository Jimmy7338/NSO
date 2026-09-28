"""Metric/threshold equality tests, not approximate-neighbor accuracy tests."""
import unittest

import numpy as np

from nso.exact_support_distance import exact_nearest_distances
from nso.observed_instances_v41 import _distances


class ExactSupportDistanceTests(unittest.TestCase):
    def assert_exact(self,points,support):
        expected=_distances(points,support)
        actual=exact_nearest_distances(points,support)
        np.testing.assert_array_equal(actual,expected)
        for threshold in (.05,.12,.4,.6):
            np.testing.assert_array_equal(actual<=threshold,expected<=threshold)
            np.testing.assert_array_equal(actual>threshold,expected>threshold)

    def test_random_metre_geometry_matches_every_distance_bit(self):
        random=np.random.default_rng(472023)
        self.assert_exact(random.normal(size=(513,3))*4.,random.normal(size=(1025,3))*4.)

    def test_threshold_neighbors_on_both_sides(self):
        points=[]
        for threshold in (.05,.12,.4,.6):
            for value in (np.nextafter(threshold,0.),threshold,np.nextafter(threshold,np.inf)):
                points.append([value,0.,0.])
        self.assert_exact(np.asarray(points),np.array([[0.,0.,0.],[10.,10.,10.]]))

    def test_many_exact_and_near_ties(self):
        support=[]
        for axis in range(3):
            for sign in (-1.,1.):
                for offset in (-1.,0.,1.):
                    point=np.zeros(3)
                    point[axis]=sign*np.nextafter(.4,0. if offset<0 else np.inf) if offset else sign*.4
                    support.append(point)
        support=np.asarray(support)
        self.assert_exact(np.array([[0.,0.,0.],[1e-16,-2e-16,3e-16]]),support)
        self.assert_exact(np.array([[0.,0.,0.],[1e-16,-2e-16,3e-16]]),support[::-1])

    def test_duplicate_support_and_already_measured_points(self):
        support=np.array([[0.,0.,0.],[.2,.1,.3],[.2,.1,.3],[.4,.2,.6]])
        self.assert_exact(support.copy(),support)

    def test_empty_and_single_support(self):
        points=np.array([[0.,0.,0.],[1.,2.,3.]])
        self.assert_exact(points,np.empty((0,3)))
        self.assert_exact(np.empty((0,3)),points)
        self.assert_exact(points,np.array([[.1,.2,.3]]))

    def test_float32_preserves_original_metric_rounding(self):
        random=np.random.default_rng(711)
        self.assert_exact(random.normal(size=(27,3)).astype(np.float32),
                          random.normal(size=(41,3)).astype(np.float32))

    def test_large_offset_and_extreme_metric_regimes(self):
        self.assert_exact(np.array([[1e12,1e12,1e12]]),
                          np.array([[1e12+.05,1e12,1e12],[1e12,1e12-.05,1e12]]))
        self.assert_exact(np.array([[0.,0.,0.],[1e-160,0.,0.]]),
                          np.array([[2e-160,0.,0.],[3e-160,0.,0.]]))
        with np.errstate(over='ignore',invalid='ignore'):
            self.assert_exact(np.array([[1e155,0.,0.]]),
                              np.array([[0.,0.,0.],[-1e155,0.,0.]]))


if __name__=='__main__':
    unittest.main()
