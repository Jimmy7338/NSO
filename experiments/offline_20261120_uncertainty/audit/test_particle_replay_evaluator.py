import unittest
import numpy as np
from evaluate_particle_replay import point_tracks,ARMS

class ParticleAttributionTests(unittest.TestCase):
    def arrays(self):
        return dict(G_points=np.zeros((5,3)),G_object_ids=np.array([0,0,0,1,1]),G_support_tracks=np.array([12,12,12,99,99]),
            G_hypothesis_ids=np.array([0,7,8,10,11]),G_particle_ids=np.array([0,3,4,0,0]),G_shape_ids=np.array([0,1,0,0,1]))
    def test_many_particles_do_not_become_extra_objects(self):
        tracks=point_tracks(self.arrays(),'G',[dict(support_track=12),dict(support_track=99)])
        np.testing.assert_array_equal(tracks,[12,12,12,99,99])
    def test_inconsistent_declared_track_is_rejected(self):
        a=self.arrays();a['G_support_tracks'][2]=99
        with self.assertRaises(AssertionError):point_tracks(a,'G',[dict(support_track=12),dict(support_track=99)])
    def test_empty_prediction_basis_keeps_explicit_empty_tracks(self):
        a={k:v[:0]for k,v in self.arrays().items()}
        self.assertEqual(len(point_tracks(a,'G',[])),0)
    def test_declared_five_arms_preserve_matched_structure_modes(self):
        self.assertEqual(ARMS['G_fixed'][1],ARMS['S_fixed'][1]);self.assertEqual(ARMS['G_uncertainty'][1],ARMS['S_uncertainty'][1])
        self.assertEqual(ARMS['S_mean_collapse'],('S','mean_collapse'))
if __name__=='__main__':unittest.main(verbosity=2)
