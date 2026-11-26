import copy,unittest
import numpy as np
from evaluate import independent_joint,verify_arrays,winner,rho


def fixture():
 obj=dict(support_track=1,particles=[dict(depth_m=.4),dict(depth_m=.8)],prior_mass=[.4,.6],
          energy_m2=[[.001,.0003],[.0004,.001]],shape_prior=[.1,.9],conditional_geometry=dict(width=.9),
          fit_input_points_sha256='p',fit_input_normals_sha256='n',fit_input_labels_sha256='l')
 prior=copy.deepcopy(obj);prior['joint_probability_mass']=independent_joint(prior,1.)[0].tolist()
 newobj=copy.deepcopy(obj);mass=independent_joint(newobj,16.)[0]
 newobj.update(joint_probability_mass=mass.tolist(),shape_probability_mass=mass.sum(axis=0).tolist())
 areas=np.array([.2,.3,.4,.5]);history=np.array([1,1,0,1],bool)
 ids=np.arange(4);particle=ids//2;shape=ids%2
 base_old=areas*np.asarray(prior['joint_probability_mass'])[particle,shape]
 base=areas*mass[particle,shape];oldweight=base_old*history;weight=base*history
 visibility=np.array([[1,0,1,1],[0,1,0,1]],float)
 old=dict(G_points=np.arange(12).reshape(4,3),G_history_keep=history,G_object_ids=np.zeros(4,int),
          G_particle_ids=particle,G_hypothesis_ids=ids,G_base_weights=base_old,G_weights=oldweight,
          G_route_union_gain_matrix=visibility*oldweight,candidate_ids=np.array([0,1]))
 new={k.removeprefix('G_'):v.copy()for k,v in old.items()};new.update(base_weights=base,weights=weight,route_union_gain_matrix=visibility*weight)
 return new,old,newobj,prior


class AuditTests(unittest.TestCase):
 def test_independent_mass_area_and_visibility_check(self):
  n,o,a,b=fixture();r=verify_arrays(n,o,'G',[a],[b],16.)
  self.assertTrue(r['conditional_geometry_identical']);self.assertTrue(r['history_exact'])
 def test_changed_visibility_is_rejected(self):
  n,o,a,b=fixture();n['route_union_gain_matrix'][0,1]=n['weights'][1]
  with self.assertRaises(AssertionError):verify_arrays(n,o,'G',[a],[b],16.)
 def test_changed_history_or_geometry_is_rejected(self):
  for key in ('history_keep','points'):
   n,o,a,b=fixture();n[key]=n[key].copy()
   if key=='history_keep':n[key].flat[0]=not n[key].flat[0]
   else:n[key].flat[0]+=1
   with self.assertRaises(AssertionError):verify_arrays(n,o,'G',[a],[b],16.)
 def test_changed_posterior_is_rejected(self):
  n,o,a,b=fixture();a['joint_probability_mass'][0][0]+=.01
  with self.assertRaises(AssertionError):verify_arrays(n,o,'G',[a],[b],16.)
 def test_winner_ties_and_negative_outcome_ranks(self):
  self.assertEqual(winner([dict(candidate_id=2,score=1.,cost=5),dict(candidate_id=1,score=1.,cost=5),dict(candidate_id=0,score=.9,cost=1)]),1)
  self.assertAlmostEqual(rho([1,2,3],[-.2,-.1,.5]),1.)
  self.assertIsNone(rho([0,0],[-1,1]))


if __name__=='__main__':unittest.main(verbosity=2)
