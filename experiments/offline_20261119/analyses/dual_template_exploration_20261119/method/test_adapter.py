"""Mechanism and source-parity tests; synthetic fixtures are evaluator-only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,importlib.util,unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(ROOT))
FIXTURE=Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO/utils/inspection_benchmark.py')
spec=importlib.util.spec_from_file_location('independent_inspection_fixture',FIXTURE);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
import open3d as o3d
from env.virtual3d_v2 import VirtualConfigV2
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.semantic_completion_v3 import ObjectCompletionModel
from nso.competition_candidates_v8 import measured_assets
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules,ExplorationObservedGainCalibration
from utils.reconstruction_metrics import ray_scene
from method.model import FullTemplateModel,measured_groups
from method import adapter

class LegacyLabels:
 def __init__(self,m):self.m=m
 def __getattr__(self,k):return getattr(self.m,k)
 def quality_evidence(self,max_points=1600):
  q=self.m.quality_evidence(max_points)
  if q is None:return None
  q=dict(q);a=q['label'];q['label']=np.where(a==2,3,np.where(a==3,2,a));return q

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.world=bench.InspectionWorld(VirtualConfigV2(voxel_m=.03,width_px=120,height_px=90),223,'box')
  cls.frames=[]
  for i,az in enumerate([-90,0,90,180]):
   pose,_=cls.world.pose(1.4,az);f=cls.world.capture(pose,i,'none')
   # Fixture dictionary is native V3; public adapter input is current dictionary.
   labels=np.where(f.semantic==2,3,np.where(f.semantic==3,2,0)).astype(np.uint8)
   cls.frames.append(replace(f,semantic=labels))
 def mapper(self,n=1,absent=False):
  m=ObservedRuntimeMapperV10(self.world.shape,self.world.config)
  for f in self.frames[:n]:m.update(replace(f,semantic=np.zeros_like(f.semantic))if absent else f)
  return m
 def target(self,model):return min(model.objects,key=lambda x:np.linalg.norm(x['center'][:2]-[4,3]))
 def test_original_constructor_parity_and_label_polarity(self):
  m=self.mapper(4)
  for semantic in [False,True]:
   old=ObjectCompletionModel(LegacyLabels(m),semantic);new=FullTemplateModel(m,semantic)
   self.assertGreater(len(old.objects),0)
   np.testing.assert_array_equal(old.points,new.points);np.testing.assert_array_equal(old.hypothesis_ids,new.hypothesis_ids)
   np.testing.assert_allclose(old.weights,new.weights,rtol=0,atol=0)
   for a,b in zip(old.objects,new.objects):
    np.testing.assert_array_equal(a['dims'],b['dims']);self.assertEqual(a['shelf_probability'],b['shelf_probability'])
  self.assertEqual(self.target(FullTemplateModel(m,True))['shelf_prior_probability'],.1)
 def test_geometry_fit_really_changes_posterior_but_not_geometry(self):
  m=self.mapper(4);full=FullTemplateModel(m,False);nofit=FullTemplateModel(m,False,fit_update=False)
  self.assertLess(self.target(full)['shelf_probability'],.5)
  self.assertEqual(self.target(nofit)['shelf_probability'],.5)
  np.testing.assert_array_equal(full.points,nofit.points)
  self.assertNotEqual(self.target(full)['consumed_geometry_log_likelihood_ratio'],0)
  self.assertEqual(self.target(nofit)['consumed_geometry_log_likelihood_ratio'],0)
 def test_absent_labels_equal_geometry_mixture(self):
  m=self.mapper(4,absent=True);g=FullTemplateModel(m,False);s=FullTemplateModel(m,True)
  np.testing.assert_array_equal(g.points,s.points);np.testing.assert_array_equal(g.weights,s.weights)
 def test_class_only_changes_template_weights(self):
  m=self.mapper(1);before=np.asarray(m.mesh().vertices).copy();g=FullTemplateModel(m,False);s=FullTemplateModel(m,True)
  np.testing.assert_array_equal(g.points,s.points)
  self.assertGreater(float(np.abs(g.weights-s.weights).sum()),0)
  np.testing.assert_array_equal(before,np.asarray(m.mesh().vertices))
 def test_history_consumes_and_contradicts_hypothesis_points(self):
  m=self.mapper(4);full=FullTemplateModel(m,True);none=FullTemplateModel(m,True,history_rejection=False)
  self.assertGreater(full.audit['removed_by_free_space']+full.audit['removed_as_measured_or_depth_tolerance'],0)
  self.assertEqual(none.audit['removed_by_free_space'],0);self.assertEqual(none.audit['removed_as_measured_or_depth_tolerance'],0)
  self.assertLess(full.audit['points_after_history'],none.audit['points_after_history'])
  for a,b in zip(full.objects,none.objects):self.assertEqual(a['shelf_probability'],b['shelf_probability'])
 def test_same_evidence_is_not_recursively_reapplied(self):
  m=self.mapper(4);a=FullTemplateModel(m,True);b=FullTemplateModel(m,True)
  np.testing.assert_array_equal(a.weights,b.weights)
  self.assertEqual([x['shelf_probability']for x in a.objects],[x['shelf_probability']for x in b.objects])
 def test_per_hypothesis_self_occlusion_is_real(self):
  model=FullTemplateModel.__new__(FullTemplateModel);model.mapper=SimpleNamespace(config=self.world.config,shape=(30,40))
  model.points=np.array([[3.5,3.,.85],[4.5,3.,.85]]);model.weights=np.ones(2);model.hypothesis_ids=np.zeros(2,int)
  model.hypothesis_rays=[ray_scene(o3d.geometry.TriangleMesh.create_box(1,1,1.5).translate((3.5,2.5,0)))]
  pose=bench.look_at_xy(np.array([2.,3.,.85]),np.array([4.,3.,.85]))
  with patch('nso.semantic_completion_v3.camera_pose',return_value=pose):np.testing.assert_array_equal(model.gain((15,10),0,None),[1.,0.])
 def test_route_union_does_not_count_repeated_view_twice(self):
  model=FullTemplateModel(self.mapper(),True,history_rejection=False);states=[(15,10,1),(15,30,3),(5,20,2),(25,20,0)]
  areas=[adapter.route_union_area(model,[p],None)[0]for p in states];best=states[int(np.argmax(areas))]
  one,c1=adapter.route_union_area(model,[best],None);twice,c2=adapter.route_union_area(model,[best,best],None)
  self.assertGreater(one,0);self.assertEqual(one,twice);self.assertEqual(c2,1)
 def test_shared_partitions_retain_unknown_and_original_quality_keys(self):
  m=self.mapper(1);q=m.quality_evidence(max_points=10000)
  rows=[(k,v)for k,v in m.quality.items()if .12<v['point'][2]<1.8]
  assigned={k:(1 if v['label']==3 else 0)for k,v in rows}
  class Index:
   def memberships_for_keys(self,keys):return np.array([assigned[tuple(k)]for k in keys],np.int64)
  m.instance_support_index=Index();m.asset_grouping_profile='measured_instance_support_v1'
  groups=measured_groups(m,q);self.assertTrue(any(t==0 for _,_,t in groups));self.assertTrue(any(t==1 for _,_,t in groups))
  ids=np.concatenate([x[1]for x in groups]);self.assertEqual(len(ids),len(set(ids.tolist())))
  model=FullTemplateModel(m,True);self.assertTrue(any(o['support_track']==1 for o in model.objects))
 def score_state(self,m):
  assets=measured_assets(m);gain=ExplorationObservedGainCalibration(m.shape,consume_predicted_attempts=False)
  return dict(mapper=m,assets=assets,gain=gain,ledger=SimpleNamespace(planning_camera_poses=lambda:[f.world_from_camera for f in m.keyframes]),
   planning_direction_support=[[float(np.mean((a['bits']&(1<<i))!=0))for i in range(8)]for a in assets])
 def test_real_score_replaces_prior_keeps_common_and_excludes_return(self):
  m=self.mapper();adapter.initialize(m,self.frames[0]);adapter.install()
  args=SimpleNamespace(cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,cpu_score_mode='G',dual_template_arm='G')
  backend=ExplorationJointCPUFourModules(args,1,m.shape);state=self.score_state(m)
  routes=[dict(candidate_id=0,outbound_states=[[15,10,1],[15,11,1]],outbound_cost=1,return_actions=['left','left'],cost=3)]
  old,old_audit=adapter._original(backend,state,routes,np.zeros(m.shape,bool))
  scores,audit=backend._scores_v10_1(state,routes,np.zeros(m.shape,bool));a=audit[0]
  self.assertEqual(a['common_total_proxy'],old_audit[0]['common_total_proxy']);self.assertEqual(scores['C'],old['C'])
  self.assertEqual(scores['N'][0],a['common_total_proxy'])
  self.assertEqual(scores['G'][0],a['common_total_proxy']+a['template_expected_visible_area_m2']['G'])
  self.assertFalse(a['analytic_prior_used']);self.assertEqual(a['legacy_analytic_prior_discarded'],old_audit[0]['prior_potentials'])
  rec=adapter.receipt(m)['latest_call'];self.assertEqual(rec['routes'][0]['outbound_successors'],1)
  self.assertEqual(rec['routes'][0]['excluded_return_actions'],2);self.assertEqual(rec['source_gain_new_calls'],{'G':1,'S':1})
  state_cache=m._dual_template_state['cache'];backend._scores_v10_1(state,routes,np.zeros(m.shape,bool));self.assertIs(state_cache,m._dual_template_state['cache'])
  self.assertEqual(adapter.receipt(m)['latest_call']['source_gain_new_calls'],{'G':0,'S':0})
 def test_observe_does_not_fuse_or_allow_premature_scoring(self):
  m=self.mapper();adapter.initialize(m,self.frames[0]);before=m.frames
  adapter.observe(m,self.frames[1]);self.assertEqual(m.frames,before)
  with self.assertRaises(ValueError):adapter._cache(m,'S')
  m.update(self.frames[1]);self.assertEqual(adapter._cache(m,'S')['key'],(2,'S'))
  with self.assertRaises(ValueError):adapter.observe(m,self.frames[1])

if __name__=='__main__':unittest.main(verbosity=2)
