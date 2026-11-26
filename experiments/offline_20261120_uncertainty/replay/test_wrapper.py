"""Prediction file/route contract with a minimal deterministic bank fixture."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import importlib.util,sys,tempfile,unittest,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
sys.path.insert(0,str(SOURCE))
p=Path(__file__).with_name('run_particle_predictions.py');s=importlib.util.spec_from_file_location('B8_wrapper_test',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)

class ContractTest(unittest.TestCase):
 def test_complete_five_arm_matrices_share_bank_and_exclude_return(self):
  mapper=SimpleNamespace(frames=1,belief=np.zeros((2,2),np.int8),quality={},mesh=lambda:o3d.geometry.TriangleMesh())
  calls=[];built=[]
  class Bank:
   observed_input_sha256='fixture'
   def __init__(self):self.points={mode:np.array([[i,0,0]for i in range(n)],float) for mode,n in [('fixed',2),('uncertainty',4),('mean_collapse',3)]};self.ids={mode:np.arange(len(v))%2 for mode,v in self.points.items()}
   def distribution(self,**kw):return {'tracks':[]}
  def build(mapper):
   bank=Bank();built.append(bank);return bank
  class Model:
   def __init__(self,mapper,semantic,sigma_m,extent_mode,bank):
    self.bank=bank;self.points=bank.points[extent_mode];n=len(self.points);self.hypothesis_ids=bank.ids[extent_mode]
    self.object_ids=np.zeros(n,int);self.particle_ids=np.arange(n)//2;self.weights=np.arange(1,n+1,dtype=float)*(2 if semantic else 1)
    self.base_weights=self.weights.copy();self.history_keep=np.ones(n,bool);self.objects=[{'support_track':7,'conditional_geometry':{},'joint_probability_mass':[[.5,.5]]}]
    self.audit={'objects':self.objects};self.semantic=semantic;self.mode=extent_mode
   def gain(self,cell,heading,ray):
    calls.append((self.semantic,self.mode,cell,heading));return self.weights*(1 if cell==(1,0) else .5)
  module=SimpleNamespace(build_particle_bank=build,FullTemplateModel=Model,observed_input_signature=lambda m:'fixture',digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest())
  routes=[{'candidate_id':4,'outbound_states':[(0,0,0),(1,0,0),(1,0,0)],'outbound_actions':['forward','forward'],'return_actions':['back']},
          {'candidate_id':8,'outbound_states':[(0,0,0),(2,0,0)],'outbound_actions':['forward'],'return_actions':['back','back']}]
  common=[{'common_total_proxy':10.,'coverage_only_proxy':3.},{'common_total_proxy':20.,'coverage_only_proxy':4.}]
  with tempfile.TemporaryDirectory()as name:
   out=Path(name);captures=r.capture_models(module,{'mapper':mapper,'routes':routes,'common_rows':common},out,.01)
   self.assertEqual(set(captures),set(r.ARMS));self.assertEqual(len(built),1)
   for arm,flags in r.ARMS.items():
    group=flags['group'];d=dict(np.load(out/(arm+'_predictions.npz'),allow_pickle=False));receipt=json.loads((out/(arm+'_predictions.json')).read_text())
    np.testing.assert_array_equal(d['candidate_ids'],[4,8]);np.testing.assert_array_equal(d[group+'_route_union_gain_matrix'][0],d[group+'_gain_4'])
    np.testing.assert_array_equal(d[group+'_route_union_gain_matrix'][1],d[group+'_gain_8'])
    np.testing.assert_array_equal(d[group+'_gain_4'],d[group+'_weights']);np.testing.assert_array_equal(d[group+'_gain_8'],.5*d[group+'_weights'])
    self.assertTrue(np.all(d[group+'_support_tracks']==7))
    self.assertEqual(receipt['routes'][0]['scores'][group],10+float(d[group+'_weights'].sum()))
   self.assertEqual(len(calls),10) # two unique outbound poses per arm, no return-query
   self.assertTrue(json.loads((out/'observed_mapper_unchanged.json').read_text())['unchanged'])

if __name__=='__main__':unittest.main(verbosity=2)
