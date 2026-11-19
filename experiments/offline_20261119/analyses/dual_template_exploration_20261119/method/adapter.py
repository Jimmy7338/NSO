"""Declared full-template gain replacement inside the preserved v10 candidate loop.

No mapping, candidate generation, safety, action accounting or reward update is
performed here. The common observed score is returned by the original scorer;
its analytic AABB prior is discarded and replaced, never added twice.
"""
from copy import deepcopy
import hashlib,inspect
from pathlib import Path
import numpy as np
from nso.cpu_sensor_contract_v10 import json_value
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules
from utils.reconstruction_metrics import ray_scene
from .model import FullTemplateModel,ORIGINAL_SOURCE_SHA256

SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
ARMS=('C','J','G','S','S_no_fit','S_no_history')
MODE={'C':'C','J':'N','G':'G','S':'S','S_no_fit':'S','S_no_history':'S'}
_original=None

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity():
 return dict(schema='dual_template_joint_score_adapter_v1',original_method_equivalence=False,
  replaced_term='entire analytic measured-AABB prior',common_terms='original coverage posterior plus measured quality',
  template_gain_coefficient=1.,coefficient_source='original v10 prior outer coefficient and common square-meter convention; not V3 planner weight4/normalizer',
  candidates='unchanged v10_3_1 generator with shared measured-instance frontend; ObjectJointPlannerV3.candidates not called',
  arms=list(ARMS),source_sha256={str(SOURCE/'nso/semantic_completion_v3.py'):_sha(SOURCE/'nso/semantic_completion_v3.py'),
   str(SOURCE/'nso/cpu_four_modules_v10.py'):_sha(SOURCE/'nso/cpu_four_modules_v10.py'),
   str(SOURCE/'nso_bridge/exploration_joint_policy.py'):_sha(SOURCE/'nso_bridge/exploration_joint_policy.py')},
  method_sha256={str(Path(__file__).resolve()):_sha(__file__),str(Path(__file__).with_name('model.py').resolve()):_sha(Path(__file__).with_name('model.py'))},
  reused_model_source_sha256=ORIGINAL_SOURCE_SHA256,
  history='original full-scene measured depth keyframes; no instance cropping; no recursively accumulated posterior',
  mapping_patch_scope='in-process ExplorationJointCPUFourModules._scores_v10_1 only, and only when args.dual_template_arm is set')

def _frame_hash(frame):
 frame.validate();h=hashlib.sha256()
 for arr in [frame.depth_m,frame.color_rgb,frame.semantic,frame.intrinsic,frame.world_from_camera]:
  a=np.ascontiguousarray(arr);h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
 h.update(repr(float(frame.timestamp_s)).encode());return h.hexdigest()

def initialize(mapper,frame):
 if hasattr(mapper,'_dual_template_state'):raise ValueError('dual template history already initialized')
 if mapper.frames!=1:raise ValueError('initialize requires exactly one original fused cold-start frame')
 mapper._dual_template_state=dict(frames=[dict(timestamp_s=float(frame.timestamp_s),sha256=_frame_hash(frame))],
  initialized_mapper_frames=1,cache=None,latest_call=None,score_calls=[])
 return receipt(mapper)

def observe(mapper,frame):
 s=mapper._dual_template_state
 if mapper.frames!=len(s['frames']):raise ValueError('previous frame not fused exactly once')
 if float(frame.timestamp_s)<=s['frames'][-1]['timestamp_s']:raise ValueError('nonmonotone/duplicate observed frame')
 s['frames'].append(dict(timestamp_s=float(frame.timestamp_s),sha256=_frame_hash(frame)));s['cache']=None
 return dict(registered_frames=len(s['frames']),expected_mapper_frames_after_native_fusion=len(s['frames']),fusion_performed=False)

def receipt(mapper):
 s=mapper._dual_template_state
 return json_value(dict(registered_frames=len(s['frames']),mapper_frames=mapper.frames,
  input_sequence_sha256=hashlib.sha256(''.join(x['sha256']for x in s['frames']).encode()).hexdigest(),
  retained_source_keyframes=len(mapper.keyframes),score_call_count=len(s['score_calls']),latest_call=s['latest_call'],
  fusion_performed=False,action_budget_modified=False))

def audit_log(mapper):return json_value(mapper._dual_template_state['score_calls'])
def frame_receipts(mapper):return deepcopy(mapper._dual_template_state['frames'])

def _cache(mapper,arm):
 s=mapper._dual_template_state
 if mapper.frames!=len(s['frames']):raise ValueError('template score before registered observation has been fused')
 key=(int(mapper.frames),arm)
 if s['cache'] is not None and s['cache']['key']==key:return s['cache']
 # Every new map state starts again from the original prior and current aggregate
 # geometry; no old posterior is reused as a new prior.
 G=FullTemplateModel(mapper,semantic=False)
 S=FullTemplateModel(mapper,semantic=True,fit_update=arm!='S_no_fit',history_rejection=arm!='S_no_history')
 if arm!='S_no_history':
  np.testing.assert_array_equal(G.points,S.points);np.testing.assert_array_equal(G.hypothesis_ids,S.hypothesis_ids)
 mesh=mapper.mesh();ray=ray_scene(mesh)if len(mesh.triangles)else None
 value=dict(key=key,models={'G':G,'S':S},ray=ray,gains={'G':{},'S':{}},gain_calls={'G':0,'S':0},
  state_sha256=hashlib.sha256((''.join(x['sha256']for x in s['frames'])+arm).encode()).hexdigest())
 s['cache']=value;return value

def route_union_area(model,states,ray,cache=None):
 """Source gain calls, pointwise union over only committed outbound successors."""
 cache={}if cache is None else cache;union=np.zeros(len(model.points));calls=0
 for state in states:
  key=tuple(map(int,state))
  if key not in cache:
   cache[key]=model.gain(key[:2],key[2],ray);calls+=1
  np.maximum(union,cache[key],out=union)
 area=float(union.sum())
 if not np.isfinite(area)or area<0 or area>float(model.weights.sum())+1e-8:raise ValueError('invalid predicted visible-area union')
 return area,calls

def _score(self,s,routes,attempted):
 arm=getattr(self.args,'dual_template_arm',None)
 if arm is None:return _original(self,s,routes,attempted)
 if arm not in ARMS or getattr(self.args,'cpu_score_mode',None)!=MODE[arm]:raise ValueError('dual template arm/score-mode mismatch')
 # This computes the exact existing common score; none of its prior is retained.
 legacy_scores,legacy_rows=_original(self,s,routes,attempted)
 mapper=s['mapper'];cache=_cache(mapper,arm);scores={m:[]for m in ['C','N','G','S']};rows=[]
 call=dict(arm=arm,mapper_frames=int(mapper.frames),state_sha256=cache['state_sha256'],
  model_audits={name:{**model.audit,'objects':model.objects}for name,model in cache['models'].items()},
  reused_original_gain_function=True,source_gain_new_calls={'G':0,'S':0},routes=[],
  prior_replaced_not_added=True,candidate_count=len(routes),template_coefficient=1.,
  G_S_semantic_dictionary='current2open/current3closed, translated only at template prior',
  original_V3_candidate_method_used=False)
 for route,old in zip(routes,legacy_rows):
  values={}
  for name,model in cache['models'].items():
   value,n=route_union_area(model,route['outbound_states'][1:],cache['ray'],cache['gains'][name]);values[name]=value
   cache['gain_calls'][name]+=n;call['source_gain_new_calls'][name]+=n
  common=old['common_total_proxy'];scores['C'].append(old['coverage_only_proxy']);scores['N'].append(common)
  scores['G'].append(common+values['G']);scores['S'].append(common+values['S'])
  row={k:v for k,v in old.items()if k not in ['prior_potentials','incremental_aperture','measured_direction_support','measured_novelty_floor']}
  row.update(prior_potentials={'N':0.,'G':values['G'],'S':values['S']},
   template_expected_visible_area_m2=values,template_state_sha256=cache['state_sha256'],
   template_gain_coefficient=1.,analytic_prior_used=False,legacy_analytic_prior_discarded=old['prior_potentials'],
   template_variant=arm,template_prior_active_in_selected_score=arm not in ['C','J'])
  rows.append(row);call['routes'].append(dict(candidate_id=route['candidate_id'],outbound_successors=len(route['outbound_states'])-1,
   excluded_return_actions=len(route['return_actions']),G_area_m2=values['G'],S_area_m2=values['S']))
 state=mapper._dual_template_state;state['latest_call']=json_value(call);state['score_calls'].append(state['latest_call'])
 return scores,rows

def install():
 global _original
 current=ExplorationJointCPUFourModules._scores_v10_1
 if current is _score:return identity()
 if _original is not None:raise RuntimeError('unexpected scorer binding after installation')
 if not Path(inspect.getfile(current)).resolve().is_relative_to(SOURCE.resolve()):raise RuntimeError('scorer must come from frozen scientific source')
 if _sha(SOURCE/'nso/semantic_completion_v3.py')!=ORIGINAL_SOURCE_SHA256:raise RuntimeError('original full-template source changed')
 _original=current;ExplorationJointCPUFourModules._scores_v10_1=_score
 return identity()
