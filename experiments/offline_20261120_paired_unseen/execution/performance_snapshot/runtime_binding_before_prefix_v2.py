"""B11 old/new prediction binding; inherited complete policy stays unchanged."""
from pathlib import Path
from types import SimpleNamespace
import dataclasses,hashlib,importlib.util,json,os,sys,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=ROOT.parent
B=BASE/'dual_template_exploration_20261119';B8=BASE/'dual_template_uncertainty_20261120';B9=BASE/'dual_template_followup_20261120'
ARMS={'S_visibility':{'score_mode':'S','predictor':'visibility'},'S_acquisition':{'score_mode':'S','predictor':'acquisition'}}
_installed=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def protocol():return json.loads((HERE/'protocol.json').read_text())
def write(p,data):
 p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda v:v.item()if isinstance(v,np.generic)else v.tolist()if isinstance(v,np.ndarray)else str(v)if isinstance(v,Path)else (_ for _ in ()).throw(TypeError(type(v).__name__)))+'\n');temp.replace(p)
def resource(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def install():
 global _installed,old,scene,model,revision
 if _installed:return old
 old=load('B11_frozen_original_policy',B/'run.py');old.dependencies()
 scene=load('B11_unseen_world',ROOT/'scene/unseen_world.py')
 model=load('B11_frozen_full_particle_model',B8/'method/model.py')
 revision=load('B11_frozen_T16_reweight',B9/'structure/replay.py')
 from utils.reconstruction_metrics import ray_scene
 from full_posterior_forecast import FullPosteriorForecast,digest
 old.method.ARMS=tuple(ARMS);old.method.MODE={a:'S'for a in ARMS};old.method.FullTemplateModel=model.FullTemplateModel
 init_original,observe_original=old.method.initialize,old.method.observe
 def initialize(mapper,frame):
  value=init_original(mapper,frame);mapper._b11_acquired_frames=[frame];return value
 def observe(mapper,frame):
  value=observe_original(mapper,frame);mapper._b11_acquired_frames.append(frame);return value
 old.method.initialize=initialize;old.method.observe=observe
 def cache(mapper,arm):
  saved=mapper._dual_template_state;assert mapper.frames==len(saved['frames'])==len(mapper._b11_acquired_frames)
  key=(int(mapper.frames),arm)
  if saved['cache'] is not None and saved['cache']['key']==key:return saved['cache']
  bank=model.build_particle_bank(mapper);models={}
  for name in ('G','S'):
   m=model.FullTemplateModel(mapper,semantic=name=='S',bank=bank,sigma_m=.01,extent_mode='uncertainty',fit_update=True,history_rejection=True,measured_surface_consumption=True)
   revision.reweight(m,1.,None);revision.reweight(m,16.,resource(B9/'structure/audit/calibration_attempt_001/selected_temperature.json'));models[name]=m
  for field in ('points','hypothesis_ids','history_keep'):np.testing.assert_array_equal(getattr(models['G'],field),getattr(models['S'],field))
  mesh=mapper.mesh();ray=ray_scene(mesh)if len(mesh.triangles)else None
  value=dict(key=key,models=models,ray=ray,gains={'G':{},'S':{}},gain_calls={'G':0,'S':0},state_sha256=hashlib.sha256(''.join(f['sha256']for f in saved['frames']).encode()).hexdigest(),full_engine=None,predictor=ARMS[arm]['predictor'])
  for name,m in models.items():m._b11_cache=value;m._b11_prior=name
  saved['cache']=value;return value
 old.method._cache=cache
 union_original=old.method.route_union_area
 def union(m,states,ray,cache=None):
  saved=m._b11_cache
  if saved['predictor']=='visibility':return union_original(m,states,ray,cache)
  if saved['full_engine'] is None:saved['full_engine']=FullPosteriorForecast(m.mapper,saved['models'],m.mapper._b11_acquired_frames)
  result=saved['full_engine'].acquisition(states)
  return result['gross_m2'][m._b11_prior],0
 old.method.route_union_area=union
 cls=old.method.ExplorationJointCPUFourModules;score_original=cls._scores_v10_1
 def score(self,state,routes,attempted):
  result,rows=score_original(self,state,routes,attempted)
  saved=state['mapper']._dual_template_state;cached=saved['cache'];g=cached['models']['G'];s=cached['models']['S'];engine=cached['full_engine']
  saved['latest_call'].update(B11_predictor=cached['predictor'],extent_mode='uncertainty',sigma_m=.01,structural_temperature=16.,conditional_mean_or_fine_basis=False,shared_G_S_full_basis=True,points_sha256=digest(g.points),G_base_weights_sha256=digest(g.base_weights),S_base_weights_sha256=digest(s.base_weights),history_keep_sha256=digest(g.history_keep),acquisition_diagnostics=None if engine is None else engine.assert_unchanged(),original_candidate_provider=True,full_policy_cost_band_applied=False,gain_revision='original B2 visibility/history'if engine is None else'same full posterior/basis; independent predicted RGBD+TSDF gross-new at5cm, original strict-free retained, no old10cm double consumption')
  return result,rows
 cls._scores_v10_1=score
 old.world_and_packet=world_and_packet;old.protocol=protocol
 _installed=True;return old

def world_and_packet(layout):
 from nso.cpu_sensor_contract_v10 import GridTransform
 world=scene.make_world(layout,seed=protocol()['environment']['noise_seed'])
 tr=GridTransform(tuple(world.shape),world.config.resolution_m)
 return world,tr,old.packet(world,tr,world.sense(),layout)

def bootstrap_scripted_prefix(layout,prefix_id):
 """External acquisition history, explicitly charged; no policy chooses it."""
 install()
 from nso.components import NSO_Components
 from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules,ExplorationJointRuntimeIntegration
 world,tr,first=world_and_packet(layout);packets=[first]
 actions=scene.sealed_prefix_actions(layout,prefix_id)
 for action in actions:
  frame,collision,done=world.step(action)
  if collision or done:raise RuntimeError('Frozen scripted acquisition prefix failed')
  packets.append(old.packet(world,tr,frame,layout,action,collision,done))
 cfg=protocol();effective=dict(cfg['policy']['args']);effective.update(cpu_score_mode='S',dual_template_arm='S_visibility',run_id=f'B11-script-{layout}-{prefix_id}')
 args=SimpleNamespace(**effective);comp=NSO_Components(args);comp.initialize('cpu',1,*world.shape,*world.shape)
 comp._cpu_backend=ExplorationJointCPUFourModules(args,1,world.shape);comp.capabilities=dict(comp._cpu_backend.capabilities)
 runtime=ExplorationJointRuntimeIntegration(comp,1,world.shape)
 runtime.start_sensor_episode(0,config=world.config,transform=tr,packets=packets,total_budget=cfg['policy']['paid_action_budget'],return_anchor=(*first.position,first.heading),paid_prefix_actions=len(actions))
 mapper=runtime.states[0]['mapper'];state=comp._cpu_backend.scenes[0]
 mapper.instance_support_index=old.front.isp.InstanceSupportIndex();mapper.asset_grouping_profile='measured_instance_support_v1'
 owners=[old.front.associate(mapper,p.frame,p.frame_id)for p in packets]
 mapper._dual_template_state=dict(frames=[dict(timestamp_s=float(p.frame.timestamp_s),sha256=old.method._frame_hash(p.frame))for p in packets],initialized_mapper_frames=len(packets),cache=None,latest_call=None,score_calls=[])
 mapper._b11_acquired_frames=[p.frame for p in packets]
 state['planning_direction_support']=None;comp._cpu_backend.update_semantic(0)
 return dict(world=world,transform=tr,packets=packets,mapper=mapper,state=state,backend=comp._cpu_backend,runtime=runtime,owners=owners,scripted_actions=list(actions),prefix_id=prefix_id,layout=layout)

def native_candidates(context):
 """Identical native generate_options and inherited region consistency rule."""
 from nso_bridge.exploration_joint_policy import generate_options
 s=context['state'];b=context['backend'];p=s['packet'];history=s['ledger'].planning_camera_poses();attempted=s['gain'].attempted_camera_mask(s['mapper'],history)
 routes,audit=generate_options(s['mapper'],p.position,p.heading,s['ledger'].remaining_budget,s['return_anchor'],max_candidates=int(getattr(b.args,'cpu_max_candidates',12)),coverage_strategy='total_diverse',coverage_slots=int(getattr(b.args,'cpu_coverage_slots',4)),attempted_camera_mask=attempted)
 graph=s['graph'];routes=[r for r in routes if graph['current_region']>0 and graph['current_region']==graph['anchor_region']==int(graph['regions'][tuple(r['pose'][:2])])]
 _,common=old.method._original(b,s,routes,attempted)
 return routes,common,audit
