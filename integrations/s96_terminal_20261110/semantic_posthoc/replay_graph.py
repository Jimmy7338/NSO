#!/usr/bin/env python3
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import json,pathlib,sys,hashlib,types,copy,traceback
import numpy as np
sys.path.insert(0,'/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge')
from nso_bridge.joint_engine import JointSession
from nso_bridge.exploration_joint_policy import _geometry_assets
from nso.competition_candidates_v8_1 import aperture_support,navigation_rear_boundary,XY_DIRECTIONS
from env.virtual3d import camera_pose
from utils.grid_geometry import inflated_obstacles
from nso.route_coverage_v2 import orientation_graph
from scipy.sparse.csgraph import dijkstra
out=pathlib.Path(__file__).parent;report=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261109/coverage_S96_20261109_01')
raw=(report/'joint/decisions.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()];init=copy.deepcopy(rows[0]['request']['params']);init.pop('semantic_configuration',None);init['execution_allowed']=False
cache={}
for p in (report/'joint/semantics').glob('*.json'):
 r=json.loads(p.read_text());cache[r['source_file_sha256']]=(p.with_suffix('.npz'),r)
session=JointSession(**init)
def cached(self,z,meta,sha):
 p,r=cache[sha];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['mask_file_sha256']
 with np.load(p,allow_pickle=False) as n:labels=n['semantic'].copy()
 assert hashlib.sha256(labels.tobytes()).hexdigest()==r['canonical_semantic_sha256'];self._instance_owner_map=None;return labels
session._semantic_labels=types.MethodType(cached,session)
def compare(x,y,path=''):
 diff=[]
 if isinstance(x,dict) and isinstance(y,dict):
  if x.keys()!=y.keys():diff.append(path+'.keys')
  for k in x.keys()&y.keys():diff+=compare(x[k],y[k],path+'.'+k)
 elif isinstance(x,list) and isinstance(y,list):
  if len(x)!=len(y):diff.append(path+'.len')
  for i,(a,b) in enumerate(zip(x,y)):diff+=compare(a,b,path+f'[{i}]')
 elif isinstance(x,(int,float)) and isinstance(y,(int,float)):
  if not np.isclose(x,y,atol=1e-8,rtol=1e-8):diff.append(path)
 elif x!=y:diff.append(path)
 return diff
result={'status':'running','baseline':[],'checkpoints':[],'source_decisions_sha256':hashlib.sha256(raw).hexdigest(),'execution_allowed':False,'configuration':init['configuration']}
def save(): (out/'replay_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def diagnose(sel,paid):
 s=session.backend.scenes[0];mapper=s['mapper'];cfg=mapper.config;packet=s['packet'];current=(*packet.position,packet.heading);anchor=s['return_anchor'];remaining=s['ledger'].remaining_budget
 safe=~inflated_obstacles(mapper.belief!=0,cfg.robot_radius_m/cfg.resolution_m);graph,cells,ids=orientation_graph(safe);start=int(ids[current[:2]])*4+current[2];home=int(ids[tuple(anchor[:2])])*4+anchor[2]
 d=dijkstra(graph,directed=True,indices=start);r=dijkstra(graph.T.tocsr(),directed=True,indices=home);cost=d+r;connected=np.isfinite(cost)&(d>0);feasible=connected&(cost<=remaining)
 nodes=np.arange(len(d));rc=cells[nodes//4];xy=np.column_stack(((rc[:,1]+.5)*cfg.resolution_m,(mapper.shape[0]-rc[:,0]-.5)*cfg.resolution_m));head=nodes%4;assets=_geometry_assets(mapper,mapper.quality_evidence(max_points=10000));regions=s['graph']['regions'];region_ok=(s['graph']['current_region']>0)&(s['graph']['current_region']==s['graph']['anchor_region'])&(regions[rc[:,0],rc[:,1]]==s['graph']['current_region'])
 used={tuple(x['pose']) for x in sel['candidate_audit']['common_role_audit'] if x.get('pose') is not None};unused=np.array([tuple((*p,int(h))) not in used for p,h in zip(rc,head)])
 diag={'paid':paid,'remaining_budget':remaining,'current_pose':current,'return_anchor':anchor,'safe_cells':int(safe.sum()),'orientation_nodes':len(d),'current_and_anchor_connected_outbound_nodes':int(connected.sum()),'budget_feasible_nodes':int(feasible.sum()),'belief_sha256':hashlib.sha256(mapper.belief.tobytes()).hexdigest(),'safe_sha256':hashlib.sha256(safe.tobytes()).hexdigest(),'graph_regions':{'current':int(s['graph']['current_region']),'anchor':int(s['graph']['anchor_region'])},'original_candidate_audit':sel['candidate_audit'],'assets':[]}
 for ai,a in enumerate(assets):
  rear=(xy-a['rear_boundary_xy'])@a['back_axis'];rear_ok=rear>.15;wanted=int(np.argmax(XY_DIRECTIONS@a['front_axis']));aperture=np.zeros(len(d),bool)
  # Original rear early return makes every excluded node's aperture exactly zero.
  for n in np.flatnonzero(rear_ok):aperture[n]=bool(aperture_support(a,camera_pose(tuple(rc[n]),int(head[n]),cfg,mapper.shape[0]),cfg).max()>0)
  connected_ap=connected&aperture;budget_ap=feasible&aperture;heading_ap=budget_ap&(head==wanted);unused_ap=heading_ap&unused;desired=navigation_rear_boundary(mapper,a)+.4*a['back_axis'];errors=np.linalg.norm(xy-desired,axis=1);projection=unused_ap&(errors<=.35)
  def witness(mask,metric):
   ix=np.flatnonzero(mask)
   if not len(ix):return None
   n=int(ix[np.argmin(metric[ix])]);return {'pose':[int(rc[n,0]),int(rc[n,1]),int(head[n])],'outward_cost':int(d[n]) if np.isfinite(d[n]) else None,'return_cost':int(r[n]) if np.isfinite(r[n]) else None,'total_cost':int(cost[n]) if np.isfinite(cost[n]) else None,'rear_offset_m':float(rear[n]),'entry_projection_error_m':float(errors[n])}
  counts={'all_safe_rear':int(rear_ok.sum()),'all_safe_aperture':int(aperture.sum()),'connected_rear':int((connected&rear_ok).sum()),'connected_aperture_all_heading':int(connected_ap.sum()),'connected_aperture_wanted_heading':int((connected_ap&(head==wanted)).sum()),'budget_aperture_all_heading':int(budget_ap.sum()),'budget_aperture_wanted_heading':int(heading_ap.sum()),'unused_budget_heading_aperture':int(unused_ap.sum()),'entry_projection_pass':int(projection.sum()),'entry_projection_and_region_pass':int((projection&region_ok).sum())}
  if not connected_ap.any():cause='no_aperture_in_current_anchor_connected_safe_graph'
  elif not budget_ap.any():cause='aperture_exists_but_reserved_return_budget_insufficient'
  elif not heading_ap.any():cause='budget_aperture_exists_but_original_heading_rejects'
  elif not unused_ap.any():cause='original_coverage_or_rotation_already_uses_pose'
  elif not projection.any():cause='entry_projection_error_exceeds_original_0_35m'
  else:cause='entry_geometric_admission_exists_inspect_original_cap_and_region'
  diag['assets'].append({'asset_index':ai,'group':a['group'],'aabb_center':a['aabb_center'].tolist(),'back_axis':a['back_axis'].tolist(),'wanted_heading':wanted,'desired_entry_xy':desired.tolist(),'counts':counts,'max_connected_rear_m':float(rear[connected].max()) if connected.any() else None,'minimum_cost_aperture_witness':witness(connected_ap,cost),'minimum_cost_wanted_heading_aperture_witness':witness(connected_ap&(head==wanted),cost),'nearest_budget_entry_witness':witness(unused_ap,errors),'earliest_blocking_stage':cause})
  print('asset',paid,ai,counts,cause,flush=True)
 (out/f'paid{paid}_graph.json').write_text(json.dumps(diag,indent=2,allow_nan=False)+'\n');result['checkpoints'].append({'paid':paid,'file':f'paid{paid}_graph.json'})
try:
 for row in rows[1:]:
  method=row['request']['method'];expected=row['response'].get('result');params=copy.deepcopy(row['request']['params'])
  if method=='observe':
   params['path']=str(report/'joint_inputs'/pathlib.Path(params['path']).name);session.observe(**params)
  elif method=='next_action':
   actual=session.next_action();paid=session.last_packet.action_id;diff=compare(expected.get('selection'),actual.get('selection'),'selection')
   if expected.get('action')!=actual.get('action'):diff.append('action')
   result['baseline'].append({'paid':paid,'request_id':row['request']['id'],'difference_count':len(diff),'differences':diff[:20]});save()
   if diff:raise RuntimeError('baseline mismatch '+str(diff[:10]))
   print('matched',paid,flush=True)
   if paid in (0,36):diagnose(actual['selection'],paid);save()
   if paid==36:break
 result['status']='baseline_matched_through_paid36';save()
except Exception:
 result['status']='error';result['error']=traceback.format_exc();save();raise
