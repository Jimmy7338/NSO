import json,sys,math
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge/joint_vendor'));from utils.grid_geometry import supercover_line
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261109/coverage_S96_20261109_01');O=Path(__file__).parent;cfg=json.loads((P/'joint_configuration.json').read_text())['configuration'];T=np.array(cfg['planning_from_odom']);targets=[(197,208),(196,208),(196,207)];out={}
def cell(p):return(399-int(np.floor(p[1]/.2)),int(np.floor(p[0]/.2)))
z=np.load(P/'joint_inputs/joint-1794224925706198868.npz');A=T@z['world_from_laser'];origin=cell(A[:3,3]);rays={c:[] for c in targets}
for i,d in enumerate(z['ranges']):
 if not np.isfinite(d) or d<=0:continue
 d=min(float(d),4.);a=float(z['angle_min'])+i*float(z['angle_increment']);pt=A@np.array([np.cos(a)*d,np.sin(a)*d,0,1]);end=cell(pt);ray=supercover_line(end[0]-origin[0],end[1]-origin[1]);cells=[(origin[0]+dr,origin[1]+dc) for dr,dc in ray]
 for c in targets:
  if c in cells:rays[c].append(dict(beam=i,raw_range=float(z['ranges'][i]),adapted_range=d,end_cell=end,position='endpoint' if end==c else 'free_ray_interior'))
mask=[]
for r in range(195,206):
 for c in range(203,214):
  if math.hypot(r-200,c-208)*.2<=cfg['robot_radius_m']+.2/math.sqrt(2):mask.append((r,c))
vals=[int(z['joint_belief_grid'][c]) for c in mask];bounded=np.load(O/'bounded_belief_86.npy');out['target_footprint86']=dict(cells=len(mask),nav_counts={str(k):vals.count(k) for k in set(vals)},bounded_native_counts={str(k):sum(int(bounded[c])==k for c in mask) for k in [-1,0,1]});out['target_cells86']=[dict(cell=c,nav=int(z['joint_belief_grid'][c]),bounded_native=int(bounded[c]),scan_rays=rays[c]) for c in targets]
rows=[json.loads(s) for s in (P/'joint/decisions.jsonl').read_text().splitlines()]
for ix,row in enumerate(rows):
 result=row['response'].get('result',{})
 if row['request']['method']=='observe' and result.get('state',{}).get('action_id')==86:
  n=rows[ix+1]['response']['result'];out['selected86']=dict(pending=n['pending'],selected=n['selected'],candidate_audit=n['selection']['candidate_audit']);target=n['pending']['target_odom_pose'];pt=T@np.array([target[0],target[1],0,1]);out['target_planning_xy']=pt[:2].tolist()
 if row['request']['method']=='observe' and result.get('state',{}).get('action_id')==87:out['actual87']=result['state']
actual=np.array(out['actual87']['actual_base_pose_planning'][:2]);out['target_xy_error_m']=float(np.linalg.norm(actual-out['target_planning_xy']));out['user_field_fact']='User directly confirmed a fixed physical object at the attributed return location, relayed by root. This is not a physical collision report.'
(O/'transition_contract.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
