import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
P=Path(__file__).resolve().parent.parent;S=P/'source';sys.path.insert(0,str(S))
from env.virtual3d_competition_v9 import create_competition_world
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.route_coverage_v2 import orientation_graph,recover_actions
from utils.grid_geometry import visible_mask
F=P/'frontier_baseline/matrix';D=Path(__file__).parent
protocol=json.loads((F/'protocol.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(F/'runner.py')==protocol['runner_sha256'];assert all(sha(S/k)==v for k,v in protocol['source_sha256'].items())
rows=[]
for arr,budget in protocol['conditions']:
 f=F/f'{arr}_{budget}';r=json.loads((f/'result.json').read_text());steps=json.loads((f/'steps.json').read_text());assert len(r['interventions'])==1;reject=r['interventions'][0];n=reject['step'];remaining=budget-n
 w=create_competition_world('Q0',arr);tr=GridTransform(tuple(w.shape),w.config.resolution_m);frame=w.sense();scan=w.scan();anchor=(*map(int,w.position),int(w.heading));packet=SensorPacket('Q0_'+arr,'fresh-first-frame','frame-0',0,frame,scan,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config);assert packet.sha256()==r['initial_packet_sha256'] and list(anchor)==r['anchor'];m=ObservedRuntimeMapperV10(w.shape,w.config);m.update(frame,scan)
 for record in steps[:n]:
  frame,col,done=w.step(record['action']);scan=w.scan();packet=SensorPacket('Q0_'+arr,'fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),frame,scan,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',record['action'],bool(col),bool(done)).validate(tr,w.config)
  assert packet.sha256()==record['packet_sha256'] and list(w.position)==record['position'] and w.heading==record['heading'] and not col and not done
  m.update(frame,scan)
 pos=tuple(map(int,w.position));heading=int(w.heading);guard=ObservedExecutionGuard(w.config.resolution_m,w.config.robot_radius_m);safe=guard.safe_grid(m.belief);ret=guard.return_plan(m.belief,pos,heading,anchor,remaining);assert ret.available
 actions={}
 for action in ('forward','left','right'):
  a=guard.assess(m.belief,pos,heading,action,remaining);h=(heading+(1 if action=='right' else -1 if action=='left' else 0))%4;back=guard.return_plan(m.belief,a.target,h,anchor,remaining-1) if a.allowed else None
  actions[action]=dict(footprint_allowed=a.allowed,footprint_reason=a.reason,next_pose=[*a.target,h],return_available=bool(back and back.available),return_cost=None if back is None else back.paid_cost,return_reason=None if back is None else back.reason,within_return_contract=bool(a.allowed and back.available))
 assert not actions[reject['action']]['within_return_contract'] and actions[reject['action']]['return_reason']==reject['reason']
 # Enumerate all orientation poses reachable through exactly the same observed-safe graph.
 # Gain uses the baseline's belief-only visibility approximation, not truth or future sensing.
 graph,cells,ids=orientation_graph(safe);start=int(ids[pos])*4+heading;home=int(ids[anchor[:2]])*4+anchor[2]
 outward,prev=dijkstra(graph,directed=True,indices=start,return_predecessors=True);homeward=dijkstra(graph.T.tocsr(),directed=True,indices=home)
 possible=np.flatnonzero(np.isfinite(outward)&np.isfinite(homeward)&(outward>=1)&(outward+homeward<=remaining))
 targets=[]
 for q in possible:
  cell=tuple(map(int,cells[int(q)//4]));hd=int(q)%4;gain=int(np.count_nonzero(visible_mask(m.belief==1,cell,hd,int(w.config.max_depth_m/w.config.resolution_m),w.config.fov_deg)&(m.belief==-1)))
  if gain:
   path=recover_actions(prev,start,int(q),cells);targets.append(dict(pose=[*cell,hd],outbound_cost=int(outward[q]),return_cost=int(homeward[q]),total_cost=int(outward[q]+homeward[q]),budget_slack=int(remaining-outward[q]-homeward[q]),predicted_unknown_cells=gain,first_action=path[0],outbound_actions=path))
 targets.sort(key=lambda x:(-x['predicted_unknown_cells'],-x['budget_slack'],x['pose']))
 # Exact shortest-return state sequence; flag target states off that witness path.
 path=list(ret.actions);state=(*pos,heading);witness=[list(state)]
 from utils.grid_geometry import DIRECTIONS
 for a in path:
  rr,cc,hh=state
  if a=='forward':dr,dc=DIRECTIONS[hh];rr+=int(dr);cc+=int(dc)
  else:hh=(hh+(1 if a=='right' else -1))%4
  state=(rr,cc,hh);witness.append(list(state))
 for target in targets:target['on_one_shortest_return_witness']=target['pose'] in witness
 extra=[x for x in targets if not x['on_one_shortest_return_witness']]
 row=dict(arrangement=arr,budget=budget,replayed_actions=n,initial_packet_sha256=r['initial_packet_sha256'],latest_packet_sha256=packet.sha256(),position=list(pos),heading=heading,anchor=anchor,remaining_budget=remaining,return_cost=ret.paid_cost,current_slack=remaining-ret.paid_cost,rejected=reject,actions=actions,belief_sha256=hashlib.sha256(m.belief.tobytes()).hexdigest(),safe_sha256=hashlib.sha256(safe.tobytes()).hexdigest(),reachable_orientation_states_within_budget=int(len(possible)),positive_gain_target_count=len(targets),positive_gain_off_return_witness_count=len(extra),max_predicted_gain=max((x['predicted_unknown_cells'] for x in targets),default=None),max_slack_among_positive_targets=max((x['budget_slack'] for x in targets),default=None),best_predicted_gain_target=targets[0] if targets else None,best_off_return_witness_target=extra[0] if extra else None,shortest_return_actions=path,targets=targets)
 rows.append(row);print(arr,budget,'done',flush=True)
(D/'audit_recomputed.json').write_text(json.dumps(dict(scope='retrospective prefix replay and observed-map feasibility; no policy rollout or truth planning; positive gain is camera-only predicted proxy, not actual measured gain',results=rows),indent=2))
