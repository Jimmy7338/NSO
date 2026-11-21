#!/usr/bin/env python3
"""Inspect a hash-matched saved real-observation grid, without replay or ROS."""
import sys,json,hashlib,pathlib
import numpy as np
from scipy.ndimage import label,distance_transform_edt
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'source'))
from utils.grid_geometry import inflated_obstacles,visible_mask
from nso.route_coverage_v2 import orientation_graph
D=pathlib.Path(__file__).resolve().parent/'inputs';O=pathlib.Path(__file__).parent
old=json.loads((D/'object_only_continuation.json').read_text());stage=old['stages'][0]; audit=stage['profiles'][0]['candidate_audit']; assets=stage['profiles'][0]['assets'];fronts={a['track']:a for a in json.loads((D/'first_observed_front.json').read_text())['assets']}; conf=json.loads((D/'offline_configuration.json').read_text())['configuration']; rec=json.loads((D/'footprint_audit.json').read_text())['records'][6];grid=np.load(D/'footprint_maps.npz')['maps'][6];sha=lambda x:hashlib.sha256(x.tobytes()).hexdigest();assert rec['action_id']==6 and rec['pose']==stage['pose'] and sha(grid)==rec['belief_sha256']==audit['belief_sha256']
res=.2;radius=conf['robot_radius_m'];safe=~inflated_obstacles(grid!=0,radius/res);assert sha(safe)==audit['safe_sha256'];blocked_unknown=inflated_obstacles(grid==-1,radius/res);blocked_occupied=inflated_obstacles(grid==1,radius/res);labs,_=label(safe);pos=tuple(stage['pose'][:2]);home=tuple(conf['return_anchor'][:2]);assert labs[pos] and labs[pos]==labs[home];connected=labs==labs[pos]
graph,cells,ids=orientation_graph(safe);start=int(ids[pos])*4+stage['pose'][2];h=int(ids[home])*4+conf['return_anchor'][2];fw=dijkstra(graph,directed=True,indices=start);bw=dijkstra(graph.T.tocsr(),directed=True,indices=h);states=np.flatnonzero(np.isfinite(fw)&np.isfinite(bw)&(fw>0));assert len(states)==stage['reachable_states']
r,c=np.indices(grid.shape);X=(c+.5)*res;Y=(grid.shape[0]-r-.5)*res
out=[]
for a in assets:
 if not a['marked_points']:continue
 f=fronts[a['track']];center=np.asarray(f['center_xy']);front=np.asarray(f['global_front']);back=-front;rear=center+back*a['measured_depth_m']/2
 margin=(X-rear[0])*back[0]+(Y-rear[1])*back[1];half=margin>.15;near=half&((X-center[0])**2+(Y-center[1])**2<=4.0**2)
 def counts(m):return dict(unknown=int(np.sum(m&(grid==-1))),occupied=int(np.sum(m&(grid==1))),free=int(np.sum(m&(grid==0))),safe=int(np.sum(m&safe)),connected=int(np.sum(m&connected)))
 assert not np.any(half&connected)
 # Observable unknown rear cells from return-connected, budget-feasible states.
 target=near&(grid==-1);best=None
 if np.any(target):
  # Candidate states ordered by distance to the finite unknown rear region. Only evaluate 80 nearest cells.
  dist=distance_transform_edt(~target);coords=np.argwhere(connected);order=np.argsort(dist[connected]); coords=coords[order[:80]]
  for cell in coords:
   for heading in range(4):
    sid=int(ids[tuple(cell)])*4+heading;cost=fw[sid]+bw[sid]
    if not np.isfinite(cost) or fw[sid]<=0 or cost>stage['remaining_budget']:continue
    vis=visible_mask(grid==1,tuple(cell),heading,20,conf['sensor_config']['fov_deg']); n=int(np.count_nonzero(vis&target))
    if n and (best is None or n>best['unknown_rear_cells_visible']):best=dict(cell=cell.tolist(),heading=heading,xy_m=[float(X[tuple(cell)]),float(Y[tuple(cell)])],outbound_actions=int(fw[sid]),return_actions=int(bw[sid]),unknown_rear_cells_visible=n,interpretation='2D occupied-grid ray model only; unknown is transparent; no evidence physical depth or semantic support will be acquired')
 out.append(dict(track=a['track'],group=a['group'],rear_boundary_xy=rear.tolist(),back_axis=back.tolist(),whole_grid=counts(half),within_4m_center=counts(near),known_free_removed_by_inflation_within_4m=int(np.sum(near&(grid==0)&~safe)),safe_disconnected_within_4m=int(np.sum(near&safe&~connected)), inflation_sources_free_cells=dict(unknown_only=int(np.sum(near&(grid==0)&blocked_unknown&~blocked_occupied)),occupied_only=int(np.sum(near&(grid==0)&~blocked_unknown&blocked_occupied)),both=int(np.sum(near&(grid==0)&blocked_unknown&blocked_occupied))),observable_unknown_example=best))
result=dict(scope='action 6 saved observed map, hash matched to semantic replay; half-plane rear margin >0.15m at camera/cell center; 4m circle is descriptive window, not exact 3D FOV',belief_sha256=sha(grid),safe_sha256=sha(safe),resolution_m=res,radius_m=radius,remaining_actions=stage['remaining_budget'],assets=out,limitations=['Unknown treated as blocked by safety inflation; free-cell removal does not prove occupied obstacle is the cause.','Unknown example is an illustrative 2D ray model over at most 80 nearest safe cells, not exhaustive and not a policy recommendation.','Occlusion by occupied 2D grid considered only for illustrative rays; true 3D occlusion and depth return remain unverified.'])
(O/'layers.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(out,indent=2))
