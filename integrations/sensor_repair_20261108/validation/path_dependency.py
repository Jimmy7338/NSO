#!/usr/bin/env python3
"""Read frozen final mapper grids; quantify one counterfactual return path."""
import json,pathlib,sys,math
import numpy as np
from scipy.sparse.csgraph import dijkstra
W=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge/joint_vendor'));sys.path.insert(0,str(W/'nso_bridge'));from nso.route_coverage_v2 import orientation_graph
from utils.grid_geometry import inflated_obstacles
D=pathlib.Path(__file__).resolve().parent
with np.load(D/'final_grid_replay.npz',allow_pickle=False) as z:b=z['baseline'];q=z['candidate'];nav=z['nav']
radius=.7212185868375829/.2;safe=~inflated_obstacles(q!=0,radius);graph,cells,ids=orientation_graph(safe);cur=(211,200,1);anchor=(200,200,0);start=int(ids[cur[:2]])*4+cur[2];end=int(ids[anchor[:2]])*4+anchor[2];cost,prev=dijkstra(graph,directed=True,indices=start,return_predecessors=True);assert np.isfinite(cost[end]);nodes=[];i=end
while i!=start:nodes.append(int(i));i=int(prev[i]);assert i>=0
nodes.append(start);nodes.reverse();states=[[*map(int,cells[n//4]),int(n%4)] for n in nodes];changed=np.argwhere((b==1)&(q!=1));threshold=radius+math.sqrt(.5);blocks=[]
for r,c in changed:
 affected=sum(math.hypot(int(r)-s[0],int(c)-s[1])<=threshold+1e-12 for s in states)
 if affected:blocks.append(dict(cell=[int(r),int(c)],candidate=int(q[r,c]),last_nav=int(nav[r,c]),path_states_affected=affected))
assert blocks
out={'scope':'One shortest counterfactual path on candidate grid; offline graph calculation, not executable or safe certification.','current':cur,'anchor':anchor,'path_cost':float(cost[end]),'states':states,'baseline_occupied_candidate_nonoccupied_total':len(changed),'changed_occupied_cells_inflating_this_path':blocks,'last_nav_occupied_among_path_blockers':sum(x['last_nav']==1 for x in blocks),'limits':['Candidate grid also introduces new occupied cells.','A path that uses these cells does not establish a unique minimal cut or rule out other routes.','Original scan and navigation packet values remain unchanged, not a guarantee that real obstacles are preserved.']};(D/'path_dependency.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'cost':out['path_cost'],'changed_total':len(changed),'path_blockers':len(blocks),'last_nav_occupied':out['last_nav_occupied_among_path_blockers']}))
