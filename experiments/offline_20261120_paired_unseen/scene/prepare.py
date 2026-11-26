"""Geometry legality, topology and fixed-prefix validation; never policy scoring."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,time
import numpy as np
from unseen_world import make_world,LAYOUTS,SPECS,sha,OLD,old
HERE=Path(__file__).resolve().parent

def room_of(xy,q):
 x,y=xy
 return ('A_southwest'if x<q['split_x']else'B_southeast')if y<q['split_y']else('D_northwest'if x<q['split_x']else'C_northeast')
def prepare(layout):
 t=time.monotonic();world=make_world(layout);q=SPECS[layout];folder=HERE/'prepared'/layout;folder.mkdir(parents=True,exist_ok=True)
 edges=set();roomcounts={n:0 for n in q['room_graph']['rooms']}
 for rr,cc in np.argwhere(world._reachable):
  cell=(int(rr),int(cc));a=room_of(world.xy(cell),q);roomcounts[a]+=1
  for h,(dr,dc)in enumerate(old.DIRECTIONS):
   if world._edge_valid[cell+(h,)]:
    b=room_of(world.xy((cell[0]+dr,cell[1]+dc)),q)
    if a!=b:edges.add(tuple(sorted([a,b])))
 expected={tuple(sorted(e))for e in q['room_graph']['edges']};assert edges==expected,(layout,edges,expected);assert all(roomcounts.values())
 prefixes=[];frame=world.sense();scan=world.scan();initial=dict(shape=list(world.shape),anchor=[*world.start,world.heading],actual_config=vars(world.config),layout=layout)
 frame.save(folder/'initial_sensor.npz');scan.save(folder/'initial_scan.npz');np.savez_compressed(folder/'initial_instances.npz',visible_instances=world.sense_visible_instances())
 np.savez_compressed(folder/'evaluator_truth.npz',**world.evaluator_truth())
 for p in q['sealed_prefixes']:
  world.position=world.start;world.heading=q['heading'];world.step_count=world.collisions=world.moves=0
  for action in p['actions']:
   frame,collision,done=world.step(action);assert not collision,(layout,p['prefix_id'],world.step_count,action)
  assert np.allclose(world.xy(world.position),p['expected_xy']);assert world.heading==p['expected_heading']
  prefixes.append(dict(prefix_id=p['prefix_id'],paid_actions=world.step_count,position=list(world.position),heading=world.heading,xy=list(world.xy(world.position)),collisions=world.collisions,valid_depth_pixels=int((frame.depth_m>0).sum()),valid_visible_instance_pixels=int((world.sense_visible_instances()>0).sum())))
 receipt=dict(status='geometry_graph_and_scripted_prefixes_validated',layout=layout,geometry=world.geometry_receipt,expected_room_graph=q['room_graph'],actual_room_edges=[list(e)for e in sorted(edges)],room_reachable_cell_counts=roomcounts,graph_cycle_rank=len(edges)-len(roomcounts)+1,prefixes=prefixes,source_objects=world._records,scene_code_sha256=sha(HERE/'unseen_world.py'),parent_sensor_motion_sha256=sha(OLD),prefix_selected_using_gain=False,source_truth_returned_in_actor_packet=False,candidate_scores_computed=False,policy_executed=False,elapsed_s=time.monotonic()-t)
 (folder/'initial_state.json').write_text(json.dumps(initial,indent=2)+'\n');(folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(layout=layout,status=receipt['status'],elapsed_s=receipt['elapsed_s'],cycle_rank=receipt['graph_cycle_rank'],prefix_actions=[p['paid_actions']for p in prefixes])),flush=True)
 return receipt
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--layout',choices=LAYOUTS,required=True);x=a.parse_args();prepare(x.layout)
