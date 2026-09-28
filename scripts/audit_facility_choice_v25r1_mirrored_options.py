#!/usr/bin/env python3
"""Single P00 reflected-suffix legality check; no physical task, mapping or Q."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
import zipfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'audit_results/facility_choice_v25r1_mirrored_options_20260915'
PROTOCOL=ROOT/'docs/research/V25_R1_MIRRORED_OPTIONS_STATIC_PROTOCOL_20260915.md'
COST=ROOT/'audit_results/facility_choice_v25_service_cost_20260915'
STATIC=ROOT/'audit_results/facility_choice_v25_static_r1_20260915'
EXPECTED={
 'cost_manifest':'0fbb550993440c90011a4561fc3abd8ee943308eec4968d5fe3ea3d768338c9f',
 'cost_result':'6c5e1029e9ba9746be46c10f51c19a91b15985dcb95433506e34fc3e29e691fe',
 'r1_manifest':'3f9394de7c77adab8733d8cd9ee88ac54cd63a9e3d02513f9c2aa416d403cf13',
 'r1_result':'7a4acce52e8a5294a8e7d89122d04ef5b8fec5b01d59b1d4aa3d665c1f7019c8',
}
CAP=256*1024
RESERVE=64*1024*1024
FAILURE_SPACE=16*1024
LIMIT_SECONDS=180
DIRECTIONS_OWN=((-1,0),(0,1),(1,0),(0,-1))
COUNTERS={'world_constructions':0,'clean_ray_queries':0,'nominal_depth_pixel_queries':0,
 'forbidden_step_calls':0,'forbidden_sense_calls':0,'forbidden_scan_calls':0,
 'new_physical_tasks':0,'new_physical_actions':0,'new_sensor_packets':0,
 'new_tsdf_fusions':0,'new_quality_evaluations':0,'new_route_searches':0,'new_cost_optimizations':0}

class GateFailure(ValueError): pass
class BoundedStop(BaseException): pass

def require(value,message):
 if not value:raise GateFailure(message)
def stop(signum,frame):raise BoundedStop('signal '+str(signum))
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def capacity(amount=0,emergency=False):
 ancestor=OUT if OUT.exists() else OUT.parent
 used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
 require(shutil.disk_usage(ancestor).free-amount-4096>=RESERVE,'64 MiB free-space reserve')
 require(used+amount+4096 <= (CAP if emergency else CAP-FAILURE_SPACE),'256 KiB output cap')
def write(name,value,emergency=False):
 blob=(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
 capacity(len(blob),emergency)
 path=OUT/name;tmp=OUT/(name+'.tmp')
 with tmp.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def verify(mapping):
 for name,expected in mapping.items():require(sha(ROOT/name)==expected,'Input SHA changed: '+name)

def sources_before_import():
 inputs={}
 for folder,key in ((COST,'cost'),(STATIC,'r1')):
  for name,suffix in (('manifest.json','manifest'),('result.json','result')):
   require(sha(folder/name)==EXPECTED[key+'_'+suffix],'Unexpected frozen '+key+' '+name)
  manifest=read(folder/'manifest.json');require(manifest['status']=='complete','Incomplete input '+key)
  inventory=read(folder/'artifact_hashes.json')
  for name,expected in inventory.items():
   path=folder/name;require(sha(path)==expected,'Original artifact mismatch '+str(path))
   inputs[str(path.relative_to(ROOT))]=expected
  inputs[str((folder/'artifact_hashes.json').relative_to(ROOT))]=sha(folder/'artifact_hashes.json')
  for field in ('source_sha256','protected_source_sha256'):
   for path,expected in manifest.get(field,{}).items():
    require(path not in inputs or inputs[path]==expected,'Conflicting frozen source '+path)
    inputs[path]=expected
 verify(inputs)
 return inputs

def own_states(start,actions):
 """Forward tuple interpreter independent of cost graph/BFS/validate_route."""
 state=list(start);states=[state.copy()]
 for action in actions:
  r,c,h=state
  if action=='forward':dr,dc=DIRECTIONS_OWN[h];state=[r+dr,c+dc,h]
  elif action=='left':state=[r,c,(h-1)%4]
  elif action=='right':state=[r,c,(h+1)%4]
  else:raise GateFailure('Unknown action '+str(action))
  states.append(state.copy())
 return states

def mirrored_catalog(parent,config):
 original=parent['route_catalog'];a=original['A'];b=original['B']
 require(a['prefix_actions']==b['prefix_actions'] and a['prefix_states']==b['prefix_states'],'Unequal original prefixes')
 require(len(a['prefix_actions'])==234 and parent['total_budget']==400,'Changed prefix/budget')
 require(own_states(parent['anchor'],a['prefix_actions'])==a['prefix_states'],'Original prefix state mismatch')
 centers=[original[x]['shared_body_center_xy_m'] for x in ('A','B')]
 require(centers[0][1]==centers[1][1],'Body centers not reflection-compatible')
 axis=(centers[0][0]+centers[1][0])/2;offset=2*axis/config['resolution_m']-1
 require(abs(offset-round(offset))<1e-9,'Reflection does not preserve the cell-center lattice')
 offset=round(offset);require(abs(axis-8.1)<1e-9 and offset==80,'Unexpected P00 reflection')
 def reflect(q):
  r,c,h=q;return [r,offset-c,{0:0,1:3,2:2,3:1}[h]]
 require(reflect(parent['anchor'])==parent['anchor'],'Anchor/heading is not reflection fixed')
 answer={}
 for source,target in (('A','B'),('B','A')):
  old=original[source];name=target+'_mirror'+source
  require(old['actions'][:234]==old['prefix_actions'],'Prefix/action mismatch')
  require(own_states(parent['anchor'],old['actions'])==old['states'],'Original full state mismatch')
  suffix=old['actions'][234:]
  require(suffix==old['continuation_actions']+old['return_actions'],'Incomplete original suffix')
  mapped=[{'forward':'forward','left':'right','right':'left'}[x] for x in suffix]
  suffix_states=[reflect(q) for q in old['states'][234:]]
  require(own_states(parent['anchor'],mapped)==suffix_states,'Mirror actions disagree with mirror states')
  actions=old['prefix_actions']+mapped;states=old['prefix_states']+suffix_states[1:]
  require(own_states(parent['anchor'],actions)==states,'Full independent interpreter mismatch')
  views=[reflect(q) for q in old['selected_view_states']]
  arrivals=[next(i for i,q in enumerate(states) if i>234 and q==v) for v in views]
  require(arrivals==old['arrival_frame_indices'] and len(set(arrivals))==2,'Distinct paid first arrivals changed')
  expected=354 if source=='A' else 340
  require(len(actions)==expected and expected<=400,'Mirrored action count/budget changed')
  split=len(old['continuation_actions']);frame_schedule=[{**row,'action':actions[i]} for i,row in enumerate(old['frame_schedule'])]
  require(len(frame_schedule)==len(actions),'Frame schedule length')
  answer[name]=dict(option=name,source_option=source,target_role=target,target_asset_id='AB'.index(target),
   paired_family='A_original__B_mirrorA' if source=='A' else 'A_mirrorB__B_original',
   prefix_actions=old['prefix_actions'],prefix_states=old['prefix_states'],
   continuation_actions=mapped[:split],continuation_states=suffix_states[:split+1],
   return_actions=mapped[split:],return_states=suffix_states[split:],
   actions=actions,states=states,frame_schedule=frame_schedule,actions_sha256=digest(actions),states_sha256=digest(states),
   selected_view_states=views,arrival_frame_indices=arrivals,paid_total=len(actions),paid_prefix_actions=234,
   supplemental_paid_actions=len(mapped),total_budget=400,shared_body_center_xy_m=original[target]['shared_body_center_xy_m'],
   planned_leg_costs=old['planned_leg_costs'],acquisition_contract=old['acquisition_contract'],
   original_source_actions_sha256=old['actions_sha256'],actual_execution=False,
   autonomous_policy=False,new_quality_guarantee=False,scripted_development_arm=True,
   construction='complete frozen suffix reflection; prefix unchanged; no search/reorder/truncation')
 return answer,dict(axis_x_m=axis,column_offset=offset,column_transform='c_prime = 80 - c',
  heading_transform={'0':0,'1':3,'2':2,'3':1},prefix_mirrored=False)

def validate_case(world,route,index):
 require(tuple(world.shape)==(50,80),'Unexpected P00 map shape')
 require(world.prefix_proposal['actions']==route['prefix_actions'] and world.prefix_proposal['states']==route['prefix_states'],'World prefix differs from catalog')
 expected=own_states(route['states'][0],route['actions']);require(expected==route['states'],'Own interpreter state mismatch')
 for i,(r,c,h) in enumerate(expected):
  require(0<=r<world.shape[0] and 0<=c<world.shape[1] and h in range(4),f'{world.assignment}/{route["option"]}: outside map/heading at {i}')
  require(not bool(world._blocked[r,c]),f'{world.assignment}/{route["option"]}: inflated obstacle at {i} {[r,c,h]}')
  require(bool(world.reachable[r,c]),f'{world.assignment}/{route["option"]}: unreachable state at {i} {[r,c,h]}')
 anchor=[*world.start,world.start_heading]
 require(expected[0]==expected[234]==expected[-1]==anchor,'Exact anchor/heading failure')
 require(world.step_count==0 and world.collisions==0,'World was physically executed')
 return dict(index=index,parent=world.parent,assignment=world.assignment,option=route['option'],
  target_role=route['target_role'],target_is_complex=world.objects[route['target_asset_id']]['category']==3,
  safety_passed=True,independent_action_interpretation_passed=True,states_checked=len(expected),
  paid_actions_if_executed=len(route['actions']),budget=400,budget_fits=True,
  start_prefix_end_and_final_anchor=anchor,arrival_frame_indices=route['arrival_frame_indices'],
  views=[dict(state=s,first_paid_frame=f,physically_executed=False) for s,f in zip(route['selected_view_states'],route['arrival_frame_indices'])],
  actions_sha256=route['actions_sha256'],states_sha256=route['states_sha256'],new_physical_actions=0)

def visibility(world,route,np,projected):
 asset=route['target_asset_id'];require(world.objects[asset]['category']==3,'Visibility requires actual complex counterpart')
 primitives=np.asarray(world._solid_primitives,float)
 ids=np.flatnonzero(primitives[:,6]==world.objects[asset]['owner']).tolist()[1:]
 require(len(ids)==2,'Expected two physical external attachments')
 def ray(state,boxes=None):
  COUNTERS['clean_ray_queries']+=1
  require(COUNTERS['clean_ray_queries']<=12,'Clean query scope exceeded')
  COUNTERS['nominal_depth_pixel_queries']+=world.config.width_px*world.config.height_px
  depth,_=projected(world,state,boxes)
  return depth
 rows=[];xy=[]
 for view_index,state in enumerate(route['selected_view_states']):
  x=(state[1]+.5)*world.config.resolution_m;y=(world.shape[0]-state[0]-.5)*world.config.resolution_m;xy.append([x,y])
  full=ray(state);attachments=[]
  for local,primitive_id in enumerate(ids):
   absent=ray(state,np.delete(primitives[:,:6],primitive_id,axis=0))
   own=((full>.15)&(full<=2.)&((absent==0)|(full<absent)))
   count=int(own.sum());interval=None if not count else [float(full[own].min()),float(full[own].max())]
   item=dict(attachment_index=local,primitive_id=primitive_id,true_unique_near_first_hit_pixels=count,
    axial_depth_range_m=interval,passes_16_pixel_contract=count>=16,
    full_depth_sha256=hashlib.sha256(np.ascontiguousarray(full).tobytes()).hexdigest(),
    removed_attachment_depth_sha256=hashlib.sha256(np.ascontiguousarray(absent).tobytes()).hexdigest())
   attachments.append(item)
   if count<16:
    # Include the actual failing query before stopping any later check.
    raise GateFailure(json.dumps(dict(option=route['option'],assignment=world.assignment,view=state,failure='fewer_than_16_unique_near_pixels',completed_views=rows,current_attachments=attachments)))
  rows.append(dict(view_index=view_index,state=state,attachments=attachments))
 cx,cy=route['shared_body_center_xy_m'];bearings=[math.degrees(math.atan2(y-cy,x-cx)) for x,y in xy]
 sep=math.dist(*xy);angle=abs((bearings[1]-bearings[0]+180)%360-180)
 require(sep>=.4-1e-9 and angle>=15.-1e-9,'Original two-view distance/bearing contract failed')
 require(world.step_count==0 and world.collisions==0,'Visibility changed physical world counters')
 return dict(option=route['option'],complex_counterpart=world.assignment,views=rows,xy_m=xy,
  separation_m=sep,horizontal_bearings_deg=bearings,horizontal_bearing_difference_deg=angle,
  all_original_acquisition_gates_passed=True,clean_static_only=True,no_noisy_observation_or_Q_guarantee=True,
  simple_counterpart_uses_this_visibility_contract_without_fictional_attachments=True)


def execute(progress):
 inputs=sources_before_import();progress['input_sha256']=inputs
 sources={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__)),str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
 progress['source_sha256']=sources
 capacity(CAP-32*1024)
 OUT.mkdir();progress['created_output']=True
 manifest=dict(status='running',protocol=str(PROTOCOL.relative_to(ROOT)),source_sha256=sources,input_sha256=inputs,
  parent='D25-P00',new_option_count=2,route_case_count=4,new_physical_tasks_authorized=0,
  maximum_world_constructions=2,maximum_clean_ray_queries=12,maximum_seconds=LIMIT_SECONDS,
  output_cap_bytes=CAP,disk_reserve_bytes=RESERVE,post_result_development_control=True)
 write('manifest.json',manifest)
 stream=io.BytesIO()
 with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
  for name in sources:z.writestr(name,(ROOT/name).read_bytes())
 blob=stream.getvalue();capacity(len(blob))
 with (OUT/'sources.zip').open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 # Frozen imports only after full source-chain verification and own-source seal.
 import numpy as np
 from scripts.audit_facility_choice_v25_service_cost import StaticWorld,projected
 class ForbiddenPhysicalWorld(StaticWorld):
  def step(self,*a,**k):COUNTERS['forbidden_step_calls']+=1;raise GateFailure('step forbidden')
  def sense(self,*a,**k):COUNTERS['forbidden_sense_calls']+=1;raise GateFailure('sense forbidden')
  def scan(self,*a,**k):COUNTERS['forbidden_scan_calls']+=1;raise GateFailure('scan forbidden')
 result=read(COST/'result.json');parent=next(p for p in result['parents'] if p['parent']=='D25-P00')
 static=read(STATIC/'result.json');config=next(c for c in static['cases'] if c['parent']=='D25-P00')['original_sensor_config']
 routes,reflection=mirrored_catalog(parent,config);progress['phase']='routes_constructed_without_search'
 write('route_catalog.json',dict(status='fixed_mirrored_routes_check_result_for_validation',parent='D25-P00',reflection=reflection,route_catalog=routes,
  original_options_reference=str((COST/'result.json').relative_to(ROOT))+'#/parents/D25-P00/route_catalog',
  families=[['A_original','B_mirrorA'],['A_mirrorB','B_original']],select_family_using_Q_forbidden=True))
 worlds=[]
 for assignment in ('A_complex_B_simple','A_simple_B_complex'):
  progress.update(phase='construct_world',assignment=assignment)
  COUNTERS['world_constructions']+=1;world=ForbiddenPhysicalWorld('D25-P00',assignment)
  require(world.step_count==0 and world.collisions==0,'Physical action during construction')
  require(world.config.width_m==config['width_m'] and world.config.height_m==config['height_m']
   and world.config.resolution_m==config['resolution_m'],'Frozen dimensions differ')
  worlds.append(world)
 progress['cases']=[];progress['visibility']=[]
 for world in worlds:
  for route in routes.values():
   progress.update(phase='validate_route',assignment=world.assignment,option=route['option'])
   progress['cases'].append(validate_case(world,route,len(progress['cases'])))
   write('partial_result.json',dict(status='partial',cases=progress['cases'],visibility=progress['visibility'],counters=COUNTERS))
 for route in routes.values():
  world=next(w for w in worlds if w.objects[route['target_asset_id']]['category']==3)
  progress.update(phase='check_fixed_view_visibility',assignment=world.assignment,option=route['option'])
  progress['visibility'].append(visibility(world,route,np,projected))
  write('partial_result.json',dict(status='partial',cases=progress['cases'],visibility=progress['visibility'],counters=COUNTERS))
 require(COUNTERS['world_constructions']==2 and COUNTERS['clean_ray_queries']==12,'Unexpected scope counts')
 require(all(w.step_count==0 and w.collisions==0 for w in worlds),'Nonzero world execution counters')
 require(all(COUNTERS[k]==0 for k in ('forbidden_step_calls','forbidden_sense_calls','forbidden_scan_calls')),'Forbidden method attempted')
 verify(inputs);verify(sources)
 final=dict(status='complete_static_validation',all_gates_passed=True,parent='D25-P00',reflection=reflection,
  cases=progress['cases'],visibility=progress['visibility'],counters=COUNTERS,
  input_hashes_rechecked_after_execution=True,new_sources_rechecked=True,
  route_catalog='route_catalog.json',mirror_controls_remove_only_declared_view_and_length_asymmetry=True,
  actual_execution_and_noisy_pairing_verified=False,autonomous_policy_evaluated=False,semantic_benefit_proven=False,
  independent_confirmation=False,new_four_task_matrix_started=False,
  residual_limits=['Common prefix and observation history are not left/right reflected.',
    'Whole-world geometry/noise reflection equivalence is not assumed.',
    'Clean fixed-view attachment visibility does not certify instance separation or Q.',
    'Strong geometry baseline must retain active probing and switching.'])
 write('result.json',final)
 # Validation is authoritative in result.json; avoid a second large catalog temp copy.
 manifest['status']='complete';write('manifest.json',manifest)
 print(json.dumps({'status':final['status'],'all_gates_passed':True,'counters':COUNTERS,'output':str(OUT)}),flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if not args.run:parser.error('Explicit --run required; one fixed output, no automatic repeat')
 require(not OUT.exists(),'Existing success/failure evidence cannot be overwritten or repeated')
 # Pin this sole process to one permitted CPU; library thread caps set above.
 if hasattr(os,'sched_getaffinity'):
  allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(allowed)})
 progress={'created_output':False,'phase':'verify_inputs'};started=perf_counter();code=0
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,stop)
 signal.setitimer(signal.ITIMER_REAL,LIMIT_SECONDS)
 try:execute(progress)
 except BaseException as err:
  signal.setitimer(signal.ITIMER_REAL,0);code=124 if isinstance(err,BoundedStop) else 1
  failure={'status':'bounded_incomplete' if code==124 else 'failed','error':repr(err),'traceback':traceback.format_exc(),
   'progress':{k:v for k,v in progress.items() if k not in ('input_sha256','source_sha256')},'counters':COUNTERS,
   'elapsed_s':perf_counter()-started,'process_id':os.getpid(),'no_retry_or_repair':True}
  try:
   verify(progress.get('input_sha256',{}));verify(progress.get('source_sha256',{}));failure['input_and_new_source_hashes_still_match']=True
  except BaseException as e:failure['input_and_new_source_hashes_still_match']=False;failure['hash_error']=repr(e)
  print(json.dumps(failure,ensure_ascii=False),file=sys.stderr,flush=True)
  if progress['created_output']:
   try:
    for tmp in OUT.glob('*.tmp'):tmp.rename(tmp.with_name('interrupted_'+tmp.name))
    write('failure.json',failure,True);m=read(OUT/'manifest.json');m['status']=failure['status'];write('manifest.json',m,True)
   except BaseException as e:print('Unable to preserve failure receipt within capacity: '+repr(e),file=sys.stderr)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  if progress['created_output']:
   try:
    write('timing.json',dict(elapsed_s=perf_counter()-started,process_id=os.getpid(),exit_code=code,
      cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None),code!=0)
    inv={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    write('artifact_hashes.json',inv,code!=0)
   except BaseException as e:print('Final sealing failed: '+repr(e),file=sys.stderr);code=code or 1
 raise SystemExit(code)
if __name__=='__main__':main()
