#!/usr/bin/env python3
"""Twelve synthetic checks: independent forward pose/global-bit BFS vs reverse BFS.
No world module import, construction, motion, sensor, fusion, or quality evaluation.
"""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';sys.dont_write_bytecode=True
import ast,collections,hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path('/root/NSO')
TARGET=ROOT/'scripts/audit_facility_choice_v25_service_cost.py'
GEOMETRY=ROOT/'utils/grid_geometry.py'
OUTPUT=ROOT/'audit_results/facility_choice_v25_service_solver_check_20260915'
NAMES=('require','encode','decode','make_graph','validate_route','ServiceReturnSolver')
MOVES={'forward':None,'left':-1,'right':1}
OWN_DIRECTIONS=((-1,0),(0,1),(1,0),(0,-1))

def digest(b):return hashlib.sha256(b).hexdigest()
def capture():
 b=TARGET.read_bytes();tree=ast.parse(b);nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in NAMES}
 assert set(nodes)==set(NAMES)
 return b,tree,nodes,{n:digest(ast.dump(nodes[n],include_attributes=False).encode()) for n in NAMES}

def own_next(grid,pose,action):
 r,c,h=pose
 if action=='forward':
  dr,dc=OWN_DIRECTIONS[h];nr,nc=r+dr,c+dc
  if not (0<=nr<len(grid) and 0<=nc<len(grid[0]) and grid[nr][nc]):return None
  return nr,nc,h
 return r,c,(h+MOVES[action])%4

def oracle(case):
 grid=case['grid'];start=tuple(case['start']);anchor=tuple(case['anchor']);initial=case['initial'];required=case['required'];seen=case['observations']
 if seen.get(start,0)&required&~initial:raise ValueError('missing already-paid current observation')
 origin=(*start,initial);q=collections.deque([origin]);parent={origin:None}
 while q:
  cur=q.popleft();pose=cur[:3];mask=cur[3]
  if pose==anchor and mask&required==required:
   states=[];actions=[];k=cur
   while parent[k] is not None:
    states.append(list(k[:3]));prev,action=parent[k];actions.append(action);k=prev
   states.append(list(start));states.reverse();actions.reverse()
   return {'cost':len(actions),'states':states,'actions':actions,'final_mask':mask,'forward_visited_states':len(parent)}
  for action in MOVES:
   nxt=own_next(grid,pose,action)
   if nxt is None:continue
   node=(*nxt,mask|seen.get(nxt,0))
   if node not in parent:parent[node]=(cur,action);q.append(node)
 return None

def validate_independently(case,route):
 states=route['states'];actions=route['actions'];assert len(states)==len(actions)+1==route['cost']+1
 assert tuple(states[0])==tuple(case['start']) and tuple(states[-1])==tuple(case['anchor'])
 mask=case['initial'];first_done=0 if mask&case['required']==case['required'] else None
 for i,(u,action,v) in enumerate(zip(states,actions,states[1:]),1):
  assert action in MOVES and own_next(case['grid'],tuple(u),action)==tuple(v)
  mask|=case['observations'].get(tuple(v),0)
  if first_done is None and mask&case['required']==case['required']:first_done=i
 assert mask&case['required']==case['required']
 if 'final_template_service_mask' in route:assert route['final_template_service_mask']==mask
 return {'independent_paid_union':mask,'first_service_complete_action':first_done,'exact_heading_return':True,'all_actions_legal':True}

def check_graph(case,graph):
 grid=case['grid'];nr,nc=len(grid),len(grid[0]);expected=np.full((nr*nc*4,3),-1,np.int32)
 def idx(p):return (p[0]*nc+p[1])*4+p[2]
 for r in range(nr):
  for c in range(nc):
   for h in range(4):
    if not grid[r][c]:continue
    for ai,act in enumerate(MOVES):
     nxt=own_next(grid,(r,c,h),act)
     if nxt is not None:expected[idx((r,c,h)),ai]=idx(nxt)
 assert np.array_equal(expected,graph['neighbors'])
 incoming=np.full_like(expected,-1)
 for node in range(len(expected)):
  for action,dest in enumerate(expected[node]):
   if dest>=0:
    assert incoming[dest,action]==-1,'same-action predecessor not unique'
    incoming[dest,action]=node
 assert np.array_equal(incoming,graph['incoming'])
 return int((expected>=0).sum())

def cases():
 full23=[[1,1,1],[1,1,1]];full33=[[1,1,1],[1,1,1],[1,1,1]];isolated=[[1,0,0],[0,0,0]];line=[[1,1,1],[0,0,0]];split=[[1,0,1],[1,0,1],[1,0,1]]
 def c(name,grid,start,anchor,obs,required,initial=0,expected=None):return dict(name=name,grid=grid,start=start,anchor=anchor,observations=obs,required=required,initial=initial,expected=expected)
 return [
  c('already_paid_service_home_zero_cost',full23,(1,0,0),(1,0,0),{(1,0,0):1},1,1,0),
  c('zero_service_still_requires_heading_return',full23,(1,0,1),(1,0,3),{},0,0,2),
  c('turn_observation_and_return_heading',isolated,(0,0,0),(0,0,0),{(0,0,1):1},1,0,2),
  c('corridor_forward_turns_and_return',line,(0,0,1),(0,0,1),{(0,2,1):1},1,0,8),
  c('two_heading_specific_bits_same_cell',isolated,(0,0,0),(0,0,0),{(0,0,2):1,(0,0,3):2},3,0,4),
  c('last_service_bit_on_return_arrival',full23,(0,2,3),(0,0,3),{(0,1,3):1,(0,0,3):2},3,0,2),
  c('paid_prefix_bits_persist_four_required',full33,(2,0,0),(2,0,0),{(2,0,0):3,(0,0,0):4,(0,2,1):8},15,3),
  c('sparse_required_bits_and_irrelevant_observations',full23,(1,0,1),(1,0,1),{(1,1,1):12,(1,2,1):3},5),
  c('blocked_center_detour_no_backward_action',[[1,1,1],[1,0,1],[1,1,1]],(2,0,0),(2,0,0),{(0,2,1):1,(0,2,2):2},3),
  c('disconnected_required_service_unreachable',split,(2,0,0),(2,0,0),{(0,2,0):1},1,0,'unreachable'),
  c('paid_service_but_return_component_unreachable',split,(2,2,0),(2,0,0),{},1,1,'unreachable'),
  c('missing_initial_paid_bit_rejected',full23,(1,0,0),(1,0,0),{(1,0,0):1},1,0,'reject_initial'),
 ]

def main():
 start=time.time();before,tree,nodes,ast_hashes=capture();gb=GEOMETRY.read_bytes();gtree=ast.parse(gb)
 dnode=next(n for n in gtree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DIRECTIONS' for t in n.targets));directions=ast.literal_eval(dnode.value);assert directions==OWN_DIRECTIONS
 assert any(isinstance(n,ast.ImportFrom) and n.module=='utils.grid_geometry' and any(a.name=='DIRECTIONS' for a in n.names) for n in tree.body)
 namespace={'np':np,'deque':collections.deque,'DIRECTIONS':directions,'ACTIONS':tuple(MOVES)}
 # Execute only the exact selected production AST definitions; no imports or top-level execution from the target.
 selected=ast.Module(body=[nodes[n] for n in NAMES],type_ignores=[]);exec(compile(selected,str(TARGET),'exec'),namespace)
 rows=[];failures=[];checks=cases();assert len(checks)==12
 for case in checks:
  try:
   safe=np.asarray(case['grid'],bool);graph=namespace['make_graph'](safe);edges=check_graph(case,graph);masks=np.zeros(safe.size*4,np.uint8)
   for (r,c,h),bits in case['observations'].items():masks[(r*safe.shape[1]+c)*4+h]=bits
   solver=namespace['ServiceReturnSolver'](graph,masks,case['required'],list(case['anchor']))
   if case['expected']=='reject_initial':
    errs=[]
    for call in (lambda:oracle(case),lambda:solver.route(list(case['start']),case['initial'])):
     try:call()
     except ValueError as e:errs.append(str(e))
    assert len(errs)==2,'invalid initial history accepted'
    outcome={'status':'passed','both_reject_missing_paid_initial_bit':True,'errors':errs}
   else:
    expected=oracle(case);actual=solver.route(list(case['start']),case['initial']);assert (expected is None)==(actual is None),'reachability disagreement'
    if expected is None:
     assert case['expected']=='unreachable';outcome={'status':'passed','reachable':False,'forward_cost':None,'reverse_cost':None}
    else:
     assert expected['cost']==actual['cost'],('cost disagreement',expected,actual)
     if isinstance(case['expected'],int):assert actual['cost']==case['expected'],'explicit hand-check cost mismatch'
     own_check=validate_independently(case,expected);actual_check=validate_independently(case,actual)
     outcome={'status':'passed','reachable':True,'forward_cost':expected['cost'],'reverse_cost':actual['cost'],'forward_witness':expected,'reverse_witness':actual,'independent_route_validation':actual_check}
     if case['name']=='last_service_bit_on_return_arrival':assert actual_check['first_service_complete_action']==actual['cost']==2
   outcome.update(name=case['name'],safe_grid=case['grid'],start=case['start'],anchor=case['anchor'],required_mask=case['required'],initial_paid_mask=case['initial'],observations=[{'pose':list(k),'mask':v} for k,v in sorted(case['observations'].items())],verified_graph_edges=edges);rows.append(outcome)
  except BaseException as e:
   failures.append({'name':case['name'],'type':type(e).__name__,'detail':str(e)});rows.append({'name':case['name'],'status':'failed','error':str(e)})
 after,_,_,after_ast=capture();assert ast_hashes==after_ast,'checked function AST changed while running'
 result={'status':'passed' if not failures else 'failed','conditions':12,'passed_conditions':sum(r['status']=='passed' for r in rows),'failures':failures,'scope':'Synthetic 2x3/3x3 safe grids only. Forward BFS on position, heading and uncompressed global acquired mask is independent of DUT graph/search. Compare exact unit-action service+exact-heading return costs and paid-arrival mask union. No world/sensor/Q/TSDF construction or run.','target_file':str(TARGET.relative_to(ROOT)),'target_file_sha256_before':digest(before),'target_file_sha256_after':digest(after),'checked_ast_sha256':ast_hashes,'grid_geometry_sha256':digest(gb),'directions_assignment_ast_sha256':digest(ast.dump(dnode,include_attributes=False).encode()),'formatting_scope':'Hashes use ast.dump(include_attributes=False): whitespace, comments and line-number changes outside semantic AST do not affect the function-level conclusion. File SHA separately records the concrete reviewed version. Changes to literal values, helpers, globals, algorithms or direction convention require renewed review.','not_proved':['Physical world service-mask visibility or no-GT policy input','Actual reconstruction quality or semantic information advantage','Uniform-turn physical collision model beyond supplied safe cells','Global lower bound beyond this specified unit-action graph and required observation-bit contract'], 'elapsed_s':time.time()-start,'cases':rows}
 if not OUTPUT.exists():OUTPUT.mkdir(parents=True)
 assert not (OUTPUT/'result.json').exists()
 data=json.dumps(result,ensure_ascii=False,indent=2).encode();assert len(data)<64*1024
 s=os.statvfs(ROOT);assert s.f_bavail*s.f_frsize>=64*1024**2+len(data)+32768
 with (OUTPUT/'result.json').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 print(json.dumps({k:result[k] for k in ['status','conditions','passed_conditions','failures','elapsed_s','target_file_sha256_before','target_file_sha256_after','checked_ast_sha256']},indent=2))
 for c in rows:print(c['name'],c['status'],c.get('forward_cost'),c.get('reverse_cost'))
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
