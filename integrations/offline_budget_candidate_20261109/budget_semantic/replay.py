#!/usr/bin/env python3
"""Offline frozen historical-input budget audit; no ROS, model inference or actuators."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,pathlib,hashlib,types,copy,time,traceback
import numpy as np
ROOT=pathlib.Path('/home/ouc/nso_robot/analyses/budget_semantic_20261109')
SRC=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge')
sys.path.insert(0,str(SRC))
from nso_bridge.joint_engine import JointSession
arm=sys.argv[1];report=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261109')/({'C':'native_timing_C40_20261109_01','S':'matched_SG_20261109_S'}[arm]);out=ROOT/arm;out.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (report/'joint/decisions.jsonl').read_text().splitlines()]
init=copy.deepcopy(rows[0]['request']['params']);init.pop('semantic_configuration',None);init['execution_allowed']=False
cache={}
for p in (report/'joint/semantics').glob('*.json'):
 r=json.loads(p.read_text());cache[r['source_file_sha256']]=(p.with_suffix('.npz'),r)
session=JointSession(**init)
def cached_labels(self,z,meta,sha):
 p,r=cache[sha];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['mask_file_sha256']
 with np.load(p,allow_pickle=False) as n:labels=n['semantic'].copy()
 assert hashlib.sha256(labels.tobytes()).hexdigest()==r['canonical_semantic_sha256']
 self._instance_owner_map=None
 return labels
session._semantic_labels=types.MethodType(cached_labels,session)
result={'arm':arm,'report':str(report),'cached_semantics':True,'source_sha256':{},'baseline':[],'states':[],'status':'running'}
for p in [pathlib.Path(__file__),SRC/'nso_bridge/joint_engine.py',SRC/'nso_bridge/exploration_joint_policy.py',SRC/'joint_vendor/nso/cpu_four_modules_v10.py',report/'joint/decisions.jsonl']:
 result['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def save(): (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def compare(x,y,path=''):
 diffs=[]
 if isinstance(x,dict) and isinstance(y,dict):
  if x.keys()!=y.keys():diffs.append(path+'.keys')
  for k in x.keys()&y.keys():diffs+=compare(x[k],y[k],path+'.'+k)
 elif isinstance(x,list) and isinstance(y,list):
  if len(x)!=len(y):diffs.append(path+'.length')
  for i,(a,b) in enumerate(zip(x,y)):diffs+=compare(a,b,path+f'[{i}]')
 elif isinstance(x,(float,int)) and isinstance(y,(float,int)):
  if not np.isclose(x,y,rtol=1e-8,atol=1e-8):diffs.append(path)
 elif x!=y:diffs.append(path)
 return diffs
try:
 for record in rows[1:]:
  method=record['request']['method'];params=copy.deepcopy(record['request']['params']);expected=record['response']['result']
  if method=='observe':
   params['path']=str(report/'joint_inputs'/pathlib.Path(params['path']).name);actual=session.observe(**params);aid=params['action_id']
   if aid in (0,20):
    s=session.backend.scenes[0];saved={k:copy.deepcopy(s.get(k)) for k in ('selected','plans','last_selection')};ncalls=len(session.backend.calls)
    state={'paid':aid,'budget_comparisons':[],'measured_labels':actual['summary']['semantic_diagnostics']['quality_labels']}
    for budget in (40,80,96):
     s['ledger'].total_budget=budget
     session.backend.select_target(0);selection=copy.deepcopy(s['last_selection']);(out/f'paid{aid}_budget{budget}_selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n')
     scores=selection['scores'];routes=selection['candidates'];audit=selection['candidate_audit']
     tops={}
     for mode in ('C','G','S'):
      i=min(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
      tops[mode]=None if i is None else {'id':routes[i]['candidate_id'],'score':scores[mode][i],'pose':routes[i]['pose'],'role':routes[i]['group'],'cost':routes[i]['cost']}
     state['budget_comparisons'].append({'total_budget':budget,'remaining_budget':s['ledger'].remaining_budget,'feasible_states':audit['feasible_states'],'candidate_count':len(routes),'roles':[r['group'] for r in routes],'top':tops,'G_equals_S':scores['G']==scores['S'],'target_audit':audit['target_audit'],'all_aperture_zero':all(all(x==0 for x in r['incremental_aperture']) for r in selection['score_audit'])})
     print(arm,aid,budget,audit['feasible_states'],len(routes),tops,flush=True)
     for k,v in saved.items():s[k]=copy.deepcopy(v)
     del session.backend.calls[ncalls:]
    s['ledger'].total_budget=40;result['states'].append(state);save()
   if aid==20:
    # Check actual historical decision from exactly this state before stopping.
    pass
  elif method=='next_action':
   actual=session.next_action();diffs=compare(expected.get('selection'),actual.get('selection'),'selection')
   if expected.get('action')!=actual.get('action'):diffs.append('action')
   result['baseline'].append({'request_id':record['request']['id'],'paid':session.last_packet.action_id,'diff_count':len(diffs),'diffs':diffs[:30]})
   save()
   if diffs:raise RuntimeError('baseline diverged '+str(diffs[:10]))
   if session.last_packet.action_id==20:break
  elif method=='shutdown':break
 result['status']='matched_baseline_to_paid20';save()
except Exception as e:
 result['status']='error';result['error']=traceback.format_exc();save();raise
