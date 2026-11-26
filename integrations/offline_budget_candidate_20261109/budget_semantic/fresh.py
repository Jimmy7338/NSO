#!/usr/bin/env python3
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,pathlib,json,hashlib,copy
sys.path.insert(0,'/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge')
from nso_bridge.joint_engine import JointSession
root=pathlib.Path('/home/ouc/nso_robot/analyses/budget_semantic_20261109');out=root/'fresh';out.mkdir(exist_ok=True)
p=pathlib.Path('/home/ouc/nso_robot/analyses/reboot_recovery_20261109_02/fresh');config=json.loads((p/'configuration.json').read_text());source=json.loads((p/'source.json').read_text());config['execution_allowed']=False;config['semantic_output']=str(out/'semantics')
s=JointSession(**config);observed=s.observe(path=source['joint_input'],action_id=0,action=None);(out/'observation.json').write_text(json.dumps(observed,indent=2)+'\n')
r={'source':source,'configuration':config,'paid':0,'scope':'new epoch frozen actual packet; one natural frontend inference; observed mapper and original score; no motion','budget_comparisons':[]}
state=s.backend.scenes[0]
for budget in (40,80,96):
 state['ledger'].total_budget=budget;s.backend.select_target(0);sel=copy.deepcopy(state['last_selection']);(out/f'budget{budget}_selection.json').write_text(json.dumps(sel,indent=2,allow_nan=False)+'\n')
 routes=sel['candidates'];scores=sel['scores'];tops={}
 for mode in ('C','G','S'):
  i=min(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
  tops[mode]=None if i is None else {'id':routes[i]['candidate_id'],'score':scores[mode][i],'pose':routes[i]['pose'],'role':routes[i]['group'],'cost':routes[i]['cost']}
 a=sel['candidate_audit'];r['budget_comparisons'].append({'total_budget':budget,'remaining_budget':state['ledger'].remaining_budget,'feasible_states':a['feasible_states'],'candidate_count':len(routes),'roles':[x['group'] for x in routes],'top':tops,'G_equals_S':scores['G']==scores['S'],'target_audit':a['target_audit'],'all_aperture_zero':all(all(x==0 for x in row['incremental_aperture']) for row in sel['score_audit'])})
 print(budget,r['budget_comparisons'][-1],flush=True)
r['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'configuration.json',p/'source.json',pathlib.Path(source['joint_input']),pathlib.Path(__file__))};(out/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
