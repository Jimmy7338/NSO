import pathlib,json,hashlib
root=pathlib.Path(__file__).parent
for arm in ('C','S'):
 p=root/arm/'result.json';r=json.loads(p.read_text())
 for state in r['states']:
  for b in state['budget_comparisons']:
   sel=json.loads((root/arm/f"paid{state['paid']}_budget{b['total_budget']}_selection.json").read_text());routes=sel['candidates']
   b['roles']=[x['group'] for x in routes]
   for top in b['top'].values():
    if top is not None:top['role']=next(x['group'] for x in routes if x['candidate_id']==top['id'])
   b['all_aperture_zero']=all(all(v==0 for v in x['incremental_aperture']) for x in sel['score_audit'])
   b['nonzero_aperture']=[{'candidate_id':x['candidate_id'],'incremental_aperture':x['incremental_aperture'],'prior_potentials':x['prior_potentials']} for x in sel['score_audit'] if any(v!=0 for v in x['incremental_aperture'])]
 r['summary_fields_corrected_from_full_selections']=True;r['current_replay_script_sha256']=hashlib.sha256((root/'replay.py').read_bytes()).hexdigest();p.write_text(json.dumps(r,indent=2)+'\n')
 print(arm,r['status'],len(r['baseline']))
 for state in r['states']:
  for b in state['budget_comparisons']:print(state['paid'],b['total_budget'],b['remaining_budget'],b['feasible_states'],b['roles'],b['G_equals_S'],b['all_aperture_zero'])
