import json,hashlib
from pathlib import Path
D=Path(__file__).parent;W=D/'execution/G_weight2';B=D.parent/'two_room_diagnostic/frozen_execution/two_room_joint';P=json.loads((D/'portable_protocol.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(D/'runner.py')==P['runner_sha256'] and sha(D/'weight2.py')==P['wrapper_sha256']; data={}
for name,root,suffix in [('G',B,'G'),('S',B,'S'),('G_weight2',W,'G_weight2')]:
 prefix=f'two_room_shelf_west_96_{suffix}';r=json.loads((root/f'{prefix}_result.json').read_text());steps=json.loads((root/f'{prefix}_steps.json').read_text());choices=json.loads((root/f'{prefix}_choices.json').read_text());den=1008*.2**2;cov=[r['before_metrics']['coverage_2d']]+[x['covered_area_m2']/den for x in steps]
 data[name]=dict(result=r,steps=steps,choices=choices,summary=dict(used=len(steps),thresholds={str(x):next((i for i,v in enumerate(cov) if v>=x/100-1e-10),None) for x in (90,95,99)},terminal_coverage=r['after_metrics']['coverage_2d'],coverage_gain_m2=r['coverage_gain_m2'],new_surface_m2=r['new_surface_m2'],f1_05cm=r['after_metrics']['f1_05cm'],path_m=.2*sum(s['action']=='forward' and not s['collision'] for s in steps),collisions=r['collisions'],termination=r['termination'],planning_calls=len(choices)))
a=data['G_weight2'];assert all(a['result']['initial_packet_sha256']==data[x]['result']['initial_packet_sha256'] and a['result']['anchor']==data[x]['result']['anchor'] and a['result']['before_metrics']==data[x]['result']['before_metrics'] for x in ('G','S'))
for call in a['choices']:
 for row,value in zip(call['outputs']['score_audit'],call['outputs']['scores']['G']):
  v=row['posthoc_G_weight2'];assert v['multiplier']==2 and abs(v['common']+2*v['prior_G']-value)<1e-9
pairs={}
for x in ('G','S'):
 b=data[x];first_action=next((i+1 for i,(u,v) in enumerate(zip(a['steps'],b['steps'])) if (u['action'],u['position'],u['heading'])!=(v['action'],v['position'],v['heading'])),None)
 # Compare target choices at matching pre-action step, never pair unequal planning times.
 calls_a={96-c['inputs']['remaining_budget']:c for c in a['choices']};calls_b={96-c['inputs']['remaining_budget']:c for c in b['choices']};common=sorted(set(calls_a)&set(calls_b));different=[]
 for t in common:
  u=calls_a[t]['outputs']['selected'];v=calls_b[t]['outputs']['selected'];pu=None if u is None else u['pose'];pv=None if v is None else v['pose']
  if pu!=pv:different.append(dict(completed_actions=t,weight2_pose=pu,other_pose=pv))
 prefix_equal=all(u['packet_sha256']==v['packet_sha256'] for u,v in zip(a['steps'][:(first_action or min(len(a['steps']),len(b['steps']))) - (1 if first_action else 0)],b['steps'][:(first_action or min(len(a['steps']),len(b['steps']))) - (1 if first_action else 0)]))
 pairs[x]=dict(first_action_difference=first_action,prefix_packet_hashes_equal_before_action_difference=prefix_equal,first_common_time_target_difference=different[0] if different else None,shared_planning_times=common)
res=dict(scope='one post-hoc lambda=2 control, after observing G/S and lambda audit; no tuning, not prospective independent evidence',exact_initial_packet_anchor_and_metrics_match=True,weight2_score_formulas_verified_all_recorded_candidates=True,summaries={k:v['summary'] for k,v in data.items()},pairwise_vs_weight2=pairs,planning_at_action19={k:next((dict(selected=c['outputs']['selected']['pose'] if c['outputs']['selected'] else None,candidate_count=len(c['outputs']['candidates'])) for c in v['choices'] if c['inputs']['remaining_budget']==77),None) for k,v in data.items()});(D/'comparison_recomputed.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
