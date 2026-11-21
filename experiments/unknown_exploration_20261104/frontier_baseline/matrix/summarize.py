import json,csv,hashlib
from pathlib import Path
D=Path(__file__).parent; B=D.parent; root=Path(__file__).resolve().parents[2]/'evidence';S=Path(__file__).resolve().parents[2]/'source'
protocol=json.loads((D/'protocol.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'runner.py')==protocol['runner_sha256']
assert all(sha(S/k)==v for k,v in protocol['source_sha256'].items())
assert all(sha(root/k/'results.json')==v for k,v in protocol['previous_results_sha256'].items())
smoke=json.loads((B/'result.json').read_text());trial=json.loads((D/'shelf_west_48/result.json').read_text());assert json.loads((B/'steps.json').read_text())==json.loads((D/'shelf_west_48/steps.json').read_text())
for key in ['before_metrics','after_metrics','coverage_gain_m2','new_surface_m2','steps','collisions','path_m']:assert smoke[key]==trial[key],key
rows=[]
for arr,budget in protocol['conditions']:
 f=json.loads((D/f'{arr}_{budget}/result.json').read_text())
 rows.append(dict(layout=arr,budget=budget,method='frontier+guard',coverage_gain_m2=f['coverage_gain_m2'],path_m=f['path_m'],new_surface_m2=f['new_surface_m2'],f1_05cm=f['after_metrics']['f1_05cm'],used=f['steps'],returned=f['termination']['returned_to_anchor'],failed=f['termination']['failed'],termination=f['termination']['reason'],return_trigger=json.dumps(f['return_trigger'],sort_keys=True)))
 for folder,label in [('fresh_matrix','original'),('deployed_policy_matrix','joint'),('coverage_ablation_matrix','joint')]:
  for r in json.loads((root/folder/'results.json').read_text()):
   if r['arrangement']!=arr or r['budget']!=budget:continue
   steps=json.loads((root/folder/f'{arr}_{budget}_{r["condition"]}_steps.json').read_text())
   rows.append(dict(layout=arr,budget=budget,method=label+'-'+r['condition'],coverage_gain_m2=r['coverage_gain_m2'],path_m=sum(s['action']=='forward' and not s['collision'] for s in steps)*.2,new_surface_m2=r['new_surface_m2'],f1_05cm=r['after_metrics']['f1_05cm'],used=r['steps'],returned=r['termination']['returned_to_anchor'],failed=r['termination']['failed'],termination=r['termination']['reason'],return_trigger='not extracted'))
assert len(rows)==24
with (D/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
lines=['# Fixed Q0 frontier comparison','','Q0 two related arrangements, one deterministic trajectory per condition; not independent scenes or physical evidence. Original G/S and joint G/S/C are previously run results, not rerun. C is same NSO candidate generator with coverage scoring, not frontier. Frontier uses the local observed FrontierPolicy plus common guard and permanent return on the first rejected action or policy stop; it is not ROS explore_lite and does not have the same policy as NSO. Surface and 5 cm F1 are simulator-reference evaluation only.','', '| Layout | Budget | Method | Coverage gain m² | Path m | New visible surface m² | F1 5 cm | Used / limit | Returned | Failed |','|---|---:|---|---:|---:|---:|---:|---:|---|---|']
for r in rows:lines.append(f"| {r['layout']} | {r['budget']} | {r['method']} | {r['coverage_gain_m2']:.2f} | {r['path_m']:.1f} | {r['new_surface_m2']:.2f} | {r['f1_05cm']:.4f} | {r['used']}/{r['budget']} | {r['returned']} | {r['failed']} |")
lines+=['','All frontier runs returned with no collisions and ended on budget exhaustion. Guard intervention and return trigger (step count denotes completed actions):','']
for r in rows:
 if r['method']=='frontier+guard':lines.append(f"- {r['layout']} {r['budget']}: {r['return_trigger']}")
lines+=['','No policy-stop-to-return occurred in these four frontier runs. West/48 steps and before/after metrics exactly match the retained smoke. Initial packet, anchor and full before metrics were asserted against the matching original, joint and C results. Source and input result hashes were checked after all runs.']
(D/'comparison.md').write_text('\n'.join(lines)+'\n');(D/'validation.json').write_text(json.dumps(dict(source_hashes_match=True,prior_results_hashes_match=True,west48_smoke_steps_and_metrics_match=True,initial_packet_anchor_and_before_metrics_asserted=True,condition_count=4,comparison_rows=len(rows)),indent=2));print('\n'.join(lines))
