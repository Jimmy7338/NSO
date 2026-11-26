from pathlib import Path
import json, hashlib, csv
P=Path(__file__).resolve().parent
B=Path('/home/ouc/nso_robot/analyses')
src=[B/'dual_template_uncertainty_20261120/audit/policy_rollouts/summary.json',B/'dual_template_followup_20261120/structure/policy_audit/summary.json']
old,new=[json.loads(p.read_text()) for p in src]
assert not new['pending'] and not new['failures']
assert len([v for d in new['rows'].values() for v in d.values()])==4
assert all(v['status']=='completed_independently_evaluated' for d in new['rows'].values() for v in d.values())
rows=[]
for layout in ['shelf_left','shelf_right']:
 for temperature,s,arms in [(1,old,['G_uncertainty','S_uncertainty']),(16,new,['G_temperature','S_temperature'])]:
  for arm in arms:
   r=s['rows'][layout][arm];m=r['surface_metrics'];t=m['target_surface_metrics']
   rows.append(dict(layout=layout,temperature=temperature,policy=arm[0],new_run=temperature==16,actions_to_100=r['first_free_correct_action']['1.0'],paid_actions=r['paid_actions'],path_m=r['path_length_m'],coverage_percent=r['free_correct_fraction']*100,F1_percent=m['f1_05cm']*100,shelf_recall_percent=t['open31']['recall_05cm']*100,cabinet_recall_percent=t['closed118']['recall_05cm']*100,collisions=r['collisions'],returned=r['final_pose_equals_initial_anchor'],trajectory_sha256=r['trajectory_sha256'],native_mesh_sha256=r['native_TSDF_mesh_sha256']))
with (P/'temperature_policy_table.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
lines=[r'\begin{table}[!htbp]',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\caption{Complete unknown-exploration comparisons. All runs start at 42.606\% correct-free coverage, execute 256 actions, finish at 100\%, return and record zero simulated collisions. $A_{100}$ is the first full-coverage action. $T=1$ reuses four previously frozen distribution runs; $T=16$ adds four runs on the same two inspected furniture permutations, not new independent scenes. Surface metrics use measured TSDFs at 5\,cm.}',r'\label{tab:temperature-policy}',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Layout & Policy & $T$ & $A_{100}$ & Path (m) & F1 (\%) & Shelf (\%) & Cabinet (\%) \\',r'\midrule']
for i,r in enumerate(rows):
 if i==4:lines.append(r'\midrule')
 lines.append(f"{'Left' if r['layout']=='shelf_left' else 'Right'} & {r['policy']} & {r['temperature']} & {r['actions_to_100']} & {r['path_m']:.1f} & {r['F1_percent']:.3f} & {r['shelf_recall_percent']:.3f} & {r['cabinet_recall_percent']:.3f} \\\\")
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(P/'temperature_policy_table.tex').write_text('\n'.join(lines)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(P/'policy_table_receipt.json').write_text(json.dumps({'status':'complete_four_new_runs_plus_four_frozen_controls','sources':{str(p):sha(p) for p in src},'builder':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},'outputs':{str(P/n):sha(P/n) for n in ['temperature_policy_table.csv','temperature_policy_table.tex']},'rows':8,'new_runs':4,'new_independent_layouts':0},indent=2)+'\n')
