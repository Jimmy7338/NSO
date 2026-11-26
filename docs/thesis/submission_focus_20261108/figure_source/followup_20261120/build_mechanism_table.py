#!/usr/bin/env python3
"""All twelve controlled mechanism comparisons, including adverse outcomes."""
from pathlib import Path
import hashlib,json,csv
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'semantic/mechanism_v2_meshfix/audit_attempt_001/summary.json';s=json.loads(SOURCE.read_text())
assert s['status']=='complete_all_native_branches_all_twelve_conditions' and s['actual_branches']==8 and len(s['conditions'])==12
rows=[]
for item in s['conditions']:
 g,z=item['selected']['G'],item['selected']['S']
 rows.append(dict(layout=item['layout'],temperature=item['temperature'],cue=item['cue'],G_candidate=g['candidate_id'],S_candidate=z['candidate_id'],
   G_object_net_m2=g['object_net_area_m2'],S_object_net_m2=z['object_net_area_m2'],delta_object_net_m2=z['object_net_area_m2']-g['object_net_area_m2'],
   G_global_net_m2=g['global_net_area_m2'],S_global_net_m2=z['global_net_area_m2'],delta_global_net_m2=z['global_net_area_m2']-g['global_net_area_m2'],
   G_correct_free_gain=g['correct_free_coverage_increment'],S_correct_free_gain=z['correct_free_coverage_increment'],delta_correct_free_pp=100*(z['correct_free_coverage_increment']-g['correct_free_coverage_increment']),
   G_actions=g['completed_actions'],S_actions=z['completed_actions'],delta_actions=z['completed_actions']-g['completed_actions'],
   G_path_m=g['path_length_m'],S_path_m=z['path_length_m'],delta_path_m=z['path_length_m']-g['path_length_m']))
with(OUT/'mechanism_conditions.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'mechanism_all_branches.json').write_text(json.dumps(s['branches'],indent=2)+'\n')
tex=[r'\begin{table}[tbp]',r'\centering\small',r'\caption{All 12 controlled mechanism conditions. Each layout retains its four original candidate routes; eight actual outbound branches (114 actions) supply all outcomes. Columns report S minus G after selecting from the same candidate set. Surface increments use native TSDF distance at 5\,cm against fixed visible source triangles; coverage is correct-free area in percentage points. Within each layout, cue and temperature interventions share exactly the same acquired bank. The two layouts have disclosed float32 rendering differences and are not exact sensor counterfactuals. These are short fixed-route branches, not complete exploration trials.}',r'\label{tab:b9-mechanism}',r'\begin{tabular}{lllrrrrr}',r'\toprule',r'Layout & $T$ & Cue & G$\to$S & $\Delta A_{obj}$ (m$^2$) & $\Delta A_{all}$ (m$^2$) & $\Delta C$ (pp) & $\Delta$ actions \\',r'\midrule']
for r in rows:
 layout='Left'if r['layout']=='shelf_left'else'Right'
 tex.append(f"{layout} & {r['temperature']} & {r['cue'].capitalize()} & {r['G_candidate']} $\\to$ {r['S_candidate']} & {r['delta_object_net_m2']:+.3f} & {r['delta_global_net_m2']:+.3f} & {r['delta_correct_free_pp']:+.2f} & {r['delta_actions']:+d} "+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(OUT/'mechanism_conditions.tex').write_text('\n'.join(tex)+'\n')
def res(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(OUT/'mechanism_table_receipt.json').write_text(json.dumps(dict(status='generated_all_twelve_conditions_no_positive_selection',source=res(SOURCE),generator=res(Path(__file__).resolve()),outputs=[res(OUT/n)for n in ('mechanism_conditions.csv','mechanism_conditions.tex','mechanism_all_branches.json')],independent_input_layouts=2,new_complete_exploration_trials=0,actual_branches=8,actual_actions=114),indent=2)+'\n')
