#!/usr/bin/env python3
"""Frozen independent structural-temperature test tables, no policy claims."""
from pathlib import Path
import hashlib,json,csv
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'structure/audit/summary.json';s=json.loads(SOURCE.read_text())
assert s['status']=='completed_independent_structural_temperature_calibration_and_test'
assert s['temperature']==16 and s['test_objects']==24
methods=s['summary']['test'];rows=[]
for split, variants in s['summary'].items():
 for variant,modes in variants.items():
  for mode,data in modes.items():
   for group,v in [('all_conditions',data['all_conditions'])]+list(data['by_condition'].items()):
    rows.append(dict(split=split,variant=variant,mode=mode,group=group,**v))
with(OUT/'structure_temperature.csv').open('w')as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tex=[r'\begin{table}[tbp]',r'\centering\small',r'\caption{Structural-temperature calibration on 24 held-out procedural objects (96 dependent observations), with temperature selected on 24 separate objects. G has a neutral category prior; S receives a correct 9:1 category prior as a controlled diagnostic, not a detector output. Exact class ties count one half in accuracy. All eight statistical fallbacks remain in the denominator. Depth columns describe changed marginal distributions, not the separate interval correction.}',r'\label{tab:b9-structure}',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Prior & $T$ & NLL $\downarrow$ & Brier $\downarrow$ & Acc. (\%) & Wrong $>0.9$ & CRPS (m) & MAE (m) \\',r'\midrule']
for variant,modes in methods.items():
 for mode,data in modes.items():
  m=data['all_conditions'];t='1'if variant=='T1'else'16';label='G neutral'if mode=='G'else'S correct cue'
  tex.append(f"{label} & {t} & {m['nll']:.3f} & {m['binary_brier']:.3f} & {100*m['accuracy']:.2f} & {m['high_confidence_wrong_cases']} & {m['marginal_depth_crps_m']:.4f} & {m['marginal_depth_median_MAE_m']:.4f} "+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(OUT/'structure_temperature.tex').write_text('\n'.join(tex)+'\n')
labels={'front':'Front','front_side':'Front + side','around':'Around','occluded':'Occluded'}
tex=[r'\begin{table}[tbp]',r'\centering\small',r'\caption{Neutral structural predictions by observation condition on the same 24 test objects. Every row retains 24 dependent condition cases. Temperature improves aggregate NLL while worsening the already informative around-view condition.}',r'\label{tab:b9-structure-conditions}',r'\begin{tabular}{llrrr}',r'\toprule',r'Condition & $T$ & NLL $\downarrow$ & Brier $\downarrow$ & Acc. (\%) \\',r'\midrule']
for condition,label in labels.items():
 for variant,modes in methods.items():
  m=modes['G']['by_condition'][condition];t='1'if variant=='T1'else'16'
  tex.append(f"{label} & {t} & {m['nll']:.3f} & {m['binary_brier']:.3f} & {100*m['accuracy']:.2f} "+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(OUT/'structure_temperature_conditions.tex').write_text('\n'.join(tex)+'\n')
def resource(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(OUT/'structure_table_receipt.json').write_text(json.dumps(dict(status='generated_from_complete_independent_test',source=resource(SOURCE),generator=resource(Path(__file__).resolve()),outputs=[resource(OUT/n)for n in ('structure_temperature.csv','structure_temperature.tex','structure_temperature_conditions.tex')],policy_results_included=False),indent=2)+'\n')
