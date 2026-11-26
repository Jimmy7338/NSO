#!/usr/bin/env python3
"""Measured B9 interval tables from one frozen held-out evaluation."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'uncertainty/audit/summary.json'
S=json.loads(SOURCE.read_text())
assert S['status']=='completed_independent_report_only_interval_diagnostic'
assert S['units']['test_objects']==24 and not S['new_test_used_for_fit_or_selection']
def res(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
rows=[]
for group,v in S['interval_test'].items():
 for kind,m in v.items():rows.append(dict(group=group,method=kind,**m))
with (OUT/'interval_results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
labels={'all_conditions':'All conditions','condition_front':'Front','condition_front_side':'Front + side','condition_around':'Around','condition_occluded':'Occluded'}
tex=[r'\begin{table}[tbp]',r'\centering\small',r'\caption{Report-only 95\% interval correction on 24 independent test objects, each with four dependent observations. Calibration uses 24 separate objects and the frozen neutral depth predictor. Case coverage and all-four-condition object coverage have distinct denominators. Width and interval score are in metres; lower values are better. The corrected intervals leave the original probability distribution and navigation unchanged.}',r'\label{tab:b9-intervals}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Condition & Interval & Cases covered & Objects covered & Width $\downarrow$ & Score $\downarrow$ \\',r'\midrule']
for group,label in labels.items():
 for kind in ('base','calibrated'):
  m=S['interval_test'][group][kind]
  cases=f"{m['covered_cases']}/{m['cases']}"
  objs=f"{m['simultaneous_covered_objects']}/{m['simultaneous_complete_objects']}" if m['simultaneous_complete_objects'] else '--'
  tex.append(f"{label} & {'Raw' if kind=='base' else 'Corrected'} & {cases} & {objs} & {m['equal_object_mean_width_m']:.3f} & {m['equal_object_mean_interval_score_m']:.3f} "+r'\\')
 if group=='all_conditions':tex.append(r'\midrule')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'interval_results.tex').write_text('\n'.join(tex)+'\n')
prows=[]
for method,record in S['original_distribution_test'].items():
 for group,m in [('all_conditions',record['all_conditions'])]+list(record['by_condition'].items()):
  prows.append(dict(method=method,group=group,objects=m['objects'],cases=m['cases'],crps_m=m['crps_m'],median_absolute_error_m=m['median_absolute_error_m'],coverage95=m['intervals']['0.95']['equal_object_coverage_fraction'],width95_m=m['intervals']['0.95']['equal_object_mean_width_m']))
with (OUT/'interval_original_pmf.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(prows[0]));w.writeheader();w.writerows(prows)
tex=[r'\begin{table}[tbp]',r'\centering\small',r'\caption{Unchanged neutral depth distributions on the same 24 test objects (96 dependent cases). All eight evaluator fallbacks remain in the denominator. CRPS and median error are in metres. The set correction in Table~\ref{tab:b9-intervals} does not change these probability scores.}',r'\label{tab:b9-original-pmf}',r'\begin{tabular}{lrr}',r'\toprule',r'Prediction & CRPS $\downarrow$ & Median error $\downarrow$ \\',r'\midrule']
names={'posterior':'Frozen neutral posterior','prior_only':'Conditional prior only','fixed_0.65':'Fixed 0.65 m'}
for k,v in S['original_distribution_test'].items():
 m=v['all_conditions'];tex.append(f"{names[k]} & {m['crps_m']:.4f} & {m['median_absolute_error_m']:.4f} "+r'\\')
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(OUT/'interval_original_pmf.tex').write_text('\n'.join(tex)+'\n')
outputs=[OUT/n for n in ('interval_results.csv','interval_results.tex','interval_original_pmf.csv','interval_original_pmf.tex')]
(OUT/'interval_table_receipt.json').write_text(json.dumps(dict(status='generated_from_completed_once_frozen_test_evaluation',source=res(SOURCE),generator=res(Path(__file__).resolve()),outputs=[res(p)for p in outputs],independent_objects=24,dependent_cases=96,all_failures_retained=True),indent=2)+'\n')
