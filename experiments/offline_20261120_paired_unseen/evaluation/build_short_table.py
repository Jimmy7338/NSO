"""Create staged B11 tables; missing states and failures remain explicit."""
from pathlib import Path
import argparse,csv,hashlib,json
from paired_metrics import summarize_layout
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(v):return '--'if v is None else f'{v:.4f}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--state-results',type=Path,nargs='+',required=True);p.add_argument('--layouts',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 inputs=[];states=[]
 for path in a.state_results:
  d=read(path);assert d['status']=='complete_independent_short_state_evaluation';states.append(d);inputs.append(dict(path=str(path),sha256=sha(path)))
 rows=[];layoutrows=[];priorrows=[]
 for layout in a.layouts:
  group=[s for s in states if s['layout']==layout];seen=[s['prefix_id']for s in group]
  if len(seen)!=len(set(seen)):raise ValueError('Repeated prefix result would duplicate sample')
  for st in sorted(group,key=lambda x:x['prefix_id']):
   main=st['paired_by_prior']['S'];row=dict(layout=layout,topology=st['topology'],prefix_id=st['prefix_id'],old_selected=main['old']['selected_candidate'],new_selected=main['new']['selected_candidate'],old_area_m2=main['old']['selected_observed_area_m2'],new_area_m2=main['new']['selected_observed_area_m2'],best_observed_area_m2=main['old']['best_observed_area_m2'],old_gap_m2=main['old']['observed_gap_m2'],new_gap_m2=main['new']['observed_gap_m2'],delta_m2=main['new_minus_old_selected_area_m2'],unchanged=main['target_unchanged'],best_fully_observed=main['old']['candidate_best_fully_observed'],incomplete_candidates=len(main['old']['incomplete_or_failed_candidates']),old_MAE_m2=main['old']['MAE_m2'],new_MAE_m2=main['new']['MAE_m2']);rows.append(row)
   for estimator,x in st['category_prior_diagnostic'].items():priorrows.append(dict(layout=layout,prefix_id=st['prefix_id'],estimator=estimator,**x))
  summary=summarize_layout([s['paired_by_prior']['S']for s in sorted(group,key=lambda x:x['prefix_id'])]);complete=len(group)==2
  if len(group)>2:raise ValueError('More than two predeclared states')
  layoutrows.append(dict(layout=layout,topology=group[0]['topology']if group else'unknown',both_states_observed=complete,**summary))
 for name,data in [('states.csv',rows),('layouts.csv',layoutrows),('prior_conditions.csv',priorrows)]:
  if data:
   with(a.output/name).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 lines=['| Layout | States | Old selected object area (m²) | New selected object area (m²) | Best of 3 (m²) | Old gap (m²) | New gap (m²) | Unchanged | Failed/incomplete |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 tex=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Layout & Topology & Old & New & Best of3 & Gap old & Gap new & Ties \\',r'\midrule']
 for x in layoutrows:
  vals=[num(x[k])if x['both_states_observed']else'--'for k in ['old_selected_area_m2','new_selected_area_m2','candidate_best_observed_area_m2','old_observed_gap_m2','new_observed_gap_m2']]
  lines.append('| '+ ' | '.join([x['layout'],str(x['states']),*vals,str(x['unchanged_states']),str(x['incomplete_candidate_events'])])+' |')
  tex.append(x['layout'].replace('_',r'\_')+' & '+x['topology']+' & '+' & '.join(vals)+f" & {x['unchanged_states']}/2"+r' \\')
 tex += [r'\bottomrule',r'\end{tabular}'];(a.output/'table.md').write_text('\n'.join(lines)+'\n\nEach layout is the equal mean of its two frozen decision states. The best value is limited to three attempted candidates; incomplete states/branches are retained. Areas are actual source-reference gross-new object surface at5cm, not predicted or whole-scene area.\n');(a.output/'table.tex').write_text('\n'.join(tex)+'\n')
 write(a.output/'summary.json',dict(status='complete_staged_short_table',layouts=a.layouts,complete_layouts=sum(x['both_states_observed']for x in layoutrows),state_results=len(states),unique_actual_branches=sum(len(s['candidate_ids'])for s in states),layout_rows=layoutrows,state_rows=rows,inputs=inputs,source=__file__,source_sha256=sha(__file__),claim='Descriptive matched selection outcomes;dependent states/branches do not increase independent layout count.'))
 print(json.dumps(dict(status='complete_staged_short_table',layouts=len(layoutrows),states=len(states),output=str(a.output))))
if __name__=='__main__':main()
