"""Produce full-run paper tables and coverage figure from frozen evaluation."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
LAYOUTS=('chain_a','cycle_a','chain_b','cycle_b');ARMS=('S_visibility','S_acquisition')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def esc(s):return s.replace('_',r'\_')
def val(x,digits=2):return '--'if x is None else f'{x:.{digits}f}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=read(a.summary);assert not d['pending'];assert d['design']['planned_runs']==8 and d['design']['layout_units']==4
 a.output.mkdir(parents=True,exist_ok=False);inputs={str(a.summary.resolve()):sha(a.summary)};rows=[]
 for layout in LAYOUTS:
  for arm in ARMS:
   r=d['rows'].get(layout,{}).get(arm)
   if r is None:rows.append(dict(layout=layout,arm=arm,status='failed_or_missing'));continue
   for f,h in r['input_sha256'].items():assert sha(f)==h,f
   row=dict(layout=layout,arm=arm,status='complete',A95=r['A95'],A100=r['A100'],censoring_actions=r['censoring_actions'],known_percent=100*r['final_coverage']['coverage_fraction'],correct_free_percent=100*r['final_coverage']['free_correct_fraction'],false_occupied_percent=100*r['final_coverage']['false_occupied_fraction'],object_weighted_recall_percent=100*r['object_area_weighted_recall_5cm'],object_macro_recall_percent=100*r['object_macro_recall_5cm'],scene_F1_percent=100*r['surface_metrics']['f1_05cm'],path_m=r['path_length_m'],return_to_anchor=r['returned_to_anchor'],paid_actions=r['paid_actions'],collisions=r['collisions'],elapsed_s=r['elapsed_s'],decision_wall_s_total=r['decision_wall_s_total'],global_planning_calls=r['global_planning_calls'],termination=r['termination'])
   for cm,suffix in [(5,'05cm'),(10,'10cm')]:
    for obj in ('open31','closed118'):row[f'{obj}_recall{cm}_percent']=100*r[f'object_recall_{cm}cm'][obj]
    for key in ('precision','recall','f1'):row[f'scene_{key}_{cm}_percent']=100*r['surface_metrics'][f'{key}_{suffix}']
   rows.append(row)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with(a.output/'full_paper_values.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 table=[r'\begin{table}[t]',r'\centering',r'\caption{Complete unknown exploration in four new layouts with a shared 256-action budget including return. V and A denote the visibility and acquisition forecasts with the same category prior, fitted full geometry posterior, and original candidate generator. $C_{\rm free}$ is terminal correct-free coverage; $R_{\rm obj}$ is object recall weighted by fixed visible source area. Recall and scene F1 use a 5\,cm tolerance. A95/A100 are first actions reaching 95\%/100\% correct-free source-reachable coverage. Unreached thresholds remain ``--'' and are censored at executed actions, rather than filled with the budget.}',r'\label{tab:b11-complete-policy}',r'\small',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{llrrrrrrl}',r'\toprule',r'Layout & Arm & $C_{\rm free}$ (\%) & A95 & A100 & $R_{\rm obj}$ (\%) & F1 (\%) & Path (m) & Return \\',r'\midrule']
 for r in rows:
  if r['status']!='complete':table.append(esc(r['layout'])+' & '+('V'if r['arm']==ARMS[0]else'A')+r' & \multicolumn{7}{l}{Failed or incomplete} \\');continue
  cells=[esc(r['layout']),'V'if r['arm']==ARMS[0]else'A',val(r['correct_free_percent']),str(r['A95'])if r['A95']is not None else'--',str(r['A100'])if r['A100']is not None else'--',val(r['object_weighted_recall_percent']),val(r['scene_F1_percent']),val(r['path_m'],1),'Yes'if r['return_to_anchor']else'No'];table.append(' & '.join(cells)+r' \\')
 table += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(a.output/'complete_policy_table.tex').write_text('\n'.join(table)+'\n')
 def detail_table(filename,caption,columns,headers,digits=None):
  lines=[r'\begin{table}[t]',r'\centering',r'\caption{'+caption+'}',r'\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{ll'+('r'*len(columns))+'}',r'\toprule','Layout & Arm & '+' & '.join(headers)+r' \\',r'\midrule']
  for r in rows:
   cells=[esc(r['layout']),'V'if r['arm']==ARMS[0]else'A']
   if r['status']!='complete':cells+=['--']*len(columns)
   else:cells += [val(r[k],(digits or {}).get(k,2))for k in columns]
   lines.append(' & '.join(cells)+r' \\')
  lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(a.output/filename).write_text('\n'.join(lines)+'\n')
 detail_table('complete_object_recall_table.tex','Every prescribed complete-policy object-recall outcome. All entries are percent source-reference recall at the stated distance. Shelf and cabinet refer to the two fixed modeled instances; these are reconstruction completeness measures, not object precision or semantic classification accuracy.',['open31_recall5_percent','open31_recall10_percent','closed118_recall5_percent','closed118_recall10_percent','object_macro_recall_percent'],['Shelf 5\,cm','Shelf 10\,cm','Cabinet 5\,cm','Cabinet 10\,cm','Macro 5\,cm'])
 detail_table('complete_scene_metrics_table.tex','Whole-scene native TSDF metrics for all eight prescribed episodes, in percent. Fixed independently sampled source references and the original native area sampler are used without post-hoc registration.',['scene_precision_5_percent','scene_recall_5_percent','scene_f1_5_percent','scene_precision_10_percent','scene_recall_10_percent','scene_f1_10_percent'],['$P_5$','$R_5$','$F_{1,5}$','$P_{10}$','$R_{10}$','$F_{1,10}$'])
 detail_table('complete_performance_table.tex','Complete recorded execution and computation costs. Decisions include local action requests and global selection. Wall times are measured with up to three concurrent single-library-thread processes; they are not logical action durations, robot travel times, isolated latency, or real-time guarantees.',['paid_actions','global_planning_calls','decision_wall_s_total','elapsed_s','collisions'],['Paid actions','Global calls','Decision wall (s)','Total wall (s)','Collisions'],{'paid_actions':0,'global_planning_calls':0,'decision_wall_s_total':1,'elapsed_s':1,'collisions':0})
 detail_table('complete_coverage_detail_table.tex','Known, correctly free and falsely occupied fractions of the fixed start-reachable robot-free reference cells, in percent. A cell classified occupied is known but does not count as correct free coverage.',['known_percent','correct_free_percent','false_occupied_percent'],['Known','Correct free','False occupied'])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(2,2,figsize=(6.8,4.8),sharex=True,sharey=True);plotrows=[]
 for ax,layout in zip(axes.flat,LAYOUTS):
  for arm,color,style,label in [(ARMS[0],'#335f8c','-','Visibility'),(ARMS[1],'#16938b','--','Acquisition')]:
   r=d['rows'].get(layout,{}).get(arm)
   if r is None:continue
   f=Path(r['attempt'])/'steps.json';inputs[str(f)]=sha(f);trace=read(f);x=[0]+[p['step']for p in trace];y=[r['initial_correct_free_fraction']]+[p['free_correct_fraction']for p in trace];ax.plot(x,np.array(y)*100,color=color,ls=style,lw=1.5,label=label);ax.plot(x[-1],y[-1]*100,'o',ms=3,color=color)
   plotrows += [dict(layout=layout,arm=arm,paid_action=n,correct_free_fraction=c)for n,c in zip(x,y)]
  ax.axhline(95,color='#b7bfc5',lw=.65,ls=':',zorder=0);ax.set_title(layout.replace('_',' '));ax.set_xlim(0,256);ax.set_ylim(0,102);ax.grid(color='#ebedef',lw=.5);ax.set_axisbelow(True)
 for ax in axes[:,0]:ax.set_ylabel('Correct-free coverage (%)')
 for ax in axes[1,:]:ax.set_xlabel('Paid actions (including return)')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',frameon=False,ncol=2,bbox_to_anchor=(.5,1.025));fig.tight_layout()
 for suffix in ('pdf','svg','png'):fig.savefig(a.output/('complete_coverage.'+suffix),bbox_inches='tight',dpi=220)
 plt.close(fig)
 with(a.output/'complete_coverage.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(plotrows[0])if plotrows else['layout','arm','paid_action','correct_free_fraction']);w.writeheader();w.writerows(plotrows)
 complete=[r for r in rows if r['status']=='complete'];full_coverage=sum(r['correct_free_percent']==100 for r in complete);returns=sum(r['return_to_anchor']for r in complete);collisions=sum(r['collisions']for r in complete);pairs=d['paired_differences'];changed=sum(not x['same_trajectory']for x in pairs.values());changes=[100*x['deltas']['scene_F1_5cm']for x in pairs.values()]
 body=[r'\paragraph{Complete Exploration in New Layouts.}',f'The prescribed comparison contains eight episodes in four new layouts. {len(complete)} episodes have complete independently evaluated records; {full_coverage} attain full correct-free coverage, {returns} return to the initial anchor, and {collisions} simulated collisions are recorded. The forecast replacement changes the complete trajectory in {changed} of {len(pairs)} available layout pairs. Table~\\ref{{tab:b11-complete-policy}} reports every episode and unreached threshold; Figure~\\ref{{fig:b11-complete-coverage}} shows the complete action histories.']
 if len(changes)==4:body.append(f'Across the four fixed layout pairs, the mean acquisition-minus-visibility scene-F1 difference is {statistics.mean(changes):+.3f} percentage points, with individual differences ranging from {min(changes):+.3f} to {max(changes):+.3f} points. This is a descriptive paired layout average without stochastic replication or a population-level significance claim.')
 (a.output/'complete_results.tex').write_text('\n'.join(body)+'\n');caption=r'''\begin{figure}[t]
\centering\includegraphics[width=\linewidth]{figures/b11_complete_coverage.pdf}
\caption{Correct-free coverage over the complete paid-action histories of eight prescribed episodes in four new layouts. Both forecasts use the category prior, the full fitted geometry posterior, and the original complete-policy candidates. All action costs, including return, count toward the shared 256-action budget. Curves end at the actual recorded termination; they are not extended to imply unexecuted actions. The dotted line denotes 95\% coverage. The four layouts, rather than the eight curves or their per-action samples, are the comparison units.}
\label{fig:b11-complete-coverage}
\end{figure}
''';(a.output/'complete_coverage_caption.tex').write_text(caption)
 receipt=dict(status='complete_policy_paper_materials_built',input_sha256=inputs,source_sha256=sha(__file__),declared_layouts=4,declared_runs=8,completed_runs=len(complete),available_layout_pairs=len(pairs),full_correct_free_episodes=full_coverage,returned_episodes=returns,simulated_collisions=collisions,trajectory_changed_pairs=changed,independent_layouts=4,new_actor_actions=0,files_sha256={p.name:sha(p)for p in a.output.iterdir()if p.is_file()});(a.output/'FULL_PAPER_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
