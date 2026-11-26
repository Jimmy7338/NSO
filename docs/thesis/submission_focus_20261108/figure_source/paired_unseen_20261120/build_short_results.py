"""Render manuscript tables/prose directly from all eight frozen state results."""
from pathlib import Path
import argparse,hashlib,json,statistics
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return '--'if x is None else f'{x:.3f}'
def avg(xs):return statistics.mean(xs)if xs and all(x is not None for x in xs)else None
def esc(s):return s.replace('_',r'\_')
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=read(a.summary)
 assert d['state_results']==8 and d['unique_actual_branches']==24 and d['complete_layouts']==4
 states=[];inputs={str(a.summary.resolve()):sha(a.summary)}
 for ref in d['inputs']:
  f=Path(ref['path']);assert sha(f)==ref['sha256'];s=read(f);states.append(s);inputs[str(f)]=sha(f)
 pairs=[s['paired_by_prior']['S']for s in states];changed=sum(not x['target_unchanged']for x in pairs);deltas=[x['new_minus_old_selected_area_m2']for x in pairs];wins=sum(x is not None and x>1e-12 for x in deltas);losses=sum(x is not None and x< -1e-12 for x in deltas);ties=sum(x is not None and abs(x)<=1e-12 for x in deltas);failed=sum(v['status']!='completed'for s in states for v in s['actual_by_candidate'].values())
 mae_old=avg([x['old']['MAE_m2']for x in pairs]);mae_new=avg([x['new']['MAE_m2']for x in pairs]);gs={mode:sum(s['category_prior_diagnostic'][mode]['target_changed']for s in states)for mode in ('visibility','acquisition')}
 table=[r'\begin{table}[t]',r'\centering',r'\caption{Actual gross-new object surface in four new layouts. Values are the equal mean of two fixed decision states, in m$^2$. Both forecasts share the same fitted full posterior, quadrature, and other score terms. Best denotes the best object-surface outcome among the three measured candidate routes, not the optimum of the joint coverage objective. V denotes visibility and A denotes acquisition. Gap is best minus selected area. Unchanged counts refer to dependent states within each layout.}',r'\label{tab:paired-gain-selected}',r'\small',r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Layout & Topology & V $\uparrow$ & A $\uparrow$ & Best3 & Gap V $\downarrow$ & Gap A $\downarrow$ & Same \\',r'\midrule']
 for row in d['layout_rows']:
  vals=[fmt(row[k])for k in ['old_selected_area_m2','new_selected_area_m2','candidate_best_observed_area_m2','old_observed_gap_m2','new_observed_gap_m2']];table.append(esc(row['layout'])+' & '+row['topology']+' & '+' & '.join(vals)+f" & {row['unchanged_states']}/2"+r' \\')
 table += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(a.output/'selected_area_table.tex').write_text('\n'.join(table)+'\n')
 paragraphs=[r'\paragraph{Paired Selection Outcomes.}']
 if changed==0:
  paragraphs.append(f'The two forecasts select the same route in all eight prescribed states. Their chosen routes therefore produce identical actual object-surface outcomes in this matched diagnostic. Table~\\ref{{tab:paired-gain-selected}} reports the remaining gap to the best measured route. This result provides no selected-route reconstruction advantage for the acquisition forecast in these four layouts.')
 else:
  paragraphs.append(f'The acquisition forecast changes {changed} of eight prescribed route selections. Actual gross-new object area is higher in {wins} states, lower in {losses}, and equal in {ties}; these states are dependent checkpoints from four layouts. Table~\\ref{{tab:paired-gain-selected}} reports the per-layout paired values and gaps to the best of the three measured routes.')
 paragraphs += [r'\paragraph{Forecast Error and Category Prior.}',f'Across the same twenty-four actual branch records, descriptive object-area MAE is {fmt(mae_old)}\\,m$^2$ for visibility and {fmt(mae_new)}\\,m$^2$ for acquisition. Switching only from the neutral to the category-conditioned prior changes {gs["visibility"]} of eight visibility selections and {gs["acquisition"]} of eight acquisition selections. These re-rankings share the recorded routes and actual outcomes; they are not additional complete exploration episodes. The candidate branches do not provide independent scene samples or between-run uncertainty estimates.',r'\paragraph{Completion and Timing.}',f'All twenty-four prescribed branches are represented, with {failed} incomplete or failed branches retained. Each native reconstruction and execution receipt is kept separately. Recorded state wall time combines acquisition-prefix reconstruction, shared forecast computation, and three actual branch executions; prediction-only runtimes of the old and new estimators were not separately measured. Concurrent wall time is not interpreted as an isolated real-time benchmark.']
 (a.output/'short_results.tex').write_text('\n'.join(paragraphs)+'\n')
 supplemental=[r'\begin{table}[t]',r'\centering',r'\caption{Per-state forecast accuracy and prior sensitivity. MAE uses the same three actual gross-new object-surface areas. The neutral/category columns record whether the selected candidate changes; repeated states are not independent layouts.}',r'\label{tab:paired-gain-state-errors}',r'\begin{tabular}{llrrll}',r'\toprule',r'Layout & State & MAE V (m$^2$) & MAE A (m$^2$) & G/S V & G/S A \\',r'\midrule']
 for s in sorted(states,key=lambda x:(x['layout'],x['prefix_id'])):
  x=s['paired_by_prior']['S'];pv=s['category_prior_diagnostic']['visibility'];pa=s['category_prior_diagnostic']['acquisition'];pair=lambda z:f"{z['neutral_selected']}/{z['category_selected']}";supplemental.append(esc(s['layout'])+' & '+s['prefix_id']+' & '+fmt(x['old']['MAE_m2'])+' & '+fmt(x['new']['MAE_m2'])+' & '+pair(pv)+' & '+pair(pa)+r' \\')
 supplemental += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(a.output/'state_errors_table.tex').write_text('\n'.join(supplemental)+'\n')
 receipt=dict(status='all_24_short_branch_paper_materials_built',independent_layouts=4,dependent_states=8,unique_actual_branches=24,changed_states=changed,actual_gain_state_wins=wins,actual_gain_state_losses=losses,actual_gain_state_ties=ties,failed_or_incomplete_branches=failed,old_MAE_m2=mae_old,new_MAE_m2=mae_new,neutral_category_target_changes=gs,input_sha256=inputs,source_sha256=sha(__file__),files_sha256={p.name:sha(p)for p in a.output.iterdir()if p.is_file()},full_policy_results_not_included=True)
 (a.output/'PAPER_SHORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
