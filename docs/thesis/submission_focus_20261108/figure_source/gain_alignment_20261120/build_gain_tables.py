"""Generate draft measured-result tables; never change a manuscript or actor."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
INPUTS={}
def read(p):INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def table(path,caption,label,columns,header,rows):
 text='\\begin{table}[!htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
 text+='\n'.join(' & '.join(row)+' \\\\'for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n';path.write_text(text)
def main():
 replay=read(ROOT/'policy_audit/independent_replay_review.json');rows=[]
 for arm in ('G','S'):
  a=replay['aggregate_equal_state_mean'][arm]
  for mode in ('old','new'):
   rows.append([arm,'History'if mode=='old'else'History + native',*[f'{a[mode+key]:.4f}'for key in ('_MAE_object_gross_m2','_MAE_object_net_m2','_rho_object_gross','_rho_object_net')]])
 table(HERE/'gain_replay_table.tex','All 66 reused candidate branches at six development prefixes. Errors compare predicted object area with actual object gross or net newly reconstructed area at 5\\,cm; candidates are averaged within each prefix, then prefixes equally. Smaller MAE and larger rank correlation are preferable. All twelve prefix--policy winners remain unchanged.','tab:gain-replay','llrrrr','Prior & Consumption & Gross MAE & Net MAE & Gross $\\rho$ & Net $\\rho$',rows)
 original=read(ROOT/'geometry_audit/independent_validation/serialization_retry_002/result.json')
 alternatives=read(ROOT/'geometry_audit/replacement_development/result.json')
 assert original['objects_completed']==12 and original['routes_evaluated']==36
 old=[]
 for obj in original['objects']:
  for c in obj['candidates']:old.append(dict(object_id=obj['object_id'],route=c['route'],prediction=c['predicted_before_m2'],actual=c['actual_source_gross_new_m2'],error=c['absolute_error_before_m2']))
 summary={}
 for route in ('all','revisit','side','around'):
  rs=[r for r in old if route=='all'or r['route']==route]
  summary[route]=dict(n=len(rs),mean_absolute_error_m2=float(np.mean([r['error']for r in rs])),mean_signed_error_m2=float(np.mean([r['prediction']-r['actual']for r in rs])))
 rows=[]
 for key,title in [('original','Original'),('A_append','Append native'),('B1_replace_coarse','Replace history'),('B2_replace_fine','Replace + finer quadrature')]:
  s=summary if key=='original'else alternatives['summary'][key]
  rows.append([title,*[f"{s[r]['mean_absolute_error_m2']:.4f}"for r in ('all','revisit','side','around')],f"{s['all']['mean_signed_error_m2']:+.4f}"])
 table(HERE/'gain_fixture_ablation_table.tex','Known-geometry acquisition probes on twelve objects (six closed, six open), each with three fixed routes. MAE and signed error are in m$^2$ against gross newly reconstructed source area. Append native was specified before these observations; replacement variants reuse the same objects as development diagnostics. These are sensor/fusion probes, not inferred-geometry or complete-exploration tests.','tab:gain-fixture-ablation','lrrrrr','Reward accounting & All MAE & Revisit & Side & Around & Signed error',rows)
 policy_path=ROOT/'policy_audit/completed_rollouts/summary.json'
 if policy_path.exists():
  policy=read(policy_path)
  if policy['status']=='completed_independent_four_run_evaluation':
   rows=[]
   for layout in ('shelf_left','shelf_right'):
    for arm in ('G_native_novelty','S_native_novelty'):
     r=policy['rows'][layout][arm];m=r['surface_metrics'];delta=policy['matched_T16_control_differences'][layout][arm]
     rows.append([layout.replace('shelf_','').capitalize(),arm[0],str(r['first_free_correct_action'].get('1.0')),f"{r['path_length_m']:.1f}",f"{100*m['f1_05cm']:.3f}",f"{100*m['target_surface_metrics']['open31']['recall_05cm']:.3f}",f"{100*m['target_surface_metrics']['closed118']['recall_05cm']:.3f}",'Yes'if delta['trajectory_identical']and delta['native_TSDF_mesh_identical']else'No'])
   table(HERE/'native_consumption_policy_table.tex','Four complete 256-action native-consumption runs on the same two development layouts. Full denotes the first action at 100\\% correct-free coverage. F1 and object recalls are percentages at 5\\,cm. The final column tests exact trajectory and native-mesh identity with the corresponding frozen $T=16$ control. Three externally interrupted partial attempts are retained separately and excluded from complete-run rows.','tab:native-consumption-policy','llrrrrrl','Layout & Prior & Full & Path (m) & F1 & Shelf & Cabinet & Same as control',rows)
 (HERE/'table_sources.json').write_text(json.dumps(dict(status='draft_tables_from_completed_evidence',inputs_sha256=INPUTS,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_manuscript_modified=False),indent=2)+'\n')
if __name__=='__main__':main()
