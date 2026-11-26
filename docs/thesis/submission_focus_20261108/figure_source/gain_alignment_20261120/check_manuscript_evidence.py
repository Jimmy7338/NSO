"""Independent read-only check of six B10 tables and their completed evidence."""
from pathlib import Path
import csv, hashlib, json, re
import numpy as np
from scipy.stats import spearmanr
B=Path('/home/ouc/nso_robot/analyses/dual_template_gain_alignment_20261120')
P=Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO/docs/thesis/submission_focus_20261108')
HERE=Path(__file__).resolve().parent
sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
 p=B/rel;sources[str(p)]=sha(p);return json.loads(p.read_text())
def vals(name):
 t=(P/name).read_text().split('\\midrule',1)[1].split('\\bottomrule',1)[0]
 rows=[]
 for l in t.splitlines():
  if '&' not in l:continue
  v=[]
  for c in l.removesuffix('\\\\').split('&'):
   c=c.strip()
   if re.fullmatch(r'[+-]?\d+(?:\.\d+)?',c):v.append(c)
  rows.append(v)
 return rows
checks=[]
def verify(name,expected):
 actual=vals(name);assert actual==expected,(name,actual,expected)
 checks.append(dict(table=name,rows=len(actual),numeric_cells=sum(map(len,actual))))
# Original independent six-prefix result, not output of the manuscript builder.
x=read('policy_audit/independent_replay_review.json')
assert x['reused_actual_branches']==66 and x['prediction_rows']==132
assert x['new_actor_actions']==0
expected=[]
for arm in ('G','S'):
 for tag in ('old','new'):
  z=x['aggregate_equal_state_mean'][arm]
  expected.append([f'{z[tag+k]:.4f}' for k in ('_MAE_object_gross_m2','_MAE_object_net_m2','_rho_object_gross','_rho_object_net')])
verify('gain_replay_table.tex',expected)
# Four policies: complete outputs, exact controls, retained failure attempts.
x=read('policy_audit/completed_rollouts/summary.json');assert not x['failures'] and not x['pending']
expected=[];timings=[];identities=[]
for layout in ('shelf_left','shelf_right'):
 for arm in ('G_native_novelty','S_native_novelty'):
  z=x['rows'][layout][arm];d=x['matched_T16_control_differences'][layout][arm]
  assert z['paid_actions']==256 and z['free_correct_fraction']==1 and z['collisions']==0
  assert z['termination']['returned_to_anchor'] and d['trajectory_identical'] and d['native_TSDF_mesh_identical']
  attempt=Path(z['attempt']);native=attempt/'native_tsdf.ply'
  if not native.exists():
   candidates=list(attempt.glob('*.ply'));assert len(candidates)==1,candidates;native=candidates[0]
  assert sha(native)==z['native_TSDF_mesh_sha256']
  m=z['surface_metrics'];expected.append([str(z['first_free_correct_action']['1.0']),f"{z['path_length_m']:.1f}",f"{100*m['f1_05cm']:.3f}",f"{100*m['target_surface_metrics']['open31']['recall_05cm']:.3f}",f"{100*m['target_surface_metrics']['closed118']['recall_05cm']:.3f}"])
  timings.append(z['elapsed_s']);identities.append(dict(layout=layout,arm=arm,trajectory_sha256=z['trajectory_sha256'],native_mesh_sha256=z['native_TSDF_mesh_sha256']))
verify('native_consumption_policy_table.tex',expected)
assert f'{min(timings):.2f}'=='423.65' and f'{max(timings):.2f}'=='500.94'
# Original and replacement fixture metrics recomputed from object/route rows.
original=read('geometry_audit/independent_validation/serialization_retry_002/result.json')
replacements=read('geometry_audit/replacement_development/result.json')
assert len(original['objects'])==len(replacements['objects'])==12
all_old=[dict(route=c['route'],err=c['predicted_before_m2']-c['actual_source_gross_new_m2']) for o in original['objects'] for c in o['candidates']]
def fixture_row(rows):
 assert len(rows)==36
 return [f"{np.mean([abs(r['err']) for r in rows if route=='all' or r['route']==route]):.4f}" for route in ('all','revisit','side','around')]+[f"{np.mean([r['err'] for r in rows]):+.4f}"]
expected=[fixture_row(all_old)]
for mode in ('A_append','B1_replace_coarse','B2_replace_fine'):
 rs=[dict(route=c['route'],err=c['predicted_m2']-c['actual_gross_new_m2']) for o in replacements['objects'] for c in o['variants'][mode]['candidates']]
 expected.append(fixture_row(rs))
verify('gain_fixture_ablation_table.tex',expected)
# All 72 known-geometry routes, with independently sampled source reference.
known={s:read('geometry_audit/forecast_development/'+s+'_attempt_001/result.json') for s in ('development','independent_test')}
kmeans={};expected=[];route_directions={}
for split,x in known.items():
 assert x['independent_objects']==12 and x['dependent_route_rows']==36
 assert x['all_predicted_native_meshes_match_actual_under_known_geometry'] and x['not_fitted_geometry']
 rs=[c for o in x['objects'] for c in o['candidates']];assert len(rs)==36
 for c in rs:assert c['exact_predicted_and_actual_acquisition_mesh']
 for mode in ('fine_original','fine_native','forward'):
  def pred(c):return c['forecast']['predicted_gross_new_m2'] if mode=='forward' else c['matched_'+mode+'_visibility_m2']
  es=[dict(route=c['route'],err=pred(c)-c['actual_source_gross_new_m2']) for c in rs]
  kmeans[split,mode]=float(np.mean([abs(r['err']) for r in es]));expected.append(fixture_row(es))
 diff=np.array([abs(c['forecast']['predicted_gross_new_m2']-c['actual_source_gross_new_m2'])-abs(c['matched_fine_original_visibility_m2']-c['actual_source_gross_new_m2']) for c in rs])
 route_directions[split]=dict(improved=int((diff< -1e-12).sum()),worsened=int((diff>1e-12).sum()),tied=int((abs(diff)<=1e-12).sum()))
 for c in rs:
  if c['route']=='revisit':assert c['actual_source_gross_new_m2']==c['forecast']['predicted_gross_new_m2']==0
verify('acquisition_fixture_table.tex',expected)
assert route_directions['development']==dict(improved=30,worsened=1,tied=5)
assert route_directions['independent_test']==dict(improved=33,worsened=1,tied=2)
# Fitted errors/ranks recomputed from all 88 score rows (44 forecasts, 11 actual branches).
fit=read('policy_audit/fitted_C_attempt_001/summary.json')
assert fit['independent_prefixes']==1 and fit['reused_actual_branches']==11 and fit['unique_forecasts']==44 and fit['comparison_rows']==88
cp=B/'policy_audit/fitted_C_attempt_001/all_candidates.csv';sources[str(cp)]=sha(cp)
with cp.open() as f:rows=list(csv.DictReader(f))
assert len(rows)==88
metrics={};expected=[]
for mode in ('original_T16','fine_original','fine_native','forward'):
 for arm in ('G','S'):
  rs=[r for r in rows if r['mode']==mode and r['arm']==arm];assert len(rs)==11 and len({r['candidate_id'] for r in rs})==11
  pred=np.array([float(r['template_gain_m2']) for r in rs]);gross=np.array([float(r['object_gross_m2']) for r in rs]);net=np.array([float(r['object_net_m2']) for r in rs])
  chosen=[r for r in rs if r['selected']=='True'];assert len(chosen)==1;c=chosen[0]
  met=[float(np.mean(abs(pred-gross))),float(np.mean(abs(pred-net))),float(spearmanr(pred,gross).statistic)];metrics[mode,arm]=met
  assert int(c['candidate_id'])==(0 if mode=='original_T16' and arm=='S' else 8)
  expected.append([f'{met[0]:.4f}',f'{met[1]:.4f}',f'{met[2]:.3f}',c['candidate_id'],f"{float(c['object_net_m2']):.4f}",f"{float(c['global_net_m2']):.3f}",c['outbound_actions']])
verify('acquisition_fitted_table.tex',expected)
expected=[[f'{kmeans["independent_test",mode]:.4f}'] for mode in ('fine_original','forward')]
for mode in ('original_T16','fine_original','fine_native','forward'):
 g=metrics[mode,'G'];s=metrics[mode,'S'];expected.append([f'{g[0]:.4f}',f'{s[0]:.4f}',f'{g[2]:.3f}',f'{s[2]:.3f}'])
verify('acquisition_main_table.tex',expected)
# Table source receipts match science inputs; draft labels do not change source meaning.
for rel,key in [('paper_update/table_sources.json','inputs_sha256'),('paper_update/acquisition_table_sources.json','inputs')]:
 receipt=read(rel)
 for path,h in receipt[key].items():assert sha(Path(path))==h,path
# Historical fitted all44 binding receipt, with an independent second calculation.
review=read('execution_audit/acquisition_forecast/independent_evaluation_review.json')
abstract=(P/'main.tex').read_text().split('\\begin{abstract}',1)[1].split('\\end{abstract}',1)[0]
files=['main.tex','main_discussion_20261120.tex','acquisition_method.tex','acquisition_results.tex','acquisition_supplement_20261120.tex']+[c['table'] for c in checks]
result=dict(status='passed_independent_manuscript_evidence_review',required_corrections=[],review_scope='B10 additions and their integration into unknown-exploration abstract/discussion; scientific evidence review, not an independent visual-layout sign-off',tables=checks,table_rows=sum(c['rows'] for c in checks),numeric_cells=sum(c['numeric_cells'] for c in checks),known_geometry_route_directions=route_directions,complete_policy_identities=identities,abstract_whitespace_words=len(abstract.split()),manuscript_files_sha256={str(P/f):sha(P/f) for f in files},science_inputs_sha256=sources,checker_sha256=sha(Path(__file__)),claims_reviewed=[
 'Unknown coverage remains the primary task; supplied known geometry is restricted to acquisition-operator probes.',
 'Six prefixes and 66 reused branches are not new independent environments; fitted C has one development prefix, 44 hypothetical forecasts and 11 reused actual branches.',
 'Twelve new known-geometry test objects each have three dependent routes; prediction/actual native geometry equality is an oracle consistency result, not shape-estimation accuracy.',
 'Conditional-mean geometry and fine quadrature differ from the original distribution before forward acquisition is substituted; the S0-to8 change cannot be attributed to forward fusion.',
 'Fitted C improves rank relative to matched fine visibility but worsens gross MAE, and the original distribution remains better; no C complete-exploration policy was executed.',
 'Four additive-native complete runs preserve corresponding T16 trajectories/native mesh hashes; three externally interrupted partial attempts are retained separately and excluded.',
 'Full coverage is first achievement within the 256-action return contract, not an early stopping rule.',
 'Original measured map is never fused with hypothetical future depth; original strict-free veto is retained and the forward method replaces measured-surface consumption with before/after native geometry.',
 'Object-only source support, whole-scene support and posterior-weighted template area are explicitly distinguished.',
 'No broad semantic efficiency, physical robot accuracy, natural-detector generalization or real-time performance claim is made.'])
(HERE/'independent_manuscript_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],table_rows=result['table_rows'],numeric_cells=result['numeric_cells'],abstract_words=result['abstract_whitespace_words'],receipt_sha256=sha(HERE/'independent_manuscript_review.json')),indent=2))
