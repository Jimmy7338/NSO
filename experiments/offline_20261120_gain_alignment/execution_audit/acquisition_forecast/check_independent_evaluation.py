"""Second evaluator check: raw44 arrays,88 rows and original11 actual branches."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
P=HERE/'fitted_attempt_001';E=ROOT/'policy_audit/fitted_C_attempt_001'
B9=ROOT.parent/'dual_template_followup_20261120'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def assertclose(x,y):assert abs(float(x)-float(y))<1e-12,(x,y)
def rank(x):
 x=np.asarray(x);out=np.empty(len(x),float)
 for v in np.unique(x):
  ids=np.flatnonzero(x==v);out[ids]=np.count_nonzero(x<v)+(len(ids)+1)/2
 return out
summary=read(E/'summary.json');pred=read(P/'result.json');cfg=read(HERE/'protocol.json')
for p,h in cfg['files_sha256'].items():assert sha(p)==h,p
assert summary['inputs']['prediction']['sha256']==sha(P/'result.json')
assert summary['unique_forecasts']==44 and summary['reused_actual_branches']==11 and summary['comparison_rows']==88
with (E/'all_candidates.csv').open()as f:csvrows=list(csv.DictReader(f))
assert len(csvrows)==88
old=next(x for x in read(B9/'structure/replay_audit/summary.json')['states']if x['layout']=='shelf_left'and x['state_action']==54)
actual={(a,int(r['candidate_id'])):r for a,d in old['arms'].items()for r in d['all_candidates']};assert len(actual)==22
hyp=[];pairs=set();point_totals=0
for resource in pred['hypothesis_outputs']:
 assert sha(resource['path'])==resource['sha256'];r=read(resource['path']);npz=r['arrays'];assert sha(npz['path'])==npz['sha256']
 assert (r['object_index'],r['shape_id'])not in {(h['object_index'],h['shape_id'])for h in hyp}
 with np.load(npz['path'],allow_pickle=False)as z:
  point_totals+=len(z['points'])
  for x in r['rows']:
   cid=x['candidate_id'];pairs.add((r['object_index'],r['shape_id'],cid))
   gain=z[f'candidate_{cid}_forward_gain'];loss=z[f'candidate_{cid}_forward_loss'];assert np.isfinite(gain).all()and np.isfinite(loss).all()
   assertclose(gain.sum(),x['predicted_gross_new_m2']);assertclose(loss.sum(),x['predicted_lost_m2']);assertclose((gain-loss).sum(),x['predicted_net_new_m2'])
   assert np.all(gain[z['strict_free']]==0)and np.all(gain[~z['native_keep']]==0)
   assert x['forecast_receipt']['past_frames']==55
 hyp.append(r)
assert pairs=={(o,h,c)for o in (0,1)for h in (0,1)for c in range(11)}
checks=0
for groupname,group in summary['groups'].items():
 arm=groupname[-1];mode=groupname[:-2];rows=[r for r in csvrows if r['arm']==arm and r['mode']==mode]
 assert len(rows)==11 and {int(x['candidate_id'])for x in rows}==set(range(11))
 for row in rows:
  cid=int(row['candidate_id']);a=actual[arm,cid];v=a['actual'];assertclose(row['object_gross_m2'],v['actual_object_gross_new_m2']);assertclose(row['object_lost_m2'],v['actual_object_lost_m2']);assertclose(row['object_net_m2'],v['actual_object_net_new_m2'])
  assertclose(row['global_gross_m2'],v['actual_global_gross_new_m2']);assertclose(row['global_net_m2'],v['actual_global_net_new_m2']);assertclose(row['common_score'],a['common_score']);assertclose(row['score'],float(row['template_gain_m2'])+float(row['common_score']))
  ex=a['execution_receipt'];assert sha(ex['path'])==ex['sha256']
  if mode!='original_T16':
   key={'forward':'predicted_gross_new_m2','fine_original':'matched_fine_original_visibility_m2','fine_native':'matched_fine_native_visibility_m2'}[mode]
   expected=sum(h['shape_probability_mass'][arm]*next(y for y in h['rows']if y['candidate_id']==cid)[key]for h in hyp);assertclose(row['template_gain_m2'],expected)
   if mode=='forward':
    loss=sum(h['shape_probability_mass'][arm]*next(y for y in h['rows']if y['candidate_id']==cid)['predicted_lost_m2']for h in hyp);assertclose(row['forecast_lost_m2'],loss);assertclose(row['forecast_net_m2'],expected-loss)
  checks+=1
 x=np.array([float(r['template_gain_m2'])for r in rows]);winner=min(rows,key=lambda r:(-float(r['score']),float(r['cost']),int(r['candidate_id'])))
 assert int(winner['candidate_id'])==group['selected_candidate']
 for target in ('object_gross_m2','object_net_m2','global_gross_m2','global_net_m2','coverage_gain_pp'):
  y=np.array([float(r[target])for r in rows]);rho=float(np.corrcoef(rank(x),rank(y))[0,1])if np.ptp(y)>1e-12 else None
  if rho is not None:assertclose(rho,group['template_rho_vs_'+target])
  assertclose(max(y)-float(winner[target]),group['selected_regret_'+target])
  if target.startswith('object_'):assertclose(np.abs(x-y).mean(),group['MAE_vs_'+target]);assertclose((x-y).mean(),group['bias_vs_'+target])
 if mode=='forward':assertclose(np.mean([abs(float(r['forecast_net_m2'])-float(r['object_net_m2']))for r in rows]),group['forecast_net_MAE_vs_object_net_m2'])
receipt=dict(status='passed_second_readonly_fitted_C_evaluation_review',review_scope='Independent checks of evaluator equations, native-branch binding, all44 prediction combinations and all88 candidate comparison rows; not an independent simulation sample.',hypothesis_candidate_pairs=len(pairs),unique_actual_branches=11,comparison_rows_checked=checks,hypothesis_basis_points=point_totals,groups=8,checks=['All4hypotheses×11candidates retained exactly once; no duplicate or missing combinations','Raw NPZ gross/lost/net andstrict-free/native novelty masks match scalar receipts','Per-object shape probabilities reconstructG/S forecast and matched comparators','All88 source metrics match frozenB9 actual rows; execution receipt SHA matches','MAE andbias denominator11 pergroup; independent average-rank correlation andchosen regret recomputed','Selectedcandidate tie rule uses original total score,cost,ID','Sourceprotocol/input/model hashes remain unchanged'],required_corrections=[],conclusion='C improves ranking againstmatched fine visibility but slightly worsensgross MAE; originalfullT16 remains stronger. Allthreefine/C variants select8, so originalS0-to8 change is not attributable solely toforwardfusion.',sources={str(p):sha(p)for p in (Path(__file__),ROOT/'policy_audit/evaluate_fitted_C.py',ROOT/'policy_audit/diagnose_alignment.py',E/'summary.json',E/'all_candidates.csv',P/'result.json',HERE/'protocol.json',B9/'structure/replay_audit/summary.json')})
q=HERE/'independent_evaluation_review.json';q.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'path':str(q),'sha256':sha(q),'rows':checks},indent=2))
