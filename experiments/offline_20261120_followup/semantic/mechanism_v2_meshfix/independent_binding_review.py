#!/usr/bin/env python3
"""Independent read-only binding review after all mechanism outcomes exist."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(j):return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def resource(p):return dict(path=str(p),sha256=sha(p))
checks=[];conditions=[];files=[]
parent=read(ROOT/'candidate_attempt_002/result.json');summary=read(ROOT/'audit_attempt_001/summary.json')
assert parent['status']=='completed_native_candidates_both_temperatures_all_six_cues'
assert summary['status']=='complete_all_native_branches_all_twelve_conditions'
for folder in [ROOT,ROOT/'branches_attempt_001/shelf_left',ROOT/'branches_attempt_001/shelf_right',ROOT/'audit_attempt_001']:
 names=['scene_freeze.json','candidate_extraction_freeze_attempt002.json']if folder==ROOT else ['freeze.json']if 'branches' in str(folder)else ['evaluation_freeze.json']
 for name in names:
  p=folder/name;freeze=read(p)
  for path,h in freeze['files_sha256'].items():assert sha(path)==h,path
  files.append(resource(p))
for layout in parent['layouts']:
 name=layout['layout'];folder=ROOT/'candidate_attempt_002'/name
 candidates=read(folder/'fixed_candidates.json');candidate_hash=digest(candidates)
 manifest=read(ROOT/'branches_attempt_001'/name/'manifest.json');assert manifest['all_candidates_complete']
 stages={s['candidate_id']:s for s in manifest['stages']};assert set(stages)=={c['candidate_id']for c in candidates}
 branch={r['candidate_id']:r for r in summary['branches']if r['layout']==name}
 prefix_hashes=set();original_points=None;G_scores={};all_input_hash=set();all_common_hash=set()
 for cid,stage in stages.items():
  for field in ('before_mesh','after_mesh','execution'):
   r=stage[field];assert sha(r['path'])==r['sha256']
  prefix_hashes.add(stage['before_mesh']['sha256']);execution=read(stage['execution']['path'])
  route=next(c for c in candidates if c['candidate_id']==cid)
  assert execution['planned_actions']==len(route['outbound_actions'])
  assert execution['completed_actions']==len(execution['steps'])
  assert execution['endpoint_reached'] and execution['collisions']==0
  assert [s['action']for s in execution['steps']]==route['outbound_actions']
  assert [s['pose']for s in execution['steps']]==route['outbound_states'][1:]
  assert branch[cid]['execution_sha256']==stage['execution']['sha256']
 assert len(prefix_hashes)==1
 for item in layout['conditions']:
  T=int(item['temperature']);cue=item['cue'];path=folder/f'T{T}_{cue}_predictions.json';pred=read(path)
  for field in ('G_winner','S_winner','candidate_sha256','common_score_sha256','observed_input_sha256','temperature','cue'):
   assert pred[field]==item[field],field
  assert pred['candidate_sha256']==candidate_hash
  all_input_hash.add(pred['observed_input_sha256']);all_common_hash.add(pred['common_score_sha256'])
  G=[r['G_total']for r in pred['candidates']]
  if T in G_scores:np.testing.assert_array_equal(G,G_scores[T])
  else:G_scores[T]=G
  if cue=='neutral':np.testing.assert_array_equal(G,[r['S_total']for r in pred['candidates']])
  with np.load(folder/f'T{T}_{cue}_predictions.npz',allow_pickle=False)as a:
   np.testing.assert_array_equal(a['G_points'],a['S_points'])
   if original_points is None:original_points=a['G_points'].copy()
   else:np.testing.assert_array_equal(original_points,a['G_points'])
  evaluated=next(r for r in summary['conditions']if r['layout']==name and r['temperature']==T and r['cue']==cue)
  assert evaluated['prediction_receipt_sha256']==sha(path)
  for mode in ('G','S'):
   selected=evaluated['selected'][mode];cid=pred[mode+'_winner'];assert selected['candidate_id']==cid
   for metric in ('object_net_area_m2','object_new_area_m2','global_net_area_m2','completed_actions','path_length_m','correct_free_coverage_increment','endpoint_reached','collisions'):
    assert selected[metric]==branch[cid][metric],metric
  conditions.append(dict(layout=name,temperature=T,cue=cue,G_winner=pred['G_winner'],S_winner=pred['S_winner']))
 assert len(all_input_hash)==len(all_common_hash)==1
 checks.append(dict(layout=name,branches=len(stages),same_original_candidate_digest=True,same_acquired_bank_all_six_conditions=True,
   same_points_all_cues_temperatures=True,frozen_winners_match_all_prediction_and_evaluation_rows=True,each_branch_native_mesh_and_execution_bound=True))
assert len(conditions)==12 and summary['actual_branches']==8 and summary['actual_actions']==114
out=dict(status='independent_readonly_mechanism_binding_review_passed',reviewer='geometry_replay_runner',checks=checks,
 frozen_sources=files,source=resource(ROOT/'audit_attempt_001/summary.json'),candidate_result=resource(ROOT/'candidate_attempt_002/result.json'),
 verified_conditions=conditions,actual_branch_count=8,actual_actions=114,all_endpoints_reached=True,collisions=0,
 limitations=['Only within-layout cue/temperature comparisons share exactly the same acquired geometry. Cross-layout float32 sensor differences are explicitly retained.',
 'Fixed outbound candidate branches are not complete autonomous exploration or natural detector tests.',
 'Positive object area changes can coincide with worse global surface coverage or greater cost; all12 conditions must remain available.'],
 code=resource(Path(__file__).resolve()))
p=ROOT/'audit_attempt_001/independent_binding_review.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(resource(p)))
