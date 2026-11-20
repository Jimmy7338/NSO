"""Independent cost-only amendment audit; never imports the actor or world."""
from pathlib import Path
import hashlib,json,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 override=ROOT/'preparation_v2/prefix_override.json';d=read(override);p=read(ROOT/'PREFIX_PREPARATION_V2.json');inputs={}
 def checked(path,h=None):
  path=Path(path);actual=sha(path)
  if h is not None:assert actual==h,str(path)
  inputs[str(path.resolve())]=actual;return read(path)
 checked(ROOT/'scene/freeze.json',d['scene_freeze_sha256']);checked(ROOT/'PREFIX_PREPARATION_V2.json',d['preparation_protocol_sha256']);checked(ROOT/'preparation_v2/selector_freeze.json',d['selector_freeze_sha256'])
 checked(ROOT/'preparation_v2/finalize_override.py',d['finalizer_sha256']) if False else None
 assert sha(ROOT/'preparation_v2/finalize_override.py')==d['finalizer_sha256']
 inputs[str(ROOT/'preparation_v2/finalize_override.py')]=d['finalizer_sha256']
 assert set(d['overrides'])=={'chain_a','chain_b'} and not d['scene_geometry_modified'] and not d['script_paths_modified'] and not d['cost_band_width_modified']
 assert not any(d[k]for k in ['visibility_or_acquisition_template_gain_evaluated','actual_branch_outcomes_evaluated','candidate_winners_selected','source_reference_surface_metrics_read'])
 original=['forward']*5+['right']+['forward']*10+['left'];rows=[]
 for layout,endx in [('chain_a',7.1),('chain_b',6.5)]:
  full=original+['right']+['forward']*int(round((endx-3.1)/.2))+['left'];entry=d['overrides'][layout]['p1'];n=entry['paid_actions']
  assert entry['actions']==full[:n] and len(original)<n<=len(full)-p['p2_minimum_separation_actions']
  record=checked(entry['preparation_record']['path'],entry['preparation_record']['sha256']);assert record['search_range_inclusive']==[18,len(full)-4]
  attempts=[x for x in d['all_attempts']if x['layout']==layout]
  assert [x['paid_actions']for x in attempts]==list(range(18,n+1))
  for item in attempts:
   raw=checked(item['record']['path'],item['record']['sha256']);assert not raw['gain_called']and not raw['winner_selected']and not raw['branch_executed']
   assert raw['actions']==full[:raw['paid_actions']] and raw['frames']==len(raw['actions'])+1
   band=item['cost_band'];counts={int(k):v for k,v in band['all_outbound_action_costs'].items()}
   candidates=[(-len(ids),low,tuple(ids))for low in sorted(set(counts.values()))for ids in [sorted(k for k,v in counts.items()if low<=v<=low+3)]]
   size,low,ids=min(candidates);assert list(ids)==band['selected_pool_ids']and low==band['lower_actions'] and band['upper_actions']==low+3
   assert (len(ids)>=3)==(raw['paid_actions']==n)
   assert not band['selection_used_predictions']and not band['selection_used_future_outcomes']
  assert record['selected']['paid_actions']==n and record['selected']['actions']==entry['actions']
  rows.append(dict(layout=layout,original_p1=17,selected_p1=n,cost_only_states_audited=len(attempts),original_p2=len(full),p2_separation=len(full)-n,candidate_costs=entry['selected_cost_band']['all_outbound_action_costs']))
 checked(d['retained_v1_preflight']['path'],d['retained_v1_preflight']['sha256'])
 assert set(d['retained_states'])=={'chain_a/p2','chain_b/p2','cycle_a/p1','cycle_a/p2','cycle_b/p1','cycle_b/p2'}
 inputs[str(override)]=sha(override);inputs[str(Path(__file__).resolve())]=sha(__file__)
 result=dict(status='independent_cost_only_v2_preparation_review_passed',created_unix_s=time.time(),rows=rows,all_intermediate_states_and_original_failures_retained=True,earliest_eligible_verified=True,original_script_prefix_verified=True,other_six_states_unchanged=True,source_scene_unchanged=True,no_gain_or_future_outcome_used=True,new_actor_actions=0,inputs_sha256=inputs,
  limitations=['Inherited native-candidate helper computes shared common proxies, but selection reads only outbound costs; those proxies were not used for state selection.','Source code and signed records are audited without re-executing sensor acquisition; packet and native hashes are checked by the runtime when each branch context is rebuilt.'])
 (HERE/'preparation_v2_review.json').write_text(json.dumps(result,indent=2)+'\n')
 originalfreeze=HERE/'freeze.json';checked(originalfreeze,'2fa64bb8cd5c2f88fb5c012f8eb4980a058c77be8b0009a410fb261d6eb3507e')
 files=[Path(__file__).resolve(),HERE/'preparation_v2_review.json',ROOT/'PREFIX_PREPARATION_V2.json',override,ROOT/'preparation_v2/selector_freeze.json',originalfreeze]
 freeze=dict(status='frozen_preparation_v2_addendum_before_new_short_predictions_or_outcomes',created_unix_s=time.time(),original_evaluator_sources_unchanged=True,source_scene_freeze_unchanged=True,files_sha256={str(f):sha(f)for f in files})
 (HERE/'preparation_v2_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
 print(json.dumps(dict(status=result['status'],rows=rows,freeze_sha256=sha(HERE/'preparation_v2_freeze.json'))))
if __name__=='__main__':main()
