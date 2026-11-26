"""Post hoc score diagnostics from fixed recorded candidates, no rescoring policy."""
from pathlib import Path
import argparse,csv,hashlib,json
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def choose(rows,field):return min(rows,key=lambda r:(-r[field],r['cost'],r['candidate_id']))
def main():
 p=argparse.ArgumentParser();p.add_argument('--state-results',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];inputs={}
 for path in a.state_results:
  state=read(path);inputs[str(path.resolve())]=sha(path);prediction=Path(state['prediction_binding']['path']);assert sha(prediction)==state['prediction_binding']['sha256'];inputs[str(prediction)]=sha(prediction);d=read(prediction)
  common=d['conditions']['visibility_S']['rows'];common_winner=choose(common,'common_score');runner_up=sorted(common,key=lambda r:(-r['common_score'],r['cost'],r['candidate_id']))[1];margin=common_winner['common_score']-runner_up['common_score'];actual=state['actual_by_candidate']
  manifest_path=Path(state['input_sha256']and next(k for k in state['input_sha256']if k.endswith('/manifest.json')));manifest=read(manifest_path);inputs[str(manifest_path)]=sha(manifest_path)
  for name,c in d['conditions'].items():
   term=c['rows'];pure=choose(term,'predicted_gain_m2');selected=choose(term,'score');byid={x['candidate_id']:x for x in term}
   margins=[common_winner['common_score']-x['common_score']-(x['predicted_gain_m2']-byid[common_winner['candidate_id']]['predicted_gain_m2'])for x in term if x['candidate_id']!=common_winner['candidate_id']]
   rows.append(dict(layout=state['layout'],prefix_id=state['prefix_id'],condition=name,selected=selected['candidate_id'],common_only_selected=common_winner['candidate_id'],common_margin=margin,completion_min=min(x['predicted_gain_m2']for x in term),completion_max=max(x['predicted_gain_m2']for x in term),minimum_common_winner_score_margin=min(margins),template_only_selected=pure['candidate_id'],template_only_actual_object_gross_m2=actual[str(pure['candidate_id'])].get('object_gross_new_m2'),selected_actual_object_gross_m2=actual[str(selected['candidate_id'])].get('object_gross_new_m2'),state_wall_s=manifest.get('elapsed_s'),engine_initialization_s=d['shared_basis'].get('initialization_seconds'),independent_old_predictor_wall_s=None,independent_new_predictor_wall_s=None,cache_fallbacks=d['shared_basis']['counts'].get('computational_fallbacks')))
 with(a.output/'decomposition.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 receipt=dict(status='posthoc_fixed_candidate_score_decomposition',policy_rerun=False,new_scores_or_weights_applied=False,new_actor_actions=0,rows=rows,input_sha256=inputs,source_sha256=sha(__file__),limits=['Template-only order is a diagnostic on recorded routes, not an executed policy or separate method comparison.','State wall time includes shared prediction, prefix preparation and three actual branches; old/new prediction-only runtimes were not separately instrumented and remain null.','Score margins apply only to the three recorded candidates and the frozen current coefficient.'])
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=receipt['status'],conditions=len(rows),output=str(a.output))))
if __name__=='__main__':main()
