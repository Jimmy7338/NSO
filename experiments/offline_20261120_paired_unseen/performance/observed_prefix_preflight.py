"""Pure observed-bank and cost feasibility audit; no surface gain or future."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'execution'))
import runtime_binding as R
from short_branches import cost_band
rows=[];start=time.monotonic();scene_freeze=json.loads((ROOT/'scene/freeze.json').read_text())
for p,h in scene_freeze['code_and_data_sha256'].items():assert R.sha(p)==h,p
for layout in ['chain_a','chain_b','cycle_a','cycle_b']:
 for prefix in ['p1','p2']:
  t=time.monotonic()
  try:
   c=R.bootstrap_scripted_prefix(layout,prefix);mapper=c['mapper'];bank=R.model.build_particle_bank(mapper);data=bank.materialize('uncertainty');routes,common,audit=R.native_candidates(c);pool,band=cost_band(routes)
   row=dict(layout=layout,prefix=prefix,status='observations_and_costs_prepared',frames=mapper.frames,paid_prefix_actions=len(c['scripted_actions']),fit_object_count=len(bank.objects),fit_support_tracks=[o['support_track']for o in bank.objects],unmodeled_tracks=bank.unmodeled_tracks,full_particle_counts=[len(o['variants']['uncertainty']['particles'])for o in bank.objects],hypothesis_terms=len(data['slices']),quadrature_points=len(data['points']),observed_input_sha256=bank.observed_input_sha256,candidate_count=len(routes),cost_band=band,common_values_not_retained=True,new_surface_predictor_called=False,candidate_winner_selected=False,actual_branch_executed=False,elapsed_s=time.monotonic()-t)
  except BaseException as e:row=dict(layout=layout,prefix=prefix,status='preparation_failed',error=repr(e),traceback=traceback.format_exc(),scene_or_prefix_replaced=False,elapsed_s=time.monotonic()-t)
  rows.append(row);R.write(HERE/'observed_prefix_preflight_progress.json',dict(completed=len(rows),planned=8,rows=rows));print(json.dumps({k:row[k]for k in ['layout','prefix','status']}),flush=True)
for p,h in scene_freeze['code_and_data_sha256'].items():assert R.sha(p)==h,p
R.write(HERE/'observed_prefix_preflight.json',dict(status='complete_eight_prefix_observation_and_cost_preflight',scene_freeze_sha256=R.sha(ROOT/'scene/freeze.json'),source=R.resource(Path(__file__)),rows=rows,completed_states=len(rows),failures=sum(r['status']=='preparation_failed'for r in rows),empty_fitted_banks=sum(r.get('fit_object_count')==0 for r in rows),insufficient_cost_bands=sum(r.get('cost_band',{}).get('status')=='insufficient_cost_matched_routes'for r in rows),new_surface_predictor_called=False,candidate_winner_selected=False,actual_branches_executed=0,scene_or_prefix_changes=0,elapsed_s=time.monotonic()-start))
