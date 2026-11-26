"""Prospectively declared v2 cost-only prefix preparation, no gain/outcome."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,traceback,argparse
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT/'execution'))
import runtime_binding as R
from short_branches import cost_band

def run(layout):
 assert layout in('chain_a','chain_b');out=HERE/layout;out.mkdir(exist_ok=False);started=time.monotonic();R.install()
 original_actions=R.scene.sealed_prefix_actions;original1=original_actions(layout,'p1');full=original_actions(layout,'p2');minimum=len(original1)+1;maximum=len(full)-4
 protocol=ROOT/'PREFIX_PREPARATION_V2.json';frozen=json.loads((ROOT/'scene/freeze.json').read_text());inputs=[Path(__file__),protocol,ROOT/'scene/freeze.json',ROOT/'execution/runtime_binding.py',ROOT/'execution/protocol.json',ROOT/'execution/short_branches.py'];hashes={str(p):R.sha(p)for p in inputs}
 for p,h in frozen['code_and_data_sha256'].items():assert R.sha(p)==h,p
 rows=[];selected=None
 for paid in range(minimum,maximum+1):
  begun=time.monotonic();actions=full[:paid]
  def planned_actions(target,prefix_id):
   return list(actions)if target==layout and prefix_id=='p1'else original_actions(target,prefix_id)
  try:
   R.scene.sealed_prefix_actions=planned_actions
   try:context=R.bootstrap_scripted_prefix(layout,'p1')
   finally:R.scene.sealed_prefix_actions=original_actions
   mapper=context['mapper'];bank=R.model.build_particle_bank(mapper);data=bank.materialize('uncertainty');routes,common,_=R.native_candidates(context);pool,band=cost_band(routes)
   assert R.model.observed_input_signature(mapper)==bank.observed_input_sha256
   mesh=mapper.mesh();digest=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
   row=dict(layout=layout,prefix_id='p1',paid_actions=paid,actions=actions,frames=mapper.frames,position=list(context['world'].position),heading=context['world'].heading,packet_sha256=[p.sha256()for p in context['packets']],observed_input_sha256=bank.observed_input_sha256,native_vertices_sha256=digest(np.asarray(mesh.vertices)),native_triangles_sha256=digest(np.asarray(mesh.triangles)),fit_object_count=len(bank.objects),fit_support_tracks=[o['support_track']for o in bank.objects],unmodeled_tracks=bank.unmodeled_tracks,full_particle_counts=[len(o['variants']['uncertainty']['particles'])for o in bank.objects],quadrature_points=len(data['points']),cost_band=band,candidate_count=len(routes),candidate_routes_sha256=hashlib.sha256(json.dumps(routes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),status='eligible_cost_pool'if len(pool)>=3 else'insufficient_cost_pool',gain_called=False,winner_selected=False,branch_executed=False,elapsed_s=time.monotonic()-begun)
  except BaseException as e:row=dict(layout=layout,prefix_id='p1',paid_actions=paid,actions=actions,status='preparation_error',error=repr(e),traceback=traceback.format_exc(),gain_called=False,winner_selected=False,branch_executed=False,elapsed_s=time.monotonic()-begun)
  R.write(out/f'state_{paid:03d}.json',row);rows.append(row);print(json.dumps(dict(layout=layout,paid_actions=paid,status=row['status'])),flush=True)
  if row['status']=='eligible_cost_pool':selected=row;break
 for p,h in hashes.items():assert R.sha(p)==h,('Input changed during cost-only preparation',p)
 for p,h in frozen['code_and_data_sha256'].items():assert R.sha(p)==h,p
 R.write(out/'result.json',dict(status='selected_earliest_cost_feasible_prefix'if selected else'no_eligible_state_in_predeclared_range',layout=layout,original_paid_actions=len(original1),search_range_inclusive=[minimum,maximum],selected=selected,all_attempts=rows,inputs_sha256=hashes,gain_or_future_outcome_used=False,scene_or_script_modified=False,remaining_six_states_unchanged=True,elapsed_s=time.monotonic()-started))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--layout',required=True,choices=['chain_a','chain_b']);run(p.parse_args().layout)
