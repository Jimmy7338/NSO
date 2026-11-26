"""Complete frozen-run timing and first divergence from existing logs only."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;B9=HERE.parents[1];B8=B9.parent/'dual_template_uncertainty_20261120';NEW=B9/'structure/rollouts/temperature';OLD=B8/'rollouts/joint_depth'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def attempt(root,layout,arm):return root/'layouts'/layout/'runs'/arm/'attempt_001'
def timing(folder):
 r=read(folder/'result.json');steps=read(folder/'steps.json');a=np.array([s['decision_wall_s']for s in steps])
 assert abs(a.sum()-r['decision_wall_s_total'])<1e-8
 return dict(result=ref(folder/'result.json'),steps=ref(folder/'steps.json'),elapsed_s=r['elapsed_s'],decision_total_s=float(a.sum()),decision_median_s=float(np.median(a)),decision_p95_s=float(np.percentile(a,95)),decision_max_s=float(a.max()),decision_max_step=steps[int(np.argmax(a))]['step'],paid_actions=r['paid_actions'])
def divergence(a,b):
 x=read(a/'steps.json');y=read(b/'steps.json');xc=read(a/'choices.json');yc=read(b/'choices.json')
 first=next((i for i,(u,v)in enumerate(zip(x,y))if(u['action'],u['position'],u['heading'])!=(v['action'],v['position'],v['heading'])),None)
 allsame=first is None and len(x)==len(y)
 common=len(x)if allsame else first
 if common is not None:
  for u,v in zip(x[:common],y[:common]):
   for key in ('packet_sha256','nonsemantic_packet_sha256','owner_sha256'):assert u[key]==v[key]
 byx={256-c['inputs']['remaining_budget']:c for c in xc};byy={256-c['inputs']['remaining_budget']:c for c in yc};cd=None
 for paid in sorted(set(byx)&set(byy)):
  if first is not None and paid>first:break
  u,v=byx[paid],byy[paid]
  if u['outputs']['selected']['pose']!=v['outputs']['selected']['pose']:
   assert u['inputs']==v['inputs'];assert u['outputs']['candidates']==v['outputs']['candidates']
   for key in ('C','N'):np.testing.assert_array_equal(u['outputs']['scores'][key],v['outputs']['scores'][key])
   callsx=read(a/'template_audit.json');callsy=read(b/'template_audit.json')
   cx=next(c for c in callsx if c['mapper_frames']==paid+1);cy=next(c for c in callsy if c['mapper_frames']==paid+1)
   def bank(call):return [{k:o[k]for k in ('support_track','particles','prior_mass','energy_m2','conditional_geometry','fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256')}for o in call['model_audits']['G']['objects']]
   assert bank(cx)==bank(cy)
   cd=dict(paid_actions_before_decision=paid,shared_candidates_and_common_scores_exact=True,shared_acquired_particles_geometry_likelihood_exact=True,
    old_candidate=u['outputs']['selected']['candidate_id'],new_candidate=v['outputs']['selected']['candidate_id'],old_pose=u['outputs']['selected']['pose'],new_pose=v['outputs']['selected']['pose'],
    old_scores=u['outputs']['scores'],new_scores=v['outputs']['scores'],old_mode=u['outputs']['mode'],new_mode=v['outputs']['mode'],
    old_structural_masses={n:[o['shape_probability_mass']for o in cx['model_audits'][n]['objects']]for n in ('G','S')},new_structural_masses={n:[o['shape_probability_mass']for o in cy['model_audits'][n]['objects']]for n in ('G','S')})
   break
 return dict(old_steps=ref(a/'steps.json'),new_steps=ref(b/'steps.json'),trajectory_identical=allsame,common_acquired_action_prefix=common,
  common_sensor_packet_and_owner_prefix_exact=True,first_differing_action_step=None if first is None else x[first]['step'],
  first_step_detail=None if first is None else{n:{k:r[k]for k in ('step','action','position','heading','selected_target_pose')}for n,r in [('old',x[first]),('new',y[first])]},first_differing_selected_target=cd)
def main():
 summary=read(HERE/'summary.json');assert summary['status']=='completed_independent_four_run_evaluation'
 timings={};pairs={}
 for layout in ('shelf_left','shelf_right'):
  timings[layout]={};pairs[layout]={}
  for arm in ('G_temperature','S_temperature'):
   p=attempt(NEW,layout,arm);timings[layout][arm]=timing(p)
   oldarm=arm[0]+'_uncertainty';op=attempt(OLD,layout,oldarm);timings[layout][oldarm]=timing(op)
   pairs[layout][arm+'_vs_T1']=divergence(op,p)
  pairs[layout]['S_vs_G_T16']=divergence(attempt(NEW,layout,'G_temperature'),attempt(NEW,layout,'S_temperature'))
 write(HERE/'timing_and_first_divergence.json',dict(status='complete_existing_log_postprocessing',summary=ref(HERE/'summary.json'),timings=timings,first_divergences=pairs,
  limits=['Elapsed/decision wall times are descriptive, not controlled runtime benchmarks; actor processes shared host resources and concurrency varied.',
   'Library thread limits were1, which does not imply one OS thread/process. Paid actions are algorithmic simulation cost rather than wall time.',
   'First divergence is audited only while the acquired sensor trajectory remains common. Later branches have distinct observed evidence and cannot be isolated as same-state posterior interventions.',
   'Original geometry/detector limitations and dependent development layouts persist; observed branch change does not establish a universal efficiency gain.'],code=ref(Path(__file__))))
 print(json.dumps(dict(status='done',output=ref(HERE/'timing_and_first_divergence.json'))))
if __name__=='__main__':main()
