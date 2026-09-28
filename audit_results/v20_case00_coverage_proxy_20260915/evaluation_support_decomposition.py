#!/usr/bin/env python3
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import hashlib,json
import numpy as np
SOURCE=Path('/root/NSO/audit_results/v20_case00_coverage_proxy_20260915/analyze_saved_history.py')
code=SOURCE.read_text();observed={'__file__':str(SOURCE),'__name__':'observed_only_no_artifact_writes'}
# The source prefix contains every saved-sensor, ledger and candidate check;
# exclude its final artifact-writing block because the data disk is full.
exec(compile(code.split('from io import StringIO\n')[0],str(SOURCE),'exec'),observed)
from env.facility_documentation_v19 import FacilityWorldV19
from utils.grid_geometry import inflated_obstacles
OUT=SOURCE.parent
before_budget_sha=observed['sha'](OUT/'decisions.json')
def forbidden(*args,**kwargs):raise AssertionError('New sensing/actions forbidden in this audit')
FacilityWorldV19.sense=forbidden;FacilityWorldV19.step=forbidden
case=observed['result']['case']
world=FacilityWorldV19(case['parent'],case['assignment'],sensor_model=case['sensor_model'],noise_seed=case['noise_seed'])
ledger=observed['ledger'];b=observed['mapper'].belief;k=b!=-1
p=ledger.state['possible'];s=ledger.state['safe'];r=world.reachable
confirmed=(ledger.radar_hit_counts>=2)&(b==1)
blocked=inflated_obstacles(confirmed,world.config.robot_radius_m/world.config.resolution_m)
assert world.step_count==0 and world.collisions==0
assert r.shape==p.shape and np.array_equal(world.shape,observed['shape'])
def count(mask):return int(np.count_nonzero(mask))
rn,pn=count(r),count(p);kr,kp=count(k&r),count(k&p)
truth=kr/rn;proxy=kp/pn
assert abs(truth-observed['result']['after']['2026']['coverage_2d'])<1e-15
intersection=r&p;missing=r&~p;extra=p&~r
missing_inflated=missing&blocked;missing_disconnected=missing&~blocked
assert count(missing)==count(missing_inflated)+count(missing_disconnected)
def split(mask):return dict(cells=count(mask),known=count(mask&k),unknown=count(mask&~k),free=count(mask&(b==0)),occupied=count(mask&(b==1)))
result=dict(status='passed',scope='single case00 endpoint, evaluator-only R; observed budgets completed before R was instantiated',
 world_sensing_or_steps_called=False,new_sensor_frames=0,ground_truth_supplied_to_planner=False,
 exact_stored_endpoint_evaluator_coverage=True,observed_budget_audit_sha256=before_budget_sha,
 R_count=rn,P_count=pn,P_minus_R_net_count=pn-rn,known_R=kr,known_P=kp,
 R_intersect_P=split(intersection),R_without_P=split(missing),P_without_R=split(extra),
 R_without_P_inflated_confirmed_obstacle=split(missing_inflated),
 R_without_P_component_exclusion=split(missing_disconnected),
 S_outside_R=split(s&~r),confirmed_observed_occupied_outside_true_occupancy=split(confirmed&~world.occupancy),
 P_is_true_reachable_superset=bool(np.all(~r|p)),
 coverage=dict(evaluator=truth,known_P=proxy,safe_P=count(s)/pn,true_minus_known_P=truth-proxy),
 exact_ordered_decomposition=dict(formula='C_R-C_KP = (known_R-known_P)/R + known_P*(1/R-1/P)',
 numerator_support_difference_contribution=(kr-kp)/rn,
 denominator_difference_contribution=kp*(1/rn-1/pn),
 decomposition_order_dependent_not_causal_estimate=True),
 residual_support_identity=dict(known_R_minus_known_P=kr-kp,
 known_R_without_P=count(k&missing),known_P_without_R=count(k&extra)),
 R_sha256=observed['digest'](r),P_sha256=observed['digest'](p),S_sha256=observed['digest'](s),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 saved_history_script_sha256=observed['sha'](SOURCE),
 saved_history_execution_ends_before_artifact_writes_because_data_disk_full=True)
assert abs(sum([result['exact_ordered_decomposition']['numerator_support_difference_contribution'],
 result['exact_ordered_decomposition']['denominator_difference_contribution']])-(truth-proxy))<1e-15
assert observed['sha'](OUT/'decisions.json')==before_budget_sha
assert all(observed['sha'](observed['ROOT']/n)==v for n,v in observed['sources'].items())
Path('/dev/shm/v20_case00_terminal_support_decomposition.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True),flush=True)
