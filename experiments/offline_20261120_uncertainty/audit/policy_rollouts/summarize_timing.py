#!/usr/bin/env python3
"""Read existing actor timing fields without modifying actors or evaluator."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parents[1];ROOT=STUDY/'rollouts/joint_depth'
ARMS=('G_fixed','S_fixed','G_uncertainty','S_uncertainty','S_mean_collapse')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def stats(values):
    a=np.asarray(values,float)
    return dict(count=len(a),sum_s=float(a.sum()),mean_s=float(a.mean()),median_s=float(np.median(a)),p95_s=float(np.percentile(a,95)),max_s=float(a.max()))if len(a)else None
def main():
    rows=[];pending=[]
    for layout in ('shelf_left','shelf_right'):
        for arm in ARMS:
            p=ROOT/'layouts'/layout/'runs'/arm/'attempt_001';rp=p/'result.json'
            if not rp.exists():pending.append([layout,arm]);continue
            r=read(rp);steps=read(p/'steps.json');assert r['status']=='completed_execution'
            choices=read(p/'choices.json');plan_steps=sorted({r['paid_action_budget']-c['inputs']['remaining_budget']for c in choices})
            assert all(0<=i<=len(steps)for i in plan_steps)
            row=dict(layout=layout,arm=arm,elapsed_s=r['elapsed_s'],decision_wall_s_total=r['decision_wall_s_total'],paid_actions=r['paid_actions'],global_planning_calls=r['global_planning_calls'],
                recorded_action_decision=stats([s['decision_wall_s']for s in steps]),recorded_sensor=stats([s['sensor_wall_s']for s in steps]),recorded_observe=stats([s['observe_wall_s']for s in steps]),
                decision_calls_with_global_replanning=stats([steps[i]['decision_wall_s']for i in plan_steps if i<len(steps)]),
                replanning_calls_without_following_paid_action=sum(i==len(steps)for i in plan_steps),
                terminal_decision_not_attached_to_action_s=r['decision_wall_s_total']-sum(s['decision_wall_s']for s in steps),source_sha256={str(x):sha(x)for x in (rp,p/'steps.json')})
            assert row['terminal_decision_not_attached_to_action_s']>=-1e-8
            rows.append(row)
    result=dict(status='complete_existing_timing_report'if not pending else'partial_existing_timing_report',rows=rows,pending=pending,
        batch_concurrency=3,shared_logical_cpu_count=8,library_threads_per_actor=1,
        interpretation=['Observed wall times from original frozen actor fields, under three concurrent actors sharing eight CPUs plus other processing.',
            'Not controlled single-thread latency, simulator physical duration or real robot travel time; no inferential speedup claim.',
            'Per-action next_local_action timing mixes cheap action dispatch and expensive replanning. Total includes a final terminal call not attached to a paid action.',
            'Replanning-containing call indices are recovered from the original remaining action budget in choices. These are whole decision calls containing replanning, not separately instrumented pure planner CPU time.',
            'Sensor and observation times are separately retained; elapsed additionally includes initialization, output and final export. No actor instrumentation or repeats were added.'],
        script_sha256=sha(__file__))
    (HERE/'timing_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if rows:
        flat=[dict(layout=r['layout'],arm=r['arm'],elapsed_s=r['elapsed_s'],decision_wall_s_total=r['decision_wall_s_total'],global_planning_calls=r['global_planning_calls'],**r['recorded_action_decision'])for r in rows]
        with(HERE/'timing_results.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(completed=len(rows),pending=pending)))
if __name__=='__main__':main()
