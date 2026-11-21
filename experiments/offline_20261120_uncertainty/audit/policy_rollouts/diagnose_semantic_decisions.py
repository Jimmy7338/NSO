#!/usr/bin/env python3
"""Distinguish changed class-prior scores from changed executed selections."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parents[1];ROOT=STUDY/'rollouts/joint_depth'
ARMS=('G_fixed','S_fixed','G_uncertainty','S_uncertainty','S_mean_collapse');TOL=1e-12
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def winner(out,name):
    routes=out['candidates'];values=out['scores'][name]
    assert len(routes)==len(values)
    if not routes:return None
    i=min(range(len(routes)),key=lambda i:(-values[i],routes[i]['cost'],routes[i]['candidate_id']))
    return int(routes[i]['candidate_id'])if values[i]>0 else None
def main():
    rows=[];pending=[]
    for layout in ('shelf_left','shelf_right'):
        for arm in ARMS:
            p=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
            if not(p/'result.json').exists():pending.append([layout,arm]);continue
            result=read(p/'result.json');choices=read(p/'choices.json');audits=read(p/'template_audit.json');by_frame={}
            for a in audits:by_frame.setdefault(a['mapper_frames'],[]).append(a)
            plans=[]
            for c in choices:
                out=c['outputs'];g=np.asarray(out['scores']['G']);s=np.asarray(out['scores']['S']);assert g.shape==s.shape
                frame=result['paid_action_budget']-c['inputs']['remaining_budget']+1
                match=[a for a in by_frame.get(frame,[])if [r['candidate_id']for r in a['routes']]==[r['candidate_id']for r in out['candidates']]]
                assert len(match)==1 or not out['candidates']
                template_deltas=[r['S_area_m2']-r['G_area_m2']for r in match[0]['routes']]if match else[]
                gw,sw=winner(out,'G'),winner(out,'S');selected=out['selected'];actual=int(selected['candidate_id'])if selected else None
                assert actual==(sw if arm.startswith('S_')else gw)
                plans.append(dict(plan_number=out['plan_number'],paid_action=frame-1,candidates=len(g),
                    scores_exactly_equal=bool(np.array_equal(g,s)),score_difference_above_tolerance=bool(np.any(abs(s-g)>TOL)),
                    template_difference_above_tolerance=bool(any(abs(v)>TOL for v in template_deltas)),
                    max_absolute_score_difference=float(np.max(abs(s-g))) if len(g) else 0.,
                    max_absolute_template_area_difference_m2=max(map(abs,template_deltas),default=0.),G_winner=gw,S_winner=sw,
                    counterfactual_winner_differs=gw!=sw,executed_selected_candidate=actual))
            rows.append(dict(layout=layout,arm=arm,planning_calls=len(plans),nonempty_planning_calls=sum(p['candidates']>0 for p in plans),
                score_exact_equal_calls=sum(p['scores_exactly_equal']for p in plans),score_nonzero_calls=sum(p['score_difference_above_tolerance']for p in plans),
                score_tiny_nonzero_calls=sum(not p['scores_exactly_equal'] and not p['score_difference_above_tolerance']for p in plans),
                template_nonzero_calls=sum(p['template_difference_above_tolerance']for p in plans),counterfactual_winner_differs_calls=sum(p['counterfactual_winner_differs']for p in plans),
                max_score_difference=max((p['max_absolute_score_difference']for p in plans),default=0.),plans=plans,
                source_sha256={str(x):sha(x)for x in (p/'result.json',p/'choices.json',p/'template_audit.json')}))
    output=dict(status='complete_semantic_decision_diagnostic'if not pending else'partial_semantic_decision_diagnostic',rows=rows,pending=pending,numerical_tolerance=TOL,
        boundaries=['Scores and template gain can differ while the same candidate remains optimal; this is distinct from a non-functioning class prior.',
            'Counterfactual winners are evaluated on each actually acquired prefix with the same original cost/candidate-ID tie rule and non-positive-score handling.',
            'Counterfactual alternative winners are not separately executed and do not prove a better complete policy.',
            'Identical G/S trajectories and native meshes are assessed separately in the main independent evaluation.',
            'Mean-collapse diagnostic G/S each collapse their own joint distribution; raw collapsed point sets may differ as declared.'],script_sha256=sha(__file__))
    (HERE/'semantic_decision_summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(completed=len(rows),pending=pending)))
if __name__=='__main__':main()
