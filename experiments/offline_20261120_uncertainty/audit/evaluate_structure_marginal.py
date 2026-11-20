#!/usr/bin/env python3
"""Post-calibration neutral class diagnostic; never selects or changes sigma."""
import json,math
from pathlib import Path
import numpy as np
from evaluate_calibration import read,write,resource,load_resource,prediction_bank,CONDITIONS
from distribution_metrics import joint_posterior
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit/structure_diagnostic'

def aggregate(rows):
    if not rows:return None
    objects=sorted({r['object_id']for r in rows})
    def equal_mean(key):return float(np.mean([np.mean([r[key]for r in rows if r['object_id']==obj])for obj in objects]))
    return dict(objects=len(objects),condition_cases=len(rows),binary_brier=equal_mean('binary_brier'),nll_nats=equal_mean('nll_nats'),
        accuracy_ties_half=equal_mean('accuracy_ties_half'),ties=sum(r['exact_probability_tie']for r in rows),
        fallback_cases=sum(r['fallback']for r in rows),wrong_confidence_above_0_9=sum(r['wrong_high_confidence']for r in rows),
        wrong_high_confidence_case_ids=[r['case_id']for r in rows if r['wrong_high_confidence']],
        nll_clipped_cases=sum(r['true_probability']<1e-15 for r in rows))

def main():
    assert not OUT.exists();OUT.mkdir()
    dp=ROOT/'generator/dataset_manifest.json';ip=ROOT/'calibration/inference_index.json';sp=ROOT/'audit/calibration_attempt_001/selected_scale.json'
    dataset=read(dp);index=read(ip);scale=read(sp);sigma=scale['sigma_m']
    assert sigma==.01 and scale['opened_truth_splits']==['train']
    by={r['case_id']:r for r in index['cases']};rows=[]
    for c in dataset['cases']:
        truth=load_resource(c['truth']);family=truth['construction']['family'];assert family in ('closed','open')
        target=int(family=='open');item=by[c['case_id']];prediction=load_resource(item['prediction']);failure=None
        try:
            track,d,q,e=prediction_bank(prediction);mass=joint_posterior(e,q,sigma).sum(axis=0)
        except Exception as exc:failure=repr(exc);mass=np.array([.5,.5])
        p=float(mass[1]);pt=float(mass[target]);tie=bool(p==.5)
        predicted=int(p>.5);accuracy=.5 if tie else float(predicted==target)
        rows.append(dict(case_id=c['case_id'],object_id=c['object_id'],condition=c['condition'],split=c['split'],true_family=family,
            neutral_shape_probability=mass.tolist(),true_probability=pt,binary_brier=(p-target)**2,nll_nats=-math.log(max(pt,1e-15)),
            accuracy_ties_half=accuracy,exact_probability_tie=tie,wrong_high_confidence=bool(not tie and predicted!=target and max(mass)>.9),
            fallback=failure is not None,failure_reason=failure,prediction=item['prediction'],truth=c['truth']))
    splits={}
    for split in ('train','validation','heldout'):
        subset=[r for r in rows if r['split']==split]
        splits[split]=dict(all_conditions=aggregate(subset),valid_only=aggregate([r for r in subset if not r['fallback']]),
            informative=aggregate([r for r in subset if r['condition'] in ('front_side','around')]),
            by_condition={c:aggregate([r for r in subset if r['condition']==c])for c in CONDITIONS},
            per_object={o:aggregate([r for r in subset if r['object_id']==o])for o in sorted({r['object_id']for r in subset})})
    output=dict(status='post_calibration_diagnostic_no_retuning',sigma_m=sigma,rows=rows,splits=splits,
        neutral_prior_baseline=dict(binary_brier=.25,nll_nats=math.log(2),accuracy_ties_half=.5),
        definitions=['Neutral G class marginal uses frozen joint weights, integrating depth and position.',
            'Binary Brier=(P(shelf)-true_shelf)^2; NLL uses true class probability clipped below1e-15 for numerical reporting only.',
            'Exact50/50 ties receive half accuracy and are counted explicitly; not deterministic class-label wins.',
            'Wrong-high-confidence means wrong unique most-probable class with probability>0.9; descriptive, no tuning or new promotion threshold.',
            'All96cases retained; the same7 geometric/association failures as depth evaluation receive neutral0.5/0.5 evaluator-only fallback.',
            'Equal conditions within object, then equal objects.24independent procedural objects; repeated conditions are not independent objects.',
            'Sigma selected by depthCRPS is not selected for class likelihood; this report does not change it or the existing acceptance criteria.'],
        provenance=dict(dataset=resource(dp),inference=resource(ip),selected_scale=resource(sp),script=resource(Path(__file__))))
    write(OUT/'summary.json',output);print(json.dumps(splits,indent=2))
if __name__=='__main__':main()
