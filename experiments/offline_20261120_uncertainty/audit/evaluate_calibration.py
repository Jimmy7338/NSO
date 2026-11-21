#!/usr/bin/env python3
"""Train-only scale selection and split-separated independent depth evaluation.

No actor/generator imports. Only this evaluator opens private geometry.
"""
import argparse,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np
from distribution_metrics import SIGMA_GRID_M,joint_posterior,metrics,projected_truth,equal_object_summary
ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=('front','front_side','around','occluded')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def load_resource(r):
    assert sha(r['path'])==r['sha256'],r['path']
    return read(r['path'])
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def code_receipt():
    return {str(ROOT/'audit'/f):sha(ROOT/'audit'/f) for f in ('distribution_metrics.py','evaluate_calibration.py','CALIBRATION_EVALUATION_SPEC.md')}

def validate_manifest(dataset):
    objects=dataset['objects'];cases=dataset['cases']
    assert len(objects)==24 and len(cases)==96
    oid={x['object_id']:x for x in objects};assert len(oid)==24
    for split,expected in [('train',12),('validation',6),('heldout',6)]:assert sum(x['split']==split for x in objects)==expected
    seen=set()
    for case in cases:
        assert case['case_id'] not in seen;seen.add(case['case_id'])
        assert case['split']==oid[case['object_id']]['split']
        assert case['case_id'] in oid[case['object_id']]['cases']
    for obj in objects:
        subset=[c for c in cases if c['object_id']==obj['object_id']]
        assert len(subset)==4 and {c['condition'] for c in subset}==set(CONDITIONS)

def prediction_bank(prediction):
    """Frozen-interface adapter; every particle and track is accounted for."""
    assert prediction['schema']=='B8_joint_depth_particle_distribution_v1'
    assert prediction['semantic'] is False and prediction['extent_mode']=='uncertainty'
    assert prediction['old_surface_likelihood_applied_again'] is False
    assert prediction['source_geometry_consumed'] is False
    if prediction.get('unmodeled_tracks'):raise ValueError('Observed tracks lack a conditional template')
    tracks=prediction['tracks']
    if len(tracks)!=1:raise ValueError(f'Expected one observed instance; got {len(tracks)}')
    track=tracks[0]
    if not track['status'].get('distribution_calibratable',False):raise ValueError('Declared non-calibratable boundary or invalid geometry fallback')
    d=np.asarray([p['depth_m'] for p in track['particles']],float)
    q=np.asarray(track['prior_mass'],float)
    surface=np.asarray(track['surface_energy_m2'],float);free=np.asarray(track['free_energy_m2'],float)
    assert track['shape_prior']==[.5,.5]
    np.testing.assert_allclose(surface+free,track['energy_m2'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(q,[p['prior_mass'] for p in track['particles']],rtol=0,atol=1e-14)
    assert abs(float(q.sum())-1)<1e-12
    if surface.shape!=(len(d),2) or free.shape!=surface.shape or q.shape!=d.shape:
        raise ValueError('Invalid particle array shapes')
    if not np.isfinite(d).all() or np.any(d<=0):raise ValueError('Invalid particle depths')
    # Validate the same joint energy independent of the candidate sigma.
    joint_posterior(surface+free,q,.06)
    return track,d,q,surface+free

def fallback_bank(reason):
    depths=np.round(np.arange(.2,1.40001,.1),10)
    edges=np.r_[.2,(depths[:-1]+depths[1:])/2,1.4]
    prior=np.diff(edges)/1.2
    track=dict(status=dict(status='evaluation_prior_fallback',failure_reason=reason,distribution_calibratable=False),depth_axis_world=[0.,1.,0.])
    return track,depths,prior,np.zeros((len(depths),2))

def load_cases(dataset,index,split):
    by={r['case_id']:r for r in index['cases']};assert len(by)==len(index['cases']),'Duplicate prediction case ID'
    cases=[];failures=[];opened_truth=set();truth_cache={}
    for entry in dataset['cases']:
        if entry['split']!=split:continue
        cid=entry['case_id'];tr=entry['truth'];tp=tr['path']
        if tp not in truth_cache:truth_cache[tp]=load_resource(tr);opened_truth.add(tp)
        assert truth_cache[tp]['object_id']==entry['object_id']
        item=by.get(cid,dict(case_id=cid,status='missing_prediction_index',prediction=None))
        prediction_resource=item.get('prediction');failure=None
        # Existing resource hash mismatches invalidate the evaluation itself.
        if prediction_resource and Path(prediction_resource['path']).exists():
            assert sha(prediction_resource['path'])==prediction_resource['sha256'],'Prediction input hash changed'
        try:
            if not prediction_resource:raise ValueError('Prediction resource unavailable')
            prediction=load_resource(prediction_resource)
            track,depths,prior,energy=prediction_bank(prediction)
            target=projected_truth(truth_cache[tp],track['depth_axis_world'])
        except Exception as exc:
            failure=repr(exc)
            track,depths,prior,energy=fallback_bank(failure)
            target=projected_truth(truth_cache[tp],[0.,1.,0.])
            failures.append(dict(case_id=cid,object_id=entry['object_id'],condition=entry['condition'],error=failure,
                fallback='unconditional_uniform_density_0.2_1.4_world_y',not_an_inferred_distribution=True))
        target['fallback_axis']=failure is not None
        cases.append(dict(entry=entry,index=item,track=track,depths=depths,prior=prior,energy=energy,target=target,failure=failure))
    return cases,failures,sorted(opened_truth)

def case_metrics(cases,sigma):
    output={method:[] for method in ('posterior','prior_only','fixed_0.65')}
    for c in cases:
        target=c['target'];truth=target['projected_true_extent_m']
        weight=joint_posterior(c['energy'],c['prior'],sigma).sum(axis=1)
        for name,d,p in [('posterior',c['depths'],weight),('prior_only',c['depths'],c['prior']),('fixed_0.65',[.65],[1.])]:
            output[name].append(dict(case_id=c['entry']['case_id'],object_id=c['entry']['object_id'],condition=c['entry']['condition'],
                sigma_m=float(sigma) if name=='posterior' else None,metrics=metrics(d,p,truth),target=target,
                prediction=c['index'].get('prediction'),truth=c['entry']['truth'],inference_status=c['track']['status'],
                failure_reason=c['failure'],evaluator_prior_fallback=c['failure'] is not None))
    return output

def summarize(rows):
    output={}
    for name,items in rows.items():
        output[name]=dict(all_conditions=equal_object_summary(items),by_condition={condition:equal_object_summary([r for r in items if r['condition']==condition]) for condition in CONDITIONS},
            informative=equal_object_summary([r for r in items if r['condition'] in ('front_side','around')]),
            axis_diagnostics=dict(swapped_axis_cases=sum(r['target']['swapped_width_depth_axis'] for r in items),
                mean_error_to_true_depth_axis_deg=float(np.mean([r['target']['error_to_true_depth_axis_deg'] for r in items])),
                mean_best_horizontal_axis_error_deg=float(np.mean([r['target']['best_horizontal_axis_error_deg'] for r in items])),
                mean_axis_elevation_deg=float(np.mean([r['target']['elevation_from_horizontal_deg'] for r in items]))),
            numerical_boundary_fallback_cases=sum(not r['inference_status'].get('distribution_calibratable',False) for r in items),
            truth_outside_original_depth_prior_cases=sum(not (.2<=r['target']['projected_true_extent_m']<=1.4) for r in items))
        valid=[r for r in items if not r['evaluator_prior_fallback']]
        informative_valid=[r for r in valid if r['condition'] in ('front_side','around')]
        output[name]['fallback_cases']=len(items)-len(valid)
        output[name]['valid_only_all_conditions']=equal_object_summary(valid) if valid else None
        output[name]['valid_only_informative']=equal_object_summary(informative_valid) if informative_valid else None
    return output

def choose_scale(scores):
    best=min(x['training_equal_object_crps_m'] for x in scores)
    return max(x['sigma_m'] for x in scores if x['training_equal_object_crps_m']<=best+1e-12)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['calibrate','evaluate'])
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--selected-scale',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        dataset=read(args.dataset);index=read(args.predictions);validate_manifest(dataset)
        assert index['status'] in ('completed_isolated_banks','completed_with_failures'),'Inference batch is not finalized'
        assert not index.get('pending_case_ids'),'Inference cases still pending'
        assert len(index['cases'])==96 and {r['case_id'] for r in index['cases']}=={r['case_id'] for r in dataset['cases']},'Final index must retain all 96 outcomes'
        if 'freeze' in index:
            frozen=load_resource(index['freeze'])
            assert all(sha(p)==s for p,s in frozen['files_sha256'].items()),'Inference frozen files changed'
        shared=dict(dataset=resource(args.dataset),inference_index=resource(args.predictions),evaluator_files_sha256=code_receipt(),
            units=dict(objects=24,train=12,validation=6,heldout=6,cases_per_object=4,condition_cases_not_independent_objects=True))
        if args.mode=='calibrate':
            assert args.selected_scale is None
            cases,failures,truth_paths=load_cases(dataset,index,'train')
            assert len(cases)==48
            scores=[]
            for sigma in SIGMA_GRID_M:
                rows=case_metrics(cases,sigma);summary=summarize(rows)
                score=dict(sigma_m=sigma,training_equal_object_crps_m=summary['posterior']['all_conditions']['crps_m'],summary=summary,rows=rows)
                scores.append(score)
            selected=choose_scale(scores)
            write(args.output/'training_scores.json',dict(status='completed_train_only_scale_comparison',split='train',opened_truth_paths=truth_paths,failures=failures,fallback_case_count=len(failures),valid_inferred_cases=len(cases)-len(failures),scales=scores,**shared))
            write(args.output/'selected_scale.json',dict(status='selected_provisional_on_training_with_fallback' if failures else 'selected_on_training_only',fallback_case_count=len(failures),sigma_m=selected,grid_m=list(SIGMA_GRID_M),objective='CRPS equal conditions within object then equal objects',tie_rule='within 1e-12 metres choose larger sigma',training_scores=resource(args.output/'training_scores.json'),opened_truth_splits=['train'],**shared))
            print(json.dumps(dict(status='selected',sigma_m=selected,selected_scale=resource(args.output/'selected_scale.json'))))
        else:
            if args.selected_scale is None:parser.error('--selected-scale required for evaluation')
            selected=read(args.selected_scale);assert selected['status'] in ('selected_on_training_only','selected_provisional_on_training_with_fallback')
            assert selected['dataset']==shared['dataset'] and selected['inference_index']==shared['inference_index']
            assert selected['evaluator_files_sha256']==shared['evaluator_files_sha256'],'Evaluator changed after scale selection'
            load_resource(selected['training_scores']);sigma=selected['sigma_m'];assert sigma in SIGMA_GRID_M
            results={}
            for split in ('validation','heldout'):
                cases,failures,truth_paths=load_cases(dataset,index,split)
                rows=case_metrics(cases,sigma);summary=summarize(rows) if cases else None
                enough=len(cases)==24
                crps_ready=False
                if enough:
                    scores={m:s['informative']['crps_m'] for m,s in summary.items()}
                    crps_ready=scores['posterior']<scores['prior_only'] and scores['posterior']<scores['fixed_0.65']
                results[split]=dict(status=('complete_with_evaluation_fallback' if failures else 'complete') if enough else 'inference_incomplete',failures=failures,fallback_case_count=len(failures),expected_cases=24,evaluable_cases=len(cases),valid_inferred_cases=len(cases)-len(failures),opened_truth_paths=truth_paths,
                    summary=summary,rows=rows,informative_crps_improves_both_baselines=crps_ready)
                write(args.output/f'{split}.json',dict(split=split,sigma_m=sigma,selected_scale=resource(args.selected_scale),**results[split],**shared))
            ready=all(r['informative_crps_improves_both_baselines'] for r in results.values())
            write(args.output/'assessment.json',dict(status='evaluated_without_retuning',sigma_m=sigma,distribution_metric_criterion_passes=ready,
                requires_separate_counterfactual_and_method_integrity_checks=True,policy_launch_authorized_by_this_file=False,
                split_results={s:resource(args.output/f'{s}.json') for s in results},selected_scale=resource(args.selected_scale),
                interpretation='Small synthetic object splits; CRPS calibration does not establish physical accuracy or semantic policy superiority. Old-six-prefix diagnostic permitted if criterion fails; no new closed-loop promotion.',**shared))
            print(json.dumps(dict(status='evaluated',sigma_m=sigma,distribution_metric_criterion_passes=ready)))
    except BaseException as exc:
        write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),mode=args.mode));raise

if __name__=='__main__':main()
