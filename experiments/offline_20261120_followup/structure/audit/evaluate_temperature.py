"""Independent, split-separated B9 structural-temperature evaluation.

Reads private truth only in the requested split. No actor execution, fitting
of geometry, candidate changes, clipping of log-likelihood, or track selection.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
UNCERTAINTY=ROOT.parent/'uncertainty'
DATASET=UNCERTAINTY/'generator/dataset_manifest.json'
INDEX=UNCERTAINTY/'inference/inference_index.json'
B8_METRICS=Path('/home/ouc/nso_robot/analyses/dual_template_uncertainty_20261120/audit/distribution_metrics.py')
CONDITIONS=('front','front_side','around','occluded')
TEMPERATURES=(.25,.5,1.,2.,4.,8.,16.,32.,64.,128.)
SIGMA=.01


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(ref):
    assert sha(ref['path'])==ref['sha256'],ref['path']
    return read(ref['path'])


def numerical_components(energy, prior):
    energy=np.asarray(energy,float);prior=np.asarray(prior,float)
    if energy.shape!=(len(prior),2) or not len(prior) or not np.isfinite(energy).all() or not np.isfinite(prior).all():
        raise ValueError('Finite K by 2 energy and K prior required')
    if np.any(energy<0) or np.any(prior<0) or not np.isclose(prior.sum(),1.):
        raise ValueError('Nonnegative energies and normalized prior required')
    with np.errstate(divide='ignore'):log_terms=np.log(prior)[:,None]-energy/(2*SIGMA**2)
    max_log=log_terms.max(axis=0);rescaled=np.exp(log_terms-max_log)
    totals=rescaled.sum(axis=0)
    conditional=rescaled/totals
    evidence=max_log+np.log(totals)
    return conditional,float(evidence[1]-evidence[0])


def class_metrics(log_difference, target, temperature, semantic=False):
    if target not in (0,1) or not math.isfinite(log_difference) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Finite evidence, positive temperature and binary truth required')
    prior_logodds=(2*target-1)*math.log(9.) if semantic else 0.
    logit=log_difference/temperature+prior_logodds
    # Avoid probability clipping: wrong confident examples retain their real NLL.
    log_p1=-float(np.logaddexp(0.,-logit));log_p0=-float(np.logaddexp(0.,logit))
    p1,p0=math.exp(log_p1),math.exp(log_p0)
    nll=float(np.logaddexp(0.,(1-2*target)*logit))
    exact_tie=logit==0.
    predicted=None if exact_tie else int(logit>0)
    confidence=max(p0,p1)
    return dict(nll=nll,binary_brier=(p1-target)**2,
                accuracy=.5 if exact_tie else float(predicted==target),
                exact_tie=exact_tie,predicted_class=predicted,
                high_confidence_wrong=bool(not exact_tie and predicted!=target and confidence>.9),
                confidence=confidence,logit=logit,shape_probability_mass=[p0,p1],
                correct_category_prior_supplied=bool(semantic))


def select(rows):
    if not rows or any(r.get('split')!='calibration' for r in rows):raise ValueError('Calibration rows only')
    groups={}
    for row in rows:
        oid=row['object_id'];cond=row['condition']
        if cond not in CONDITIONS or cond in groups.setdefault(oid,{}):raise ValueError('Duplicate or invalid condition')
        groups[oid][cond]=row
    if any(set(block)!=set(CONDITIONS) for block in groups.values()):raise ValueError('Four conditions per object required')
    curve=[]
    for t in TEMPERATURES:
        per_object={oid:float(np.mean([class_metrics(r['log_evidence_difference'],r['target'],t)['nll'] for r in block.values()]))for oid,block in sorted(groups.items())}
        curve.append(dict(temperature=t,equal_object_nll=float(np.mean(list(per_object.values()))),per_object=per_object))
    best=min(curve,key=lambda r:(r['equal_object_nll'],abs(math.log2(r['temperature'])),-r['temperature']))
    return dict(temperature=best['temperature'],objects=len(groups),cases=len(rows),curve=curve,
                criterion='calibration_equal_object_neutral_class_NLL',
                tie_rule='Exact NLL tie: closest log2 distance to 1, then larger T')


def checked_inputs():
    audit_freeze=read(Path(__file__).with_name('freeze.json'))
    for path,h in audit_freeze['files_sha256'].items():assert sha(path)==h,path
    dataset=read(DATASET);index=read(INDEX)
    assert len(dataset['objects'])==48 and len(dataset['cases'])==192
    assert index['status']=='completed_isolated_banks' and not index['pending_case_ids']
    assert len(index['cases'])==192
    by={r['case_id']:r for r in index['cases']};assert len(by)==192
    assert set(by)=={r['case_id'] for r in dataset['cases']}
    for split in ('calibration','test'):
        objects={o['object_id']for o in dataset['objects']if o['split']==split}
        cases=[c for c in dataset['cases']if c['split']==split]
        assert len(objects)==24 and len(cases)==96
        for oid in objects:assert {r['condition']for r in cases if r['object_id']==oid}==set(CONDITIONS)
    assert not ({o['object_id']for o in dataset['objects']if o['split']=='calibration'} & {o['object_id']for o in dataset['objects']if o['split']=='test'})
    frozen=load(index['freeze'])
    for path,h in frozen['files_sha256'].items():assert sha(path)==h,path
    protocol=read(ROOT/'protocol.json');assert protocol['temperatures']==list(TEMPERATURES) and protocol['sigma_m']==SIGMA
    for name,h in protocol['code_sha256'].items():assert sha(ROOT/name)==h,name
    return dataset,by


def metrics_module():
    spec=importlib.util.spec_from_file_location('B9_structural_depth_metrics',B8_METRICS)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def bank(prediction):
    assert prediction['schema']=='B8_joint_depth_particle_distribution_v1'
    assert prediction['semantic'] is False and prediction['extent_mode']=='uncertainty'
    assert not prediction['source_geometry_consumed'] and not prediction['old_surface_likelihood_applied_again']
    if prediction.get('unmodeled_tracks'):raise ValueError('Observed track has no conditional template')
    if len(prediction['tracks'])!=1:raise ValueError('Expected one observed track, no truth-based track selection')
    track=prediction['tracks'][0]
    assert track['shape_prior']==[.5,.5]
    if not track['status'].get('distribution_calibratable',False):raise ValueError('Noncalibratable out-of-support/invalid-geometry status')
    d=np.asarray([r['depth_m']for r in track['particles']],float);q=np.asarray(track['prior_mass'],float)
    np.testing.assert_allclose(q,[r['prior_mass']for r in track['particles']],rtol=0,atol=1e-14)
    energy=np.asarray(track['energy_m2'],float)
    np.testing.assert_allclose(energy,np.asarray(track['surface_energy_m2'])+np.asarray(track['free_energy_m2']),rtol=0,atol=1e-14)
    if d.shape!=q.shape or not np.isfinite(d).all() or np.any(d<=0):raise ValueError('Invalid depths')
    conditional,diff=numerical_components(energy,q)
    return track,d,q,conditional,diff


def fallback(reason):
    d=np.round(np.arange(.2,1.40001,.1),10);edges=np.r_[.2,(d[:-1]+d[1:])/2,1.4];q=np.diff(edges)/1.2
    track=dict(status=dict(status='evaluation_prior_fallback',reason=reason,distribution_calibratable=False),depth_axis_world=[0.,1.,0.])
    return track,d,q,np.repeat(q[:,None],2,axis=1),0.


def load_split(dataset,by,split):
    assert split in ('calibration','test')
    metric=metrics_module();truths={};cases=[]
    for entry in dataset['cases']:
        if entry['split']!=split:continue
        tr=entry['truth'];path=tr['path']
        if path not in truths:truths[path]=load(tr)
        truth=truths[path];assert truth['object_id']==entry['object_id']
        family=truth['construction']['family'];assert family in ('closed','open');target=int(family=='open')
        row=by[entry['case_id']];ref=row.get('prediction');reason=None
        if ref and Path(ref['path']).exists():assert sha(ref['path'])==ref['sha256'],'Prediction hash mismatch is an integrity failure'
        try:
            if not ref:raise ValueError('Missing prediction resource')
            track,d,q,conditional,diff=bank(load(ref))
            projected=metric.projected_truth(truth,track['depth_axis_world'])
        except Exception as exc:
            reason=repr(exc);track,d,q,conditional,diff=fallback(reason)
            projected=metric.projected_truth(truth,[0.,1.,0.])
        cases.append(dict(entry=entry,prediction=ref,target=target,family=family,
                          shape_regime=truth['construction']['shape_regime'],track=track,
                          depths=d,prior=q,conditional=conditional,log_evidence_difference=diff,
                          projected=projected,fallback_reason=reason))
    assert len(cases)==96 and len(truths)==24
    return cases,sorted(truths)


def evaluate_cases(cases,temperature):
    metric=metrics_module();outputs=[]
    for case in cases:
        for tag,t in (('T1',1.),('selected',temperature)):
            for name,semantic in (('G',False),('S_correct_category',True)):
                c=class_metrics(case['log_evidence_difference'],case['target'],t,semantic)
                p=case['conditional']@np.asarray(c['shape_probability_mass'])
                m=metric.metrics(case['depths'],p,case['projected']['projected_true_extent_m'])
                outputs.append(dict(case_id=case['entry']['case_id'],object_id=case['entry']['object_id'],condition=case['entry']['condition'],
                                    split=case['entry']['split'],variant=tag,temperature=t,mode=name,
                                    target_class=case['target'],family=case['family'],shape_regime=case['shape_regime'],
                                    log_evidence_difference=case['log_evidence_difference'],class_metrics=c,depth_metrics=m,
                                    inference_status=case['track']['status'],evaluator_prior_fallback=case['fallback_reason']is not None,
                                    fallback_reason=case['fallback_reason'],target=case['projected'],prediction=case['prediction'],truth=case['entry']['truth']))
    return outputs


def aggregate(rows):
    if not rows:raise ValueError('No rows')
    groups={}
    for row in rows:groups.setdefault(row['object_id'],[]).append(row)
    def mean(metric):return float(np.mean([np.mean([metric(r)for r in block])for block in groups.values()]))
    return dict(objects=len(groups),cases=len(rows),nll=mean(lambda r:r['class_metrics']['nll']),
                binary_brier=mean(lambda r:r['class_metrics']['binary_brier']),accuracy=mean(lambda r:r['class_metrics']['accuracy']),
                high_confidence_wrong_cases=sum(r['class_metrics']['high_confidence_wrong']for r in rows),
                exact_tie_cases=sum(r['class_metrics']['exact_tie']for r in rows),
                marginal_depth_crps_m=mean(lambda r:r['depth_metrics']['crps_m']),
                marginal_depth_median_MAE_m=mean(lambda r:r['depth_metrics']['median_absolute_error_m']),
                fallback_cases=sum(r['evaluator_prior_fallback']for r in rows))


def summaries(rows):
    result={}
    for variant in ('T1','selected'):
        result[variant]={}
        for mode in ('G','S_correct_category'):
            block=[r for r in rows if r['variant']==variant and r['mode']==mode]
            result[variant][mode]=dict(all_conditions=aggregate(block),by_condition={c:aggregate([r for r in block if r['condition']==c])for c in CONDITIONS})
    return result


def code_receipt():
    return {str(p):sha(p)for p in (Path(__file__),Path(__file__).with_name('test_evaluator.py'),ROOT/'protocol.json',ROOT/'model.py',B8_METRICS)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('calibrate','evaluate'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selected-temperature',type=Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:
        dataset,by=checked_inputs();common=dict(dataset=resource(DATASET),inference_index=resource(INDEX),code_sha256=code_receipt(),protocol=resource(ROOT/'protocol.json'),sigma_m=SIGMA)
        if args.mode=='calibrate':
            assert args.selected_temperature is None
            cases,opened=load_split(dataset,by,'calibration')
            scalars=[dict(split='calibration',object_id=c['entry']['object_id'],condition=c['entry']['condition'],target=c['target'],log_evidence_difference=c['log_evidence_difference'])for c in cases]
            selection=select(scalars);assert selection['objects']==24 and selection['cases']==96
            # Check independent selection against the separately frozen method API.
            spec=importlib.util.spec_from_file_location('B9_frozen_structural_temperature',ROOT/'model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            counterpart=module.select_temperature(scalars);assert counterpart['temperature']==selection['temperature']
            np.testing.assert_allclose([r['equal_object_nll']for r in counterpart['curve']],[r['equal_object_nll']for r in selection['curve']],rtol=0,atol=1e-12)
            rows=evaluate_cases(cases,selection['temperature'])
            write(args.output/'calibration_results.json',dict(status='completed_calibration_only_evaluation',rows=rows,summary=summaries(rows),selection_curve=selection['curve'],opened_truth_paths=opened,**common))
            write(args.output/'selected_temperature.json',dict(status='frozen_from_new_calibration_only',**selection,opened_truth_splits=['calibration'],opened_truth_paths=opened,
                  scalar_calibration_rows=scalars,calibration_result=resource(args.output/'calibration_results.json'),independent_formula_matches_method=True,**common))
            print(json.dumps(dict(status='selected',temperature=selection['temperature'],receipt=resource(args.output/'selected_temperature.json'))))
        else:
            assert args.selected_temperature is not None
            selected=read(args.selected_temperature);assert selected['status']=='frozen_from_new_calibration_only' and selected['opened_truth_splits']==['calibration']
            assert selected['dataset']==common['dataset'] and selected['inference_index']==common['inference_index'] and selected['code_sha256']==common['code_sha256']
            load(selected['calibration_result']);cases,opened=load_split(dataset,by,'test')
            assert not set(opened)&set(selected['opened_truth_paths'])
            rows=evaluate_cases(cases,selected['temperature']);summary=summaries(rows)
            improved=summary['selected']['G']['all_conditions']['nll']<summary['T1']['G']['all_conditions']['nll']
            write(args.output/'test_results.json',dict(status='completed_once_without_retuning',rows=rows,summary=summary,opened_truth_paths=opened,selected_temperature=resource(args.selected_temperature),**common))
            write(args.output/'assessment.json',dict(status='completed_independent_structural_temperature_evaluation',temperature=selected['temperature'],test_primary_neutral_NLL_improves=improved,
                  selected_temperature=resource(args.selected_temperature),test_result=resource(args.output/'test_results.json'),summary=summary,
                  limitations=['Correct-category S uses a controlled oracle category prior, not natural detector performance.','One shared structural temperature cannot correct the neutral log-evidence sign.','Conditional geometry distributions are fixed but marginal depth distributions may change.','B9 G-neutral T1 interval-calibration guarantees do not transfer to these reweighted distributions.','No planning superiority follows from these scalar class/depth diagnostics.'],**common))
            print(json.dumps(dict(status='evaluated',temperature=selected['temperature'],test_primary_neutral_NLL_improves=improved,assessment=resource(args.output/'assessment.json'))))
    except BaseException as exc:
        write(args.output/'failure.json',dict(status='failed',mode=args.mode,error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
