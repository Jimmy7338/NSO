"""Independent scalar metrics for B11 paired acquired-route experiments.

No actor, source renderer, policy or map mutation is imported here. Source
reference distances must come from a separately frozen evaluation process.
"""
from __future__ import annotations
import math
from typing import Iterable, Mapping
import numpy as np

TOLERANCE_M=.05

def surface_increment(before_distance,after_distance,area_weights,tolerance_m=TOLERANCE_M):
    before=np.asarray(before_distance,float);after=np.asarray(after_distance,float);area=np.asarray(area_weights,float)
    if before.ndim!=1 or after.shape!=before.shape or area.shape!=before.shape:
        raise ValueError('paired one-dimensional reference distances and area weights required')
    if np.isnan(before).any() or np.isnan(after).any() or np.any(before<0)or np.any(after<0):
        raise ValueError('nonnegative non-NaN distances required;infinity denotes no reconstructed surface')
    if not np.isfinite(area).all()or np.any(area<0):raise ValueError('finite nonnegative reference areas required')
    if not np.isfinite(tolerance_m)or tolerance_m<0:raise ValueError('finite nonnegative tolerance required')
    tol=float(np.float32(tolerance_m));was=before<=tol;now=after<=tol
    gross=float(area[~was&now].sum());lost=float(area[was&~now].sum());net=gross-lost
    denominator=float(area.sum());old_area=float(area[was].sum());new_area=float(area[now].sum())
    if not np.isclose(net,new_area-old_area,atol=1e-11,rtol=1e-11):raise AssertionError('paired area conservation failed')
    return dict(reference_points=len(area),reference_area_m2=denominator,before_recovered_area_m2=old_area,after_recovered_area_m2=new_area,
      gross_new_area_m2=gross,lost_area_m2=lost,net_new_area_m2=net,
      before_recall=old_area/denominator if denominator else None,after_recall=new_area/denominator if denominator else None,
      gross_new_points=int((~was&now).sum()),lost_points=int((was&~now).sum()),before_points=int(was.sum()),after_points=int(now.sum()))

def winner(rows):
    """Original maximization, then lower complete-route cost, then identifier."""
    rows=list(rows)
    if not rows:return None
    for r in rows:
        if not math.isfinite(float(r['score']))or int(r['cost'])<0:raise ValueError('finite score and nonnegative route cost required')
    return min(rows,key=lambda r:(-float(r['score']),int(r['cost']),int(r['candidate_id'])))['candidate_id']

def rank_correlation(x,y):
    from scipy.stats import spearmanr
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.shape!=y.shape or x.ndim!=1:raise ValueError('paired scalar vectors required')
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<2 or np.ptp(x[valid])==0 or np.ptp(y[valid])==0:return None
    return float(spearmanr(x[valid],y[valid]).statistic)

def summarize_condition(predictions,actual_by_candidate,*,metric='object_gross_new_m2'):
    """Evaluate all available observations without treating failures as zero.

    A failed-but-measured route retains its area and status. No output here
    turns partial observation into a successful complete route.
    """
    predictions=list(predictions);ids=[int(x['candidate_id'])for x in predictions]
    if len(ids)!=len(set(ids)):raise ValueError('candidate IDs must be unique')
    selected=winner(predictions);actuals={int(k):v for k,v in actual_by_candidate.items()}
    missing=[cid for cid in ids if cid not in actuals or actuals[cid].get(metric)is None]
    measured=[cid for cid in ids if cid not in missing]
    for cid in measured:
        if not math.isfinite(float(actuals[cid][metric])):raise ValueError('measured areas must be finite')
    incomplete=[cid for cid in ids if actuals.get(cid,{}).get('status')!='completed']
    best=max(measured,key=lambda cid:(float(actuals[cid][metric]),-cid))if measured else None
    val=lambda cid:float(actuals[cid][metric])if cid in measured else None
    pairs=[(float(x['predicted_gain_m2']),val(int(x['candidate_id'])))for x in predictions if int(x['candidate_id'])in measured]
    if any(not math.isfinite(x)for x,y in pairs):raise ValueError('finite predicted area required')
    errors=np.array([x-y for x,y in pairs])
    chosen=val(selected);best_value=val(best)
    return dict(selected_candidate=selected,selected_observed_area_m2=chosen,best_observed_candidate=best,best_observed_area_m2=best_value,
      observed_gap_m2=best_value-chosen if chosen is not None and best_value is not None else None,
      candidate_count=len(ids),observed_candidate_count=len(measured),missing_area_candidates=missing,incomplete_or_failed_candidates=incomplete,
      candidate_best_fully_observed=not missing and not incomplete,
      MAE_m2=float(abs(errors).mean())if len(errors)else None,bias_m2=float(errors.mean())if len(errors)else None,
      template_rank_rho=rank_correlation([x for x,y in pairs],[y for x,y in pairs])if pairs else None,
      selected_status=actuals.get(selected,{}).get('status','missing'))

def paired_summary(old_predictions,new_predictions,actual_by_candidate):
    old_ids=[int(x['candidate_id'])for x in old_predictions];new_ids=[int(x['candidate_id'])for x in new_predictions]
    if old_ids!=new_ids:raise ValueError('paired predictors must see the same ordered candidates')
    for old,new in zip(old_predictions,new_predictions):
        for key in ('cost','common_score'):
            if old[key]!=new[key]:raise ValueError('paired common score and execution cost must be identical')
        for row in (old,new):
            if not np.isclose(row['score'],row['common_score']+row['predicted_gain_m2'],atol=1e-11,rtol=0):raise ValueError('candidate score differs from common plus forecast gain')
    old=summarize_condition(old_predictions,actual_by_candidate);new=summarize_condition(new_predictions,actual_by_candidate)
    delta=(new['selected_observed_area_m2']-old['selected_observed_area_m2']
      if old['selected_observed_area_m2']is not None and new['selected_observed_area_m2']is not None else None)
    return dict(old=old,new=new,target_unchanged=old['selected_candidate']==new['selected_candidate'],new_minus_old_selected_area_m2=delta)

def summarize_layout(state_pairs):
    """Equal decision-state means; retain state rows separately, never impute."""
    rows=list(state_pairs)
    fields={'old_selected_area_m2':lambda x:x['old']['selected_observed_area_m2'],
            'new_selected_area_m2':lambda x:x['new']['selected_observed_area_m2'],
            'candidate_best_observed_area_m2':lambda x:x['old']['best_observed_area_m2'],
            'old_observed_gap_m2':lambda x:x['old']['observed_gap_m2'],
            'new_observed_gap_m2':lambda x:x['new']['observed_gap_m2'],
            'new_minus_old_selected_area_m2':lambda x:x['new_minus_old_selected_area_m2']}
    means={}
    for k,fun in fields.items():
        values=[fun(x)for x in rows];means[k]=float(np.mean(values))if values and all(v is not None for v in values)else None
    return dict(**means,states=len(rows),unchanged_states=sum(x['target_unchanged']for x in rows),
      all_three_candidates_fully_observed=all(x['old']['candidate_best_fully_observed']for x in rows),
      incomplete_candidate_events=sum(len(x['old']['incomplete_or_failed_candidates'])for x in rows),
      independent_unit='layout;states and candidate branches are dependent')
