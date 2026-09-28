#!/usr/bin/env python3
"""Saved-candidate single-instance posterior interventions; no world or policy run."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'audit_results/article_stage_20260928'
RUNS = tuple('dev_'+name+'_b160_n92801' for name in ('AISLE_G','AISLE_B','AISLE_S','AISLE_NBV','CELL_B'))
TOL = 1e-12
NAMES = ('planar','recessed','louvered','open_frame')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def target(value):
    return (value['node'], int(value['heading']))


def label(value):
    return None if value is None else f'{value[0]}:{value[1]}'


def best(scores):
    """Exact executed lexicographic rule over unique targets (epsilon 1e-12)."""
    if not scores or max(scores.values()) <= 0:
        return dict(target=None, score=0., margin=None, ties=0)
    maximum = max(scores.values())
    tied = sorted(key for key,value in scores.items() if value >= maximum-TOL)
    chosen = tied[0]
    alternatives = [value for key,value in scores.items() if key != chosen]
    return dict(target=chosen, score=scores[chosen],
                margin=None if not alternatives else scores[chosen]-max(alternatives), ties=len(tied))


def numeric_vector(value, *, probability=False):
    require(isinstance(value,list) and len(value)==4 and
            all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in value),
            'four nonnegative finite values required')
    if probability:
        require(abs(sum(value)-1.) < 1e-10, 'posterior not normalized')
    return value


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


class SealedInputs:
    def __init__(self):
        self.pins={}

    def read(self,path,expected=None,expected_bytes=None):
        path=Path(path)
        require(path.is_file() and not path.is_symlink(), 'plain existing input required')
        data=path.read_bytes(); pin=digest(data)
        if expected is not None:require(pin==expected,'input hash differs: '+str(path))
        if expected_bytes is not None:require(len(data)==expected_bytes,'input size differs: '+str(path))
        key=str(path.relative_to(ROOT))
        old=self.pins.get(key)
        row=dict(sha256=pin,bytes=len(data))
        require(old is None or old==row,'input changed within analysis: '+key)
        self.pins[key]=row
        return data

    def json(self,path,**kwargs):
        return json.loads(self.read(path,**kwargs))

    def artifact(self,episode,manifest,name):
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'relative artifact required')
        row=manifest['files'][name]
        data=self.read(episode/relative,expected=row['sha256'],expected_bytes=row['bytes'])
        return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)


def declare(output):
    require(not output.exists(),'fresh output required')
    ledger_path=STAGE/'development_v1/start_ledger.json'
    ledger_bytes=ledger_path.read_bytes(); ledger=json.loads(ledger_bytes)
    entries={r['run_id']:r for r in ledger['entries']}
    rows=[]
    for run in RUNS:
        row=entries[run]
        require(row['status']=='controller_stop' and row['qualified'] is True,'only reviewed terminal runs')
        episode=STAGE/'development_v1/episodes'/run
        review=STAGE/'episode_reviews_v1'/run
        manifest=(episode/'artifact_manifest.json').read_bytes()
        require(digest(manifest)==row['artifact_manifest_sha256'],'terminal manifest mismatch')
        review_manifest=(review/'manifest.json').read_bytes(); rm=json.loads(review_manifest)
        payload=(review/'review.json').read_bytes(); r=json.loads(payload)
        require(digest(payload)==rm['files']['review.json']['sha256'],'review SHA differs')
        require(r['status']=='reviewed' and r['all_checks_passed'] is True and r['qualified'] is True,
                'passed independent review required')
        require(r['run_id']==run and r['input_manifest_sha256']==digest(manifest)==rm['input_episode_manifest_sha256'],
                'independent review/episode binding differs')
        rows.append(dict(run_id=run,terminal_entry=row,episode=str(episode.relative_to(ROOT)),
            review=str(review.relative_to(ROOT)),review_manifest_sha256=digest(review_manifest)))
    plan=dict(schema='article.saved_direct_oracle_protocol.v1',
        declared_utc=datetime.now(timezone.utc).isoformat(),script_sha256=digest(Path(__file__).read_bytes()),
        selected_runs=rows,ledger_snapshot_sha256=digest(ledger_bytes),
        fixed_inputs='Saved direct candidate poses, costs, per-instance area vectors, posterior marginals, and discovery remainder',
        intervention='One observed instance at a time: replace its posterior with each of four one-hot vectors; other marginals and discovery unchanged',
        score='s_a + inspection_weight*(A_ai[h] - dot(A_ai,p_i))/cost_a',
        tie_rule='Maximum score per unique executable target, lexicographic among score >= maximum-1e-12; no target if maximum <= 0',
        zero_missing_rule='Zero only when a complete per-instance allocation explicitly excludes the candidate; allocated missing forecast is unavailable',
        reconstruction='Verify forecast expected area and posterior, direct expected_gain/cost; infer and hold nonnegative discovery remainder; check visited XY zero, same-XY invariance and occupancy cell-area multiple',
        discovery_limit='Per-replan occupancy/discovery not independently saved; remainder consistency is not an independent replay of discovery geometry',
        missing_record_rule='Mark replan unavailable; do not fill missing allocated forecasts or infer hidden structures',
        outputs=['per-instance four-hypothesis top1 changes and ranks','positive/structure-dependent/all-zero forecasts',
                 'individual structure-only area and area/cost argmax','actual selection kind and before/after first saved return directive'],
        scope='Optimistic instantaneous posterior intervention in recorded direct candidate pools only; no joint-instance oracle, diagnostic reranking, future route, actual true structure, quality estimate or new experiment',
        new_worlds=0,new_policy_runs=0,new_sensor_queries=0,new_tsdf_integrations=0,new_surface_evaluations=0)
    output.mkdir(parents=True)
    (output/'protocol.json').write_bytes(canonical(plan))
    (output/'ledger_snapshot.json').write_bytes(ledger_bytes)
    return plan


def analyze_replan(step, configuration, visited):
    selection=step['decision']['global_selection']
    rows=selection['direct_options']
    require(rows,'no recorded direct options')
    beliefs={r['instance_id']:numeric_vector(r['structure_probabilities'],probability=True)
             for r in step['controller_evidence']['structure_belief']['instances']}
    require(len(beliefs)==len(step['controller_evidence']['structure_belief']['instances']),'duplicate belief ID')
    allocations={r['instance_id']:{target(x) for x in r['candidates']}
                 for r in selection['instance_candidate_allocations']}
    require(len(allocations)==len(selection['instance_candidate_allocations']) and set(allocations)==set(beliefs),
            'complete allocation for every observed instance required')
    forecasts={r['instance_id']:r for r in selection['forecasts']}
    require(len(forecasts)==len(selection['forecasts']) and set(forecasts)<=set(beliefs),'invalid forecast IDs')
    vectors={key:{} for key in beliefs}; fallback=Counter()
    for key,allocation in allocations.items():
        if not allocation:
            require(key not in forecasts,'unexpected forecast for empty allocation');continue
        require(key in forecasts,'allocated instance forecast missing')
        forecast=forecasts[key]
        require(tuple(forecast['structure_names'])==NAMES,'structure order differs')
        fp=numeric_vector(forecast['structure_probabilities'],probability=True)
        require(max(abs(x-y) for x,y in zip(fp,beliefs[key]))<1e-10,'saved forecast posterior differs')
        for item in forecast['candidates']:
            view=item['candidate']
            node,heading=item['view_id'].rsplit(':',1); t=(node,int(heading))
            require(t not in vectors[key],'duplicate forecast candidate')
            area=numeric_vector(item['structure_new_surface_area_m2'])
            require(math.isclose(dot(area,fp),item['expected_new_surface_area_m2'],rel_tol=1e-10,abs_tol=1e-10),
                    'forecast expectation cannot be reconstructed')
            if item['fallback'] or item['repeated_view_excluded']:
                require(max(area)<=TOL,'fallback/repeated forecast must be zero')
                fallback[item.get('fallback_reason') or 'repeated_view']+=1
            vectors[key][t]=area
        require(set(vectors[key])==allocation,'forecast/allocation target inventory differs')
    costs={}; scores={}; remainder={}; snapshots=[]; per_node={}
    iw=configuration['inspection_weight']; dw=configuration['discovery_weight']
    require(iw>=0 and dw>=0,'nonnegative declared utility weights required')
    for row in rows:
        t=target(row['target']);cost=row['total_cost']
        require(t not in costs and isinstance(cost,int) and cost>0,'unique direct targets and positive integer costs')
        require(cost<=selection['remaining'],'saved direct candidate exceeds remaining budget')
        gain=row['expected_gain'];score=row['score']
        require(all(isinstance(x,(int,float)) and math.isfinite(x) for x in [gain,score]),'nonfinite direct score')
        require(math.isclose(gain/cost,score,rel_tol=1e-10,abs_tol=1e-12),'direct gain/cost differs')
        contributions={key:iw*dot(vectors[key].get(t,[0.]*4),p) for key,p in beliefs.items()}
        discovery=gain-sum(contributions.values())
        require(discovery>=-1e-9,'negative reconstructed discovery remainder')
        if t[0] in visited:require(abs(discovery)<1e-9,'visited XY has nonzero discovery remainder')
        if t[0] in per_node:require(abs(per_node[t[0]]-discovery)<1e-9,'same-XY discovery changes with heading')
        per_node[t[0]]=discovery
        if dw:
            units=discovery/(dw*.01)
            require(abs(units-round(units))<1e-6,'discovery remainder not measured-grid area multiple')
        else:require(abs(discovery)<1e-9,'disabled discovery is nonzero')
        # This is an algebraic reconstruction, not an independent occupancy replay.
        reconstructed=discovery+sum(contributions.values())
        require(math.isclose(reconstructed,gain,rel_tol=1e-10,abs_tol=1e-10),'aggregate reconstruction differs')
        costs[t]=cost;scores[t]=score;remainder[t]=max(0.,discovery)
        snapshots.append(dict(target=label(t),cost=cost,score=score,expected_gain=gain,
            inferred_discovery_remainder=max(0.,discovery),surface_contributions=contributions,
            structure_areas={key:vectors[key].get(t,[0.]*4) for key in beliefs}))
    baseline=best(scores)
    actual=selection.get('selected') or {}
    if actual.get('kind')=='direct':
        require(target(actual['target'])==baseline['target'],'saved direct winner differs from lexicographic rule')
    candidates=[]; instances=[]
    for key,p in beliefs.items():
        area={t:vectors[key].get(t,[0.]*4) for t in scores}
        span=max(max(v)-min(v) for v in area.values())
        positive=any(max(v)>TOL for v in area.values())
        hrows=[]
        for h in range(4):
            modified={t:value+iw*(area[t][h]-dot(area[t],p))/costs[t] for t,value in scores.items()}
            chosen=best(modified);base_target=baseline['target']
            raw=best({t:v[h] for t,v in area.items()})
            area_cost=best({t:v[h]/costs[t] for t,v in area.items()})
            base_after=None if base_target is None else modified[base_target]
            row=dict(instance_id=key,h=h,structure=NAMES[h],posterior=p,
                top1_changed=chosen['target']!=base_target,baseline_target=label(base_target),
                counterfactual_target=label(chosen['target']),counterfactual_score=chosen['score'],
                counterfactual_margin=chosen['margin'],counterfactual_ties=chosen['ties'],
                baseline_target_score_after=base_after,
                baseline_target_rank_after=None if base_after is None else 1+sum(v>base_after+TOL for v in modified.values()),
                maximum_score_shift=max(abs(modified[t]-scores[t]) for t in scores),
                instance_only_area_argmax=label(raw['target']),instance_only_area_cost_argmax=label(area_cost['target']),
                instance_only_area_max=raw['score'],instance_only_area_cost_max=area_cost['score'])
            hrows.append(row);candidates.append(row)
        instances.append(dict(instance_id=key,posterior=p,positive_area=positive,
            structure_dependent=span>TOL,maximum_structure_area_span=span,
            pure_h_top1_changed=any(x['top1_changed'] for x in hrows),
            changed_hypotheses=[x['h'] for x in hrows if x['top1_changed']],
            distinct_instance_area_argmax=len({x['instance_only_area_argmax'] for x in hrows}-{None}),
            distinct_instance_area_cost_argmax=len({x['instance_only_area_cost_argmax'] for x in hrows}-{None})))
    return dict(baseline_direct_target=label(baseline['target']),baseline_score=baseline['score'],
        baseline_margin=baseline['margin'],baseline_ties=baseline['ties'],
        direct_candidates=len(scores),observed_instances=len(beliefs),
        any_positive_area=any(r['positive_area'] for r in instances),
        any_structure_dependent_area=any(r['structure_dependent'] for r in instances),
        any_single_instance_pure_h_flip=any(r['pure_h_top1_changed'] for r in instances),
        instances=instances,hypotheses=candidates,candidate_snapshot=snapshots,
        fallback_counts=dict(fallback),discovery_verification='nonnegative algebraic remainder; not independent occupancy replay')


def write_csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['empty']
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def analyze(output):
    plan_path=output/'protocol.json';plan=json.loads(plan_path.read_bytes())
    require(plan['script_sha256']==digest(Path(__file__).read_bytes()),'analysis source changed after declaration')
    require(not (output/'summary.json').exists(),'fresh declared analysis required')
    ledger_bytes=(output/'ledger_snapshot.json').read_bytes()
    require(digest(ledger_bytes)==plan['ledger_snapshot_sha256'],'ledger snapshot changed')
    inputs=SealedInputs();all_replans=[];all_hypotheses=[];run_summary=[];sensitivity=[]
    for item in plan['selected_runs']:
        run=item['run_id'];episode=ROOT/item['episode'];review=ROOT/item['review']
        rm=inputs.json(review/'manifest.json',expected=item['review_manifest_sha256'])
        for name,row in rm['files'].items():
            inputs.read(review/name,expected=row['sha256'],expected_bytes=row['bytes'])
        r=inputs.json(review/'review.json');entry=item['terminal_entry']
        manifest=inputs.json(episode/'artifact_manifest.json',expected=entry['artifact_manifest_sha256'])
        require(r['input_manifest_sha256']==entry['artifact_manifest_sha256'],'review pin differs')
        result=inputs.artifact(episode,manifest,'result.json')
        require(result['status']=='controller_stop','nonterminal input')
        require(manifest['files']['result.json']['sha256']==entry['result_sha256'],'terminal result pin differs')
        configuration=inputs.artifact(episode,manifest,'controller_final.json')['configuration']
        protocol=inputs.artifact(episode,manifest,'protocol.json')
        require(manifest['protocol_sha256']==digest((episode/'protocol.json').read_bytes()),'protocol binding differs')
        slot=protocol['slots'][run];steps=[];visited=set();first_return=None
        counts=Counter();replans=[]
        for name in sorted(manifest['files']):
            if not name.startswith('steps/') or not name.endswith('.json.gz'):continue
            step=inputs.artifact(episode,manifest,name);index=step['accounting']['paid_step']
            require(index==len(steps),'nonconsecutive saved steps');steps.append(index)
            decision=step['decision'];evidence=step['controller_evidence']
            visited.add(evidence['safety']['current_state']['node'])
            reason=(decision.get('routing') or {}).get('reason')
            if reason=='return' and first_return is None:first_return=index
            selection=decision.get('global_selection')
            if not decision.get('global_replanned'):
                counts['no_global_replan_steps']+=1;continue
            selected=(selection or {}).get('selected') or {}
            kind=selected.get('kind','none');counts['actual_selected_kind:'+kind]+=1
            row=dict(run_id=run,method=slot['method'],scene_id=slot['scene_id'],paid_step=index,
                remaining_budget=slot['budget']-index,actual_selected_kind=kind,
                actual_selected_target=label(target(selected['target'])) if selected.get('target') else None,
                routing_reason=reason,global_replanned=True)
            if not selection or not selection.get('direct_options'):
                row.update(status='not_direct_comparison',reason='shared_initialization' if kind=='measurement_initialization' else 'no_direct_candidates')
            else:
                try:
                    data=analyze_replan(step,configuration,visited)
                    row.update(status='analyzed',**{k:v for k,v in data.items() if k not in ('hypotheses','candidate_snapshot')})
                    for h in data['hypotheses']:all_hypotheses.append({**{k:row[k] for k in ('run_id','method','scene_id','paid_step','remaining_budget','actual_selected_kind')},**h})
                    sensitivity.append(dict(run_id=run,paid_step=index,candidates=data['candidate_snapshot']))
                except (ValueError,KeyError,TypeError) as error:
                    row.update(status='unavailable',reason=str(error))
            replans.append(row)
        require(len(steps)==result['executed_paid_actions']+1,'saved paid-frame count differs')
        for row in replans:
            row['return_timing']='before_first_return' if first_return is None or row['paid_step']<first_return else 'at_or_after_first_return'
            counts[row['status']]+=1
            if row['status']=='analyzed':
                counts['positive_area_replans']+=row['any_positive_area']
                counts['structure_dependent_replans']+=row['any_structure_dependent_area']
                counts['single_instance_oracle_changes_direct_top1']+=row['any_single_instance_pure_h_flip']
                counts['pure_h_rank_invariant_replans']+=not row['any_single_instance_pure_h_flip']
                counts['all_zero_area_replans']+=not row['any_positive_area']
                counts['opportunity_'+row['return_timing']]+=row['any_single_instance_pure_h_flip']
        changed=[r['paid_step'] for r in replans if r.get('any_single_instance_pure_h_flip')]
        run_summary.append(dict(run_id=run,method=slot['method'],scene_id=slot['scene_id'],
            counts=dict(counts),first_return_paid_step=first_return,opportunity_paid_steps=changed,
            earliest_opportunity_paid_step=min(changed) if changed else None,
            terminal_paid_actions=result['executed_paid_actions']))
        all_replans.extend(replans)
    for row in all_hypotheses:
        parent=next(x for x in all_replans if x['run_id']==row['run_id'] and x['paid_step']==row['paid_step'])
        row['return_timing']=parent['return_timing']
    totals=Counter()
    for run in run_summary:totals.update(run['counts'])
    summary=dict(schema='article.saved_direct_oracle_summary.v1',protocol_sha256=digest(plan_path.read_bytes()),
        scope=plan['scope'],runs=run_summary,totals=dict(totals),hypothesis_interventions=len(all_hypotheses),
        changing_hypothesis_interventions=sum(r['top1_changed'] for r in all_hypotheses),
        independent_trajectories_added=0,new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        limitations=['A single-instance intervention does not establish the joint perfect-information policy.',
                    'No missing plane, unallocated candidate, different route history or diagnostic channel is repaired by this intervention.',
                    'A changed direct winner is an available nominal decision opportunity, not an executed action or quality improvement.',
                    'Rank invariance applies only to the recorded direct pool, costs and fixed other-instance beliefs.',
                    'Discovery is a verified algebraic remainder, not independently rerendered or remeasured.',
                    'Time split uses first recorded routing reason return; no unrecorded latch time is inferred.'])
    (output/'summary.json').write_bytes(canonical(summary));(output/'replans.json').write_bytes(canonical(all_replans))
    (output/'candidate_sufficient_statistics.json').write_bytes(canonical(sensitivity))
    write_csv(output/'hypothesis_interventions.csv',all_hypotheses)
    write_csv(output/'replans.csv',[{k:v for k,v in r.items() if k!='instances'} for r in all_replans])
    report_lines=['# Saved direct-target posterior opportunity analysis','',plan['scope'],'',
        '| Run | Analyzed replans | Structure-dependent | Any single-instance pure-h flip | All-zero surface gain | First return step |',
        '|---|---:|---:|---:|---:|---:|']
    for r in run_summary:
        c=r['counts'];report_lines.append(f"| {r['run_id']} | {c.get('analyzed',0)} | {c.get('structure_dependent_replans',0)} | {c.get('single_instance_oracle_changes_direct_top1',0)} | {c.get('all_zero_area_replans',0)} | {r['first_return_paid_step']} |")
    report_lines += ['',f"Unavailable recorded replans: {totals.get('unavailable',0)}. Initialization selections are separate and are never interpreted as direct-score competitions.",'',
        'Discovery is held as the saved additive remainder after independently reconstructing each forecast expectation. Nonnegativity, visited-node zero, same-XY equality, grid-area multiples and exact direct gain/cost are checked. This is not an independent occupancy replay.','',
        'All four hypothetical structures are considered without consulting which is physically true. No counterfactual action is executed. Per-structure area and area/cost argmax, direct rank margins, actual selection kind, and return timing are retained in the CSV/JSON tables.','',
        'Interpretation: zero surface forecasts identify a perception/eligibility/candidate bottleneck. Nonzero structure-dependent forecasts with unchanged one-hot winners identify invariance within the current pool. A changed winner demonstrates an available nominal decision opportunity; observed belief strength, competing discovery and route cost may still prevent a real switch. None of these establishes new reconstruction quality or scene-wide perfect-information behavior.','']
    (output/'README.md').write_text('\n'.join(report_lines))
    for name,row in inputs.pins.items():
        require(digest((ROOT/name).read_bytes())==row['sha256'],'sealed input changed before finish')
    manifest=dict(schema='article.saved_direct_oracle_manifest.v1',inputs=inputs.pins,
        outputs={p.name:dict(sha256=digest(p.read_bytes()),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()},
        script_sha256=digest(Path(__file__).read_bytes()),new_worlds=0,new_policy_runs=0)
    (output/'manifest.json').write_bytes(canonical(manifest))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('declare','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    require(output.is_relative_to(STAGE/'diagnostics'),'new diagnostic directory under article stage required')
    result=declare(output) if args.command=='declare' else analyze(output)
    print(json.dumps(result if args.command=='analyze' else {'declared':str(output),'runs':len(result['selected_runs'])},ensure_ascii=False))


if __name__=='__main__':main()
