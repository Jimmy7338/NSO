#!/usr/bin/env python3
"""Saved direct-candidate bounds under the one-peer reliability model.

No World, GT, sensor replay, learned policy, diagnostic branch or quality
measurement. Independent interval combinations are only conservative envelopes,
not an executable jointly consistent shared posterior.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_article_direct_oracle_20260928 as saved

STAGE=ROOT/'audit_results/article_stage_20260928'
GROUND_ALLOWED=('ground_AISLE_B_b160_n92801','ground_AISLE_G_b160_n92801')
TOL=1e-12
GUARD=1e-13


def posterior(g,q,likelihood,rho):
    values=[((1-rho)*a+rho*b)*l for a,b,l in zip(g,q,likelihood)]
    total=sum(values)
    saved.require(total>0 and math.isfinite(total),'positive posterior normalizer')
    return [x/total for x in values]


def endpoint_range(g,q,likelihood,utility,lo,hi):
    """A linear-fractional expectation is monotone or constant on this interval.

    f(r)=(alpha+beta*r)/(gamma+delta*r), gamma+delta*r>0;
    f'(r)=(beta*gamma-alpha*delta)/(gamma+delta*r)^2.
    This applies to signed candidate-score contrasts as well as area gains.
    """
    ends=[saved.dot(utility,posterior(g,q,likelihood,r)) for r in (lo,hi)]
    return min(ends),max(ends)


def models_from_belief(belief,*,optimistic_missing_peer=False):
    g=belief['geometry_prior'];saved.numeric_vector(g,probability=True)
    saved.require(all(x>0 for x in g),'positive geometry prior')
    instances=belief['instances'];models={}
    classes=Counter(r['observed_class'] for r in instances if r['observed_class'] is not None)
    for row in instances:
        key=row['instance_id'];label=row['observed_class'];e=row['geometry_log_evidence']
        saved.require(len(e)==4 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in e),'finite own log evidence')
        likelihood=[math.exp(v-max(e)) for v in e]
        current=saved.numeric_vector(row['structure_probabilities'],probability=True)
        if label is None:
            q=g;rho=.5;lo=hi=.5;peers=0;reason='no_current_qualified_class'
        else:
            q=belief['class_structure_priors'][label];saved.numeric_vector(q,probability=True)
            saved.require(all(x>0 for x in q),'positive category prior')
            rho=row['rho'];peers=classes[label]-1
            saved.require(peers<=1,'outside the declared at-most-one-qualified-peer regime')
            saved.require(isinstance(rho,(int,float)) and math.isfinite(rho),'finite current rho')
            if peers or optimistic_missing_peer:
                ratios=[a/b for a,b in zip(q,g)]
                lo=min(ratios)/(1+min(ratios));hi=max(ratios)/(1+max(ratios))
                reason='one_current_qualified_peer' if peers else 'hypothetical_one_peer_not_currently_eligible'
            else:
                lo=hi=.5;reason='no_current_qualified_peer'
        saved.require(lo-1e-12<=rho<=hi+1e-12,'current rho outside declared interval')
        reconstructed=posterior(g,q,likelihood,rho)
        saved.require(max(abs(a-b) for a,b in zip(reconstructed,current))<1e-10,'own evidence/current rho cannot reconstruct posterior')
        prior=[(1-rho)*a+rho*b for a,b in zip(g,q)]
        saved.require(max(abs(a-b) for a,b in zip(prior,row['active_structure_prior']))<1e-10,'saved active prior differs')
        models[key]=dict(instance_id=key,observed_class=label,qualified_peer_count=peers,
            reason=reason,g=g,q=q,own_log_evidence=e,likelihood=likelihood,
            current_posterior=current,current_rho=rho,rho_interval=[lo,hi],
            fixed_B_posterior=posterior(g,q,likelihood,.5),
            endpoint_posteriors=[posterior(g,q,likelihood,r) for r in (lo,hi)])
    return models


def bound_candidates(data,models,inspection_weight):
    candidates=data['candidate_snapshot'];targets={}
    for row in candidates:
        node,heading=row['target'].rsplit(':',1);targets[(node,int(heading))]=row
    scores={t:r['score'] for t,r in targets.items()}
    baseline=saved.best(scores);btarget=baseline['target']
    fixed_B={t:r['score']+sum(inspection_weight*saved.dot(r['structure_areas'][key],
        [a-b for a,b in zip(model['fixed_B_posterior'],model['current_posterior'])])/r['cost']
        for key,model in models.items()) for t,r in targets.items()}
    bbest=saved.best(fixed_B)
    absolute=[];pair_rows=[];single=[]
    # Absolute envelopes also establish whether some positive candidate must remain.
    for target,row in targets.items():
        low=high=row['score']
        for key,model in models.items():
            utility=[inspection_weight*x/row['cost'] for x in row['structure_areas'][key]]
            current=saved.dot(utility,model['current_posterior'])
            a,b=endpoint_range(model['g'],model['q'],model['likelihood'],utility,*model['rho_interval'])
            low+=a-current;high+=b-current
        absolute.append(dict(target=saved.label(target),lower=low,upper=high))
    def pair_envelope(reference,tag):
        if reference is None:return []
        base=targets[reference];rows=[]
        for target,row in targets.items():
            if target==reference:continue
            low=high=row['score']-base['score'];terms=[]
            for key,model in models.items():
                utility=[inspection_weight*(a/row['cost']-b/base['cost']) for a,b in
                    zip(row['structure_areas'][key],base['structure_areas'][key])]
                current=saved.dot(utility,model['current_posterior'])
                a,b=endpoint_range(model['g'],model['q'],model['likelihood'],utility,*model['rho_interval'])
                low+=a-current;high+=b-current
                if abs(b-a)>GUARD:
                    terms.append(dict(instance_id=key,lower_delta=a-current,upper_delta=b-current))
            # Sufficient, conservative tie guarantee under the executed epsilon
            # lexicographic rule. Multiple tied targets need not be ordered pairwise.
            excluded=(high < -TOL-GUARD) if target<reference else (high <= TOL-GUARD)
            rows.append(dict(reference_kind=tag,reference=saved.label(reference),challenger=saved.label(target),
                current_difference=row['score']-base['score'],difference_lower=low,difference_upper=high,
                challenger_lexicographically_earlier=target<reference,
                challenger_excluded_by_envelope=excluded,varying_instance_terms=terms))
        return rows
    actual_pairs=pair_envelope(btarget,'recorded_direct_winner')
    b_pairs=pair_envelope(bbest['target'],'same_state_fixed_B_direct_winner')
    pair_rows=actual_pairs+b_pairs
    any_positive_guaranteed=max(x['lower'] for x in absolute)>GUARD
    none_guaranteed=max(x['upper'] for x in absolute)<=0.
    def excluded(reference,pairs):
        if reference is None:return none_guaranteed
        return any_positive_guaranteed and all(x['challenger_excluded_by_envelope'] for x in pairs)
    for key,model in models.items():
        for side,rho in zip(('low','high'),model['rho_interval']):
            p=posterior(model['g'],model['q'],model['likelihood'],rho)
            changed={t:r['score']+inspection_weight*saved.dot(r['structure_areas'][key],
                [a-b for a,b in zip(p,model['current_posterior'])])/r['cost'] for t,r in targets.items()}
            winner=saved.best(changed)
            single.append(dict(instance_id=key,endpoint=side,rho=rho,
                selected_direct_target=saved.label(winner['target']),recorded_direct_top1_differs=winner['target']!=btarget,
                margin=winner['margin'],joint_realizability_asserted=False))
    return dict(recorded_direct_winner=saved.label(btarget),recorded_margin=baseline['margin'],
        same_state_fixed_B_direct_winner=saved.label(bbest['target']),same_state_fixed_B_margin=bbest['margin'],
        recorded_and_fixed_B_winner_equal=btarget==bbest['target'],
        all_recorded_direct_changes_excluded=excluded(btarget,actual_pairs),
        all_fixed_B_direct_changes_excluded=excluded(bbest['target'],b_pairs),
        individual_endpoint_top1_change=any(x['recorded_direct_top1_differs'] for x in single),
        any_positive_score_guaranteed=any_positive_guaranteed,no_positive_score_guaranteed=none_guaranteed,
        models=[{k:v for k,v in m.items() if k!='likelihood'} for m in models.values()],
        absolute_score_envelopes=absolute,pairwise_score_envelopes=pair_rows,
        one_instance_endpoint_winners=single,
        independent_endpoint_combinations_are_executable=False)


def declare(output):
    saved.require(not output.exists(),'fresh output required')
    selected=[];excluded=[];ledger_excerpts={}
    for phase,review_dir,allowed in (
        ('development_v1','episode_reviews_v1',None),
        ('ground_ablation_v2','episode_reviews_ground_v2',GROUND_ALLOWED)):
        raw=(STAGE/phase/'start_ledger.json').read_bytes();ledger=json.loads(raw)
        entries={r['run_id']:r for r in ledger['entries']}
        names=sorted(entries) if allowed is None else allowed
        ledger_excerpts[phase]=dict(original_ledger_sha256=saved.digest(raw),selected_entries=[])
        for run in names:
            row=entries.get(run);review=STAGE/review_dir/run
            if row is None or row['status']!='controller_stop' or row.get('qualified') is not True:
                excluded.append(dict(run_id=run,phase=phase,reason='not a qualified completed original run'));continue
            if not (review/'manifest.json').exists():
                excluded.append(dict(run_id=run,phase=phase,reason='independent review absent'));continue
            rmraw=(review/'manifest.json').read_bytes();rm=json.loads(rmraw)
            rrraw=(review/'review.json').read_bytes();rr=json.loads(rrraw)
            saved.require(saved.digest(rrraw)==rm['files']['review.json']['sha256'],'independent review SHA')
            saved.require(rr['status']=='reviewed' and rr['all_checks_passed'] is True and rr['qualified'] is True,'passed independent review')
            episode=STAGE/phase/'episodes'/run
            pin=saved.digest((episode/'artifact_manifest.json').read_bytes())
            saved.require(pin==row['artifact_manifest_sha256']==rr['input_manifest_sha256']==rm['input_episode_manifest_sha256'],'review/episode/ledger binding')
            selected.append(dict(run_id=run,phase=phase,episode=str(episode.relative_to(ROOT)),review=str(review.relative_to(ROOT)),
                review_manifest_sha256=saved.digest(rmraw),terminal_entry=row))
            ledger_excerpts[phase]['selected_entries'].append(row)
    plan=dict(schema='article.saved_rho_bounds_protocol.v1',declared_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=saved.digest(Path(__file__).read_bytes()),validator_source_sha256=saved.digest(Path(saved.__file__).read_bytes()),
        selected_runs=selected,excluded_runs=excluded,ground_allowed_exactly=list(GROUND_ALLOWED),
        primary_scope='Own evidence, current qualified classes, current at-most-one same-class peer, direct candidates and costs fixed.',
        secondary_scope='Optimistic one-peer capacity even if qualified peer currently absent; still requires an observed qualified class, not GT.',
        G_scope='G/NBV class=None remains geometry-only; no class supplied from GT or inferred as if G were S.',
        rho_formula='r_min=min(q/g), r_max=max(q/g); rho extrema=r/(1+r); no current peer fixes rho=.5 in primary scope.',
        proof='Each instance score/contrast is (alpha+beta*rho)/(gamma+delta*rho), positive denominator, constant derivative sign. Endpoints bound each term; sums give conservative box envelopes only.',
        joint_warning='Independent per-instance endpoints can be mutually inconsistent. Crossing/not excluded is not a realizable shared posterior, actual action, or quality gain.',
        missing_rule='Unavailable on missing allocated forecast, inconsistent posterior/gain reconstruction or more than one eligible peer.',
        direct_tie_rule='Same unique-target lexicographic epsilon1e-12; exclude only with conservative1e-13 guard.',
        initialization_diagnostics='Initialization is not a direct comparison; diagnostic-selected steps analyze only their recorded direct subproblem. No diagnostic channel or macro reranking.',
        discovery='Nonnegative algebraic remainder verified and fixed, not independently recomputed occupancy discovery.',
        new_worlds=0,new_policy_runs=0,new_surface_evaluations=0,new_tsdf_integrations=0)
    output.mkdir(parents=True)
    (output/'protocol.json').write_bytes(saved.canonical(plan))
    (output/'ledger_excerpts.json').write_bytes(saved.canonical(ledger_excerpts))
    return plan


def analyze(output):
    plan=json.loads((output/'protocol.json').read_bytes())
    saved.require(plan['script_sha256']==saved.digest(Path(__file__).read_bytes()),'analysis source changed after declaration')
    saved.require(plan['validator_source_sha256']==saved.digest(Path(saved.__file__).read_bytes()),'input reconstruction validator changed')
    saved.require(not (output/'summary.json').exists(),'fresh declared output required')
    inputs=saved.SealedInputs();rows=[];bounds=[];run_summaries=[]
    for item in plan['selected_runs']:
        episode=ROOT/item['episode'];review=ROOT/item['review'];run=item['run_id'];entry=item['terminal_entry']
        rm=inputs.json(review/'manifest.json',expected=item['review_manifest_sha256'])
        for name,pin in rm['files'].items():inputs.read(review/name,expected=pin['sha256'],expected_bytes=pin['bytes'])
        manifest=inputs.json(episode/'artifact_manifest.json',expected=entry['artifact_manifest_sha256'])
        result=inputs.artifact(episode,manifest,'result.json')
        saved.require(manifest['files']['result.json']['sha256']==entry['result_sha256'],'result/ledger pin')
        config=inputs.artifact(episode,manifest,'controller_final.json')['configuration']
        protocol=inputs.artifact(episode,manifest,'protocol.json');slot=protocol['slots'][run]
        visited=set();first_return=None;steps=0;run_rows=[]
        for name in sorted(manifest['files']):
            if not name.startswith('steps/') or not name.endswith('.json.gz'):continue
            step=inputs.artifact(episode,manifest,name);index=step['accounting']['paid_step']
            saved.require(index==steps,'consecutive saved paid frames');steps+=1
            d=step['decision'];visited.add(step['controller_evidence']['safety']['current_state']['node'])
            reason=(d.get('routing') or {}).get('reason')
            if reason=='return' and first_return is None:first_return=index
            if not d.get('global_replanned'):continue
            selection=d.get('global_selection') or {};chosen=selection.get('selected') or {}
            common=dict(run_id=run,phase=item['phase'],method=slot['method'],scene_id=slot['scene_id'],
                paid_step=index,actual_selected_kind=chosen.get('kind','none'),
                return_timing='before_first_return' if first_return is None else 'at_or_after_first_return',
                actual_selected_target=saved.label(saved.target(chosen['target'])) if 'target' in chosen else None)
            if not selection.get('direct_options'):
                run_rows.append(dict(**common,status='not_direct_comparison',reason='initialization_or_no_direct_pool'));continue
            try:
                data=saved.analyze_replan(step,config,visited)
                row=dict(**common,status='analyzed',direct_candidates=data['direct_candidates'],
                    any_positive_area=data['any_positive_area'],any_structure_dependent_area=data['any_structure_dependent_area'],
                    baseline_is_geometry_only_by_design=slot['method'] in ('G','NBV'))
                for scope,optimistic in (('current_eligibility',False),('hypothetical_one_peer',True)):
                    models=models_from_belief(step['controller_evidence']['structure_belief'],optimistic_missing_peer=optimistic)
                    bound=bound_candidates(data,models,config['inspection_weight'])
                    fields=('recorded_direct_winner','same_state_fixed_B_direct_winner','recorded_and_fixed_B_winner_equal',
                        'all_recorded_direct_changes_excluded','all_fixed_B_direct_changes_excluded','individual_endpoint_top1_change')
                    row[scope]={k:bound[k] for k in fields}
                    row[scope]['variable_instances']=sum(m['rho_interval'][0]!=m['rho_interval'][1] for m in models.values())
                    bounds.append(dict(**common,scope=scope,**bound))
                run_rows.append(row)
            except (ValueError,KeyError,TypeError) as error:
                run_rows.append(dict(**common,status='unavailable',reason=str(error)))
        saved.require(steps==result['executed_paid_actions']+1,'terminal frame count')
        counts=Counter()
        for row in run_rows:
            counts[row['status']]+=1
            if row['status']!='analyzed':continue
            counts['all_zero_surface_pools']+=not row['any_positive_area']
            for scope in ('current_eligibility','hypothetical_one_peer'):
                r=row[scope];counts[scope+':variable_pool']+=r['variable_instances']>0
                counts[scope+':direct_change_excluded']+=r['all_fixed_B_direct_changes_excluded']
                counts[scope+':direct_change_not_excluded']+=not r['all_fixed_B_direct_changes_excluded']
                counts[scope+':single_endpoint_change']+=r['individual_endpoint_top1_change']
                if row['return_timing']=='before_first_return':
                    counts[scope+':pre_return_not_excluded']+=not r['all_fixed_B_direct_changes_excluded']
        run_summaries.append(dict(run_id=run,phase=item['phase'],method=slot['method'],counts=dict(counts),
            first_return_step=first_return,
            current_eligibility_not_excluded_steps=[r['paid_step'] for r in run_rows if r.get('status')=='analyzed' and not r['current_eligibility']['all_fixed_B_direct_changes_excluded']],
            hypothetical_one_peer_not_excluded_steps=[r['paid_step'] for r in run_rows if r.get('status')=='analyzed' and not r['hypothetical_one_peer']['all_fixed_B_direct_changes_excluded']]))
        rows.extend(run_rows)
    totals=Counter()
    for row in run_summaries:totals.update(row['counts'])
    summary=dict(schema='article.saved_rho_bounds_summary.v1',runs=run_summaries,totals=dict(totals),
        protocol_sha256=saved.digest((output/'protocol.json').read_bytes()),new_worlds=0,new_policy_runs=0,
        new_surface_evaluations=0,new_tsdf_integrations=0,independent_trajectories_added=0,
        exclusions='Proof only for fixed recorded direct pools, own evidence, category eligibility and cost model.',
        not_excluded='Conservative envelope can cross; neither a jointly realizable posterior nor an actual change is asserted.',
        G_limitation='G/NBV no-category states are structurally fixed by design, not evidence that hypothetical S has zero capacity.',
        diagnostic_limitation='Current direct subproblem only, not the selected diagnostic macro or full future policy.')
    (output/'summary.json').write_bytes(saved.canonical(summary));(output/'replans.json').write_bytes(saved.canonical(rows))
    (output/'bounds.json').write_bytes(saved.canonical(bounds))
    saved.write_csv(output/'replans.csv',rows)
    lines=['# One-peer rho interval bounds on saved direct candidates','',
        'Fixed own evidence, recorded costs and area vectors. Endpoint sums are conservative envelopes; not-excluded is not an observed or executable benefit.','',
        '| Run | Direct pools | Variable current-peer pools | Current peer: changes not excluded | Hypothetical peer: changes not excluded |',
        '|---|---:|---:|---:|---:|']
    for r in run_summaries:
        c=r['counts'];lines.append(f"| {r['run_id']} | {c.get('analyzed',0)} | {c.get('current_eligibility:variable_pool',0)} | {c.get('current_eligibility:direct_change_not_excluded',0)} | {c.get('hypothetical_one_peer:direct_change_not_excluded',0)} |")
    lines+=['','G/NBV rows preserve their deliberately absent semantic class. No private class/GT information is inserted.',
        'Initialization is separated. At diagnostic-selected steps, only the direct subproblem is bounded. Missing or inconsistent records are unavailable.',
        'Single-instance score contrasts are linear fractional in rho with positive denominator; their derivative sign is constant, so extrema lie at endpoints. Summing extrema over instances is conservative and can combine incompatible endpoints.',
        'Full per-candidate contrast bounds, endpoint posteriors and exact current selection kinds are retained in bounds.json. No frozen execution source is modified.','']
    (output/'README.md').write_text('\n'.join(lines))
    for name,pin in inputs.pins.items():saved.require(saved.digest((ROOT/name).read_bytes())==pin['sha256'],'saved input changed during analysis')
    manifest=dict(schema='article.saved_rho_bounds_manifest.v1',inputs=inputs.pins,
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=saved.digest(p.read_bytes())) for p in output.iterdir() if p.is_file()},
        source_sha256={str(Path(__file__).relative_to(ROOT)):saved.digest(Path(__file__).read_bytes()),
                       str(Path(saved.__file__).relative_to(ROOT)):saved.digest(Path(saved.__file__).read_bytes())},
        new_worlds=0,new_policy_runs=0)
    (output/'manifest.json').write_bytes(saved.canonical(manifest))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('declare','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    saved.require(output.is_relative_to(STAGE/'diagnostics'),'isolated diagnostic output required')
    value=declare(output) if args.command=='declare' else analyze(output)
    print(json.dumps(dict(declared=str(output),runs=len(value['selected_runs'])) if args.command=='declare' else value,ensure_ascii=False))
