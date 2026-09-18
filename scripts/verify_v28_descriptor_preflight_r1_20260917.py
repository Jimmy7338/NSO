#!/usr/bin/env python3
"""Independent saved-JSON/hash/arithmetic review; no project runtime imports.

This verifier never creates a mapper/world, loads raw RGBD arrays, plans routes,
extracts meshes, or evaluates Q. Wait for the producer's completed result before
running it. Full-route safety remains a producer execution assertion. Stored raw support
permits independent eligibility and ranking arithmetic without geometry calls.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audit_results/v28_descriptor_preflight_r1_20260917'
OUTPUT=ROOT/'audit_results/v28_descriptor_preflight_r1_review_20260917'
CHECKPOINTS=list(range(0,401,50))
MODES={'G','O','S','X','L','S_no_reservation'}
DIFF_KEYS=('selection_S_vs_O','selection_S_vs_X','ranking_S_vs_O',
           'ranking_S_vs_X','shortlist_S_vs_O','reservation_changes_shortlist')
NULL_FIELDS=('distinct_viewpoints','effective_baseline_m','boundary_fragmentation',
             'normal_stability','calibrated_quality_uncertainty')
CAP=100*1024
RESERVE=64*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        return json.load(stream)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def close(actual,expected,message):
    if actual is None or expected is None:require(actual is expected,message)
    else:require(math.isfinite(actual) and math.isfinite(expected) and
                 math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),message)


def hash_string(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def inventory(root):
    saved=read(root/'artifact_hashes.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'artifact_hashes.json'}
    require(set(saved)==actual,'artifact set differs: '+str(root))
    for name,expected in saved.items():require(sha(root/name)==expected,'artifact SHA: '+name)
    return dict(entries=len(saved),sha256=sha(root/'artifact_hashes.json'))


def verify_sources(source,manifest):
    paths=manifest['source_sha256']
    require(sha(source/'sources.zip')==manifest['source_archive_sha256'],'preflight source ZIP SHA')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        require(len(archive.namelist())==len(paths) and set(archive.namelist())==set(paths),'ZIP file set')
        for name,expected in paths.items():
            require(sha(ROOT/name)==expected,'current producer source changed: '+name)
            require(hashlib.sha256(archive.read(name)).hexdigest()==expected,'ZIP member differs: '+name)
    return dict(count=len(paths),all_current_and_archived_sources_match=True,
                source_archive_sha256=manifest['source_archive_sha256'])


def pose(row):
    return None if row is None else tuple(row['pose'])


def classes(row):
    return {e['class_id'] for e in row['semantic_evidence'] if e['hypothesis_gain']>0}


def selected_expected(rows):
    return rows[0] if rows and rows[0]['score']>0 else None


def answer_check(mode,answer,snapshot,cues):
    rows=answer['candidates'];audit=answer['audit'];action=snapshot['action_id']
    require(len(rows)<=12 and len({pose(row) for row in rows})==len(rows),'shortlist capacity/duplicate pose')
    require(hash_string(answer['route_identity_sha256']),'route hash format')
    key=lambda row:(-row['score'],row['outbound_cost'],row['cost'],tuple(row['pose']))
    require(rows==sorted(rows,key=key),'shortlist score/tie order: '+mode)
    require(answer['selected']==selected_expected(rows),'selected not positive best shortlisted row: '+mode)
    for row in rows:
        require(len(row['pose'])==3 and all(type(x) is int for x in row['pose']),'pose schema')
        require(all(type(row[x]) is int and row[x]>=0 for x in ('outbound_cost','return_cost','cost')),'paid costs')
        require(row['cost']==row['outbound_cost']+row['return_cost'] and row['cost']<=400-action,'cost/budget sum')
        require(math.isfinite(row['score']) and math.isfinite(row['geometry_gain']),'finite geometry/score')
        evidence=row['semantic_evidence']
        close(row['semantic_gain'],sum(e['hypothesis_gain'] for e in evidence),'semantic gain sum')
        require(len({e['cue_id'] for e in evidence})==len(evidence),'duplicate cue evidence')
        for e in evidence:
            require(e['cue_id'] in cues and e['hypothesis_gain']>0,'evidence must concern observed positive cue')
            cue=cues[e['cue_id']]
            expected_class=None if mode=='O' else 5-cue['class_id'] if mode=='X' else cue['class_id']
            require(e['class_id']==expected_class,'class intervention does not match declared mode')
            require(e['calibrated'] is False and 0<=e['sector']<8,'uncalibrated sector evidence')
            require(0<e['visible_patches']<=e['local_patches'],'visible/local support')
            close(e['visible_fraction'],e['visible_patches']/e['local_patches'],'visible fraction')
            d=snapshot['descriptors'][e['cue_id']]
            require(e['local_patches']==d['patch_count'],'descriptor neighborhood count')
            require(0<=e['novel_direction_patches']<=e['visible_patches'],'novel visible support')
            close(e['candidate_direction_deficit'],e['novel_direction_patches']/e['visible_patches'],'novel/visible deficit')
            close(e['observed_direction_deficit'],e['candidate_direction_deficit'],'candidate direction deficit')
            require(0<=e['normalized_prior_weight']<=1,'prior weight range')
            if mode=='O':close(e['normalized_prior_weight'],1/8,'uniform O prior')
            close(e['hypothesis_gain'],.05*cue['confidence']*8*e['normalized_prior_weight']*
                  e['observed_direction_deficit']*e['visible_fraction'],'hypothesis arithmetic')
    require(audit['evaluation_truth_used'] is False and audit['forced_class_selection'] is False
            and audit['predictions_calibrated'] is False,'declared limits')
    retained=sorted({e['cue_id'] for row in rows for e in row['semantic_evidence']})
    selected=[] if answer['selected'] is None else [e['cue_id'] for e in answer['selected']['semantic_evidence']]
    require(audit['retained_cue_ids']==retained and audit['selected_cue_ids']==selected,'retained/selected cue arithmetic')
    eligible=audit['eligible_cue_ids']
    require(eligible==sorted(set(eligible)) and set(eligible).issubset(cues),'eligible IDs schema')
    require(set(retained).issubset(eligible),'retained cues outside producer eligible set')
    if audit['exact_geometry_fallback']:
        require(not retained and not eligible,'fallback has semantic evidence')
    else:
        require(audit['candidate_count']==len(rows) and audit['candidate_limit']==12,'candidate counts')
        require(audit['raw_pool_size']==snapshot['raw_pool_size']>=len(rows),'pool size')
        require(audit['semantic_route_queries']==snapshot['semantic_route_queries'],'query count')
        require(audit['observed_geometry_sha256']==snapshot['geometry_sha256'],'geometry audit SHA')
        require(audit['class_independent_proposal_pool'] is True,'pool declaration')
        require(audit['instance_reservation_enabled']==(mode!='S_no_reservation'),'reservation setting')
        require(audit['dropped_eligible_cue_ids']==sorted(set(eligible)-set(retained)),'dropped eligible arithmetic')
        require(set(audit['reserved_cue_ids']).issubset(eligible),'reserved cue eligibility')
        require(0<=audit['coverage_routes_retained']<=min(4,sum(r['group'].startswith('coverage_') for r in rows)),
                'retained coverage count')
        require(audit['mu_delta_Q'] is None and audit['sigma_delta_Q'] is None
                and audit['actual_C80_guaranteed'] is False,'quality/coverage overclaim')


def prior_weights(cue,mode):
    if mode=='O':return [1/8]*8
    x,y=cue['outward'][:2];norm=math.hypot(x,y);x/=norm;y/=norm
    directions=((x,y),(-x,-y),(-y,x),(y,-x))
    category=5-cue['class_id'] if mode=='X' else cue['class_id']
    masses=(.15,.35,.25,.25) if category==3 else (.6,.1,.15,.15)
    kernels=[]
    for k in range(8):
        angle=-math.pi+(k+.5)*math.pi/4
        kernels.append([max(0.,math.cos(angle)*dx+math.sin(angle)*dy)**2 for dx,dy in directions])
    sums=[sum(row[j] for row in kernels) for j in range(4)]
    weights=[sum(row[j]/sums[j]*masses[j] for j in range(4)) for row in kernels]
    total=sum(weights)
    return [weight/total for weight in weights]


def raw_pool_check(snapshot,cues,known_fraction):
    """Recompute r1 using saved candidate-relative novel/visible support.

    Raw visible and novel counts remain recorded geometric inputs; this verifier
    checks their bounds and all downstream arithmetic without reconstructing them.
    """
    pool=snapshot['raw_pool_observed_support']
    require(len(pool)==snapshot['raw_pool_size'] and len({pose(row) for row in pool})==len(pool),
            'raw observed support size/unique poses')
    checked=0;capacity=None
    for mode in ('O','S','X','S_no_reservation'):
        answer=snapshot['answers'][mode]
        if answer['audit']['exact_geometry_fallback']:continue
        rows=[]
        for raw in pool:
            gains={}
            for cue_id,geometry in raw['cue_geometry'].items():
                require(cue_id in cues and cues[cue_id]['confidence']>=.6,'raw support trusted cue')
                cue=cues[cue_id];k=geometry['sector'];d=snapshot['descriptors'][cue_id]
                require(0<=k<8 and geometry['local_patches']==d['patch_count'],'raw neighborhood/sector')
                require(0<geometry['visible_patches']<=geometry['local_patches'],'raw visible support')
                close(geometry['visible_fraction'],geometry['visible_patches']/geometry['local_patches'],'raw visible fraction')
                weight=prior_weights(cue,mode)[k]
                require(0<=geometry['novel_direction_patches']<=geometry['visible_patches'],'raw novel support')
                close(geometry['candidate_direction_deficit'],geometry['novel_direction_patches']/geometry['visible_patches'],'raw novel/visible deficit')
                gain=.05*cue['confidence']*8*weight*geometry['candidate_direction_deficit']*geometry['visible_fraction']
                if gain>0:gains[cue_id]=dict(gain=gain,weight=weight)
            rows.append(dict(raw=raw,pose=pose(raw),gains=gains,
                score=raw['geometry_score']+known_fraction*sum(g['gain'] for g in gains.values())/max(1,raw['outbound_cost'])))
        by_pose={row['pose']:row for row in rows}
        eligible=sorted({cue_id for row in rows for cue_id in row['gains']})
        require(eligible==answer['audit']['eligible_cue_ids'],'independent raw eligibility: '+mode)
        for selected in answer['candidates']:
            raw=by_pose[pose(selected)];evidence={e['cue_id']:e for e in selected['semantic_evidence']}
            require(set(evidence)==set(raw['gains']),'raw-to-shortlist cue support')
            for cue_id,amount in raw['gains'].items():
                close(evidence[cue_id]['normalized_prior_weight'],amount['weight'],'independent class prior')
                close(evidence[cue_id]['hypothesis_gain'],amount['gain'],'independent raw gain')
            close(selected['score'],raw['score'],'independent raw score')
            # Stored bits remain authoritative for exact tie ordering after
            # independently checking floating arithmetic within 1e-12.
            raw['score']=selected['score']
            for key in ('group','outbound_cost','cost'):require(selected[key]==raw['raw'][key],'raw shortlist geometry')
        order=lambda row:(-row['score'],row['raw']['outbound_cost'],row['raw']['cost'],row['pose'])
        coverage=sorted((row for row in rows if row['raw']['group'].startswith('coverage_')),key=order)[:4]
        chosen={row['pose']:row for row in coverage};reserved=[]
        if mode!='S_no_reservation':
            for cue_id in eligible:
                if any(cue_id in row['gains'] for row in chosen.values()):continue
                if len(chosen)==12:break
                candidate=min((row for row in rows if cue_id in row['gains']),key=order)
                chosen[candidate['pose']]=candidate;reserved.append(cue_id)
        for row in sorted(rows,key=order):
            if len(chosen)>=12:break
            chosen.setdefault(row['pose'],row)
        expected=[row['pose'] for row in sorted(chosen.values(),key=order)]
        require(expected==[pose(row) for row in answer['candidates']],'raw reservation/shortlist reconstruction: '+mode)
        require(reserved==answer['audit']['reserved_cue_ids'],'reserved cue arithmetic')
        if mode=='S':
            complex_ids={key for key,cue in cues.items() if cue['class_id']==3 and cue['confidence']>=.6}
            complex_rows=[row for row in rows if set(row['gains']).intersection(complex_ids)]
            retained_complex=[row for row in answer['candidates'] if 3 in classes(row)]
            witness={row['pose']:row for row in coverage}
            witness.update({row['pose']:row for row in complex_rows})
            for cue_id in eligible:
                if not any(cue_id in row['gains'] for row in witness.values()):
                    chosen_row=min((row for row in rows if cue_id in row['gains']),key=order)
                    witness[chosen_row['pose']]=chosen_row
            for row in witness.values():
                require(0<=row['raw']['outbound_cost']<=row['raw']['cost']<=400-snapshot['action_id'],
                        'stored witness paid budget')
            capacity=dict(raw_complex_positive_rows=len(complex_rows),
                retained_complex_rows=len(retained_complex),raw_complex_dropped_by_shortlist=len(complex_rows)-len(retained_complex),
                raw_complex_candidate_poses=[row['pose'] for row in complex_rows],
                retained_complex_poses=[pose(row) for row in retained_complex],
                same_length_capacity_upper_bound=min(len(complex_rows),len(answer['candidates'])),
                mandatory_coverage_rows=len(coverage),all_complex_plus_coverage_and_eligible_cues_rows=len(witness),
                all_complex_fit_same_shortlist_length=len(witness)<=len(answer['candidates']),
                mandatory_union_witness_poses=sorted(witness),
                witness_scope='Saved candidate set capacity witness only; no new route planning, safety reconstruction, or policy/quality claim.')
        checked+=1
    if capacity is None:
        capacity=dict(raw_complex_positive_rows=0,retained_complex_rows=0,raw_complex_dropped_by_shortlist=0,
            raw_complex_candidate_poses=[],retained_complex_poses=[],same_length_capacity_upper_bound=0,
            mandatory_coverage_rows=None,all_complex_plus_coverage_and_eligible_cues_rows=None,
            all_complex_fit_same_shortlist_length=True,mandatory_union_witness_poses=[],
            witness_scope='Geometry fallback: no positive semantic candidate.')
    return dict(modes_checked=checked,complex_candidate_capacity=capacity)


def snapshot_check(snapshot,source_audit,known_fraction):
    action=snapshot['action_id'];answers=snapshot['answers'];cues={c['cue_id']:c for c in snapshot['cues']}
    require(set(answers)==MODES,'six modes required')
    require(snapshot['cues']==source_audit['cues'] and snapshot['geometry_sha256']==source_audit['geometry_sha256'],
            'original cue/geometry identity')
    require(len(cues)==len(snapshot['cues']),'unique cue IDs')
    require(snapshot['sampled_patches']==min(256,snapshot['full_patches']),'sampled/full patch count')
    require(hash_string(snapshot['full_geometry_sha256']) and hash_string(snapshot['class_independent_raw_pool_sha256']),
            'full-state/pool hash formats')
    require(set(snapshot['descriptors'])==set(cues),'descriptor cue coverage')
    for cue_id,d in snapshot['descriptors'].items():
        require(d['cue_id']==cue_id and d['action_id']==action,'descriptor identity')
        require(0<=d['patch_count']<=snapshot['full_patches'],'descriptor patch count')
        require(d['directional_deficit'] is None,'r1 cue-centered deficit unavailable')
        require(len(d['patch_relative_directional_support'])==8,'eight patch-relative bins')
        require(all(0<=support<=1 for support in d['patch_relative_directional_support']),'patch support bounds')
        require(d['direction_support_reference']=='each measured patch; not cue-centered support','explicit patch origin')
        require(all(d[k] is None for k in NULL_FIELDS),'unavailable descriptor fields must be null')
        require(d['unavailable_fields_are_not_zero'] is True and all(d[k] is False for k in
            ('instance_segmentation','predicts_Q','evaluation_truth_used')),'descriptor scope')
    receipts=snapshot['counterfactual_checks']
    require(hash_string(receipts['objectness_sha256']) and receipts['objectness_sha256']==
            receipts['swapped_class_objectness_sha256'],'O class-swap equality receipt')
    require(receipts['actual_low_confidence']==.59 and receipts['low_confidence_route_queries']==0,'actual low-confidence receipt')
    require(receipts['G_equals_original'] is True and receipts['L_equals_original'] is True,'full fallback equality receipts')
    for key in ('candidates','selected','route_identity_sha256'):
        require(answers['G'][key]==answers['L'][key],'saved G/L equality: '+key)
    require(answers['G']['audit']['exact_geometry_fallback'] and answers['L']['audit']['exact_geometry_fallback'],'G/L fallback')
    for mode,answer in answers.items():answer_check(mode,answer,snapshot,cues)
    raw_modes=raw_pool_check(snapshot,cues,known_fraction)
    # Shared rows expose the same geometry and paid costs even when ranking differs.
    shared={}
    for mode in ('O','S','X','S_no_reservation'):
        for row in answers[mode]['candidates']:
            key=pose(row);geom={k:row[k] for k in ('pose','group','outbound_cost','return_cost','cost','geometry_gain')}
            if key in shared:require(geom==shared[key],'class intervention changed shared candidate geometry')
            else:shared[key]=geom
    original={pose(row):row for row in answers['S']['candidates']}
    for row in answers['S_no_reservation']['candidates']:
        if pose(row) in original:require(row==original[pose(row)],'reservation changed same candidate score/evidence')
    short=answers['S']['candidates'];eligible=answers['S']['audit']['eligible_cue_ids']
    require(snapshot['positive_eligible_cue_ids']==eligible,'eligible receipt alias')
    computed=dict(gate_eligible={c['class_id'] for c in cues.values()}=={2,3} and
        bool({c['cue_id'] for c in cues.values() if c['class_id']==3}.intersection(eligible)),
        complex_shortlist_rows=sum(3 in classes(row) for row in short),
        positive_semantic_shortlist_rows=sum(bool(row['semantic_evidence']) for row in short),
        shortlist_rows=len(short),all_eligible_cues_retained=set(eligible).issubset(answers['S']['audit']['retained_cue_ids']))
    choices={mode:pose(answer['selected']) for mode,answer in answers.items()}
    ordered={mode:[pose(row) for row in answer['candidates']] for mode,answer in answers.items()}
    computed.update(selection_S_vs_O=choices['S']!=choices['O'],selection_S_vs_X=choices['S']!=choices['X'],
        ranking_S_vs_O=ordered['S']!=ordered['O'],ranking_S_vs_X=ordered['S']!=ordered['X'],
        shortlist_S_vs_O=set(ordered['S'])!=set(ordered['O']),
        reservation_changes_shortlist=set(ordered['S'])!=set(ordered['S_no_reservation']))
    for key,value in computed.items():require(snapshot[key]==value,'snapshot arithmetic: '+key)
    return dict(action_id=action,**computed,objectness_swap_receipt_equal=True,saved_G_L_equal=True,
                selected_and_retention_arithmetic_checked=True,
                raw_eligibility_and_shortlist_modes_recomputed=raw_modes['modes_checked'],
                complex_candidate_capacity=raw_modes['complex_candidate_capacity'],
                selected_poses={mode:pose(answer['selected']) for mode,answer in answers.items()},
                selected_actual_observed_classes={mode:sorted({cues[e['cue_id']]['class_id'] for e in
                    ([] if answer['selected'] is None else answer['selected']['semantic_evidence'])})
                    for mode,answer in answers.items()})


def summarize(index,rows):
    eligible=[row for row in rows if row['gate_eligible']]
    numerator=sum(row['complex_shortlist_rows'] for row in eligible)
    denominator=sum(row['shortlist_rows'] for row in eligible)
    semantic_denominator=sum(row['positive_semantic_shortlist_rows'] for row in eligible)
    share=numerator/denominator if denominator else None
    return dict(case_index=index,snapshots=len(rows),eligible_snapshots=len(eligible),
        complex_shortlist_rows=numerator,all_shortlist_rows=denominator,
        positive_semantic_shortlist_rows=semantic_denominator,complex_share_all_shortlist=share,
        complex_share_semantic_rows=numerator/semantic_denominator if semantic_denominator else None,
        complex_20pct_gate_passed=share is not None and share>=.20,
        all_eligible_cues_retained=all(row['all_eligible_cues_retained'] for row in rows),
        ten_consecutive_plan_absence_gate='not_tested_sparse_snapshots',
        **{key:sum(row[key] for row in rows) for key in DIFF_KEYS})


def self_test():
    base=dict(gate_eligible=True,complex_shortlist_rows=2,shortlist_rows=12,
        positive_semantic_shortlist_rows=4,all_eligible_cues_retained=True,
        **{key:False for key in DIFF_KEYS})
    summary=summarize(1,[base])
    require(summary['complex_share_all_shortlist']==1/6 and summary['complex_share_semantic_rows']==.5,
            'self-test distinguishes denominator')
    require(not summary['complex_20pct_gate_passed'],'semantic denominator must not pass primary gate')
    require(summarize(1,[dict(base,complex_shortlist_rows=2,shortlist_rows=10)])['complex_20pct_gate_passed'],
            'exact .20 threshold')
    require(summarize(1,[dict(base,gate_eligible=False)])['complex_share_all_shortlist'] is None,'empty pool unavailable')
    require(summarize(1,[dict(base,all_eligible_cues_retained=False)])['all_eligible_cues_retained'] is False,'retention failure')
    require(selected_expected([dict(score=0)]) is None and selected_expected([dict(score=.1)])==dict(score=.1),
            'positive score selection')
    print('6 pure-metadata self-checks passed; no source result opened')


def run(source,output):
    # Missing/incomplete producer results never create a misleading review directory.
    result=read(source/'result.json');require(result['status']=='complete','producer not complete')
    require(not output.exists(),'refuse existing review evidence')
    require(shutil.disk_usage(output.parent).free>CAP+RESERVE,'review capacity')
    output.mkdir();started=perf_counter()
    def write(name,value):
        data=value.encode() if isinstance(value,str) else (canonical(value)+'\n').encode()
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file())+len(data)<CAP,'100 KiB review cap')
        require(shutil.disk_usage(output).free-len(data)>RESERVE,'review reserve')
        with (output/name).open('xb') as stream:stream.write(data)
    try:
        manifest=read(source/'manifest.json');source_inventory=inventory(source)
        sources=verify_sources(source,manifest)
        original=ROOT/manifest['source_root'];old=read(original/'manifest.json')
        original_inventory=inventory(original)
        require(original_inventory['sha256']==manifest['original_inventory_sha256'],'input inventory frozen link')
        require(len(old['source_sha256'])==manifest['original_frozen_sources_verified'],'old source count')
        for name,expected in old['source_sha256'].items():require(sha(ROOT/name)==expected,'old frozen source changed')
        require(manifest['checkpoints']==CHECKPOINTS and manifest['cases']==[1,3],'predeclared snapshot matrix')
        require(manifest['evaluation_fields_forwarded'] is False,'evaluation access declaration')
        require('docs/research/V28_DESCRIPTOR_PREFLIGHT_R1_PROTOCOL_20260917.md' in manifest['source_sha256'],'frozen protocol missing')
        reviews=[];summaries=[];capacity_summaries=[];selection_summaries=[]
        for index in (1,3):
            case=read(source/f'case_{index:02d}.json')
            original_record=read(original/f'case_{index:02d}/result.json')
            grid_size=math.prod(original_record['shape'])
            require([s['action_id'] for s in case['snapshots']]==CHECKPOINTS,'complete fixed nine snapshots')
            rows=[snapshot_check(snapshot,read(original/f'case_{index:02d}/audit/{snapshot["action_id"]:04d}.json.gz'),
                  original_record['trace'][snapshot['action_id']]['known_public_roi_cells']/grid_size)
                  for snapshot in case['snapshots']]
            summary=summarize(index,rows)
            require(summary==case['summary'],'case summary independently recomputed')
            require(summary==next(row for row in result['summaries'] if row['case_index']==index),'root summary')
            reviews.append(dict(case_index=index,snapshots=rows));summaries.append(summary)
            eligible_rows=[row for row in rows if row['gate_eligible']]
            upper=sum(row['complex_candidate_capacity']['same_length_capacity_upper_bound'] for row in eligible_rows)
            denominator=summary['all_shortlist_rows']
            capacity_summaries.append(dict(case_index=index,fixed_denominator=denominator,
                retained_complex_rows=summary['complex_shortlist_rows'],raw_capacity_upper_numerator=upper,
                raw_capacity_upper_share=upper/denominator if denominator else None,
                raw_capacity_permits_20pct=bool(denominator and upper/denominator>=.20),
                all_eligible_snapshots_have_all_complex_capacity_witness=all(row['complex_candidate_capacity']['all_complex_fit_same_shortlist_length'] for row in eligible_rows),
                retained_vs_capacity_gap_rows=upper-summary['complex_shortlist_rows'],
                conclusion='Observed shortlist truncation can explain a deficit relative to the raw capacity upper bound; retaining additional candidates does not establish better selection or Q.'))
            selection_summaries.append(dict(case_index=index,gate_eligible_snapshots=len(eligible_rows),
                S_selected_complex_in_gate_eligible_snapshots=sum(3 in row['selected_actual_observed_classes']['S'] for row in eligible_rows),
                S_selected_complex_actions=[row['action_id'] for row in rows if 3 in row['selected_actual_observed_classes']['S']],
                S_O_actual_selection_differences=[{key:row[key] for key in
                    ('action_id','gate_eligible','selected_poses','selected_actual_observed_classes')}
                    for row in rows if row['selection_S_vs_O']]))
        expected=dict(saved_packets=802,mapper_updates=802,geometry_plans=18,rank_calls=126,
            new_worlds=0,new_actions=0,new_sensor_packets=0,mesh_extractions=0,Q_evaluations=0)
        require(result['counters']==expected,'producer work counters')
        require(all(value==0 for value in result['world_tripwire_counters'].values()),'world tripwire violation')
        require(result['both_complex_share_gates_passed']==all(s['complex_20pct_gate_passed'] for s in summaries),'both-case gate')
        require(result['all_eligible_cues_retained']==all(s['all_eligible_cues_retained'] for s in summaries),'retention aggregate')
        require(all(result[k] is True for k in ('behavioral_checks_passed','original_inventory_unchanged','source_unchanged')),'producer completed checks')
        require(result['main_tasks_used']==12 and all(result[k] is False for k in
            ('autonomous_gain_proven','semantic_gain_proven','Q_calibrated','main_task_permission')),'scope/permission limits')
        out=dict(status='artifact_arithmetic_verified_r1_mechanism_gate_failed',source=str(source.relative_to(ROOT)),
            method_accepted=False,r1_uses_candidate_novel_visible_deficit=True,mechanism_gate_passed=False,
            source_manifest_sha256=sha(source/'manifest.json'),source_result_sha256=sha(source/'result.json'),
            source_inventory=source_inventory,source_archive_and_live_files=sources,
            original_inventory=original_inventory,original_frozen_sources_checked=len(old['source_sha256']),
            cases=reviews,summaries=summaries,complex_candidate_capacity_summaries=capacity_summaries,
            selection_diagnostics=selection_summaries,
            counterfactual_receipts_checked=18,
            producer_counters_verified=expected,review_cost=dict(worlds=0,raw_packet_loads=0,mapper_updates=0,
            plans=0,mesh_extractions=0,Q_evaluations=0,physical_actions=0,elapsed_s=perf_counter()-started),
            counter_scope='Top-level counters are the actual work record. Shared world_tripwire_counters generic mapper/TSDF/decision counters are not connected here; their zeros must not erase the 802 mapper updates and 18/126 plan/rank calls.',
            limitations=[
                'O and class-swapped-O full answers are not serialized: matching SHA receipts plus frozen executed equality assertion support invariance; this reviewer does not reconstruct those full answers.',
                'G/L compact candidates, selected records and route SHA match directly; full original-G equality additionally rests on producer execution receipts.',
                'Raw proposal observed-support rows permit independent eligibility/prior/score/reservation arithmetic; observed support counts themselves are not rebuilt. Full route states and observed map arrays are omitted, so route-hash preimages and per-cell safety are not independently reconstructed.',
                'All shares and selected/retained/dropped/ranking/shortlist arithmetic are independently recomputed, including raw eligible IDs under the frozen r1 formula; observing arithmetic agreement does not validate the underlying geometry counts or establish useful outcomes.',
                'Descriptors retain null missing measurements; no calibrated Q, semantic efficacy, autonomous gain, or ten-consecutive-decision absence result is inferred from 18 sparse development snapshots.'])
        write('result.json',out)
        table='\n'.join(f"| {s['case_index']:02d} | {s['eligible_snapshots']} | {s['complex_shortlist_rows']}/{s['all_shortlist_rows']} | {s['complex_share_all_shortlist']} | {s['complex_20pct_gate_passed']} | {s['all_eligible_cues_retained']} |" for s in summaries)
        report='''# V28 r1 描述符预检独立结果复核（2026-09-17）

18 个预定快照的证据清单、源码 ZIP 与当前源码、原输入封存、反事实收据及汇总算术均通过独立复核，**两排列共同机制门仍未通过，不能验收为有效方案**。r1 将候选方向不足改为同一可见面片集合中的 novel/visible；本次重算该公式，不重建所依赖的几何计数。首轮有已知坐标原点缺陷，证据保持独立，不混入本版。本程序只读 JSON/gzip/源码字节，不读取 RGBD 数组，不创建 mapper/world，不规划、不提网格、不评价 Q。

| case | 合格快照数 | 复杂类别短名单行 / 全部短名单行 | 固定主份额 | ≥20%门 | eligible 实例均保留 |
|---|---:|---:|---:|---|---|
'''+table+'''

每快照核对 O 与交换类别 O 的完整答案 SHA 收据相同，实际低置信 .59、零语义路线查询；直接比较保存的 G/L 候选、选择与路线摘要。从 raw_pool_observed_support、描述符与 cues 独立重算 r1 的 candidate novel/visible、prior/gain/score、raw eligibility 和保留/短名单步骤。所有选择、排序、短名单差异和 eligible/retained/dropped 算术由保存行重算；主份额始终使用预先固定的“合格快照全部短名单行”分母，正语义行分母仅作为旁报。

证据边界：未保存完整 O 反事实答案及路线状态，因此完整答案相等及逐格安全性仍依赖冻结生产脚本中的执行断言；原池 observed support 的几何真实性本次没有重建。本次不把摘要 SHA 当作独立路径重建，也不重算描述符所依赖的几何。O 使用与人工语义 detector 相同的位置输出，是去类别方向先验消融，不能等同独立 objectness 检测器。

本版生产预检实际重建 802 个已保存包、18 次几何计划、126 次排名；实际计数以顶层 counters 为准。world_tripwire_counters 中通用 mapper/TSDF/decision 计数未接线，其零值不能解释成生产端未重建；禁止世界调用的零计数只与冻结 tripwire 控制流合用。首版另有802包重建，不被本版覆盖。本复核新增这些调用全部为零。连续10次缺席门仍未验证，18个开发快照不能证明闭环、校准Q或语义性能收益，也不新增主任务授权。详见同目录 result.json。
'''
        capacity_table='\n'.join(f"| {case['case_index']:02d} | {row['action_id']} | {row['gate_eligible']} | {row['complex_candidate_capacity']['raw_complex_positive_rows']} | {row['complex_candidate_capacity']['retained_complex_rows']} | {row['shortlist_rows']} | {row['complex_candidate_capacity']['all_complex_fit_same_shortlist_length']} |" for case in reviews for row in case['snapshots'])
        report+='''

## 复杂类别原池与裁剪

以下只对已保存候选集合做容量计数；不重规划，不构成新策略或质量结论。容量见证包含原规则必须保留的覆盖候选、全部类别3正候选及每个 eligible cue 至少一条候选。能容纳不意味着原分数会选择它们。

| case | action | 纳入主门分母 | 复杂正候选原池数 | 实际保留数 | 全短名单数 | 全复杂+覆盖+cue保留可容纳 |
|---|---:|---|---:|---:|---:|---|
'''+capacity_table+'\n\n'
        for item in capacity_summaries:
            report+=f"case{item['case_index']:02d}：固定分母下实际 {item['retained_complex_rows']}/{item['fixed_denominator']}，原池容量上限 {item['raw_capacity_upper_numerator']}/{item['fixed_denominator']}={item['raw_capacity_upper_share']}；合格快照中全部复杂候选与强制保留条目可同时容纳：{item['all_eligible_snapshots_have_all_complex_capacity_witness']}。短名单容量上限与实际保留差 {item['retained_vs_capacity_gap_rows']} 行。\n\n"
        report+='case03 的20%门未通过不能仅归咎于原池没有足够类别3候选：四个合格快照各有3条，48行固定分母下可达12/48=25%，实际只保留5行。该证据定位候选裁剪/排名环节，但不支持强加类别配额，也不证明保留更多候选就能改善终点质量。\n'
        report+='\n## 最终选择\n\n'
        for item in selection_summaries:
            report+=f"case{item['case_index']:02d} 在 {item['gate_eligible_snapshots']} 个主门合格快照中，S 最终选择含类别3正证据的次数为 {item['S_selected_complex_in_gate_eligible_snapshots']}；全9快照中对应动作 {item['S_selected_complex_actions']}。S/O选择差异动作 {[row['action_id'] for row in item['S_O_actual_selection_differences']]}。\n\n"
        report+='选择类别由候选的实际观测cue身份读取；O的class_id=None不被误读为无实例证据。S选入复杂候选若与O相同，不能归因为类别方向先验。\n'
        write('REVIEW.md',report)
        write('manifest.json',dict(status='complete',script=str(Path(__file__).resolve().relative_to(ROOT)),
            script_sha256=sha(__file__),source_manifest_sha256=sha(source/'manifest.json'),
            source_result_sha256=sha(source/'result.json'),output_cap_bytes=CAP))
        write('artifact_hashes.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        print(canonical(dict(status=out['status'],summaries=summaries,elapsed_s=perf_counter()-started)))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_reconstruction_or_physical_actions=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:run(args.source,args.output)
