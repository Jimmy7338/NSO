#!/usr/bin/env python3
"""Independent saved-JSON/hash/arithmetic review; no project runtime imports.

This verifier never creates a mapper/world, loads raw RGBD arrays, plans routes,
extracts meshes, or evaluates Q. Wait for the producer's completed result before
running it. Full-route safety and omitted raw-pool eligibility remain producer
execution assertions, not independently reconstructed evidence.
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
SOURCE=ROOT/'audit_results/v28_descriptor_preflight_20260917'
OUTPUT=ROOT/'audit_results/v28_descriptor_preflight_review_20260917'
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
            close(e['observed_direction_deficit'],d['directional_deficit'][e['sector']],'direction deficit')
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
    """Recompute declared first-version formula, including its known flaw.

    Reproducing this arithmetic does not endorse the patch/cue sector-origin
    mismatch. Raw cue support is recorded observed input, not recomputed here.
    """
    pool=snapshot['raw_pool_observed_support']
    require(len(pool)==snapshot['raw_pool_size'] and len({pose(row) for row in pool})==len(pool),
            'raw observed support size/unique poses')
    checked=0
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
                gain=.05*cue['confidence']*8*weight*d['directional_deficit'][k]*geometry['visible_fraction']
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
        checked+=1
    return checked


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
        require(len(d['directional_support'])==len(d['directional_deficit'])==8,'eight directional bins')
        for support,debt in zip(d['directional_support'],d['directional_deficit']):
            require(0<=support<=1,'direction support bounds');close(debt,1-support,'support/deficit complement')
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
                selected_and_retention_arithmetic_checked=True,raw_eligibility_and_shortlist_modes_recomputed=raw_modes)


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
        require('docs/research/V28_DESCRIPTOR_PREFLIGHT_PROTOCOL_20260917.md' in manifest['source_sha256'],'frozen protocol missing')
        reviews=[];summaries=[]
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
        expected=dict(saved_packets=802,mapper_updates=802,geometry_plans=18,rank_calls=126,
            new_worlds=0,new_actions=0,new_sensor_packets=0,mesh_extractions=0,Q_evaluations=0)
        require(result['counters']==expected,'producer work counters')
        require(all(value==0 for value in result['world_tripwire_counters'].values()),'world tripwire violation')
        require(result['both_complex_share_gates_passed']==all(s['complex_20pct_gate_passed'] for s in summaries),'both-case gate')
        require(result['all_eligible_cues_retained']==all(s['all_eligible_cues_retained'] for s in summaries),'retention aggregate')
        require(all(result[k] is True for k in ('behavioral_checks_passed','original_inventory_unchanged','source_unchanged')),'producer completed checks')
        require(result['main_tasks_used']==12 and all(result[k] is False for k in
            ('autonomous_gain_proven','semantic_gain_proven','Q_calibrated','main_task_permission')),'scope/permission limits')
        out=dict(status='artifact_arithmetic_verified_method_not_accepted',source=str(source.relative_to(ROOT)),
            method_accepted=False,known_method_defect='First-version directional bits are patch-relative while the ranked deficit is indexed by cue-to-camera sector; arithmetic consistency cannot validate this mixed-origin quantity.',
            source_manifest_sha256=sha(source/'manifest.json'),source_result_sha256=sha(source/'result.json'),
            source_inventory=source_inventory,source_archive_and_live_files=sources,
            original_inventory=original_inventory,original_frozen_sources_checked=len(old['source_sha256']),
            cases=reviews,summaries=summaries,counterfactual_receipts_checked=18,
            producer_counters_verified=expected,review_cost=dict(worlds=0,raw_packet_loads=0,mapper_updates=0,
            plans=0,mesh_extractions=0,Q_evaluations=0,physical_actions=0,elapsed_s=perf_counter()-started),
            limitations=[
                'O and class-swapped-O full answers are not serialized: matching SHA receipts plus frozen executed equality assertion support invariance; this reviewer does not reconstruct those full answers.',
                'G/L compact candidates, selected records and route SHA match directly; full original-G equality additionally rests on producer execution receipts.',
                'Raw proposal observed-support rows permit independent eligibility/prior/score/reservation arithmetic; observed support counts themselves are not rebuilt. Full route states and observed map arrays are omitted, so route-hash preimages and per-cell safety are not independently reconstructed.',
                'All shares and selected/retained/dropped/ranking/shortlist arithmetic are independently recomputed, including raw eligible IDs under the frozen first-version formula; the formula has a known sector-origin defect.',
                'Descriptors retain null missing measurements; no calibrated Q, semantic efficacy, autonomous gain, or ten-consecutive-decision absence result is inferred from 18 sparse development snapshots.'])
        write('result.json',out)
        table='\n'.join(f"| {s['case_index']:02d} | {s['eligible_snapshots']} | {s['complex_shortlist_rows']}/{s['all_shortlist_rows']} | {s['complex_share_all_shortlist']} | {s['complex_20pct_gate_passed']} | {s['all_eligible_cues_retained']} |" for s in summaries)
        report='''# V28 描述符预检独立结果复核（2026-09-17）

18 个预定快照的证据清单、源码 ZIP 与当前源码、原输入封存、反事实收据及汇总算术均通过独立复核，**方法不予验收**。首轮有已知实质缺陷：观测方向 bits 以面片为原点，评分却用 cue→相机方向索引 deficit，混用了方向坐标原点；数字复现不能消除此问题。本程序只读 JSON/gzip/源码字节，不读取 RGBD 数组，不创建 mapper/world，不规划、不提网格、不评价 Q。

| case | 合格快照数 | 复杂类别短名单行 / 全部短名单行 | 固定主份额 | ≥20%门 | eligible 实例均保留 |
|---|---:|---:|---:|---|---|
'''+table+'''

每快照核对 O 与交换类别 O 的完整答案 SHA 收据相同，实际低置信 .59、零语义路线查询；直接比较保存的 G/L 候选、选择与路线摘要。从 raw_pool_observed_support、描述符与 cues 独立重算首轮公式的 prior/gain/score、raw eligibility 和保留/短名单步骤。所有选择、排序、短名单差异和 eligible/retained/dropped 算术由保存行重算；主份额始终使用预先固定的“合格快照全部短名单行”分母，正语义行分母仅作为旁报。

证据边界：未保存完整 O 反事实答案及路线状态，因此完整答案相等及逐格安全性仍依赖冻结生产脚本中的执行断言；原池 observed support 的几何真实性本次没有重建。本次不把摘要 SHA 当作独立路径重建，也不重算描述符所依赖的几何。O 使用与人工语义 detector 相同的位置输出，是去类别方向先验消融，不能等同独立 objectness 检测器。

生产预检实际重建 802 个已保存包、18 次几何计划、126 次排名；本复核新增这些调用全部为零。连续10次缺席门仍未验证，18个开发快照不能证明闭环、校准Q或语义性能收益，也不新增主任务授权。详见同目录 result.json。
'''
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
