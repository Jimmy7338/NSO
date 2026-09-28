#!/usr/bin/env python3
"""Independent saved-number/hash review. No project imports or Q recomputation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ANALYTIC=ROOT/'audit_results/v31_supported_outline_20260917'
DIAGNOSTIC=ROOT/'audit_results/v31_saved_outline_diagnostic_20260917'
OLD=ROOT/'audit_results/v30_pixel_routes_20260917'
OUT=ROOT/'audit_results/v31_supported_outline_review_20260917'
DOC=ROOT/'docs/research/V31_SUPPORTED_OUTLINE_INDEPENDENT_REVIEW_20260917.md'
CAP=2*1024**2
TAGS=('02cm','05cm','10cm')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b,message):
    require(np.isfinite([a,b]).all() and abs(float(a)-float(b))<=1e-10,message)


def inventory(root):
    entries=read(root/'artifact_hashes.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    require(set(entries)==actual,'complete inventory set '+str(root))
    for name,value in entries.items():require(sha(root/name)==value,'artifact SHA '+name)
    return dict(files=len(entries),inventory_sha256=sha(root/'artifact_hashes.json'))


def sources(root):
    m=read(root/'manifest.json');entries=m['source_sha256']
    archive_sha=m.get('source_archive_sha256',m.get('source_zip_sha256'))
    require(sha(root/'sources.zip')==archive_sha,'archive bytes '+str(root))
    with zipfile.ZipFile(root/'sources.zip') as z:
        require(len(z.namelist())==len(entries) and set(z.namelist())==set(entries),'complete source archive entries')
        for name,value in entries.items():
            require(sha(ROOT/name)==value and hashlib.sha256(z.read(name)).hexdigest()==value,'current/archived source '+name)
    for name,value in m.get('input_sha256',{}).items():require(sha(ROOT/name)==value,'saved input SHA '+name)
    return dict(count=len(entries),archive_sha256=archive_sha,manifest_sha256=sha(root/'manifest.json'),
        input_count=len(m.get('input_sha256',{})))


def score_arithmetic(score,candidate):
    rows=score['instances'];n=score['mission_asset_count']
    require(n==len(rows) and n>0,'fixed task denominator')
    require(score['missing_asset_count']==sum(row['missing'] for row in rows),'missing task count')
    require(score['prediction_crop'] is False and score['prediction_alignment'] is False,'no prediction crop/alignment')
    c=score['coverage_2d'];require(0<=c<=1,'valid inherited coverage')
    physical=bool(c>=.8 and score['returned'] and score['collisions']==0 and not score['failed']
        and score['primitive_budget_compliant'] is True)
    coefficient=1.
    if candidate:
        nonempty={r['observed_slot'] for r in score['submitted_instance_support'] if r['submitted_mesh']['triangles']>0}
        counts=Counter(r['reference_id'] for r in score['fixed_seed_association']
            if r['observed_slot'] in nonempty and r['reference_id'] is not None)
        ids={r['id'] for r in rows}
        credited={r['observed_slot'] for r in score['fixed_seed_association']
            if r['observed_slot'] in nonempty and r['reference_id'] in ids and counts[r['reference_id']]==1}
        unmatched=nonempty-credited;coefficient=len(credited)/len(nonempty) if nonempty else 0.
        require(sorted(nonempty)==score['nonempty_output_slots'],'nonempty output count')
        require(sorted(credited)==score['uniquely_matched_nonempty_slots'],'unique nonempty attribution')
        require(sorted(unmatched)==score['unmatched_nonempty_slots'],'unmatched nonempty count')
        require({r['assigned_observed_slot'] for r in rows if not r['missing']}==credited,'credited task row set')
        close(score['output_instance_precision'],coefficient,'output precision arithmetic')
        require(score['historical_task_eligible']==physical,'physical qualification unchanged')
        require(score['candidate_output_gate_passed']==(not unmatched),'separate output gate')
        require(score['eligible']==bool(physical and not unmatched),'candidate qualified conjunction')
        require(score['unique_complete_seed_association']==bool(len(credited)==n and not unmatched),'complete association field')
    else:require(score['eligible']==physical,'original physical qualification')
    for row in rows:
        require(set(row['projections'])=={'xy','xz','yz'},'all three projections kept')
        for p in row['projections'].values():
            require(0<=p['iou']<=1+1e-12 and p['predicted_area_m2']>=0 and p['reference_area_m2']>0,'area score bounds')
            if p['missing_projection']:close(p['iou'],0.,'missing area IoU zero')
            if candidate:
                receipt=p['representation'];close(receipt['area_m2'],p['predicted_area_m2'],'saved area receipt')
                require(receipt['added_connections']==0,'declared no added connection')
                require(0<=receipt['residual_line_length_m']<=receipt['support_length_m']+1e-10,'residual support length')
            for tag in TAGS:
                precision,recall,f1=(p[tag][k] for k in ('precision','recall','f1'))
                require(all(0<=v<=1+1e-12 for v in (precision,recall,f1)),'P/R/F1 range')
                close(f1,2*precision*recall/(precision+recall) if precision+recall else 0.,'saved P/R to F1')
        for tag in TAGS:
            expected=sum(min(p[tag]['f1'],p['iou']) for p in row['projections'].values())/3
            close(row[tag]['outline_quality'],expected,'min(F1,IoU) fixed three-projection macro')
            close(row[tag]['outline_f1'],sum(p[tag]['f1'] for p in row['projections'].values())/3,'three-projection F1 macro')
            if row['missing']:close(expected,0.,'missing reference fixed zero')
    for tag in TAGS:
        base=sum(r[tag]['outline_quality'] for r in rows)/n
        close(score[tag]['outline_macro_f1'],sum(r[tag]['outline_f1'] for r in rows)/n,'task F1 denominator')
        if candidate:
            close(score[tag]['base_outline_macro_quality'],base,'base fixed-task Q')
            close(score[tag]['base_joint_outline'],c*base,'base J=CQ')
        close(score[tag]['outline_macro_quality'],base*coefficient,'task Q times output precision')
        close(score[tag]['joint_outline'],c*base*coefficient,'J=C times final Q')


def analytic_review():
    result=read(ANALYTIC/'result.json');rows=read(ANALYTIC/'scores.json')
    require(result['status']=='passed' and len(rows)==26,'complete formal analytic cases')
    for name,row in rows.items():
        require(row['inputs_unchanged'] and row['qualification_fields_are_analytic_test_inputs'],'analytic-only receipt')
        score_arithmetic(row['candidate'],True);score_arithmetic(row['legacy_full_instance'],False)
        close(result['quality05'][name],row['candidate']['05cm']['outline_macro_quality'],'analytic summary Q')
        close(result['legacy_quality05'][name],row['legacy_full_instance']['05cm']['outline_macro_quality'],'analytic old summary Q')
    q=lambda name:rows[name]['candidate']['05cm']['outline_macro_quality']
    positive=['box_closed','box_vertical','box_retriangulated','box_vertical_retriangulated','box_duplicate',
        'box_vertical_duplicate','l_closed','l_vertical','rotated_closed','rotated_vertical']
    gates={name+'_near_one':q(name)>=.99 for name in positive}
    gates.update(equivalent_box_geometry_equal=max(q(n) for n in positive[:6])-min(q(n) for n in positive[:6])<=1e-8,
        rotated_closed_vertical_equal=abs(q('rotated_closed')-q('rotated_vertical'))<=1e-8,
        l_closed_vertical_equal=abs(q('l_closed')-q('l_vertical'))<=1e-8,thin_walls_near_one=q('box_four_thin_walls')>=.98)
    negative=['box_vertical_missing_side','box_expanded_030','box_shifted_030','l_notch_filled',
        'l_convex_shortcut','box_extra_detached_box','box_extra_open_strip']
    for name in negative:gates[name+'_penalized']=q('l_closed' if name.startswith('l_') else 'box_closed')-q(name)>=.01
    counts=result['counts']
    gates.update(all_missing_zero=q('empty')==q('all_instances_missing')==0.,
        one_missing_fixed_denominator=q('one_instance_missing')<=.5,duplicate_seed_zero=q('duplicate_seed')==0.,
        extra_nonempty_output_penalty=abs(q('extra_unassigned_instance')-2/3)<=1e-12 and not rows['extra_unassigned_instance']['candidate']['eligible'],
        empty_extra_not_penalized=abs(q('empty_extra_duplicate_seed')-q('both_present'))<=1e-12,
        one_mm_gap_not_closed=rows['one_mm_gap']['candidate']['instances'][0]['projections']['xy']['predicted_area_m2']==0.,
        invalid_wire_triangle_rejected_including_unassigned=counts['expected_invalid_rejections']==2,
        expected_evaluation_counts=counts['candidate_evaluations']==28 and counts['legacy_evaluations']==26)
    require(gates==result['gates'] and all(gates.values()) and len(gates)==29,'all exact preregistered gate arithmetic')
    require(result['all_gates_passed'] and result['failed_gates']==[],'gate terminal status')
    for key in ('worlds','sensor_packets','mapper_updates','TSDF_integrations','new_main_tasks'):require(counts[key]==0,'zero new physical work receipt')
    require(result['main_tasks_used']==16 and not result['semantic_efficacy_proven'],'scope and quota')
    close(q('one_mm_gap'),2/3,'exact gap remains incomplete projection')
    return dict(gates_verified=29,saved_valid_candidate_cases=26,saved_legacy_cases=26,
        expected_invalid_rejections_reported=2,counts=counts,quality05=result['quality05'],
        gap_xy=rows['one_mm_gap']['candidate']['instances'][0]['projections']['xy'],
        scope='saved gate arithmetic only; no fixture geometry or Q re-evaluation')


def diagnostic_review():
    result=read(DIAGNOSTIC/'result.json');rows=read(DIAGNOSTIC/'scores.json');original_analysis=read(OLD/'result.json')
    require(result['status']=='complete' and len(rows)==4,'complete four saved cases')
    summary=[];changes=[];mesh_count=0
    for index,row in enumerate(rows):
        old=read(OLD/f'case{index:02d}'/'result.json');require(row['case']==old['case'],'case identity')
        stage_summary={}
        for stage in ('prefix','final'):
            entry=row['stages'][stage];previous=old['stages'][stage];score=entry['candidate']
            require(entry['original_window']==previous['window']['main_observed'],'old window values copied unchanged')
            require(entry['original_full_instance']==previous['full_instance'],'old full-instance values copied unchanged')
            score_arithmetic(score,True)
            for key in ('coverage_2d','returned','collisions','failed','primitive_budget_compliant'):
                require(score[key]==previous['window']['main_observed'][key],'inherited actual physical field '+key)
            require(score['historical_task_eligible']==previous['window']['main_observed']['eligible'],'old physical eligibility')
            for slot in range(2):
                with np.load(OLD/f'case{index:02d}'/f'{stage}_slot{slot}.npz',allow_pickle=False) as f:
                    vertices,triangles=f['vertices'],f['triangles']
                    mesh_sha=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
                    old_receipt=previous['full_instance']['submitted_instance_support'][slot]['submitted_mesh']
                    new_receipt=score['submitted_instance_support'][slot]['submitted_mesh']
                    require(mesh_sha==entry['prediction_geometry_sha256'][slot]==old_receipt['sha256']==new_receipt['sha256'],'16 original full meshes unchanged')
                    require(len(vertices)==new_receipt['vertices'] and len(triangles)==new_receipt['triangles'],'complete mesh counts')
                mesh_count+=1
            delta=score['05cm']['outline_macro_quality']-previous['full_instance']['05cm']['outline_macro_quality']
            close(delta,entry['candidate_minus_original_full_Q05'],'saved diagnostic delta')
            changes.append(delta)
            stage_summary[stage]=dict(C=score['coverage_2d'],old_window_Q05=entry['original_window']['05cm']['outline_macro_quality'],
                old_full_Q05=entry['original_full_instance']['05cm']['outline_macro_quality'],candidate_Q05=score['05cm']['outline_macro_quality'],
                delta_Q05=delta,eligible=score['eligible'],output_precision=score['output_instance_precision'],
                projection_receipts=[dict(id=r['id'],projections={k:p['representation'] for k,p in r['projections'].items()}) for r in score['instances']])
        summary.append(dict(case=row['case'],stages=stage_summary))
    for tag in TAGS:
        for key in ('outline_macro_quality','joint_outline'):
            y=[row['stages']['final']['candidate'][tag][key] for row in rows]
            mean=lambda ids:sum(y[i] for i in ids)/len(ids)
            fixed_a,fixed_b=mean([0,2]),mean([1,3]);best=max(fixed_a,fixed_b)
            ids=original_analysis['observed_rule_case_indices'];complex_value=mean(ids['complex'])
            expected=dict(always_A=fixed_a,always_B=fixed_b,best_fixed=best,complex_rule=complex_value,
                swapped_rule=mean(ids['swapped']),oracle=(max(y[:2])+max(y[2:]))/2,
                complex_relative=complex_value/best-1 if best else None)
            for name,value in expected.items():
                saved=result['selection'][tag][key][name]
                if value is None:require(saved is None,'zero denominator maintained')
                else:close(value,saved,'selection arithmetic '+tag+'.'+key+'.'+name)
    close(max(map(abs,changes)),result['maximum_absolute_change_Q05'],'maximum Q change')
    for key,field in (('final_candidate_Q05','candidate_Q05'),('final_original_full_Q05','old_full_Q05'),('final_candidate_eligible','eligible')):
        require(result[key]==[row['stages']['final'][field] for row in summary],'final summary '+key)
    expected_counts=dict(candidate_evaluations=8,old_quality_evaluations=0,mesh_inputs_read=16,worlds=0,sensor_packets=0,
        mapper_updates=0,TSDF_integrations=0,mapper_mesh_extractions=0,new_main_tasks=0)
    require(result['counts']==expected_counts and mesh_count==16,'bounded saved-only diagnostic work')
    require(result['main_tasks_used']==16 and not result['observed_geometry_changed'] and not result['semantic_efficacy_proven'],'no new efficacy claim')
    require(result['physical_qualification_unchanged'] and not any(result['final_candidate_eligible']),'C80 failure preserved')
    return dict(cases=summary,counts=result['counts'],maximum_absolute_change_Q05=result['maximum_absolute_change_Q05'],
        selection=result['selection'],scope='saved mesh byte identity and numbers only; no Q geometry regenerated')


def review():
    evidence={}
    for label,path in (('analytic',ANALYTIC),('diagnostic',DIAGNOSTIC),('original_V30',OLD)):
        evidence[label]=dict(inventory=inventory(path),sources=sources(path),result_sha256=sha(path/'result.json'))
    require(evidence['analytic']['sources']['count']==43 and evidence['original_V30']['sources']['count']==54,'frozen source counts')
    evidence['original_V30']['case_inventories']=[inventory(OLD/f'case{i:02d}') for i in range(4)]
    require(evidence['original_V30']['result_sha256']=='7f05663cd004e142f22f261924a05a159cf11304d20fb303356bf21adc2e3b55','original V30 terminal identity')
    require(evidence['original_V30']['inventory']['inventory_sha256']=='8052e546fd4d5ef49ae43c4b268f2c77e6f3163c74c19c80789e73d86641ff52','original sealed V30 identity')
    analytic_result=read(ANALYTIC/'result.json');diagnostic_result=read(DIAGNOSTIC/'result.json')
    require(analytic_result['source_count']==evidence['analytic']['sources']['count'],'actual analytic source count')
    require(diagnostic_result['source_count']==evidence['diagnostic']['sources']['count']
        and diagnostic_result['input_count']==evidence['diagnostic']['sources']['input_count'],'actual diagnostic source/input counts')
    analytic=analytic_review();diagnostic=diagnostic_review()
    return dict(status='independent_saved_evidence_verified',evidence=evidence,analytic=analytic,diagnostic=diagnostic,
        review_script_sha256=sha(__file__),new_review_worlds=0,new_review_sensors=0,new_review_maps=0,new_review_TSDF=0,
        new_review_Q_evaluations=0,new_main_tasks=0,total_main_used=16,
        limitations=['reported call counts and expected rejection receipts are not independent execution tracing',
            'noding/polygonization and geometric distances not recomputed by reviewer',
            'one-millimetre positive gap still gives Q=2/3; analytic success is not noisy-loop robustness',
            'posthoc rescoring does not change observed geometry, C80 failure, or prove semantic efficacy',
            'no declaration that the whole metric is ready as a training target'])


def write(name,payload):
    if not isinstance(payload,bytes):payload=(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    require(used+len(payload)+32768<=CAP and shutil.disk_usage(OUT).free-len(payload)>=64*1024**2,'review cap/reserve')
    with (OUT/name).open('xb') as f:f.write(payload)


def report(result):
    a,d=result['analytic'],result['diagnostic']
    lines=['# V31 终态独立证据复核','',
        '只读封存源、ZIP、产物清单、16份旧实例网格数组及保存分数；未构造world、地图、TSDF或重新计算任何几何Q。',
        f"解析包 {result['evidence']['analytic']['sources']['count']} 源、诊断包 {result['evidence']['diagnostic']['sources']['count']} 源与旧V30的54源及全部库存通过。旧V30 result/root inventory与原封存SHA精确相同。",'',
        '解析26个合法案例的候选及旧值、29个预声明门的算术通过；另2次非法输入拒绝按冻结运行器收据核对，未再次执行。',
        f"准确闭合/完整竖面Q=1；2mm连续薄壁Q={a['quality05']['box_four_thin_walls']:.12f}；1mm开口Q仍为2/3，XY面积0。",'',
        '这说明候选能保留准确闭合线支持，并明确拒绝自动跨缺口补线；它不说明噪声网格一定能闭合，也不表示整套外形评价已经适合作为训练目标。',
        '输出precision、base固定任务macro、相乘后的Q/J、缺失和额外实例门已分别复核。非空extra会受罚，空extra不破坏原匹配。','',
        '| case | 阶段 | 原完整实例Q5 | 新Q5 | 差 | C | 候选资格 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for case in d['cases']:
        for stage,row in case['stages'].items():
            lines.append(f"| {case['case']['index']} | {stage} | {row['old_full_Q05']:.9f} | {row['candidate_Q05']:.9f} | {row['delta_Q05']:+.3g} | {row['C']:.6f} | {row['eligible']} |")
    lines+=['',f"8次已保存网格诊断的最大绝对Q5变化：{d['maximum_absolute_change_Q05']:.12g}。旧指标仅复制，16份实例网格字节不变；这一开发重评分不是新的建图增益或独立语义验证。",
        '四条旧任务C80失败保持；主任务仍16/36，本审查新增主任务0。原有类别选臂表只用于不合格、已经见过的固定路线诊断。',
        '完整数值与来源见同目录result.json。调用数来自封存运行器及保存案例，不是独立系统调用追踪；本复核也没有重新求投影或距离。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.print_help();return
    require(read(DIAGNOSTIC/'result.json')['status']=='complete' and (DIAGNOSTIC/'artifact_hashes.json').exists(),'await terminal diagnostic seal')
    require(not OUT.exists() and not DOC.exists(),'no overwrite or implicit retry')
    require(shutil.disk_usage(ROOT).free>=64*1024**2+CAP,'initial reserve')
    OUT.mkdir()
    try:
        write('review_script.py',Path(__file__).read_bytes());result=review();write('result.json',result)
        text=report(result);write('REVIEW.md',text.encode())
        with DOC.open('x') as f:f.write(text)
        print(json.dumps(dict(status=result['status'],output=str(OUT),max_Q_change=result['diagnostic']['maximum_absolute_change_Q05'])))
    except BaseException:
        write('failure.json',dict(traceback=traceback.format_exc(),new_main_tasks=0,new_Q_evaluations=0));raise
    finally:write('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
