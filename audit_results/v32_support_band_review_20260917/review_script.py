#!/usr/bin/env python3
"""Read-only V32 saved-number review: no fixture construction or Q evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audit_results/v32_support_band_20260917'
DIAGNOSTIC=ROOT/'audit_results/v32_saved_support_band_20260917'
OLD=ROOT/'audit_results/v30_pixel_routes_20260917'
V31=ROOT/'audit_results/v31_saved_outline_diagnostic_20260917'
OUT=ROOT/'audit_results/v32_support_band_review_20260917'
DOC=ROOT/'docs/research/V32_SUPPORT_BAND_INDEPENDENT_REVIEW_20260917.md'
TAGS=('02cm','05cm','10cm')
CAP=2*1024**2


def need(condition,reason):
    if not condition:raise ValueError(reason)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b,reason):need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-10,reason)


def inventory(path):
    inv=read(path/'artifact_hashes.json')
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    need(actual==set(inv),'exact artifact set '+str(path))
    for name,value in inv.items():need(sha(path/name)==value,'artifact SHA '+name)
    return dict(files=len(inv),inventory_sha256=sha(path/'artifact_hashes.json'))


def sources(path):
    manifest=read(path/'manifest.json');expected=manifest['source_sha256']
    zsha=manifest.get('source_archive_sha256',manifest.get('source_zip_sha256'))
    need(sha(path/'sources.zip')==zsha,'source ZIP bytes')
    with zipfile.ZipFile(path/'sources.zip') as z:
        need(len(z.namelist())==len(expected) and set(z.namelist())==set(expected),'source ZIP exact entries')
        for name,value in expected.items():
            need(sha(ROOT/name)==value and hashlib.sha256(z.read(name)).hexdigest()==value,'current/archived source '+name)
    for name,value in manifest.get('input_sha256',{}).items():need(sha(ROOT/name)==value,'input SHA '+name)
    return dict(source_count=len(expected),manifest_sha256=sha(path/'manifest.json'),source_archive_sha256=zsha,
        input_count=len(manifest.get('input_sha256',{})))


def arithmetic(s):
    rows=s['instances'];n=s['mission_asset_count'];need(n==len(rows) and n>0,'fixed task count')
    need(s['missing_asset_count']==sum(r['missing'] for r in rows),'missing tasks')
    nonempty={r['observed_slot'] for r in s['submitted_instance_support'] if r['submitted_mesh']['triangles']>0}
    assoc=s['fixed_seed_association'];counts=Counter(r['reference_id'] for r in assoc
        if r['observed_slot'] in nonempty and r['reference_id'] is not None)
    ids={r['id'] for r in rows}
    credited={r['observed_slot'] for r in assoc if r['observed_slot'] in nonempty
        and r['reference_id'] in ids and counts[r['reference_id']]==1}
    unmatched=nonempty-credited;precision=len(credited)/len(nonempty) if nonempty else 0.
    need(s['nonempty_output_slots']==sorted(nonempty) and s['uniquely_matched_nonempty_slots']==sorted(credited),'output count/attribution')
    need(s['unmatched_nonempty_slots']==sorted(unmatched),'unmatched output set')
    need({r['assigned_observed_slot'] for r in rows if not r['missing']}==credited,'assigned nonempty rows')
    close(s['output_instance_precision'],precision,'output precision arithmetic')
    physical=bool(s['coverage_2d']>=.8 and s['returned'] and s['collisions']==0 and not s['failed']
        and s['primitive_budget_compliant'] is True)
    need(s['historical_task_eligible']==physical and s['eligible']==bool(physical and not unmatched),'physical/output qualification split')
    need(s['candidate_output_gate_passed']==(not unmatched),'candidate output gate')
    need(s['unique_complete_seed_association']==bool(len(credited)==n and not unmatched),'complete association')
    need(s['completion_fraction'] is None and not s['connectivity_certified']
        and not s['observed_surface_completion_certified'] and not s['full_3d_accuracy_certified']
        and not s['candidate_ready_as_sole_training_target'],'scope: no completion/topology/full3D/training certificate')
    need(s['prediction_crop'] is False and s['prediction_alignment'] is False,'no prediction crop/alignment')
    for row in rows:
        need(set(row['projections'])=={'xy','xz','yz'},'three fixed projections')
        for p in row['projections'].values():
            close(p['iou'],p['legacy_region_iou'],'old region IoU preserved separately')
            need(not p['band_is_occupied_region'] and not p['connectivity_certified'],'band scope')
            for tag in TAGS:
                r=p[tag];pr,re,f=r['precision'],r['recall'],r['f1']
                need(all(0<=v<=1+1e-12 for v in (pr,re,f,r['band_iou'],r['quality'])),'support scores in range')
                close(f,2*pr*re/(pr+re) if pr+re else 0.,'F1 from saved P/R')
                inter,union=r['band_intersection_m2'],r['band_union_m2']
                pa,ra=r['predicted_band_area_m2'],r['reference_band_area_m2']
                need(union>0 and min(inter,pa,ra)>=0 and inter<=min(pa,ra)+1e-10,'band area bounds')
                close(union,pa+ra-inter,'band union arithmetic')
                close(r['band_iou'],inter/union,'band IoU arithmetic, not region IoU')
                close(r['quality'],min(f,r['band_iou']),'new min(F1,band_IoU)')
                close(r['missed_reference_support_m'],p['reference_boundary_length_m']*(1-re),'missed reference length estimate')
                close(r['excess_predicted_support_m'],p['predicted_boundary_length_m']*(1-pr),'excess prediction length estimate')
                if p['missing_support']:close(r['quality'],0.,'missing support remains zero')
        for tag in TAGS:
            close(row[tag]['outline_quality'],sum(p[tag]['quality'] for p in row['projections'].values())/3,'new instance mean')
            close(row[tag]['outline_f1'],sum(p[tag]['f1'] for p in row['projections'].values())/3,'instance F1 mean')
            close(row[tag]['v31_region_outline_quality'],sum(min(p[tag]['f1'],p['iou']) for p in row['projections'].values())/3,'old region definition retained')
            if row['missing']:close(row[tag]['outline_quality'],0.,'fixed missing instance zero')
    for tag in TAGS:
        base=sum(r[tag]['outline_quality'] for r in rows)/n;c=s['coverage_2d']
        close(s[tag]['base_outline_macro_quality'],base,'base task denominator')
        close(s[tag]['outline_macro_quality'],base*precision,'base times output precision')
        close(s[tag]['base_joint_outline'],c*base,'base joint')
        close(s[tag]['joint_outline'],c*base*precision,'new J=C*support Q')
        close(s[tag]['outline_macro_f1'],sum(r[tag]['outline_f1'] for r in rows)/n,'task F1 mean')
        close(s[tag]['v31_region_outline_quality'],sum(r[tag]['v31_region_outline_quality'] for r in rows)/n*precision,'old definition macro with same output precision')


def gates(rows,counts,ambiguity):
    q=lambda name:rows[name]['candidate']['05cm']['outline_macro_quality']
    correct=['box_closed','box_vertical','box_retriangulated','box_vertical_retriangulated',
        'box_duplicate','box_vertical_duplicate','l_closed','l_vertical']
    g={'anchor_'+name+'_near_one':q('anchor_'+name)>=.99 for name in correct}
    g['equivalent_box_representations']=max(q('anchor_'+n) for n in correct[:6])-min(q('anchor_'+n) for n in correct[:6])<=1e-8
    g['thin_wall_anchor']=q('anchor_box_four_thin_walls')>=.95
    g['one_mm_line_gap_stable']=abs(q('corner_gap_001mm')-q('corner_gap_000mm'))<=.01
    g['one_mm_thin_wall_gap_stable']=abs(q('thin_walls_001mm_opening')-q('thin_walls_continuous_002mm'))<=.01
    gap=[0,1,5,10,20,50,100,300]
    for width in gap[:5]:g[f'gap_{width:03d}mm_high_support']=q(f'corner_gap_{width:03d}mm')>=.98
    g['gap_ladder_nonincreasing']=all(q(f'corner_gap_{b:03d}mm')<=q(f'corner_gap_{a:03d}mm')+1e-8 for a,b in zip(gap[:-1],gap[1:]))
    for width,minimum in ((1,.95),(5,.85),(10,.75),(20,.55)):g[f'translation_{width:03d}mm_stable']=q(f'translate_x_{width:03d}mm')>=minimum
    for width,minimum in ((1,.90),(5,.75),(10,.55)):g[f'face_jitter_{width:03d}mm_stable']=q(f'independent_face_jitter_{width:03d}mm')>=minimum
    g['whole_side_missing_penalized']=q('corner_gap_000mm')-q('one_vertical_side_missing')>=.05
    for name in ('box_expanded_030','box_shifted_030','l_notch_filled','l_convex_shortcut','box_extra_detached_box','box_extra_open_strip'):
        g['anchor_'+name+'_penalized']=q('anchor_l_closed' if name.startswith('l_') else 'anchor_box_closed')-q('anchor_'+name)>=.01
    for width in (10,50,200):
        name=f'neighbors_gap_{width:03d}mm';s=rows[name]['candidate']
        g[name+'_correct']=q(name)>=.99 and s['mission_asset_count']==2 and s['unique_complete_seed_association']
    g['wide_wrong_bridge_penalized']=q('neighbors_gap_200mm')-q('neighbors_gap_200mm_wrong_bridge')>=.005
    g['empty_zero']=q('anchor_empty')==q('anchor_all_instances_missing')==0.
    g['missing_instance_fixed_denominator']=q('anchor_one_instance_missing')<=.5
    g['duplicate_seed_zero']=q('anchor_duplicate_seed')==0.
    g['extra_output_penalty']=q('anchor_extra_unassigned_instance')<=2/3+1e-12 and not rows['anchor_extra_unassigned_instance']['candidate']['candidate_output_gate_passed']
    g['empty_extra_no_penalty']=abs(q('anchor_empty_extra_duplicate_seed')-q('anchor_both_present'))<=1e-12
    g['same_observation_same_prediction_representation']=all(ambiguity.values())
    g['no_topology_surface_or_training_certification']=all(not r['candidate']['connectivity_certified']
        and not r['candidate']['observed_surface_completion_certified'] and not r['candidate']['candidate_ready_as_sole_training_target']
        and r['candidate']['completion_fraction'] is None for r in rows.values())
    g['inputs_unchanged']=all(r['inputs_unchanged'] for r in rows.values())
    g['counts_and_two_invalid_rejections']=len(rows)==56 and counts['candidate_evaluations']==58 and counts['expected_invalid_rejections']==2 and all(
        counts[k]==0 for k in ('worlds','sensor_packets','mapper_updates','TSDF_integrations','new_main_tasks'))
    return g


def verify():
    provenance=dict(inventory=inventory(SOURCE),sources=sources(SOURCE),result_sha256=sha(SOURCE/'result.json'))
    result=read(SOURCE/'result.json');rows=read(SOURCE/'scores.json')
    need(result['status'] in ('passed','failed_gates') and len(rows)==56,'complete formal run retained')
    for name,row in rows.items():
        need(row['inputs_unchanged'] and row['qualification_fields_are_analytic_only'],'synthetic evidence scope')
        arithmetic(row['candidate'])
        close(result['support_quality05'][name],row['candidate']['05cm']['outline_macro_quality'],'summary support Q')
        close(result['v31_region_quality05'][name],row['candidate']['05cm']['v31_region_outline_quality'],'summary region Q')
    aa='ambiguity_closed_solid_missing_corner';bb='ambiguity_true_open_shell_same_observation'
    need(rows[aa]['input_mesh_hashes']==rows[bb]['input_mesh_hashes'],'same prediction array hashes for ambiguous cases')
    need(rows[aa]['candidate']['reference_signature']!=rows[bb]['candidate']['reference_signature'],'different references for ambiguous cases')
    ambiguity=result['ambiguity']
    need(len(ambiguity)==16 and all(type(v) is bool for v in ambiguity.values()),'six WKB plus nine band plus array equality receipts')
    calculated=gates(rows,result['counts'],ambiguity)
    need(calculated==result['gates'] and len(calculated)==45,'all exact preregistered gate arithmetic')
    failed=[k for k,v in calculated.items() if not v]
    need(failed==result['failed_gates'] and result['all_gates_passed']==all(calculated.values()),'failed gates retained')
    need(result['status']==('passed' if not failed else 'failed_gates'),'terminal status reflects gates')
    need(result['source_count']==provenance['sources']['source_count'],'source count')
    need(result['counts']['standalone_legacy_evaluations']==0 and result['main_tasks_used']==16,'no old evaluator rerun or new task')
    summary={}
    for name,row in rows.items():
        score=row['candidate'];summary[name]=dict(
            Q_by_threshold={tag:score[tag]['outline_macro_quality'] for tag in TAGS},
            region_Q05=score['05cm']['v31_region_outline_quality'],
            output_precision=score['output_instance_precision'],
            topology_certified=score['connectivity_certified'],
            per_instance_Q05=[r['05cm']['outline_quality'] for r in score['instances']],
            xy_stats=[{tag:{k:r['projections']['xy'][tag][k] for k in
                ('precision','recall','f1','band_iou','quality','missed_reference_support_m','excess_predicted_support_m',
                 'predicted_band_components','reference_band_components')} for tag in TAGS} for r in score['instances']])
    diagnostic=saved_diagnostic()
    return dict(status='independent_saved_evidence_verified',method_gates_passed=result['all_gates_passed'],
        failed_method_gates=failed,gate_count=45,provenance=provenance,counts=result['counts'],cases=summary,
        saved_diagnostic=diagnostic,
        ambiguity_receipts=ambiguity,same_input_array_hashes_independently_compared=True,
        total_main_used=16,new_main_tasks=0,new_review_Q_evaluations=0,new_review_worlds=0,
        source_review_script_sha256=sha(__file__),
        scope='saved arithmetic/provenance only; no new geometry, noding, buffer, Q or sensor queries',
        limitations=['WKB/invalid-input/no-mutation checks rely on frozen execution receipts; not independently rerun',
            '58 evaluator attempts include56 valid plus2 expected invalid; auxiliary ambiguity calls are6 projection extractions and18 buffers',
            'support-band IoU is not occupied-region IoU; preserved hard F1 may jump at tolerance',
            'same mesh under different references cannot certify connectivity; no sole-training or full3D quality approval'])


def saved_diagnostic():
    provenance={}
    for name,path in (('V32_saved',DIAGNOSTIC),('V31_saved',V31),('original_V30',OLD)):
        provenance[name]=dict(inventory=inventory(path),sources=sources(path),result_sha256=sha(path/'result.json'))
    provenance['original_V30']['case_inventories']=[inventory(OLD/f'case{i:02d}') for i in range(4)]
    need(provenance['original_V30']['result_sha256']=='7f05663cd004e142f22f261924a05a159cf11304d20fb303356bf21adc2e3b55','original V30 result identity')
    need(provenance['original_V30']['inventory']['inventory_sha256']=='8052e546fd4d5ef49ae43c4b268f2c77e6f3163c74c19c80789e73d86641ff52','original V30 sealed inventory identity')
    need(provenance['V31_saved']['result_sha256']=='c2f7c628dfc08efcb9eea89ea065ba57c27bf4015a9eac76b9bf8455bb8acae4','original V31 diagnostic identity')
    result=read(DIAGNOSTIC/'result.json');rows=read(DIAGNOSTIC/'scores.json');old31=read(V31/'scores.json')
    need(result['status']=='complete' and len(rows)==4,'four terminal saved diagnostic cases')
    need(result['source_count']==provenance['V32_saved']['sources']['source_count']==42
        and result['input_count']==provenance['V32_saved']['sources']['input_count']==45,'saved source/input counts')
    summary=[];all_exact=True;mesh_count=0
    for index,row in enumerate(rows):
        previous=read(OLD/f'case{index:02d}'/'result.json');need(row['case']==previous['case'],'saved case identity')
        stages={}
        for stage in ('prefix','final'):
            r=row['stages'][stage];s=r['candidate'];p=previous['stages'][stage];prior=old31[index]['stages'][stage]['candidate']
            arithmetic(s)
            need(r['original_window']==p['window']['main_observed'] and r['original_full_instance']==p['full_instance'],'original scores copied unchanged')
            need(r['original_v31']=={tag:prior[tag] for tag in TAGS},'V31 saved values copied unchanged')
            for tag in TAGS:
                close(s[tag]['v31_region_outline_quality'],prior[tag]['outline_macro_quality'],'V31 region Q preserved all thresholds')
                all_exact &= s[tag]['v31_region_outline_quality']==prior[tag]['outline_macro_quality']
            for key in ('coverage_2d','returned','collisions','failed','primitive_budget_compliant'):
                need(s[key]==p['window']['main_observed'][key],'physical field inherited '+key)
            need(s['historical_task_eligible']==p['window']['main_observed']['eligible'],'old physical eligibility retained')
            for slot in range(2):
                with np.load(OLD/f'case{index:02d}'/f'{stage}_slot{slot}.npz',allow_pickle=False) as f:
                    vertices,triangles=f['vertices'],f['triangles']
                    h=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
                    new=s['submitted_instance_support'][slot]['submitted_mesh']
                    old=p['full_instance']['submitted_instance_support'][slot]['submitted_mesh']
                    need(h==new['sha256']==old['sha256']==r['prediction_geometry_sha256'][slot],'same 16 complete saved meshes')
                    need(len(vertices)==new['vertices'] and len(triangles)==new['triangles'],'complete saved mesh counts')
                mesh_count+=1
            stages[stage]=dict(C=s['coverage_2d'],support_Q05=s['05cm']['outline_macro_quality'],
                region_Q05=s['05cm']['v31_region_outline_quality'],candidate_eligible=s['eligible'],
                output_precision=s['output_instance_precision'],per_instance_support_Q05=[x['05cm']['outline_quality'] for x in s['instances']],
                xy_stats=[{k:x['projections']['xy']['05cm'][k] for k in ('precision','recall','f1','band_iou','quality',
                    'missed_reference_support_m','excess_predicted_support_m')} for x in s['instances']])
        summary.append(dict(case=row['case'],stages=stages))
    ids=read(OLD/'result.json')['observed_rule_case_indices']
    for tag in TAGS:
        for key in ('outline_macro_quality','joint_outline'):
            y=[r['stages']['final']['candidate'][tag][key] for r in rows];mean=lambda ix:sum(y[i] for i in ix)/len(ix)
            a,b=mean([0,2]),mean([1,3]);best=max(a,b);complex_value=mean(ids['complex'])
            values=dict(always_A=a,always_B=b,best_fixed=best,complex_rule=complex_value,swapped_rule=mean(ids['swapped']),
                oracle=(max(y[:2])+max(y[2:]))/2,complex_relative=complex_value/best-1 if best else None)
            for name,value in values.items():
                saved=result['selection'][tag][key][name]
                if value is None:need(saved is None,'zero denominator maintained')
                else:close(value,saved,'all thresholds fixed-rule arithmetic '+tag+'.'+key+'.'+name)
    expected_counts=dict(candidate_evaluations=8,standalone_legacy_evaluations=0,mesh_inputs_read=16,worlds=0,
        sensor_packets=0,mapper_updates=0,TSDF_integrations=0,mapper_mesh_extractions=0,new_main_tasks=0)
    need(result['counts']==expected_counts and mesh_count==16,'bounded saved diagnostic counts')
    need(result['main_tasks_used']==16 and not any(result['final_candidate_eligible']),'no new tasks and all old C80 fail')
    for field,key in [('final_support_Q05','support_Q05'),('final_v31_region_Q05','region_Q05'),('final_candidate_eligible','candidate_eligible')]:
        need(result[field]==[r['stages']['final'][key] for r in summary],'saved summary '+field)
    need(result['physical_qualification_unchanged'] and result['legacy_region_diagnostic_matches_v31'],'diagnostic provenance flags')
    for key in ('observed_geometry_changed','semantic_efficacy_proven','candidate_ready_as_sole_training_target','full_3d_accuracy_certified'):
        need(result[key] is False,'limited diagnostic claim '+key)
    return dict(provenance=provenance,cases=summary,counts=expected_counts,selection=result['selection'],
        retained_region_values_exactly_equal_all_thresholds=all_exact,
        all_old_C80_failures_retained=True,new_geometry_Q_computation_by_review=False)


def write(name,value):
    payload=value if isinstance(value,bytes) else (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    need(used+len(payload)+32768<=CAP and shutil.disk_usage(OUT).free-len(payload)>=64*1024**2,'review cap/reserve')
    with (OUT/name).open('xb') as f:f.write(payload)


def report(r):
    lines=['# V32 支持带解析合同：独立保存证据复核','',
        f"45项门算术及56个合法案例的2/5/10cm指标已核。方法门通过：{r['method_gates_passed']}；失败门：{r['failed_method_gates']}。",
        f"核验{r['provenance']['sources']['source_count']}份冻结源/ZIP与完整产物集合。无新投影、buffer、几何Q、传感、world或建图调用；主任务仍16/36。",'',
        '| 案例 | 新支持Q5 | V31区域Q5 |', '| --- | ---: | ---: |']
    names=['corner_gap_000mm','corner_gap_001mm','corner_gap_300mm','thin_walls_continuous_002mm','thin_walls_001mm_opening',
        'translate_x_049mm','translate_x_050mm','translate_x_051mm','one_vertical_side_missing',
        'neighbors_gap_200mm_wrong_bridge','ambiguity_closed_solid_missing_corner','ambiguity_true_open_shell_same_observation']
    for name in names:
        c=r['cases'][name];lines.append(f"| {name} | {c['Q_by_threshold']['05cm']:.9f} | {c['region_Q05']:.9f} |")
    lines+=['','新分数是外轮廓支持吻合度；它保留硬F1距离阈值，不能宣称位置误差处处连续。距离带相交不表示占用或实体连通。',
        '300mm角缺口与缺整侧仍可有很高的宏分：方向均值与投影不可辨识性使支持分不能认证完整性。45门通过不等于单一训练目标就绪。',
        '同预测字节的歧义两案参考不同；只要求预测表示相同，不要求最终分数相同。表示等价的WKB收据来自冻结执行，本复核不重新计算。',
        '合法56次与预期非法2次是评价器尝试数；歧义另有6次投影提取、18次buffer，不能把58称全部几何调用数。',
        '无论工程门是否通过，都没有完整3D质量、拓扑完成、单一训练目标或语义效果认证。方法失败与证据完整性分别报告；不得修改原门或省略负例。','']
    d=r['saved_diagnostic'];pr=d['provenance']
    lines += ['## 旧网格开发诊断', '',
        f"保存诊断{pr['V32_saved']['sources']['source_count']}源、{pr['V32_saved']['sources']['input_count']}输入与16原实例网格字节核验通过；旧V30的54源、264根产物及4×65案例产物保持原封存身份。8次新候选评价已完成，本复核没有重新执行。",
        f"三个阈值下旁报V31区域Q逐值完全相同：{d['retained_region_values_exactly_equal_all_thresholds']}。四案C80失败保持。",'',
        '| case | 终点支持Q5 | 原C | 资格 |','| --- | ---: | ---: | --- |']
    for case in d['cases']:
        s=case['stages']['final'];lines.append(f"| {case['case']['index']} | {s['support_Q05']:.9f} | {s['C']:.6f} | {s['candidate_eligible']} |")
    lines+=['','| 容差 | 复杂规则Q相对最佳固定 | 复杂规则J相对最佳固定 |','| --- | ---: | ---: |']
    for tag in TAGS:
        s=d['selection'][tag];lines.append(f"| {tag} | {100*s['outline_macro_quality']['complex_relative']:+.6f}% | {100*s['joint_outline']['complex_relative']:+.6f}% |")
    lines+=['','必须同时保留2/5/10cm结果，不能挑选有利阈值。这是已见、固定路线、不合格任务的新函数开发表，不是新的重建增益、强G/S比较或语义成功。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.print_help();return
    need((SOURCE/'artifact_hashes.json').exists() and read(SOURCE/'result.json')['status'] in ('passed','failed_gates'),'await sealed result')
    need((DIAGNOSTIC/'artifact_hashes.json').exists() and read(DIAGNOSTIC/'result.json')['status']=='complete','await saved diagnostic seal')
    need(not OUT.exists() and not DOC.exists(),'no overwrite/implicit retry')
    need(shutil.disk_usage(ROOT).free>=CAP+64*1024**2,'initial reserve');OUT.mkdir()
    try:
        write('review_script.py',Path(__file__).read_bytes());r=verify();write('result.json',r)
        text=report(r);write('REVIEW.md',text.encode())
        with DOC.open('x') as f:f.write(text)
        print(json.dumps(dict(status=r['status'],method_gates_passed=r['method_gates_passed'],failed=r['failed_method_gates'],output=str(OUT))))
    except BaseException:
        write('failure.json',dict(traceback=traceback.format_exc(),new_Q_evaluations=0,new_main_tasks=0));raise
    finally:write('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
