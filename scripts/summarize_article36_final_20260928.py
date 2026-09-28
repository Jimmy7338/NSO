#!/usr/bin/env python3
"""Administrative, read-only summary of all 36 declared scene episodes.

Waits by refusing incomplete inputs; does not poll, run policies, regenerate
maps, or evaluate surfaces. Original CELL/G failure stays an original failure.
S/B action claims require the caller's released complete Exposure snapshot.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import io
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_article_exposure_comparisons_20260928 as a

STAGE=a.STAGE
VERSIONS=('OriginalV1','GroundV2','ExposureV3')
METHODS=('G','B','S','NBV')
SCENES=a.SCENES
DEFAULT_OUTPUT=STAGE/'analysis_v1/article36_final_summary_v1'
DEFAULT_REPORT=ROOT/'docs/research/ARTICLE_EXPOSURE_ABLATION_RESULT_20260928.md'
GROUNDSNAPSHOT=STAGE/'analysis_v1/ground12_complete'
PROTOCOLS=dict(OriginalV1=ROOT/'configs/virtual3d/article_development_v1_20260928.json',
    GroundV2=ROOT/'configs/virtual3d/article_ground_ablation_v2_20260928.json',
    ExposureV3=ROOT/'configs/virtual3d/article_exposure_ablation_v3_20260928.json')
REVIEWS=dict(OriginalV1=STAGE/'episode_reviews_v1',GroundV2=STAGE/'episode_reviews_ground_v2',ExposureV3=STAGE/'episode_reviews_exposure_v3')
METRICS=a.METRICS
COSTS=('paid_actions','path_length_m','turns','explicit_observe_actions','observation_frames')
REQUIRE=a.require


def complete_inventory(protocol,entries):
    slots=protocol['slots']
    REQUIRE(len(slots)==12 and len({(r['scene_id'],r['method']) for r in slots.values()})==12
        and {(r['scene_id'],r['method']) for r in slots.values()}=={(s,m) for s in SCENES for m in METHODS},'complete fixed 3x4 inventory')
    REQUIRE(set(entries)==set(slots) and all(r['status']!='reserved' for r in entries.values()),
        'all twelve declared attempts must be terminal before final summary')
    REQUIRE(all(r['budget']==160 and r['noise_seed']==92801 for r in slots.values()),'same fixed scene-level budget/seed')


def sealed_review(inputs,version,run,entry,protocol,pin):
    folder=REVIEWS[version]/run
    schema='article.episode_review_manifest.v1' if version=='OriginalV1' else a.REVIEW_SCHEMAS[version][0]
    manifest=a.ground.sealed_folder(inputs,folder,{schema})
    REQUIRE(manifest is not None,'independent review not yet sealed: '+run)
    report=inputs.json(folder/'review.json')
    REQUIRE(report['run_id']==run and report['online_status']==entry['status']
        and report['qualified']==entry.get('qualified',False),'review original terminal qualification binding')
    REQUIRE(report['all_checks_passed'] is True and not report.get('failures'),'independent review has unresolved findings: '+run)
    sources=manifest['reviewer_source_sha256']
    if version=='OriginalV1':
        REQUIRE(isinstance(sources,str),'original reviewer source fingerprint')
        inputs.read(ROOT/'scripts/review_article_episode_20260928.py',pin=sources)
    else:
        REQUIRE(isinstance(sources,dict) and sources and sources==report['reviewer_source_sha256'],'reviewer source map')
        for name,sha in sources.items():inputs.read(a.old.safe(ROOT,name),pin=sha)
        REQUIRE(report['protocol_sha256']==manifest['protocol_sha256']==pin,'review/protocol binding')
        REQUIRE(report['source_archive_sha256']==protocol['source_archive_sha256'],'review frozen source archive')
    return folder,manifest,report


def check_physics(physics):
    counts=physics['action_counts']
    REQUIRE(set(counts)<={'forward','turn_left','turn_right','observe'}
        and all(type(v) is int and v>=0 for v in counts.values()),'paid physical action inventory')
    REQUIRE(sum(counts.values())==physics['paid_actions'] and physics['rgbd_frames']==physics['paid_actions']+1,
        'one measured frame per paid physical action plus initial grant')
    REQUIRE(math.isfinite(physics['translation_m']) and physics['translation_m']>=0,'finite actual translation')
    return a.motion_complete(physics)


def new_version_row(inputs,version,protocol,pin,phase,run,entry):
    slot=protocol['slots'][run];row=a.blank_row(version,run,slot,entry)
    folder,review_manifest,review=sealed_review(inputs,version,run,entry,protocol,pin)
    row.update(version=version,review_status=review['status'],review_passed=True,input_protocol_sha256=pin,
        input_review_manifest_sha256=a.old.digest(inputs.read(folder/'manifest.json')))
    if review['status']=='failed_attempt_preserved':
        REQUIRE(entry['status']=='attempt_failed' and not review['qualified'],'verified failure is retained')
        failure=phase/'episodes'/run/'attempt_failure.json'
        if not failure.is_file():failure=phase/'failures'/(run+'.json')
        REQUIRE(inputs.json(failure,pin=entry['result_sha256'])==review['failure'],'failure bytes/ledger binding')
        row.update(status='original_failed_no_quality',original_end_to_end_qualified=False)
        return row
    REQUIRE(review['status']=='reviewed','terminal episode fully reviewed')
    episode=phase/'episodes'/run
    artifacts=inputs.json(episode/'artifact_manifest.json',pin=entry['artifact_manifest_sha256'])
    REQUIRE(entry['artifact_manifest_sha256']==review_manifest['input_episode_manifest_sha256']==review['input_manifest_sha256'],
        'review ledger original artifact seal')
    REQUIRE(artifacts['protocol_sha256']==pin and artifacts['source_sha256']==protocol['source_sha256'], 'unchanged execution sources/protocol')
    REQUIRE(review['method']==slot['method'] and review['scene_id']==slot['scene_id'] and review['phase']=='ablation','reviewed declared slot')
    REQUIRE(review['metric_version']==a.VERSION and review['metric_settings']==protocol['evaluation'],'common canonical measurement')
    evaluation=json.loads(a.old.sealed_read(inputs,episode,artifacts,'evaluation.json'))
    REQUIRE(evaluation['metric_version']==a.VERSION and evaluation['metrics']==review['metrics'],'stored evaluation/review equality')
    result=json.loads(a.old.sealed_read(inputs,episode,artifacts,'result.json'))
    REQUIRE(a.old.digest(a.old.sealed_read(inputs,episode,artifacts,'result.json'))==entry['result_sha256'],'terminal result seal')
    motion=check_physics(review['physics_metrics'])
    a.ground.apply_metrics(row,review['metrics'],version=a.VERSION,motion=motion,mode='original_online_canonical_evaluation')
    a.ground.apply_costs(row,review['physics_metrics'],result)
    row.update(status='reviewed_qualified' if review['qualified'] else 'reviewed_unqualified',
        collisions=review['physics_metrics']['collisions'],returned_xy_and_yaw=review['physics_metrics']['returned_xy_and_yaw'],
        action_counts=review['physics_metrics']['action_counts'],
        input_artifact_manifest_sha256=entry['artifact_manifest_sha256'],input_evaluation_sha256=artifacts['files']['evaluation.json']['sha256'])
    return row


def original_rows(inputs,protocol,pin,phase,entries,plan,plan_pin,common_summary):
    result=[];summarized={r['run_id']:r for r in common_summary['rows']}
    REQUIRE(common_summary['schema']=='article.common_numeric_summary.v1' and common_summary['plan_sha256']==plan_pin
        and len(summarized)==12 and set(summarized)==set(protocol['slots']),'complete original common summary inventory')
    REQUIRE(plan['protocol_sha256']==pin and plan['slots']==protocol['slots'] and plan['measurement_version']==a.VERSION,'original common plan binding')
    for run,slot in sorted(protocol['slots'].items()):
        entry=entries[run];row=a.blank_row('OriginalV1',run,slot,entry)
        folder,_,review=sealed_review(inputs,'OriginalV1',run,entry,protocol,pin)
        ready=a.ground.load_off(inputs,phase,REVIEWS['OriginalV1'],a.old.safe(ROOT,plan['output_root']),plan,plan_pin,row,entry,pin)
        REQUIRE(ready is not None and row['quality_measurement_available'],'all declared original predictions have sealed common measurements')
        summary=summarized[run]
        REQUIRE(summary['original_status']==entry['status'] and summary['original_qualified']==entry.get('qualified',False)
            and summary['motion_complete']==row['motion_completion_verified'] and summary['mode']==row['quality_mode'],
            'common summary retains original outcome and measurement mode')
        for metric in METRICS:REQUIRE(abs(summary[metric]-row[metric])<=1e-12,'common summary quality: '+metric)
        prepared=a.old.safe(ROOT,plan['output_root'])/run/'prepared'
        snapshot=inputs.json(prepared/'input_snapshot.json');physics=snapshot['motion_completion']
        REQUIRE(check_physics(physics)==row['motion_completion_verified'] and physics['all_checks_passed'],'original saved-motion verification')
        REQUIRE(summary['translation_m']==row['path_length_m'] and summary['actions']==physics['action_counts'],'common summary actual costs')
        row.update(version='OriginalV1',collisions=physics['collisions'],returned_xy_and_yaw=physics['returned_xy_and_yaw'],
            action_counts=physics['action_counts'],input_protocol_sha256=pin,
            input_review_manifest_sha256=a.old.digest(inputs.read(folder/'manifest.json')),
            input_evaluation_sha256=a.old.digest(inputs.read(a.old.safe(ROOT,plan['output_root'])/run/'measurement/result.json')))
        result.append(row)
    REQUIRE(sum(r['original_end_to_end_qualified'] for r in result)==common_summary['original_qualified']==11
        and sum(r['online_status']=='attempt_failed' for r in result)==common_summary['original_failures']==1,
        'original eleven qualified and one original failure remain unchanged')
    return result


def verify_snapshot(inputs,path,schema,rows,arm_to_version):
    manifest=a.ground.sealed_folder(inputs,path,{schema});REQUIRE(manifest is not None,'released snapshot manifest required')
    summary=inputs.json(path/'summary.json')
    REQUIRE(summary['findings']==0 and inputs.json(path/'findings.json')==[],'released complete snapshot without unresolved findings')
    table=list(csv.DictReader(io.StringIO(inputs.read(path/'slots.csv').decode())))
    expected={r['run_id']:r for r in rows if r['version'] in arm_to_version.values()}
    REQUIRE(len(table)==24 and {r['run_id'] for r in table}==set(expected),'snapshot has both complete paired cohorts')
    for captured in table:
        row=expected[captured['run_id']]
        REQUIRE(arm_to_version[captured['arm']]==row['version'] and captured['scene_id']==row['scene_id']
            and captured['method']==row['method'] and captured['online_status']==row['online_status'], 'snapshot named terminal slot')
        for key in ('original_end_to_end_qualified','motion_completion_verified','quality_measurement_available'):
            REQUIRE(captured[key]==('' if row[key] is None else str(row[key])),'snapshot preserves separate qualifications: '+key)
        for key in (*METRICS,*COSTS):
            value=captured[key]
            REQUIRE((row[key] is None and value=='') or (row[key] is not None and value!='' and abs(float(value)-row[key])<=1e-12),
                'snapshot verified metric/cost: '+key)
        review_file=str((REVIEWS[row['version']]/row['run_id']/'manifest.json').relative_to(ROOT))
        REQUIRE(manifest['input_sha256'][review_file]['sha256']==row['input_review_manifest_sha256'],'same independent review as released snapshot')
    return list(csv.DictReader(io.StringIO(inputs.read(path/'paired_effects.csv').decode())))


def differences(rows):
    by={(r['version'],r['scene_id'],r['method']):r for r in rows}
    REQUIRE(len(by)==len(rows)==36,'complete thirty-six slots')
    results=[]
    for scene in SCENES:
        for method in METHODS:
            for newer,older in [('GroundV2','OriginalV1'),('ExposureV3','GroundV2'),('ExposureV3','OriginalV1')]:
                left,right=by[newer,scene,method],by[older,scene,method]
                quality=(left['quality_measurement_available'] and right['quality_measurement_available']
                    and left['motion_completion_verified'] is True and right['motion_completion_verified'] is True)
                if quality:REQUIRE(left['metric_version']==right['metric_version']==a.VERSION
                    and left['reference_fingerprint']==right['reference_fingerprint'],'common version/reference for signed difference')
                result=dict(scene_id=scene,method=method,comparison=newer+'-'+older,
                    left_run=left['run_id'],right_run=right['run_id'],quality_available=quality,
                    left_original_qualified=left['original_end_to_end_qualified'],right_original_qualified=right['original_end_to_end_qualified'])
                for key in (*METRICS,*COSTS):
                    result['delta_'+key]=(left[key]-right[key] if left[key] is not None and right[key] is not None
                        and (key not in METRICS or quality) else None)
                results.append(result)
    return results


def display(value,decimals=6):
    return '未取得' if value is None else f'{value:.{decimals}f}'


def chinese_report(rows,delta,sb,summary,output,exposure_snapshot):
    by={(r['version'],r['scene_id'],r['method']):r for r in rows}
    changes={(r['comparison'],r['scene_id'],r['method']):r for r in delta}
    lines=['# Exposure 共同曝光抵扣消融：完整场景级结果','',
        '本报告汇总既有 36 条场景级尝试：初始版本 12 条、共同地面关联修正 GroundV2 12 条、共同曝光抵扣 ExposureV3 12 条。只读已保存结果和独立复核，不产生新路线、TSDF 或表面评价。三版本均为 AISLE/CELL/LOOP × G/B/S/NBV，预算 160、噪声种子 92801；三个开发布局作为描述性比较单位，不把帧、设备或方法当成独立布局样本。','',
        '初始 CELL/G 的原流程评价失败仍保留；其运动已单独核验完成，质量来自统一版本补测。补测沿用 11 条完全相同输入的旧评价，仅对该失败样本增加过 1 次已声明表面测量；本报告没有新增测量。后两版本的原流程资格、运动完成与质量可用分别记录，空缺不填零。','',
        '|版本|声明尝试|原流程合格|运动完成|统一质量可用|','|---|---:|---:|---:|---:|']
    for version in VERSIONS:
        lines.append(f"|{version}|{summary['version_records'][version]}|{summary['original_qualified'][version]}|{summary['motion_complete'][version]}|{summary['quality_available'][version]}|")
    lines+=['',
        '## 全部十二组联合质量比较','',
        '所有数值采用 article.common_numeric_face_evaluation.v1，J = C_nav × 四实例等权宏平均 F1。† 表示原评价失败后取得的独立派生分数；差值为 Exposure 减 Ground。','',
        '|布局|方法|初始 J|Ground J|Exposure J|ΔJ|','|---|---|---:|---:|---:|---:|']
    for scene in SCENES:
        for method in METHODS:
            original,ground,exposure=(by[v,scene,method] for v in VERSIONS)
            mark='†' if original['online_status']=='attempt_failed' else ''
            difference=changes['ExposureV3-GroundV2',scene,method]['delta_J_nav']
            lines.append(f"|{scene.split('_')[1]}|{method}|{display(original['J_nav'])}{mark}|{display(ground['J_nav'])}|{display(exposure['J_nav'])}|{display(difference)}|")
    lines+=['','## Exposure 的分项质量','',
        'P、R 和 F1 分列；宏平均 F1 是四个实例各自 F1 的均值，不能用宏平均 P、R 再取调和平均替代。','',
        '|布局/方法|C_nav|P|R|F1|J|','|---|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        for method in METHODS:
            row=by['ExposureV3',scene,method]
            lines.append('|'+scene.split('_')[1]+'/'+method+'|'+'|'.join(display(row[k]) for k in METRICS)+'|')
    lines+=['','## Exposure 的实际执行成本','',
        '动作、路程、转向与额外 observe 来自独立复核的 physics_metrics。每个付费动作产生一次 RGB-D 和扫描采集，另有第 0 帧初始观测；不是每次采集都占一个独立 observe 动作。','',
        '|布局/方法|动作/预算|路程/m|转向|额外observe|原流程/返航|','|---|---:|---:|---:|---:|---|']
    for scene in SCENES:
        for method in METHODS:
            row=by['ExposureV3',scene,method]
            cost='未取得' if row['paid_actions'] is None else str(row['paid_actions'])+'/160'
            status=('合格' if row['original_end_to_end_qualified'] else '未合格')+'/'+('已核验完成' if row['motion_completion_verified'] else '未核验完成')
            lines.append(f"|{scene.split('_')[1]}/{method}|{cost}|{display(row['path_length_m'],1)}|{display(row['turns'],0)}|{display(row['explicit_observe_actions'],0)}|{status}|")
    lines+=['','## S/B 共享作用：只引用完整复核快照','',
        '下表直接绑定 root 已发布的完整 Exposure 配对快照，而不是仅凭相同端点分数推断动作相同。后验变化、评分变化、实际动作分歧与质量增益分别看待。','',
        '|版本/布局|S−B 的 J|保存动作是否一致|首个动作分歧步|共同实际输入帧数|','|---|---:|---|---:|---:|']
    for record in sb:
        raw=record['pair'];difference=raw.get('delta_J_nav','')
        divergence=raw.get('first_action_divergence_paid_step') or ('无记录分歧' if raw.get('mechanism_status')=='common_actual_prefix_only' else '未取得')
        lines.append('|'+record['version']+'/'+record['scene_id'].split('_')[1]+'|'+('未取得' if difference=='' else f'{float(difference):.6f}')
            +'|'+('一致' if raw.get('actions_identical')=='True' else '有分歧' if raw.get('actions_identical')=='False' else '未取得')
            +'|'+divergence+'|'+(raw.get('common_observation_prefix_frames') or '未取得')+'|')
    lines+=['','## 结果解释范围','',
        'Exposure 在全部方法中共同修正同光心已尝试视锥的名义新表面机会面积；它不改变实测支持、完整深度融合或评价，也不将模板表面视为已重建。该组件差分不能自动归为语义增益。完整 CSV 保留初始→Ground、Ground→Exposure、初始→Exposure 的逐布局、逐方法质量和成本差值；不混合三个不同版本做重复试验均值。','',
        '局部类别确认实验的 6.1638% 收益、场景级 B/G 类别作用、G/NBV 诊断规划作用与 S/B 同类共享作用应分别陈述。已保存 AISLE 案例包含类别改变路径但端点下降；CELL 的 G/NBV 案例确认诊断宏动作改变执行路径，但第 20 帧没有兑现新的几何后验信息。这些案例与本完整矩阵并列，不代替完整矩阵。','',
        '本报告不自动判定或启动 main 阶段，也不以某个端点正差替代预先声明的 S/B 可执行机制门。三布局均为开发场景，以上比较为描述性结果；没有 p 值或未知场景泛化结论。','',
        '## 输入与输出','',
        f'- 完整结果包：`{output.relative_to(ROOT)}`，含 `rows36.csv`、`rows36.json`、`version_differences.csv`、`summary.json`、`shared_semantic_pairs.json` 与逐输入 SHA。',
        f'- 完整 Exposure 比较：`{exposure_snapshot.relative_to(ROOT)}`。',
        '- 原始/Ground 比较：`audit_results/article_stage_20260928/analysis_v1/ground12_complete/`。',
        '- 原始共同评价汇总：`audit_results/article_stage_20260928/common_evaluation_summary_v1.json`；各原始派生结果与准备快照也已重新核对。',
        '- 每份独立 review 的 manifest 输出全部校验；执行源、协议、评价与实际成本分别绑定。仅汇总，没有新仿真、控制器回放、融合或评价调用。','']
    return '\n'.join(lines)


def summarize(output,report_path,exposure_snapshot):
    output=Path(output).resolve();report_path=Path(report_path).resolve();exposure_snapshot=Path(exposure_snapshot).resolve()
    REQUIRE(not output.exists() and output.resolve().is_relative_to(STAGE.resolve()),'fresh article summary output')
    REQUIRE(not report_path.exists() and report_path.resolve().is_relative_to(ROOT/'docs/research'),'new independent research report')
    inputs=a.old.Inputs()
    sources=(__file__,a.__file__,a.ground.__file__,a.old.__file__)
    for source in sources:inputs.read(Path(source))
    protocols={v:inputs.json(path) for v,path in PROTOCOLS.items()}
    pins={v:a.old.digest(inputs.read(path)) for v,path in PROTOCOLS.items()}
    a.validate_pairing(protocols['ExposureV3'],protocols['GroundV2'])
    for version,protocol in protocols.items():
        REQUIRE(protocol['status']=='frozen','frozen protocol');a.verify_sources(inputs,protocol)
    for version in ('GroundV2','ExposureV3'):
        for key in ('evaluation','references','mapper','asset_manifest_sha256'):
            REQUIRE(protocols[version][key]==protocols['OriginalV1'][key],'same measurement/mapper/asset: '+key)
    ledgers={v:a.ground.ledger_entries(inputs,protocols[v],pins[v]) for v in VERSIONS}
    for version in VERSIONS:complete_inventory(protocols[version],ledgers[version][0])
    plan_record=protocols['GroundV2']['common_baseline_evaluation_plan']
    plan=inputs.json(a.old.safe(ROOT,plan_record['path']),pin=plan_record['sha256'])
    for name,pin in plan['source_sha256'].items():inputs.read(a.old.safe(ROOT,name),pin=pin)
    common_summary=inputs.json(STAGE/'common_evaluation_summary_v1.json')
    entries,phase=ledgers['OriginalV1']
    rows=original_rows(inputs,protocols['OriginalV1'],pins['OriginalV1'],phase,entries,plan,plan_record['sha256'],common_summary)
    for version in ('GroundV2','ExposureV3'):
        entries,phase=ledgers[version]
        for run,entry in sorted(entries.items()):rows.append(new_version_row(inputs,version,protocols[version],pins[version],phase,run,entry))
    original_pairs=verify_snapshot(inputs,GROUNDSNAPSHOT,'article.ground_comparison_manifest.v1',rows,
        dict(GroundOff='OriginalV1',GroundOn='GroundV2'))
    exposure_pairs=verify_snapshot(inputs,exposure_snapshot,'article.exposure_comparison_manifest.v1',rows,
        dict(GroundV2='GroundV2',ExposureV3='ExposureV3'))
    sb=[]
    for version,source,comparison in [('OriginalV1',original_pairs,'GroundOff/S-B'),('GroundV2',original_pairs,'GroundOn/S-B'),
                                    ('ExposureV3',exposure_pairs,'ExposureV3/S-B')]:
        pairs=[r for r in source if r['comparison']==comparison]
        REQUIRE(len(pairs)==3 and {r['scene_id'] for r in pairs}==set(SCENES),'all three S/B pairs from complete snapshot')
        sb.extend(dict(version=version,scene_id=r['scene_id'],pair=r) for r in sorted(pairs,key=lambda x:x['scene_id']))
    delta=differences(rows)
    summary=dict(schema='article.all36_final_summary.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        declared_attempts=36,version_records={v:sum(r['version']==v for r in rows) for v in VERSIONS},metric_version=a.VERSION,
        original_qualified={v:sum(r['version']==v and r['original_end_to_end_qualified'] is True for r in rows) for v in VERSIONS},
        motion_complete={v:sum(r['version']==v and r['motion_completion_verified'] is True for r in rows) for v in VERSIONS},
        quality_available={v:sum(r['version']==v and r['quality_measurement_available'] is True for r in rows) for v in VERSIONS},
        status_counts={v:dict(Counter(r['online_status'] for r in rows if r['version']==v)) for v in VERSIONS},
        historical_common_measurement=dict(exact_input_reuses=common_summary['exact_input_reuses'],new_surface_evaluations_at_original_supplement=common_summary['new_surface_evaluations']),
        protocol_sha256=pins,complete_exposure_snapshot=str(exposure_snapshot.relative_to(ROOT)),
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0,automatic_main_approval=False)
    text=chinese_report(rows,delta,sb,summary,output,exposure_snapshot)
    inputs.unchanged();output.mkdir(parents=True)
    def write(name,value):
        with (output/name).open('xb') as stream:stream.write(a.old.canonical(value))
    write('rows36.json',rows);write('summary.json',summary);write('shared_semantic_pairs.json',sb)
    a.old.write_csv(output/'rows36.csv',rows,['version','scene_id','method','run_id','online_status','original_end_to_end_qualified','motion_completion_verified','quality_measurement_available'])
    a.old.write_csv(output/'version_differences.csv',delta,['comparison','scene_id','method','quality_available'])
    for index,payload in enumerate(inputs.captured.values()):(output/f'captured_ledger_{index:02d}.json').write_bytes(payload)
    report_path.write_text(text);(output/'REPORT.md').write_text(text)
    archive=output/'source_archive';archive.mkdir()
    source_pins={}
    for source in sources:
        path=Path(source);source_pins[str(path.relative_to(ROOT))]=a.old.digest(path.read_bytes());(archive/path.name).write_bytes(path.read_bytes())
    write('manifest.json',dict(schema='article.all36_final_manifest.v1',input_sha256=inputs.pins,source_sha256=source_pins,
        report=dict(path=str(report_path.relative_to(ROOT)),sha256=a.old.digest(report_path.read_bytes())),
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=a.old.digest(p.read_bytes())) for p in output.rglob('*') if p.is_file()}))
    return dict(output=str(output),report=str(report_path),rows=len(rows),version_differences=len(delta),summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    parser.add_argument('--exposure-analysis',type=Path,required=True)
    parser.add_argument('--release',action='store_true',help='Root release only after all terminal reviews and full snapshot')
    args=parser.parse_args()
    if not args.release:parser.error('require explicit --release; no partial final report')
    print(json.dumps(summarize(args.output,args.report,args.exposure_analysis)))


if __name__=='__main__':main()
