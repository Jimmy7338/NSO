#!/usr/bin/env python3
"""Saved ExposureV3 CELL G/NBV mechanism diagnosis, not a cohort snapshot.

Only the two terminal independently reviewed episodes are inspected. This
post-hoc diagnosis was requested after their endpoint difference was known.
It executes no policy, simulator, TSDF or surface measurement.
"""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_article_exposure_comparisons_20260928 as a

METHODS=('G','NBV')
PROTOCOL=ROOT/'configs/virtual3d/article_exposure_ablation_v3_20260928.json'
PROTOCOL_PIN='c78d7eaa0e3784f8402541809c8edd346d654fc0adc683a1c5e1ea57720f6fb7'
DEFAULT_OUTPUT=a.STAGE/'analysis_v1/exposure_cell_g_nbv_lookahead'
MAX_BYTES=1024**2


def saved_step(inputs,episode,manifest,step):
    return json.loads(gzip.decompress(a.old.sealed_read(inputs,episode,manifest,f'steps/{step:03d}.json.gz')))


def evidence_summary(record):
    evidence=record['controller_evidence']
    return dict(planning=a.planning_evidence(evidence),
        geometry_feedback=[{k:r.get(k) for k in ('instance_id','paid_step','applied','reason','observation_sha256')}
            for r in evidence.get('geometry_feedback',[])],
        newly_known_cells=evidence['mapper_receipt']['newly_known_cells'],
        tsdf_integration_count=evidence['mapper_receipt']['tsdf_integration_count'],
        no_intermediate_surface_quality_measured=True)


def candidate_areas(selection,targets):
    result=[]
    for forecast in selection['forecasts']:
        for candidate in forecast['candidates']:
            if candidate['view_id'] not in targets:continue
            result.append(dict(instance_id=forecast['instance_id'],view_id=candidate['view_id'],
                camera=candidate['candidate'],candidate_sha256=candidate['candidate_sha256'],
                area_summary=a.area_receipt(forecast,candidate,'ExposureV3'),components=candidate['components'],
                area_is_not_diagnostic_lookahead_utility=True))
    return result


def macro_audit(inputs,episode,manifest,records,commit):
    initial=records[commit]['decision'];macro=initial['macro_id']
    completed=next((step for step in sorted(records) if step>commit
        and records[step]['controller_evidence'].get('completed_macro_id')==macro),None)
    stop=completed if completed is not None else max(records)
    actions=[];distance=0.;new_cells=0
    for step in range(commit+1,stop+1):
        previous=records[step-1]['decision'];current=records[step]
        packet=json.loads(a.old.sealed_read(inputs,episode,manifest,f'packets/{step:03d}_receipt.json'))
        execution=packet['execution'];before=execution['pose_before_xyyaw_rad'];after=execution['pose_xyyaw_rad']
        travel=math.hypot(after[0]-before[0],after[1]-before[1]);distance+=travel
        new_cells+=current['controller_evidence']['mapper_receipt']['newly_known_cells']
        a.require(previous['macro_id']==macro and previous['macro_target']==initial['macro_target'],
            'the recorded committed macro persists until paid completion')
        actions.append(dict(paid_step=step,decision_step=step-1,macro_id=macro,
            controller_action=previous['action'],actual_sensor_action=execution['action'],collision=execution['collision'],
            pose_before_xyyaw_rad=before,pose_after_xyyaw_rad=after,translation_m=travel,
            remaining_before=records[step-1]['controller_evidence']['routing']['remaining'],
            safety_guard=previous['safety_guard'],return_cost_after_action=previous['routing'].get('return_cost_after_action'),
            completed_macro_id=current['controller_evidence'].get('completed_macro_id'),
            initialization_cancelled_after=current['decision'].get('initialization_cancelled')))
    if completed is not None:
        a.require(actions[-1]['actual_sensor_action']=='observe' and records[completed]['decision']['global_replanned'],
            'extra paid observation completes macro and enables next replan')
    return dict(commit_paid_step=commit,macro_id=macro,committed_target=initial['macro_target'],
        completion_paid_step=completed,status='paid_observation_completed' if completed is not None else 'no_recorded_completion',
        previous_initialization_cancelled_at_commit=initial.get('initialization_cancelled'),
        selected_macro_cancelled=False if completed is not None and all(x['initialization_cancelled_after'] is None for x in actions) else None,
        actual_paid_actions=len(actions),actual_translation_m=distance,
        action_counts=dict(Counter(x['actual_sensor_action'] for x in actions)),
        summed_newly_known_2d_cells=new_cells,macro_actions=actions,
        completion_evidence=None if completed is None else evidence_summary(records[completed]),
        next_selection=None if completed is None else records[completed]['decision'].get('global_selection',{}).get('selected'))


def diagnose(output):
    a.require(not output.exists() and output.resolve().is_relative_to(a.STAGE.resolve()),'fresh external diagnosis directory')
    inputs=a.old.Inputs();sources=(__file__,a.__file__,a.ground.__file__,a.old.__file__)
    for source in sources:inputs.read(Path(source))
    protocol=inputs.json(PROTOCOL,pin=PROTOCOL_PIN)
    a.require(protocol['schema']=='article.experiment_protocol.exposure_v3' and len(protocol['source_sha256'])==65,'frozen 65-source ExposureV3')
    a.verify_sources(inputs,protocol);entries,phase=a.ground.ledger_entries(inputs,protocol,PROTOCOL_PIN)
    loaded={}
    for method in METHODS:
        run=f'exposure_CELL_{method}_b160_n92801';entry=entries[run];slot=protocol['slots'][run]
        a.require(entry['status']!='reserved' and slot==dict(scene_id='ART1_CELL_DEV',method=method,budget=160,noise_seed=92801,
            paired_baseline_run_id=f'ground_CELL_{method}_b160_n92801'),'fixed terminal CELL pair')
        row=a.blank_row('ExposureV3',run,slot,entry)
        ready=a.load_terminal(inputs,phase,a.STAGE/'episode_reviews_exposure_v3',row,entry,protocol,PROTOCOL_PIN)
        a.require(ready is not None and row['original_end_to_end_qualified'] and row['motion_completion_verified'],'both episodes independently qualified')
        trace=a.ground.read_trace(inputs,*ready)
        records={step:saved_step(inputs,*ready,step) for step in trace['steps']}
        final=json.loads(a.old.sealed_read(inputs,*ready,'controller_final.json'))
        loaded[method]=dict(row=row,episode=ready[0],manifest=ready[1],trace=trace,records=records,final=final)
    comparison,witness=a.ground.compact_comparison(loaded['G']['trace'],loaded['NBV']['trace'])
    first=comparison['first_action_divergence_paid_step'];selected=witness['first_changed_target']
    a.require(first is not None and selected is not None and first-1<comparison['common_observation_prefix_frames'],
        'actual different action has identical physical input prefix')
    commit=selected['paid_step'];a.require(commit<comparison['common_observation_prefix_frames'],'selected macro difference before sensor divergence')
    g,n=loaded['G'],loaded['NBV']
    configs={m:loaded[m]['final']['configuration'] for m in METHODS}
    config_differences={k:dict(G=configs['G'].get(k),NBV=configs['NBV'].get(k))
        for k in set(configs['G'])|set(configs['NBV']) if configs['G'].get(k)!=configs['NBV'].get(k)}
    a.require(config_differences==dict(article_method=dict(G='G',NBV='NBV')),'only declared article method differs in saved configuration')
    selections={m:loaded[m]['records'][commit]['decision']['global_selection'] for m in METHODS}
    a.require(selections['G']['direct_options']==selections['NBV']['direct_options'],'same common direct candidates and scores')
    a.require(selections['G']['selected']['kind']=='diagnose_then_observe' and selections['NBV']['selected']['kind']=='direct',
        'saved selected diagnostic versus direct, not assumed')
    a.require(selections['NBV'].get('myopic_nbv') is True and not selections['NBV']['diagnostic_options'],'actual myopic branch receipt')
    best=max(selections['G']['direct_options'],key=lambda x:x['score']);chosen=selections['G']['selected']
    targets={str(s['selected']['target']['node'])+':'+str(s['selected']['target']['heading']) for s in selections.values()}
    for method in METHODS:
        for record in loaded[method]['records'].values():
            a.require(all(r['observed_class'] is None and not r['semantic_conditioning_used'] and not r['peer_instance_ids']
                for r in record['controller_evidence']['structure_belief']['instances']), 'geometry-only method without active semantic peers')
    report=dict(schema='article.saved_cell_lookahead_diagnosis.v1',scope='post-hoc diagnosis of two already-qualified CELL episodes; not a cohort snapshot or held-out confirmation',
        selected_after_endpoint_difference_known=True,protocol_sha256=PROTOCOL_PIN,source_files=65,
        common_prefix=comparison,first_witness=witness,configuration_differences=config_differences,
        planner_branch_note='NBV retains inherited configuration metadata future_information=true, but its executed selector is myopic_nbv and emits zero diagnostic options. Branch behavior is bound to frozen controller source and saved selection.',
        decision_commit_paid_step=commit,first_actual_action_divergence_paid_step=first,
        common_direct_options=selections['G']['direct_options'],diagnostic_option_counts={m:len(s['diagnostic_options']) for m,s in selections.items()},
        selected_score_comparison=dict(G_diagnostic=chosen['score'],best_common_direct=best['score'],
            G_before_predicted_information=chosen['before_observation_score'],G_predicted_evi=chosen['evi'],
            diagnostic_minus_best_direct=chosen['score']-best['score']),
        sides={},terminal_metrics={m:{k:loaded[m]['row'][k] for k in (*a.METRICS,*a.COSTS)} for m in METHODS},
        terminal_G_minus_NBV={k:g['row'][k]-n['row'][k] for k in a.METRICS},
        limitations=['Both methods are geometry-only; this is not S/B sharing evidence or a main-mechanism gate.',
            'The diagnostic channel is a nominal uncalibrated prediction. Its anticipated information value is not an observed measurement.',
            'Endpoint differences follow all later trajectory changes; no isolated causal attribution of terminal quality to one observation.',
            'No intermediate surface scores, p-values, new policies, World calls, TSDF integrations or quality evaluations.'],
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0)
    area_records={}
    for method in METHODS:
        data=loaded[method];records=data['records'];macro=macro_audit(inputs,data['episode'],data['manifest'],records,commit)
        relevant={commit,first-1,first,macro['completion_paid_step']}-{None}
        report['sides'][method]=dict(run_id=data['row']['run_id'],selected_macro=selections[method]['selected'],
            committed_decision=a.gate_decision(records[commit],first_recorded_return_step=None),
            first_divergence_decision=a.gate_decision(records[first-1],first_recorded_return_step=None),macro=macro,
            original_qualified=data['row']['original_end_to_end_qualified'],review_manifest_sha256=data['row']['input_review_manifest_sha256'],
            artifact_manifest_sha256=data['row']['input_artifact_manifest_sha256'],evaluation_sha256=data['row']['input_evaluation_sha256'],
            relevant_step_pins={str(data['episode'].relative_to(ROOT))+f'/steps/{step:03d}.json.gz':data['manifest']['files'][f'steps/{step:03d}.json.gz'] for step in sorted(relevant)})
        area_records[method]=candidate_areas(selections[method],targets)
    completion=report['sides']['G']['macro']['completion_evidence']
    a.require(not any(r['applied'] for r in completion['geometry_feedback']),'diagnostic completion did not apply fresh geometric feedback')
    prior=g['records'][first-1]['controller_evidence']['structure_belief']['instances']
    after=g['records'][first]['controller_evidence']['structure_belief']['instances']
    report['actual_first_diagnostic_information']=dict(geometry_feedback_applied=False,
        posterior_before_after_equal=prior==after,newly_known_2d_cells=completion['newly_known_cells'],
        conclusion='Paid diagnostic macro executed, but no fresh geometric posterior update at its completion; anticipated information gain was not demonstrated.')
    inputs.unchanged();output.mkdir(parents=True)
    def write(name,value):
        with (output/name).open('xb') as stream:stream.write(a.old.canonical(value))
    write('report.json',report);write('selected_candidate_components.json',area_records)
    actions=[dict(method=m,**row) for m in METHODS for row in report['sides'][m]['macro']['macro_actions']]
    a.old.write_csv(output/'macro_actions.csv',actions,['method','paid_step','actual_sensor_action'])
    for index,payload in enumerate(inputs.captured.values()):(output/f'captured_ledger_{index:02d}.json').write_bytes(payload)
    archive=output/'source_archive';archive.mkdir()
    source_pins={}
    for path in sources:
        path=Path(path);source_pins[str(path.relative_to(ROOT))]=a.old.digest(path.read_bytes());(archive/path.name).write_bytes(path.read_bytes())
    narrative=make_narrative(report)
    (output/'REPORT.md').write_text(narrative)
    write('manifest.json',dict(schema='article.saved_cell_lookahead_manifest.v1',source_sha256=source_pins,input_sha256=inputs.pins,
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=a.old.digest(p.read_bytes())) for p in output.rglob('*') if p.is_file()}))
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());a.require(size<=MAX_BYTES,'compact diagnosis output exceeds 1MiB')
    return dict(output=str(output),bytes=size,first_decision=commit,first_actual_action=first,
        terminal_G_minus_NBV_J=report['terminal_G_minus_NBV']['J_nav'],new_worlds=0)


def make_narrative(r):
    g,n=r['terminal_metrics']['G'],r['terminal_metrics']['NBV'];s=r['selected_score_comparison']
    return f'''# CELL 保存轨迹中的 G / NBV 前瞻规划机制

这是在端点差异已知后开展的两条已复核轨迹诊断，不是完整矩阵分析或独立测试集证据。两条运行均保持 ExposureV3 的 65 文件冻结源码、共同地面/曝光前端、传感预算和统一评价；保存配置仅 article_method 不同。G/NBV 都是几何方法，全程没有启用类别或同类共享，因此本报告不满足 S/B 语义机制门。

实际 RGB-D 与扫描数组在帧 0–{r['common_prefix']['common_observation_prefix_frames']-1} 完全相同，仅忽略运行命名的 frame_id。第 {r['decision_commit_paid_step']} 步，两者共有的 32 个直接观察候选及分数完全相同，后验也相同；G 另有 7 个付费诊断候选。G 选中 n_003_001:3 的 diagnose_then_observe，分数 {s['G_diagnostic']:.9f}；NBV 选中 n_004_001:2 的 direct，分数 {s['best_common_direct']:.9f}。所选诊断的预测信息增量为 {s['G_predicted_evi']:.9f}，将其分数从 {s['G_before_predicted_information']:.9f} 提高到高于最佳直接候选。两者均有 143 个剩余付费动作，所选首宏动作加完整返航成本分别为 28、42，预算和足迹守卫均通过。

第 17 步取消的是两者共同的旧初始化承诺，原因是已经取得平面；随后提交的宏动作 2 没有取消。两者先实际前进两步，第 20 次付费动作首次分歧：G 原地 observe，NBV turn_right。G 的宏动作 2 在帧 20 随额外付费观测完成，共 3 动作；NBV 的宏动作 2 在帧 29 完成，共 12 动作。保存逐步动作、目标、完整预算/守卫和完成记录见 macro_actions.csv 与 report.json。

这次 G 诊断观测 **没有应用新的几何反馈**，原因记录为 repeated_support_or_feedback_or_history_cap；其前后结构后验仍相同，均为均匀分布，新增已知二维格为 0。后续第 20 步实际重规划选中 n_003_002:1。故现有证据证明了“前瞻候选改变被选宏动作并改变真实路径”，没有证明本次预测的信息收益被真实反馈兑现，更不能将该次观察直接归因为最终三维质量的改善。完整深度仍按原实现融合；未测中途 F1。

|方法|付费动作|实际路程/m|转向|额外observe|C_nav|F1|J_nav|
|---|---:|---:|---:|---:|---:|---:|---:|
|G|{g['paid_actions']}|{g['path_length_m']:.1f}|{g['turns']}|{g['explicit_observe_actions']}|{g['C_nav']:.6f}|{g['F1']:.6f}|{g['J_nav']:.6f}|
|NBV|{n['paid_actions']}|{n['path_length_m']:.1f}|{n['turns']}|{n['explicit_observe_actions']}|{n['C_nav']:.6f}|{n['F1']:.6f}|{n['J_nav']:.6f}|

两条轨迹均无碰撞、完成全位姿返航并通过独立复核。G−NBV 的端点 J 差为 {g['J_nav']-n['J_nav']:+.6f}，描述这一个开发场景下全部后续路径共同作用的结果。未计算 p 值，也未运行新的 World、控制器、TSDF 或质量评价。NBV 配置中继承的 future_information=true 不能用来判断实际分支：保存选择收据明确 myopic_nbv=true 且诊断候选数为 0；冻结 controller_article_v1.py 的实际分支也区分了两方法。

## English insertion

In the CELL development scene, the saved G and myopic NBV runs share exactly the same physical observations through paid frame 19. At decision 17, their 32 direct candidates have identical scores, but G selects an additional paid diagnostic macro. Its predicted score (0.082740) exceeds the best direct score (0.078458), leading to an observed action divergence at paid action 20: G observes in place while NBV turns right. Both macros complete and both episodes return safely. However, the first diagnostic observation triggers no fresh geometric feedback and leaves the uniform structure posterior unchanged. This trace therefore establishes a lookahead-to-action link, not realization of the predicted information gain. The final joint scores are 0.786500 and 0.515457; this single-scene difference includes all subsequent route changes and is not evidence for semantic sharing or an isolated effect of that observation.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();print(json.dumps(diagnose(args.output)))


if __name__=='__main__':main()
