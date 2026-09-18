#!/usr/bin/env python3
"""Reconstruct old V27 option ownership from sealed plans/calls/trace only.

No project runtime imports, packets, mapper, world, mesh, Q, or new policy.
V27's in-memory feedback_lifecycle was not serialized by its collector. Close
reasons below are source-backed retrospective inferences, not saved events.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audit_results/observed_autonomous_v27_20260916'
OUTPUT=ROOT/'audit_results/option_lifecycle_v29_review_r1_20260917'
REPORT=ROOT/'docs/research/V29_OPTION_LIFECYCLE_AUDIT_20260917.md'
CAP=160*1024
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


def checked_file(path,inventory):
    name=str(path.relative_to(SOURCE))
    require(name in inventory and sha(path)==inventory[name],'sealed input changed: '+name)
    return read(path)


def analyze_case(index,inventory):
    folder=SOURCE/f'case_{index:02d}'
    record=checked_file(folder/'result.json',inventory)
    plans=checked_file(folder/record['plans_file'],inventory)
    calls=checked_file(folder/record['calls_file'],inventory)
    trace=record['trace'];paid=record['paid_actions']
    require(record['mode']=='S' and paid==400,'frozen feedback-enabled history')
    require([row['action_id'] for row in trace]==list(range(paid+1)),'continuous actual paid trace')
    require(not any(row['collision'] for row in trace),'this reconstruction covers the saved collision-free histories')
    require([row['call_id'] for row in calls]==list(range(1,len(calls)+1)),'one-based module call sequence')
    transitions=[row for row in calls if row['module']=='IGCR' and row['operation']=='measured_transition_feedback']
    require([row['action_id'] for row in transitions]==list(range(1,paid+1)),'one IGCR event per paid step')
    starts=[plan['audit']['action_id'] for plan in plans]
    require(starts==sorted(set(starts)) and starts[0]==0,'unique chronological global plans')
    serial=0;plan_ids=[]
    for plan in plans:
        if plan['selected'] is not None:serial+=1;plan_ids.append(serial)
        else:plan_ids.append(None)
    groups={};plan_index=0
    for event in transitions:
        action=event['action_id']
        while plan_index+1<len(plans) and starts[plan_index+1]<=action-1:plan_index+=1
        expected_id=plan_ids[plan_index]
        require(event['option_id']==expected_id,'recorded option ID differs from frozen open-order reconstruction')
        require(event['before_action_id']==action-1 and event['evaluation_Q_used'] is False,'adjacent measured feedback')
        if action>1:require(event['support_before_sha256']==transitions[action-2]['support_after_sha256'],'full support chain')
        selected=plans[plan_index]['selected']
        reached=selected is not None and selected['pose']==trace[action]['pose']
        require(event['selected_endpoint_reached']==reached,'arrival is exact target cell and heading')
        require(event['transition_scope']==('endpoint_last_paid_step' if reached else 'not_at_selected_endpoint'),'window scope')
        require(not event['directional_updates'] or reached,'directional update away from selected endpoint')
        groups.setdefault(expected_id,[]).append(event)
    rows=[];feedback_windows=[]
    for pi,plan in enumerate(plans):
        option_id=plan_ids[pi]
        if option_id is None:continue
        selected=plan['selected'];start=starts[pi]
        end=starts[pi+1] if pi+1<len(plans) else paid
        events=groups.get(option_id,[])
        require([event['action_id'] for event in events]==list(range(start+1,end+1)),'complete paid ownership interval')
        arrivals=[event['action_id'] for event in events if event['selected_endpoint_reached']]
        require(not arrivals or arrivals==[end],'arrival must close option before next paid step')
        next_plan=plans[pi+1] if pi+1<len(plans) else None
        if arrivals:
            reason='reached_after_feedback';replacement='not_replaced_unreached'
        elif next_plan is not None:
            reason='replanned_after_previous_paid_feedback'
            require(end-start==5,'unreached replacement must follow frozen five-step global replan')
            replacement=('selected_became_none' if next_plan['selected'] is None else
                'same_target_pose_reopened' if next_plan['selected']['pose']==selected['pose'] else 'different_target_pose')
        else:
            reason=record['summary']['terminal_reason'];replacement='terminal'
        evidence=[item for item in selected['semantic_evidence'] if item['hypothesis_gain']>0]
        updates=[item for event in events for item in event['directional_updates']]
        rows.append(dict(case_index=index,option_id=option_id,opening_action=start,closing_action=end,
            paid_duration=end-start,target_pose=selected['pose'],opening_outbound_cost=selected['outbound_cost'],
            opening_return_cost=selected['return_cost'],opening_total_cost=selected['cost'],
            opening_candidate_id=selected.get('candidate_id'),opening_group=selected['group'],
            positive_semantic_cue_ids=sorted({item['cue_id'] for item in evidence}),
            positive_semantic_class_ids=sorted({item['class_id'] for item in evidence}),
            inferred_close_reason=reason,replacement_kind=replacement,actually_reached=bool(arrivals),
            directional_update_rows=len(updates),updated_rows=sum(item['status']=='updated' for item in updates)))
        for event in events:
            for update in event['directional_updates']:
                feedback_windows.append(dict(case_index=index,option_id=option_id,cue_id=update['cue_id'],
                    status=update['status'],opening_action=start,feedback_before_action=event['before_action_id'],
                    feedback_after_action=event['action_id'],owned_duration=event['action_id']-start,
                    actual_feedback_window=1,preceding_owned_actions_outside_feedback=max(0,event['action_id']-start-1)))
    # Only merge consecutive same-pose ownership intervals retrospectively.
    # Cue sets/predictions can change; this does NOT create a semantic option.
    chains=[]
    for row in rows:
        if (chains and chains[-1][-1]['closing_action']==row['opening_action']
                and chains[-1][-1]['target_pose']==row['target_pose']
                and not chains[-1][-1]['actually_reached']):chains[-1].append(row)
        else:chains.append([row])
    chain_rows=[dict(option_ids=[row['option_id'] for row in chain],target_pose=chain[0]['target_pose'],
        opening_action=chain[0]['opening_action'],closing_action=chain[-1]['closing_action'],
        paid_duration=chain[-1]['closing_action']-chain[0]['opening_action'],
        semantic_prediction_identity_preserved=False,retrospective_pose_chain_only=True)
        for chain in chains if len(chain)>1]
    duration=[row['paid_duration'] for row in rows]
    reasons=Counter(row['inferred_close_reason'] for row in rows)
    replacements=Counter(row['replacement_kind'] for row in rows)
    paid_owned=sum(duration);unowned=len(groups.get(None,[]))
    require(paid_owned+unowned==paid,'every paid action must be counted including return/no-positive-goal')
    summary=dict(case_index=index,global_plans=len(plans),nonnull_selected_options=len(rows),
        no_selected_global_plans=sum(option_id is None for option_id in plan_ids),
        close_reason_counts=dict(reasons),replacement_counts=dict(replacements),
        duration_histogram=dict(Counter(duration)),minimum_duration=min(duration),maximum_duration=max(duration),
        median_duration=statistics.median(duration),owned_paid_actions=paid_owned,
        paid_actions_with_selected_none=unowned,total_paid_actions=paid,
        semantic_positive_options=sum(bool(row['positive_semantic_cue_ids']) for row in rows),
        directional_update_rows=len(feedback_windows),feedback_status_counts=dict(Counter(row['status'] for row in feedback_windows)),
        feedback_owned_duration_histogram=dict(Counter(row['owned_duration'] for row in feedback_windows)),
        feedback_window_length_actions=1,whole_option_support_comparison_executed=False,
        reached_without_directional_updates=sum(row['actually_reached'] and not row['directional_update_rows'] for row in rows),
        same_pose_multi_option_chains=len(chain_rows),
        longest_same_pose_chain=max(chain_rows,key=lambda row:row['paid_duration']) if chain_rows else None,
        lifecycle_file_saved=False,close_reasons_reconstructed_from_frozen_control_flow=True,
        source_sha256={name:sha(folder/name) for name in ('result.json',record['plans_file'],record['calls_file'])})
    return summary,rows,feedback_windows,chain_rows


def report_text(summaries):
    table='\n'.join(f"| {s['case_index']:02d} | {s['global_plans']} | {s['nonnull_selected_options']} | {s['close_reason_counts'].get('replanned_after_previous_paid_feedback',0)} | {s['replacement_counts'].get('same_target_pose_reopened',0)} | {s['replacement_counts'].get('different_target_pose',0)} | {s['close_reason_counts'].get('reached_after_feedback',0)} | {s['paid_actions_with_selected_none']} |" for s in summaries)
    return '''# V29 完整观察选项：旧生命周期审计（2026-09-17）

本审计只读取 V27 case01/03 封存的 plans、module_calls 与逐步 trace。没有读取原始 RGBD 数组、没有 mapper/世界/网格/Q 或新策略执行。旧 `feedback_lifecycle` 仅保存在内存，复用的采集器没有将它写出，因此关闭原因是依据冻结代码和实际 option_id 后置重建；不得称为原始生命周期日志或新持久选项效果。

## 实际调用链

`ObservedANSRuntimeV26.next_action` 在无目标、已到达或距离上次全局计划≥5步时重新 `plan`，深拷贝 selected。`ObservedANSRuntimeV27.next_action` 检测到新plan后关闭旧option，再分配新serial；即使目标pose相同也重新开启。与此同时，`ObservedPlannerV26.local_action` 本来就会在**每个动作前**按最新观测地图重算同目标去程与含转向返航，不能把5步全局换目标误称为唯一安全刷新。

`accept` 先校验已授权动作、episode、连续action、时间和位姿，更新一次mapper，再调用IGCR；发生到达后才关闭option。V27的`support_before/after`始终是相邻两包，只在目标位置及朝向实际到达时产生方向反馈；没有保存开端支持用于whole-option比较。开端被5步重置和反馈只看最后一步是两个需要同时处理的问题。

## 两条已有历史的后置统计

| case | 全局plan | 有目标option | 未到达即重规划关闭 | 同pose重开 | 改变pose | 实际到达 | selected=None的付费动作 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+table+'''

两例共162个有目标选项，全部持续1–5个付费动作，中位数5；127个在到达前被重规划关闭，其中56次目标pose未变、67次更换pose、4次改为无正目标。共35次实际到达，24条方向更新；更新的真实支持窗口全部只有最后1步。最长同pose连续链在case01为action59→89、30步/6个ID，case03为54→79、25步/5个ID。链只按pose后置连接，其cue集合和预测未必相同，不能当成完整语义观察选项。

800个实际付费动作中，738步归属非空selected，另62步发生在无正目标的返航/停止前阶段，均保留成本。此次没有累计新的几何收益，更没有用片段Q证明持久化一定更好。

## 新接口建议与必测条件

1. 开启时冻结option ID、目标位姿（含朝向）或明确有序观察目标、cue身份/开端预测、起始action、几何/支持摘要和预算；不可用后来新cue或类别改变追溯改写初始承诺。
2. 路线刷新与全局选臂分离。每5步可优化到原目标的路径，但不改ID、目标和起始支持。每步仍须验证已观测安全单元、当前位置与下一步足迹、剩余完整观察动作及返航朝向成本。安全失效可取消，不能为了持久化忽略阻挡、预算或未知格。
3. `issue`只允许一个pending；`accept`必须与授权动作、同episode、连续action/时间/位姿匹配。动作到账后先归原option，再判断到达/完成；不能让到达帧归新option，也不能靠重复包或免费感知获得奖励。同cell不同朝向不算到达。
4. 完成必须满足预先声明的实际观察条件，不能只凭候选预测。零付费的已在目标状态不产生新的补看收益。取消、碰撞、sensor_done、预算耗尽、返航不可行各有原因和已付成本；失败/取消不能伪装成功收益，也不能把没有观测当作零收益标签训练。
5. 完整起点→终点支持只能记为测量代理；对稳定共同key或明示新增key分别计数，避免同面片逐步重复计奖。IGCR最多结算一次，完成后不可再更新同ID；取消支持保留但单列，不强行写入成功回报。
6. 返航是独立阶段/账本，旋转同样付费。测试预算恰等于去程+观察+返航的边界、缺1步拒绝、路径被新观测封闭、当前或anchor不安全、正返航成本但预算不足，以及取消后安全返航或明确停止。

最少测试应覆盖：5步同目标刷新保持起点；到达帧归属顺序；同格不同朝向；零步/重复/乱序/跨episode；pending未consume时拒绝重选；collision/sensor_done；取消/完成双重调用；类别改变不篡改承诺；每个付费动作只归一个阶段。测试通过只能说明新生命周期契约可执行，尚不构成闭环优势。

源码依据：`nso/observed_runtime_v26.py`、`nso/observed_runtime_v27.py`、`nso/observed_planner_v26.py`、`nso/observed_planner_v27.py`、`nso/observed_feedback_v27.py` 与封存V27采集器。全部旧源码SHA仍与179项清单一致；逐option记录见同名审计目录的 options.csv，反馈窗口见 feedback_windows.csv，完整摘要见 result.json。
'''


def run(output,report):
    require(not output.exists() and not report.exists(),'refuse overwriting evidence or report')
    require(shutil.disk_usage(output.parent).free>RESERVE+CAP,'capacity')
    manifest=read(SOURCE/'manifest.json');inventory=read(SOURCE/'artifact_hashes.json')
    require(manifest['status']=='complete','complete original batch required')
    require({str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file() and p!=SOURCE/'artifact_hashes.json'}==set(inventory),'original inventory set')
    for name,digest in inventory.items():require(sha(SOURCE/name)==digest,'original inventory SHA')
    for name,digest in manifest['source_sha256'].items():require(sha(ROOT/name)==digest,'frozen source')
    require(sha(SOURCE/'sources.zip')==manifest['source_archive_sha256'],'original source archive')
    with zipfile.ZipFile(SOURCE/'sources.zip') as archive:
        require(set(archive.namelist())==set(manifest['source_sha256']),'archive source set')
        for name,digest in manifest['source_sha256'].items():require(hashlib.sha256(archive.read(name)).hexdigest()==digest,'archived source')
    output.mkdir()
    def write(name,value):
        data=value.encode() if isinstance(value,str) else (canonical(value)+'\n').encode()
        require(sum(p.stat().st_size for p in output.iterdir())+len(data)<CAP,'160KiB audit cap')
        require(shutil.disk_usage(output).free-len(data)>RESERVE,'64MiB reserve')
        with (output/name).open('xb') as stream:stream.write(data)
    try:
        summaries=[];all_rows=[];windows=[];chains=[]
        for index in (1,3):
            summary,rows,feedback,case_chains=analyze_case(index,inventory)
            summaries.append(summary);all_rows+=rows;windows+=feedback
            chains.extend(dict(case_index=index,**chain) for chain in case_chains)
        for name,rows in (('options.csv',all_rows),('feedback_windows.csv',windows)):
            buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({key:canonical(value) if isinstance(value,(dict,list)) else value for key,value in row.items()} for row in rows)
            write(name,buffer.getvalue())
        text=report_text(summaries);write('REPORT.md',text)
        with report.open('x') as stream:stream.write(text)
        result=dict(status='complete_retrospective_option_lifecycle_audit',cases=summaries,same_pose_chains=chains,
            total_global_plans=sum(s['global_plans'] for s in summaries),total_options=len(all_rows),
            total_owned_paid_actions=sum(s['owned_paid_actions'] for s in summaries),
            total_paid_actions_without_selected=sum(s['paid_actions_with_selected_none'] for s in summaries),
            total_directional_update_rows=len(windows),retrospective_only=True,new_persistent_option_executed=False,
            no_policy_improvement_claimed=True,evaluation_Q_fields_not_accessed_or_recomputed=True,
            result_container_opened_but_only_trace_plan_metadata_and_identity_fields_used=True,
            audit_new_work=dict(raw_packet_loads=0,mapper_updates=0,worlds=0,plans=0,mesh_extractions=0,Q_evaluations=0,physical_actions=0),
            source_count=len(manifest['source_sha256']),original_inventory_entries=len(inventory),
            failed_verifier_attempt='audit_results/option_lifecycle_v29_20260917',
            verifier_recovery_reason='One-based call IDs verified against saved log and frozen _record implementation; no original source or result changed.',
            original_manifest_sha256=sha(SOURCE/'manifest.json'),original_inventory_sha256=sha(SOURCE/'artifact_hashes.json'),
            original_source_archive_sha256=manifest['source_archive_sha256'],
            script_sha256=sha(__file__),report_path=str(report.relative_to(ROOT)),report_sha256=sha(report))
        write('result.json',result);write('audit_source.py',Path(__file__).read_text())
        write('artifact_hashes.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        print(canonical(dict(status=result['status'],cases=summaries)))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),physical_actions=0,mapper_updates=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--report',type=Path,default=REPORT)
    args=parser.parse_args();run(args.output,args.report)
