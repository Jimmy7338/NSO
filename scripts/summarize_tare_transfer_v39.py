#!/usr/bin/env python3
"""Read-only summary of sealed native TARE transfer evidence; never run policy.

No simulator, planner, mapper, evaluator or ROS import. Saved SensorPacket loads
only check stored observation identity. A fixed-action reconstruction replay is
not an independent execution of the asynchronous native policy.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'audit_results/v39_tare_adapter_r1_20260920'
ORIGINAL=ROOT/'audit_results/v39_tare_adapter_20260920'
OUTPUT=ROOT/'docs/thesis/figures/v39_tare'
REPORT=ROOT/'docs/research/V39_TARE_TRANSFER_RESULT_20260920.md'
NODE_SHA='59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).resolve().relative_to(ROOT))
def require(condition,message):
    if not condition: raise ValueError(message)


def seal(folder,name,tracked,exclude=None):
    path=folder/name; expected=read(path)
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
        and p!=path and not (exclude and p.relative_to(folder).parts[0]==exclude)}
    require(actual==set(expected),'sealed inventory changed: '+str(folder))
    for name,digest in expected.items():
        p=folder/name; p.resolve().relative_to(folder.resolve())
        require(sha(p)==digest,'sealed bytes changed: '+str(p))
        tracked[rel(p)]=digest
    tracked[rel(path)]=sha(path)


def audit_case(case,source,config,tracked):
    import numpy as np
    from nso.decision_replay_v13 import load_packet,array_hash
    folder=source/f"case{case['index']:02d}"
    seal(folder,'main_seal.json',tracked,'replay')
    base={k:case[k] for k in ('index','parent','hypothesis','method')}
    if (folder/'failure.json').exists():
        return dict(**base,status='failed',failure_source=rel(folder/'failure.json')),
    result,trace,evidence=(read(folder/name) for name in ('result.json','trace.json','controller.json'))
    seal(folder/'replay','seal.json',tracked)
    replay=read(folder/'replay/result.json'); policy=read(folder/'replay/policy_scope.json')
    require(replay['passed'] and replay['main_result_sha256']==sha(folder/'result.json'),'invalid main replay binding')
    require(replay['fresh_process'] and replay['all_saved_actions_sensor_packets_maps_meshes_scores_equal'],
        'fresh sensor/map/mesh/score replay certificate missing')
    require(replay['policy_reexecuted'] is False and replay['all_controller_decisions_reexecuted'] is False
        and replay['native_ROS_publications']==0 and policy['policy_reexecuted'] is False,
        'fixed-action reconstruction replay incorrectly claims policy execution')
    a,b=read(folder/'started.json'),read(folder/'replay/started.json')
    require(a['process_identity']!=b['process_identity'],'replay process identity equals main')
    require(result['native_node_sha256']==NODE_SHA and result['original_TARE_native_core_executed'],
        'original native core certificate missing')
    require(result['native_visibility_degrees']==[360,24] and not result['camera_visibility_model_adapted'],
        'native visibility transfer contract changed')
    require(not result['candidate_world_queries'] and not result['counts']['candidate_world_queries'],
        'forbidden candidate-world query')
    require(result['physical_case']==case and result['counts']==replay['counts'],'case/replay counts mismatch')
    summary=result['controller_summary']; paid=result['paid_actions']
    require(summary['adapter_shutdown']['bridge_exit_code']==0,'native bridge did not cleanly exit')
    require(summary['native_ready']['native_initial_timer_observed'],'native initialization handshake missing')
    require(summary['native_ready']['node_sha256']==NODE_SHA,'wrong native node')
    require(not summary['structural_posterior_used'],'structural posterior used by TARE')
    require(len(trace)==paid+1 and len(result['actions'])==paid and evidence['actions']==result['actions'],
        'paid-action/trace contract mismatch')
    require(summary['native_publication_count']==len(trace),'paid packet publication count mismatch')
    frozen=read(folder/'prediction_freeze.json')
    require(frozen['before_first_GT_access'] and frozen['all_controller_decisions_frozen'],'prediction freeze missing')
    for path,digest in frozen['artifact_sha256'].items():
        require(sha(folder/path)==digest,'pre-GT prediction changed')
    for step,row in enumerate(trace):
        packet=load_packet(folder/'packets'/f'{step:03d}.npz')
        require(packet.sha256()==row['packet_sha256'],'logical saved-packet identity mismatch')
        require(packet.action_id==step and row['paid']==step,'nonconsecutive paid observation')
        require(packet.action==(None if step==0 else result['actions'][step-1]),'stored action mismatch')
        ack=row['observed_ledger']
        require(ack['status']=='observed' and ack['action_id']==step
            and ack['unique_paid_observations']==step+1,'native paid observation acknowledgement mismatch')
    mesh_audit=[]
    for stage in ('prefix','final'):
        data=result['stages'][stage]
        with np.load(folder/f'{stage}_maps.npz',allow_pickle=False) as maps:
            require(array_hash(maps['belief'])==data['belief_sha256'],'stored measured map identity mismatch')
        for suffix in ('raw','extracted'):
            path=folder/f'{stage}_{suffix}.npz'
            with np.load(path,allow_pickle=False) as mesh:
                require({'vertices','triangles'}<=set(mesh.files),'mesh arrays missing')
                v,t=mesh['vertices'],mesh['triangles']
                require(v.ndim==2 and v.shape[1]==3 and t.ndim==2 and t.shape[1]==3,'mesh shape mismatch')
                require(np.isfinite(v).all() and (not t.size or (t.min()>=0 and t.max()<len(v))),
                    'invalid saved mesh')
                mesh_audit.append(dict(path=rel(path),sha256=sha(path),vertices=len(v),triangles=len(t)))
    plans=evidence['plans']; prefix=config['prefix']
    executed=[p for p in plans if prefix<=p['step']<paid and p['action'] is not None]
    require(all(p['action']==result['actions'][p['step']] for p in executed),'executed plan/action mismatch')
    require(len(executed)==paid-prefix,'autonomous plan count mismatch')
    native=[p for p in executed if p['planning']['phase']=='native_tare_waypoint_projection'
        and p['planning']['decision_reason']=='native_waypoint']
    waits=[p for p in executed if p['planning']['phase']=='adapter_paid_native_cadence_wait']
    returns=[p for p in executed if p not in native and p not in waits]
    require(len(native)+len(waits)+len(returns)+prefix==paid,'action attribution incomplete')
    require(summary['adapter_paid_cadence_turns']==len(waits),'adapter paid waits disagree')
    decisions=[p['planning'] for p in plans if p['step']>=prefix]
    require(not any(p.get('posterior_used_to_choose_action') for p in decisions),'TARE posterior decision leak')
    reasons=Counter(p.get('decision_reason','unspecified') for p in decisions)
    acks=[r['observed_ledger'] for r in trace]
    counters={a['waypoint']['output_counter'] for a in acks if a.get('waypoint')}
    measurement=result['stages']['final']['measurement']; m5=measurement['05cm']
    require(abs(m5['joint']-measurement['C_map']*m5['f1'])<1e-15,'J5 product mismatch')
    row=dict(**base,status='complete',C_map=measurement['C_map'],F5=m5['f1'],J5=m5['joint'],
        eligible=measurement['eligible'],paid_actions=paid,returned=result['returned'],collisions=result['collisions'],
        early_stop=paid<config['budget'],prefix_actions=prefix,native_target_actions=len(native),
        adapter_wait_actions=len(waits),adapter_return_actions=len(returns),
        native_fresh_observations=sum(bool(a['fresh_waypoint']) for a in acks),
        native_fresh_after_prefix=sum(bool(a['fresh_waypoint']) for a in acks[prefix:]),
        native_unique_confirmed_waypoints=len(counters),
        native_keypose_due_observations=sum(bool(a['expected_native_planning_due']) for a in acks),
        native_finished_observed=any(bool(a['exploration_finished']) for a in acks),
        native_invalid_waypoints=len(summary['rejected_waypoints']),
        projection_rejection_decisions=sum('projection_rejected' in p.get('decision_reason','') for p in decisions),
        return_budget_rejection_decisions=reasons['native_goal_not_return_affordable'],
        terminal_reason=summary['terminal_reason'],
        terminal_decision_reason=(plans[-1]['planning'].get('decision_reason')
            if plans and plans[-1]['step']==paid and plans[-1]['action'] is None
            else 'budget_end_no_policy_selection'),
        last_executed_decision_reason=executed[-1]['planning'].get('decision_reason') if executed else None,
        policy_reexecuted=False,main_source=rel(folder/'result.json'),replay_source=rel(folder/'replay/result.json'))
    details=dict(case=case,reasons_including_stop=dict(reasons),saved_mesh_audit=mesh_audit,
        packet_identities_checked=len(trace),native_receipts_checked=len(acks),
        replay_certificate=rel(folder/'replay/result.json'),
        independently_compared_new_replay_meshes_in_this_summary=False,
        explanation='fresh-run equality is attested by sealed runner receipt; summary reads saved main arrays only')
    return row,details


def collect(source):
    tracked={}; seal(source,'final_seal.json',tracked)
    config,manifest,aggregate=(read(source/name) for name in ('config.json','manifest.json','result.json'))
    require(config['scope']=='L2-TARE-transfer' and config['methods']==['TARE'],'wrong experiment scope')
    require(len(config['physical_cases'])==4 and config['budget']==42 and config['prefix']==18,'wrong frozen matrix')
    require(config['combined_original_and_continuation_main_starts']==4
        and config['combined_maximum_replay_attempts']==5
        and config['original_failed_replay_retained'] and not config['original_completed_main_rerun'],
        'comparator amendment accounting missing')
    require(aggregate['status'] in ('complete','complete_with_failures') and aggregate['all_completed_cases_replayed'],
        'all declared cases and successful reconstruction replays required')
    for path,digest in read(source/'prepare_seal.json').items():
        require(sha(source/path)==digest,'preparation changed')
    require(sha(source/'sources.zip')==manifest['source_archive_sha256'],'source archive changed')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        for name,digest in manifest['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest()==digest,'archived source hash mismatch')
    seal(ORIGINAL/'case00','main_seal.json',tracked,'replay')
    seal(ORIGINAL/'case00/replay','seal.json',tracked)
    old_failure=read(ORIGINAL/'case00/replay/failure.json')
    require('fixed-action fresh sensor/map trace differs' in old_failure['error']
        and old_failure['counts']['evaluation_stages']==0,'original comparator failure not retained')
    imported=read(source/'imported_main_receipt.json')
    require(imported['new_worlds']==imported['new_sensor_queries']==imported['new_main_starts']==0
        and imported['same_main_trajectory_counted_once'],'import created another main experiment')
    for name,digest in imported['copied_sha256'].items():
        require(sha(ORIGINAL/'case00'/name)==digest==sha(source/'case00'/name),
            'imported main differs from retained original')
    old_manifest=read(ORIGINAL/'manifest.json')
    for name in ('scripts/run_tare_routes_v39.py','scripts/tare_ros_bridge_v39.py',
                 'configs/virtual3d/v39_tare_transfer_overrides_20260920.json'):
        require(manifest['source_sha256'][name]==old_manifest['source_sha256'][name],
            'comparator continuation changed frozen algorithm or parameters')
    config['summary_actual_replay_attempts']=1+sum((source/f"case{c['index']:02d}"/'replay/started.json').is_file()
        for c in config['physical_cases'])
    require(config['summary_actual_replay_attempts']<=5,'replay amendment quota exceeded')
    for name,digest in manifest['input_sha256'].items():
        require(sha(ROOT/name)==digest,'common frozen input changed')
        tracked[name]=digest
    rows=[]; details=[]; aggregated={r['index']:r for r in aggregate['rows']}
    for case in config['physical_cases']:
        found=audit_case(case,source,config,tracked); row=found[0]; rows.append(row)
        if row['status']=='complete':
            details.append(found[1]); other=aggregated[case['index']]
            require(all(row[k]==other[v] for k,v in (('C_map','C_map'),('F5','Q5'),('J5','J5'),
                ('paid_actions','paid_actions'),('returned','returned'))),'aggregate metrics mismatch')
    require(all(p['equal'] for p in aggregate['prefix_comparisons']),'common prefix comparison failed')
    return config,rows,details,tracked


def render(config,rows,details,tracked,output,report):
    output.mkdir(parents=True,exist_ok=True); report.parent.mkdir(parents=True,exist_ok=True)
    fields=['parent','hypothesis','method','status','C_map','F5','J5','eligible','paid_actions','returned','collisions',
        'early_stop','prefix_actions','native_target_actions','adapter_wait_actions','adapter_return_actions',
        'native_fresh_observations','native_fresh_after_prefix','native_unique_confirmed_waypoints',
        'native_keypose_due_observations','native_finished_observed','native_invalid_waypoints',
        'projection_rejection_decisions','return_budget_rejection_decisions','terminal_reason','terminal_decision_reason',
        'last_executed_decision_reason',
        'policy_reexecuted','main_source','replay_source','failure_source']
    with (output/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    tex=['% L2 native TARE transfer only; never combine as full narrow-FOV ranking.',
        '% columns: condition, C_map, F5, J5, paid actions, native-target, adapter-wait, adapter-return.']
    successful=[r for r in rows if r['status']=='complete']
    for r in rows:
        condition=f"{r['parent']}/$h_{r['hypothesis']}$"
        if r['status']=='failed': tex.append(condition+r' & -- & -- & -- & -- & -- & -- & failed \\'); continue
        tex.append(condition+f" & {r['C_map']:.3f} & {r['F5']:.3f} & {r['J5']:.3f} & {r['paid_actions']}"+
            f" & {r['native_target_actions']} & {r['adapter_wait_actions']} & {r['adapter_return_actions']}"+r' \\')
    if successful:
        means={key:sum(r[key] for r in successful)/len(successful) for key in
            ('C_map','F5','J5','paid_actions','native_target_actions','adapter_wait_actions','adapter_return_actions')}
        tex.append(r'\midrule')
        tex.append(f"完成例均值（{len(successful)}/4） & {means['C_map']:.3f} & {means['F5']:.3f} & {means['J5']:.3f} & {means['paid_actions']:.1f}"+
            f" & {means['native_target_actions']:.1f} & {means['adapter_wait_actions']:.1f} & {means['adapter_return_actions']:.1f}"+r' \\')
    (output/'table_rows.tex').write_text('\n'.join(tex)+'\n')
    text=['# V39 原始 TARE 节点迁移结果','',
        f"已封存 4 个预注册条件，完成 {len(successful)} 例、失败 {4-len(successful)} 例；完成例均通过独立进程的固定主轨迹感知、TSDF 和指标复核。合计 4 次实际主实验、{config['summary_actual_replay_attempts']} 次固定轨迹复核尝试，其中原比较器序列化失败 1 次、成功 {len(successful)} 次。不是全部首次通过。失败保留 failed，未知质量为空，不代填 0；均值仅对完成例计算。",'',
        '原 case00 主轨迹逐文件 SHA 复制到 r1，计为同一次主实验；没有重跑主策略。原失败重放的 39 帧传感与前缀/终点地图、mesh 比较已通过，但在质量评价前因 cue 字典整数键与 JSON 字符串键的直接比较失败。更正只使用无容差的规范 JSON 比较，保留全部物理字段；额外一次复核明示计入成本，原失败目录与封存仍在。','',
        '## 逐条件结果','',
        '| 条件 | 状态 | C_map | F1@5cm | J5 | 动作 | 返回 | native目标/adapter等待/adapter返航 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        if r['status']=='failed':
            text.append(f"| {r['parent']}/h{r['hypothesis']} | failed | — | — | — | — | — | — |"); continue
        text.append(f"| {r['parent']}/h{r['hypothesis']} | complete | {r['C_map']:.6f} | {r['F5']:.6f} | {r['J5']:.6f} | {r['paid_actions']} | {'是' if r['returned'] else '否'} | {r['native_target_actions']}/{r['adapter_wait_actions']}/{r['adapter_return_actions']} |")
    text+=['','动作归因不含共同 18 动作前缀；每例满足前缀 + 原生目标执行 + adapter 付费等待 + adapter 安全返航 = 实际动作。原生目标动作是到实际 native waypoint 的公共图 BFS 执行，不等同于原生控制器直接发出的离散动作。终止决定不计付费动作。','',
        '## 原生输出与接口保护','',
        '| 条件 | fresh/确认目标ID | fresh（前缀后） | 规划周期 | 投影拒绝决定 | 返航预算拒绝决定 | 原生finished | 提前停止 |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in successful:
        text.append(f"| {r['parent']}/h{r['hypothesis']} | {r['native_fresh_observations']}/{r['native_unique_confirmed_waypoints']} | {r['native_fresh_after_prefix']} | {r['native_keypose_due_observations']} | {r['projection_rejection_decisions']} | {r['return_budget_rejection_decisions']} | {r['native_finished_observed']} | {r['early_stop']} |")
    text+=['','fresh 是保存回执的实际新路点事件；规划周期按非空 scan 每 5 次累计。投影/预算拒绝按决定次数统计（含终止决定），不是独立目标数。native finished 与 adapter 提前终止分列，不能将安全返航归因于 TARE 主动完成。','']
    for r in successful:
        text.append(f"- {r['parent']}/h{r['hypothesis']} 终止：`{r['terminal_decision_reason']}`；最后实际动作的选择原因 `{r['last_executed_decision_reason']}`；控制器状态 `{r['terminal_reason']}`；非法新路点 {r['native_invalid_waypoints']} 个，碰撞 {r['collisions']} 次。")
    text+=['','## 可支持的结论与限制','',
        '该结果证明固定原始 TARE 原生节点已在同一虚拟感知—建图—评分链中实际运行，并保留真实路点、执行及返航证据。仅有两个已见布局、每个两构型的迁移测试，不能据此证明跨场景泛化，也不与 L1 的 SWAP-I/VISTA-I 拼接为完整 TARE 公平排名。', '',
        '原生 360°×24°内部可见模型与真实窄视场 RGB-D 不同；surface/keypose/collision/occupancy 尺度及覆盖膨胀采用公开物理尺寸的一次性适配，其余点数阈值保留但有效面积意义随点密度改变。terrain 为公共平地 residual 近似；精确仿真位姿不涉及真实 SLAM 漂移。BFS 路点投影、每次等待最多 4 个真实付费右转与精确返航守卫属于 adapter，其影响不能全归因于原生 TARE 算法。','',
        '**重放范围：policy_reexecuted=false。** 原生 ROS 调度可能非确定，复核只按主实验记录动作重新获得传感、TSDF 和评分，不再次运行原生策略。汇总器核验主/重放 seal、冻结源码归档、共同输入 SHA、每个存储包的逻辑 SHA、地图摘要、mesh 结构与绑定到主结果的重放收据；不新建 World，不重新融合或评分，也不把复核当独立统计样本。fresh mesh 的逐数组比较由封存 runner 执行，汇总器只审其收据并读取主实验保存数组。','',
        f"来源：`{rel(SOURCE)}`；逐步动作/停止原因和 mesh 审计详见 `manifest.json`。",'']
    report.write_text('\n'.join(text))
    outputs=[output/'measurements.csv',output/'table_rows.tex',report]
    manifest=dict(scope='L2 native TARE transfer; descriptive and separate from L1 ranking',
        source=rel(SOURCE),declared_conditions=4,completed=len(successful),failed=4-len(successful),
        actual_main_experiments=4,fixed_trajectory_replay_attempts=config['summary_actual_replay_attempts'],
        retained_original_serialization_failures=1,successful_fixed_trajectory_replays=len(successful),
        original_failed_replay_retained=True,imported_main_zero_new_worlds=True,
        policy_reexecuted=False,simulator_calls=0,mapper_calls=0,quality_calls=0,
        packet_identities_checked=sum(d['packet_identities_checked'] for d in details),
        case_audits=details,input_sha256=tracked,script_sha256={rel(Path(__file__)):sha(Path(__file__))},
        output_sha256={rel(p):sha(p) for p in outputs})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=len(successful),failed=4-len(successful),
        packet_identities_checked=manifest['packet_identities_checked'],output=rel(output))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args()
    config,rows,details,tracked=collect(args.source)
    render(config,rows,details,tracked,OUTPUT,REPORT)


if __name__=='__main__': main()
