#!/usr/bin/env python3
"""Audit and report the fixed-data model × objective development ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def report(root):
    metadata=read(root/'metadata.json');assert metadata['status']=='complete'
    source=Path(metadata['source']);inputs=read(root/'input_hashes.json')
    assert all(sha(Path(path))==expected for path,expected in inputs.items())
    assert sha(source/'verification.json')==metadata['source_verification_sha256']
    with zipfile.ZipFile(root/'sources.zip') as archive:
        assert all(hashlib.sha256(archive.read(path)).hexdigest()==expected
                   for path,expected in metadata['source_sha256'].items())
    rows=[];selections={};per_history=[];exact_old=0
    for variant in metadata['variants']:
        folder=root/variant;summary=read(folder/'summary.json');choices=read(folder/'choices.json')
        selection={};selected_outcomes=[]
        for history in summary['histories']:
            pred=read(folder/history['predictions']);outcome_path=folder/history['outcomes']
            assert sha(outcome_path)==sha(source/history['outcomes'])
            outcomes={row['candidate_id']:row for row in read(outcome_path)}
            key=(history['fixture'],history['history_step'])
            selection[key]=pred['selected']
            if variant=='v3_rate':
                original=read(source/history['predictions'])
                assert pred['candidates']==original['candidates'] and pred['selected']==original['selected']
                exact_old+=1
            for scorer in pred['scorers']:
                ranking=sorted(enumerate(pred['candidates']),key=lambda p:(
                    -p[1]['scores'][scorer]['final_score'],p[1]['cost'],p[0]))
                assert pred['selected'][scorer]==ranking[0][1]['candidate_id']
            chosen=outcomes[pred['selected']['S']];selected_outcomes.append(chosen)
            per_history.append({'variant':variant,**{k:history[k] for k in (
                'fixture','history_step','seed','depth_sigma_m')},'selected':pred['selected'],
                'actual_sparse_joint_auc_05cm':chosen['branch_joint_auc_05cm'],
                'f1_gain_05cm':chosen['f1_gain_05cm'],'new_area_m2':chosen['new_area_m2']})
        selections[variant]=selection
        rows.append({'variant':variant,**summary['scorer_averages']['S'],
            'no_hidden_averages':summary['scorer_averages']['N'],
            'actual_sparse_joint_auc_05cm':float(np.mean([o['branch_joint_auc_05cm'] for o in selected_outcomes])),
            'semantic_choice_changes':sum(v['S']!=v['G'] for v in selection.values()),
            'fine_vs_objectness_choice_changes':sum(v['S']!=v['O'] for v in selection.values()),
            'all_scorers_same_choice_histories':sum(len(set(v.values()))==1 for v in selection.values())})
    changes={}
    for a,b in [('v3_rate','v5_rate'),('v3_auc','v5_auc'),('v3_rate','v3_auc')]:
        changes[f'{a}_to_{b}']=sum(selections[a][k]['S']!=selections[b][k]['S'] for k in selections[a])
    result={'scope':metadata['scope'],'physical_branches_reused':metadata['branches'],
        'new_physical_branches':0,'histories':metadata['histories'],'rows':rows,
        'selection_changes':changes,'per_history':per_history,
        'verification':{'source_inputs_checked':len(inputs),'old_prediction_histories_exact':exact_old,
                        'archived_sources_valid':True,'copied_outcomes_exact':True,'ranking_selection_consistent':True},
        'semantic_advantage_proven':False,'independent_confirmation':False,
        'auc_limit':'actual AUC uses original prefix/arrival/final checkpoints and hold; not per-frame reconstruction evaluation',
        'primary_protocol_unchanged':True}
    (root/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 固定72分支的模型 × 评分目标开发消融','',
        '所有数值重用751/752的6个可用历史、72条已执行分支；新增物理分支为0。'
        '另2个预定历史仍不可用。新方法是在已查看的数据上设计，不能解释为独立预测成功。','',
        '|模型／目标|面积/动作|平均F1增量|二维新增m²|实际稀疏联合AUC|S改变G选择|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["variant"]}|{r["area_per_action"]:.6f}|{r["f1_gain_05cm"]:.6f}|{r["coverage_gain_m2"]:.6f}|{r["actual_sparse_joint_auc_05cm"]:.6f}|{r["semantic_choice_changes"]}/6|')
    lines += ['',
        'V5模型修正没有改变最终选择；在每一种模型与评分目标下，G/O/S/X/M均同选。'
        '共同预算AUC改变5/6历史的选择，均值改善只能归因于评分目标，语义选择收益仍为零。'
        'N在AUC评分下有1个历史不同，平均面积率0.551441、F1增量0.061411均高于S；隐藏模型净贡献仍负。','',
        '失败保留：752、低噪声、step160的新选择花16动作，新增面积仍为0，F1变化为−0.000495；'
        '751、高噪声、step80的实际稀疏联合AUC由0.841055降到0.819313。均值改善不代表逐例改善。','',
        '新评分在每个付费动作之后更新共同预测并集/逐点最大质量，积分到共同48动作预算；路线结束后保持。'
        '若累计代理为u_t、u_0=0，则积分增量为Σ(48−t+0.5)Δu_t/48。'
        '这是一条路线完成后空闲的预测目标，不包含余下预算重新规划，也不等于校准的真实F1。','',
        '实际联合AUC仍使用原实验起点、到达、终点三个重建检查点插值后保持，未增加逐帧F1评价。'
        '原面积率主指标、753/754门槛及负控制要求全部保留。尚未启用这些检查种子或独立种子。','',
        f'验证：{len(inputs)}个源输入重新核对哈希；6个历史的原评分逐字段精确复现；'
        '源快照、复制的实际结果表、选择与排名一致。上述验证保证可追溯，不是效能通过。']
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows':rows,'changes':changes,'verification':result['verification']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);report(parser.parse_args().run)
