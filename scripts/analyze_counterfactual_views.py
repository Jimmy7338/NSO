#!/usr/bin/env python3
"""Describe all scheduled histories and actual selected-view utility."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path): return json.loads(path.read_text())


def analyze(run):
    meta=read(run/'metadata.json');config=read(run/'config.json');summary=read(run/'summary.json')
    assert meta['status']=='complete'
    choices=read(run/'choices.json');history_rows=[];calibration={name:[] for name in ('G','O','S','X','M')}
    for history in summary['histories']:
        pred=read(run/history['predictions']); outcomes=read(run/history['outcomes'])
        by_id={row['candidate_id']:row for row in outcomes}
        stats=pred['cluster_statistics']['aligned']
        history_rows.append({**history,'marker_supported_clusters':stats['marker_supported_clusters'],
            'accepted_clusters':stats['accepted_clusters'],
            'objectness_equals_geometry':pred['diagnostics']['objectness_degenerate_with_geometry'],
            'semantic_selected_differs_geometry':pred['selected']['S']!=pred['selected']['G'],
            'semantic_selected_differs_no_hidden':pred['selected']['S']!=pred['selected']['N'],
            'selected':pred['selected'], 'unmarked_hypothesis_area_m2':pred['diagnostics']['unmarked_potential_background_area_m2'],
            'predicted_scores_change_with_semantics':any(x['scores']['S']['score']!=x['scores']['G']['score'] for x in pred['candidates'])})
        for candidate in pred['candidates']:
            outcome=by_id[candidate['candidate_id']]
            for name in calibration:calibration[name].append((candidate['scores'][name]['hidden_area_m2'],outcome['new_area_m2']))
    calibrated={}
    for name,pairs in calibration.items():
        if not pairs:continue
        x,y=np.asarray(pairs).T;gamma=float(np.clip(x@y/(x@x),0,1)) if x@x>0 else 0.
        calibrated[name]={'sample_count':len(x),'fitted_shrink_factor':gamma,'raw_area_mae_m2':float(np.mean(abs(x-y))),
            'in_sample_scaled_area_mae_m2':float(np.mean(abs(gamma*x-y))),
            'predicted_to_actual_area_sum':float(x.sum()/y.sum()) if y.sum()>0 else None,
            'scope':'descriptive fit on the same training tranche; hidden prediction vs total new area is a mismatch diagnostic, not calibrated geometry probability or held-out improvement'}
    requested=sum(len(entry.get('checkpoints',config['checkpoints'])) for entry in config['fixtures'])
    unavailable=meta.get('unavailable_histories',[])
    no_candidates=[h for h in history_rows if h['candidate_count']==0]
    no_information=[h for h in history_rows if h['marker_supported_clusters']==0]
    assert requested==len(history_rows)+len(unavailable)
    averages=summary['scorer_averages']
    comparisons={}
    for other in ('G','O','N','X','M'):
        pairs=[]
        for h in history_rows:
            rows={r['scorer']:r for r in choices if r['fixture']==h['fixture'] and r['history_step']==h['history_step']}
            if not rows:continue
            pairs.append({'seed':h['seed'],'area_rate_delta':rows['S']['area_per_action']-rows[other]['area_per_action'],
                'f1_delta':rows['S']['f1_gain_05cm']-rows[other]['f1_gain_05cm'],
                'f1_rate_delta':rows['S']['f1_gain_per_action']-rows[other]['f1_gain_per_action']})
        comparisons[other]={'available_pairs':len(pairs),'mean_area_rate_delta':float(np.mean([r['area_rate_delta'] for r in pairs])) if pairs else None,
            'per_seed':{str(seed):{key:float(np.mean([r[key] for r in pairs if r['seed']==seed])) for key in ('area_rate_delta','f1_delta','f1_rate_delta')}
                for seed in sorted({r['seed'] for r in pairs})}}
    review={'scope':config['scope'],'requested_histories':requested,'available_histories':len(history_rows),
        'unavailable_histories':unavailable,'zero_candidate_histories':len(no_candidates),
        'histories_without_marker_supported_model_clusters':len(no_information),
        'histories_with_semantic_score_changes':sum(h['predicted_scores_change_with_semantics'] for h in history_rows),
        'histories_with_semantic_choice_changes':sum(h['semantic_selected_differs_geometry'] for h in history_rows),
        'objectness_equals_geometry_histories':sum(h['objectness_equals_geometry'] for h in history_rows),
        'histories':history_rows,'scorer_averages_over_available_histories':averages,'semantic_comparisons':comparisons,
        'descriptive_area_shrink_fit':calibrated,'held_out_development_checked':False,
        'independent_confirmation':False,'whole_system_advantage_proven':False,
        'protocol_gates_passed':False,'gate_note':'Only the initial fitting tranche has run; 753/754, negative controls, full calibration and closed-loop system gates remain pending. Unavailable histories are not assigned invented zero utility.'}
    target=run/'analysis';target.mkdir(exist_ok=True)
    (target/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n')
    lines=['# 同历史、同候选的真实观测价值：首批开发结果','',
        f'预定{requested}个历史，实际可评分{len(history_rows)-len(no_candidates)}个；不可用历史{len(unavailable)}个、零候选历史{len(no_candidates)}个。'
        f'实际{meta["branches"]}条付费动作分支。只有{len({h["seed"] for h in history_rows})}个开发种子；噪声和检查点为种子内条件。', '',
        '以下均值仅覆盖有候选的历史，不以零收益填补不可用历史。分支实际执行去程、到达与返程，所有评分器共享同一物理分支结果。','',
        '|评分器|新增可见面积 m²|面积/动作|F1@5cm增量|F1增量/动作|二维新增 m²|面积率后悔值|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('G','O','S','X','M','N'):
        if name not in averages:continue
        a=averages[name];lines.append(f'|{name}|{a["new_area_m2"]:.6f}|{a["area_per_action"]:.6f}|{a["f1_gain_05cm"]:.6f}|{a["f1_gain_per_action"]:.6f}|{a["coverage_gain_m2"]:.6f}|{a["area_rate_regret"]:.6f}|')
    lines += ['','G为共同几何后验，O为当前仅物体性实现，S为细类别，X为交换标签，M为缺失标签，N为同候选关闭隐藏面收益。', '',
        f'{len(history_rows)}个可用历史中，{len(no_information)}个没有被模型接纳的带标识几何簇；不可用历史不补造此项判断。'
        f'语义改变预测分数的历史有{review["histories_with_semantic_score_changes"]}个，改变最终选择的有{review["histories_with_semantic_choice_changes"]}个。'
        f'O与G退化为相同模型的历史有{review["objectness_equals_geometry_histories"]}个。', '',
        '新增可见面积是独立唯一外表面评价，不等于重建精度；F1增量包括新增完整性和已有表面误差的共同变化。'
        '隐藏面积与全场景新可见面积的尺度拟合仅用于发现失配，不称校准后验或验证集收益。', '',
        '本批不包含753/754固定开发检查、关系/形状负控制或独立测试，不能据此宣布协议或完整方案通过。'
        '原始动作、帧、候选、评分封存、可见位图和到达/最终网格均保留。', '',
        '逐历史、分种子差异、标识支持情况与描述性拟合见 review.json；源运行的 verification.json 单独记录回放是否完整通过。']
    (target/'results.md').write_text('\n'.join(lines)+'\n')
    inputs=['metadata.json','config.json','summary.json','choices.json']+[h[k] for h in summary['histories'] for k in ('predictions','outcomes')]
    provenance={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in inputs}}
    (target/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:review[k] for k in ('requested_histories','available_histories','zero_candidate_histories','histories_without_marker_supported_model_clusters','histories_with_semantic_choice_changes','protocol_gates_passed')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)
