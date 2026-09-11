#!/usr/bin/env python3
"""Compare original and augmented shared pools using frozen V3 scores.

No navigation, fusion, calibration or GT-based proposal. Predictions are sealed
before parsing old physical outcomes. Completion requires a separate successful
replay, explicitly acknowledged through --finalize after the replay owner agrees.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify_run(folder):
    meta=read(folder/'metadata.json');assert meta['status']=='complete'
    config=read(folder/'config.json')
    assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==meta['config_sha256']
    hashes={}
    for name,expected in read(folder/'artifact_hashes.json').items():
        path=(folder/name).resolve();assert path.is_relative_to(folder)
        actual=sha(path);assert actual==expected,str(path);hashes[str(path)]=actual
    for seal in folder.glob('*/pre_outcome_seal.json'):
        for name,expected in read(seal).items():assert sha(seal.parent/name)==expected
    return meta,hashes


def render_report(output,complete):
    summary=read(output/'summary.json');lines=['# V6增补后的完整共同候选池对照','',
        '状态：完整输入核验及新增分支独立物理回放均通过。' if complete else '状态：合并预测与结果核验已完成；新增分支独立物理回放尚待执行方确认，以下为待确认计算。','',
        '这是已有开发诊断后的候选增补，不是预注册独立测试。原751／752的6个可用历史全部保留，'
        '4个step80历史各由12条增至16条，2个step160仍为12条，共88条已执行分支（原72＋新增16）。'
        '预定但缺失的2个高噪声step160历史不补造零收益。六评分器始终共享每历史完整联合池，未使用extras-only均值代替完整比较。','',
        'G：几何后验；O：物体性；S：细类别；X：类别交换；M：标签缺失；N：关闭隐藏表面收益。'
        'rate按代理增益/动作选路；horizon_auc按48动作预算内累计代理增益选路。实际联合AUC来自原执行器的到达/终点检查点插值并保持末值，非逐帧重建指标。','']
    for objective,title in [('rate','V3 rate'),('horizon_auc','V3 horizon_auc')]:
        result=summary['objectives'][objective]
        lines += [f'## {title}','',
            '逐历史选择：`原ID→联合池ID`，ID≥12为新增候选。','',
            '|种子/噪声cm/step|候选数|G|O|S|X|M|N|','|---|---:|---|---|---|---|---|---|']
        for h in result['histories']:
            key=f"{h['seed']}/{100*h['depth_sigma_m']:g}/{h['history_step']}"
            lines.append('|'+key+'|'+str(h['union_candidates'])+'|'+'|'.join(
                f"{h['old_selected'][s]}→{h['union_selected'][s]}" for s in ('G','O','S','X','M','N'))+'|')
        lines += ['','完整联合池选择的宏均值（6历史等权；只有2个开发种子，不能视为6个独立样本）：','',
            '|评分器|新面积m²|面积/动作|ΔF1@5cm|ΔF1/动作|二维新增m²|联合增量AUC@5cm|面积率后悔值|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for s,a in result['union_means'].items():
            lines.append('|'+s+'|'+'|'.join(f'{a[k]:.6f}' for k in (
                'new_area_m2','area_per_action','f1_gain_05cm','f1_gain_per_action','coverage_gain_m2',
                'joint_gain_auc_05cm','area_rate_regret'))+'|')
        lines += ['','从原池到联合池的变化（联合减原；面积率后悔值的参照最大值也随池扩大）：','',
            '|评分器|改选历史数|Δ面积/动作|ΔF1/动作|Δ二维新增m²|Δ联合增量AUC|',
            '|---|---:|---:|---:|---:|---:|']
        for s,d in result['old_to_union'].items():
            lines.append(f"|{s}|{d['choice_changes']}|"+'|'.join(f'{d[k]:.6f}' for k in (
                'area_per_action','f1_gain_per_action','coverage_gain_m2','joint_gain_auc_05cm'))+'|')
        lines += ['','S相对各对照（同一联合池）：','',
            '|对照|选择不同历史数|S−对照面积/动作|S−对照ΔF1/动作|S−对照联合增量AUC|',
            '|---|---:|---:|---:|---:|']
        for s,d in result['semantic_comparisons'].items():
            lines.append(f"|{s}|{d['choice_differences']}|"+'|'.join(f'{d[k]:.6f}' for k in (
                'area_per_action','f1_gain_per_action','joint_gain_auc_05cm'))+'|')
        lines += ['',result['interpretation'],'']
    c=summary['checks']
    lines += ['## 完整性与原门槛','',
        f"原72条rate预测逐项一致；原72条horizon_auc预测与已冻结V3 AUC重评分逐项一致；"
        f"新增16条rate预测与增补执行前封存预测一致。原72条outcomes与各分支原始outcome.json完全一致，合并后仍使用相同记录。"
        f"6历史的prefix记录、RGB-D/雷达、地图/网格及4个fixture的参考数组均一致。输入文件哈希核验数量：{c['input_files_verified']}。",'',
        '全部合并预测先写盘并形成pre_outcome_analysis_seal.json，再解析已有outcomes。'
        '这种顺序只保证本脚本未用结果影响重评分；原结果和诊断此前已被查看，所以不能称前瞻盲测。'
        '没有重新执行导航、重新融合或重新采样参考；原指标复核是记录等同性，物理正确性依赖两批各自独立回放。','',
        '原协议的安全、共享输入、付费动作、真值隔离和指标完整性要求仍适用。'
        '候选数从12变为最多16，且规则是在诊断后设计，效果阈值只能作开发描述，不能据此判原预注册测试通过。'
        '753／754固定开发检查、类别关系/形状负控制、独立确认及完整四模块/主流基线测试均未由本比较完成；'
        '原语义增益与隐藏模型净收益门槛没有降低。详见summary.json的分种子对照、排序读数和完整哈希。','']
    (output/'results.md').write_text('\n'.join(lines))


def worker(original,extra,baseline,output,snapshot):
    sys.path.insert(0,str(snapshot))
    import numpy as np
    from scipy.stats import spearmanr
    from env.virtual3d_inspection_v4 import InspectionConfigV4
    from utils.rgbd_contract import RGBDFrame,PlanarScan
    from scripts.eval_counterfactual_views import remap,grid_config
    from nso.counterfactual_view_scoring import score_routes,CounterfactualScoreConfig
    import nso.counterfactual_view_scoring as scoring
    assert Path(scoring.__file__).resolve().is_relative_to(snapshot)
    old_meta,old_hashes=verify_run(original);new_meta,new_hashes=verify_run(extra)
    assert Path(new_meta['original_run']).resolve()==original
    assert sha(original/'verification.json')==new_meta['original_verification_sha256']
    assert read(original/'verification.json')['status']=='passed_full'
    relevant=['env/virtual3d.py','env/virtual3d_v2.py','env/virtual3d_inspection_v4.py',
              'nso/semantic_completion_v3.py','nso/mapping3d.py','nso/mapping3d_v2.py','nso/camera_mapping_v2.py',
              'utils/reconstruction_metrics.py','utils/counterfactual_surface_visibility.py','utils/rgbd_contract.py']
    for name in relevant:assert old_meta['source_sha256'][name]==new_meta['source_sha256'][name]
    input_hashes=old_hashes|new_hashes
    def checked(path):
        path=Path(path);actual=sha(path)
        if str(path) in input_hashes:assert input_hashes[str(path)]==actual
        input_hashes[str(path)]=actual;return path
    original_summary=read(original/'summary.json');extra_summary=read(extra/'summary.json')
    new_index={(h['fixture'],h['history_step']):h for h in extra_summary['histories']}
    assert set(new_index)=={(h['fixture'],h['history_step']) for h in original_summary['histories']}
    old_config=read(original/'config.json');new_config=read(extra/'config.json')
    for key in old_config:
        if key!='scope':assert old_config[key]==new_config[key],key
    rescore_meta=read(checked(baseline/'metadata.json'))
    assert rescore_meta['status']=='complete' and Path(rescore_meta['source']).resolve()==original
    assert rescore_meta['source_sha256']['nso/counterfactual_view_scoring.py']==new_meta['source_sha256']['nso/counterfactual_view_scoring.py']
    pending=[];prefix_checks=[]
    for history in original_summary['histories']:
        fixture=history['fixture'];index=history['history_step'];relative=Path(fixture)/f'history_{index:04d}'
        old_dir=original/relative;new_dir=extra/relative;target=output/relative;target.mkdir(parents=True)
        old_routes=read(old_dir/'candidates.json');new_routes=read(new_dir/'candidates.json')
        old_ids=[r['candidate_id'] for r in old_routes];new_ids=[r['candidate_id'] for r in new_routes]
        assert len(old_ids)==12 and len(new_ids)<=4 and old_ids==list(range(12))
        assert len(set(new_ids))==len(new_ids) and not(set(old_ids)&set(new_ids))
        assert new_ids==list(range(12,12+len(new_ids)))
        assert {tuple(r['pose']) for r in old_routes}.isdisjoint({tuple(r['pose']) for r in new_routes})
        assert read(old_dir/'candidate_audit.json')['safe_hash']==read(new_dir/'candidate_audit.json')['safe_hash']
        oc=read(original/fixture/'fixture.json')['environment'];nc=read(extra/fixture/'fixture.json')['environment'];assert oc==nc
        c=InspectionConfigV4(**oc)
        all_records=read(original/fixture/'prefix/records.json')
        assert all_records==read(extra/fixture/'prefix/records.json')
        for i in range(len(all_records)):
            for kind in ('frames','scans'):
                name=Path(fixture)/'prefix'/kind/f'{i:04d}.npz'
                assert input_hashes[str(original/name)]==input_hashes[str(extra/name)],str(name)
        for filename in ('prefix_map.npz','prefix_mesh.npz'):
            a=np.load(old_dir/filename);b=np.load(new_dir/filename)
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        a=np.load(original/fixture/'reference.npz');b=np.load(extra/fixture/'reference.npz')
        assert a.files==b.files
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        records=all_records[:index+1]
        frames=[RGBDFrame.load(original/fixture/'prefix/frames'/f'{i:04d}.npz') for i in range(index+1)]
        scans=[PlanarScan.load(original/fixture/'prefix/scans'/f'{i:04d}.npz') for i in range(index+1)]
        shape=(round(c.height_m/c.resolution_m),round(c.width_m/c.resolution_m))
        mappers={mode:remap(frames,scans,records,c,shape,mode) for mode in ('aligned','shuffled','absent')}
        routes=old_routes+new_routes;write(target/'candidates.json',routes)
        base=read(old_dir/'predictions.json');added=read(new_dir/'predictions.json')
        predictions={}
        for objective in ('rate','horizon_auc'):
            pred=score_routes(mappers,routes,CounterfactualScoreConfig(grid_config(c),c,
                max_route_actions=old_config['branch_actions'],score_objective=objective))
            old_prediction=base if objective=='rate' else read(checked(baseline/'v3_auc'/relative/'predictions.json'))
            assert pred['candidates'][:12]==old_prediction['candidates'],(fixture,index,objective,'old prediction changed')
            old_selected={s:next(cid for cid in pred['rankings'][s] if cid in old_ids) for s in pred['scorers']}
            assert old_selected==old_prediction['selected']
            if objective=='rate':assert pred['candidates'][12:]==added['candidates'],'extra prediction changed'
            pred['old_pool_selected']=old_selected
            write(target/f'predictions_{objective}.json',pred);predictions[objective]=pred
        prefix_checks.append({'fixture':fixture,'history_step':index,'all_prefix_records':len(all_records),
                              'prefix_physical_bytes_equal':True,'prefix_map_mesh_arrays_equal':True,'reference_arrays_equal':True,
                              'old_rate_predictions_exact':12,'old_auc_predictions_exact':12,'extra_rate_predictions_exact':len(new_routes)})
        pending.append((history,relative,old_routes,new_routes,predictions))
        print('predictions sealed for',fixture,index,'pool',len(routes),flush=True)
        del mappers
    seal={str(path.relative_to(output)):sha(path) for path in output.glob('*/history_*/*.json')}
    write(output/'pre_outcome_analysis_seal.json',{'sealed_unix':time.time(),'files':seal,
          'scope':'no outcomes parsed by this script before this seal; prior development outcomes were already known'})
    # Only now parse existing physical outcomes. No evaluator is run here.
    scorers=('G','O','S','X','M','N');choices={k:[] for k in ('rate','horizon_auc')};history_outputs={k:[] for k in choices}
    metrics=('new_area_m2','area_per_action','f1_gain_05cm','f1_gain_per_action','coverage_gain_m2',
             'joint_gain_auc_05cm','branch_joint_auc_05cm','area_rate_regret','paid_actions')
    for history,relative,old_routes,new_routes,predictions in pending:
        originals=read(original/relative/'outcomes.json');addition=read(extra/relative/'outcomes.json')
        assert [r['candidate_id'] for r in originals]==[r['candidate_id'] for r in old_routes]
        assert [r['candidate_id'] for r in addition]==[r['candidate_id'] for r in new_routes]
        before=originals[0]['before']
        for origin,route_rows,outcome_rows in ((original,old_routes,originals),(extra,new_routes,addition)):
            for route,outcome in zip(route_rows,outcome_rows):
                assert outcome==read(origin/relative/f"candidate_{route['candidate_id']:03d}"/'outcome.json')
                assert outcome['before']==before
                assert outcome['planned_actions']==route['cost'] and outcome['arrival_actions']==route['arrival_action']
                assert outcome['failure'] is None and outcome['paid_actions']==route['cost']
        combined=originals+addition;write(output/relative/'outcomes.json',combined)
        assert read(output/relative/'outcomes.json')[:12]==originals
        by_id={r['candidate_id']:r for r in combined}
        for objective,pred in predictions.items():
            history_outputs[objective].append({**history,'union_candidates':len(combined),
                'old_selected':pred['old_pool_selected'],'union_selected':pred['selected']})
            for pool,available,selected in (('old',pred['candidates'][:12],pred['old_pool_selected']),
                                             ('union',pred['candidates'],pred['selected'])):
                actual=np.array([by_id[r['candidate_id']]['area_per_action'] for r in available])
                for scorer in scorers:
                    outcome=by_id[selected[scorer]];scores=np.array([r['scores'][scorer]['score'] for r in available])
                    correlation=float(spearmanr(scores,actual).statistic) if np.ptp(scores)>0 and np.ptp(actual)>0 else None
                    pairs=[(i,j) for i in range(len(scores)) for j in range(i+1,len(scores)) if actual[i]!=actual[j]]
                    pairwise=float(np.mean([1 if (scores[i]-scores[j])*(actual[i]-actual[j])>0 else .5 if scores[i]==scores[j] else 0 for i,j in pairs])) if pairs else None
                    row={**{k:history[k] for k in ('fixture','history_step','seed','depth_sigma_m')},'pool':pool,'scorer':scorer,
                         'chosen':selected[scorer],**{k:outcome[k] for k in metrics if k not in ('joint_gain_auc_05cm','area_rate_regret')},
                         'joint_gain_auc_05cm':outcome['branch_joint_auc_05cm']-outcome['before']['joint_05cm'],
                         'area_rate_regret':float(actual.max()-outcome['area_per_action']),
                         'spearman_area_rate':correlation,'pairwise_area_rate_accuracy':pairwise}
                    choices[objective].append(row)
    objectives={}
    for objective,rows in choices.items():
        def means(pool):return {s:{m:float(np.mean([r[m] for r in rows if r['pool']==pool and r['scorer']==s])) for m in metrics} for s in scorers}
        old=means('old');union=means('union');hist=history_outputs[objective]
        differences={s:{'choice_changes':sum(h['old_selected'][s]!=h['union_selected'][s] for h in hist),
                        **{m:union[s][m]-old[s][m] for m in metrics}} for s in scorers}
        comparisons={}
        for other in ('G','O','X','M','N'):
            per_seed={}
            for seed in sorted({r['seed'] for r in rows}):
                per_seed[str(seed)]={m:float(np.mean([r[m] for r in rows if r['pool']=='union' and r['scorer']=='S' and r['seed']==seed])-
                    np.mean([r[m] for r in rows if r['pool']=='union' and r['scorer']==other and r['seed']==seed])) for m in metrics}
            comparisons[other]={'choice_differences':sum(h['union_selected']['S']!=h['union_selected'][other] for h in hist),
                **{m:union['S'][m]-union[other][m] for m in metrics},'per_seed':per_seed}
        interpretation=('S、G、O、X、M、N在全部历史选择一致；本目标下未发现可验证语义选择收益或隐藏模型净收益。'
            if all(d['choice_differences']==0 for d in comparisons.values()) else
            '评分器之间存在选择差异；具体正负以同池对照及分种子差异为准，不把增补候选的共同收益归因于语义。')
        objectives[objective]={'histories':hist,'old_means':old,'union_means':union,'old_to_union':differences,
                              'semantic_comparisons':comparisons,'interpretation':interpretation}
        write(output/f'choices_{objective}.json',rows)
    assert all(sha(name)==value for name,value in input_hashes.items())
    assert all(sha(output/name)==value for name,value in seal.items())
    write(output/'input_hashes.json',input_hashes)
    write(output/'summary.json',{'scope':'post-diagnosis development candidate augmentation on known 751/752 histories',
        'new_navigation_branches':0,'original_physical_branches':72,'added_existing_physical_branches':16,
        'objectives':objectives,'checks':{'input_files_verified':len(input_hashes),'histories':prefix_checks,
            'old_outcomes_exact':72,'shared_prefix_and_reference_exact':True,'prediction_seal_verified':True},
        'protocol_gates_passed':False,'independent_confirmation':False,'whole_system_advantage_proven':False})


def run(original,extra,baseline,output):
    original,extra,baseline,output=[p.resolve() for p in (original,extra,baseline,output)]
    assert not output.exists() and all(not output.is_relative_to(p) for p in (original,extra,baseline))
    output.mkdir(parents=True);started=time.time();new_meta=read(extra/'metadata.json')
    meta={'status':'running','original':str(original),'augmentation':str(extra),'auc_baseline':str(baseline),
          'script_sha256':sha(__file__),'scorer_archive_sha256':sha(extra/'sources.zip'),'started_unix':started,
          'augmentation_verification_at_start':read(extra/'verification.json') if (extra/'verification.json').exists() else None,
          'workers':1,'new_navigation_branches':0,'runtime_estimate_minutes':[1,3],'disk_estimate_mib':15}
    write(output/'metadata.json',meta)
    try:
        with tempfile.TemporaryDirectory(prefix='nso_union_pool_') as name:
            snapshot=Path(name)
            with zipfile.ZipFile(extra/'sources.zip') as archive:
                for member in archive.infolist():
                    target=(snapshot/member.filename).resolve();assert target.is_relative_to(snapshot)
                    if not member.is_dir():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(member))
            for name,value in new_meta['source_sha256'].items():assert sha(snapshot/name)==value,name
            env=os.environ|{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--original',str(original),
                '--augmentation',str(extra),'--auc-baseline',str(baseline),'--output',str(output),'--snapshot',str(snapshot)],
                cwd=snapshot,env=env,check=True)
        meta.update(status='awaiting_replay_confirmation',elapsed_s=time.time()-started)
    except Exception as error:
        meta.update(status='failed',error=repr(error));write(output/'metadata.json',meta);raise
    write(output/'metadata.json',meta);render_report(output,False)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})


def finalize(output):
    meta=read(output/'metadata.json');assert meta['status']=='awaiting_replay_confirmation'
    for name,value in read(output/'artifact_hashes.json').items():assert sha(output/name)==value,name
    for path,value in read(output/'input_hashes.json').items():assert sha(path)==value,path
    verification=Path(meta['augmentation'])/'verification.json';review=read(verification)
    assert review['status']=='passed_full','independent complete replay has not passed'
    meta.update(status='complete',augmentation_verification_final=review,
                augmentation_verification_sha256=sha(verification),finalized_unix=time.time(),
                completion_scope='union-pool computation plus acknowledged independent full replay; no independent efficacy claim')
    write(output/'metadata.json',meta);render_report(output,True)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path);p.add_argument('--augmentation',type=Path)
    p.add_argument('--auc-baseline',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--snapshot',type=Path);p.add_argument('--finalize',action='store_true')
    a=p.parse_args()
    if a.finalize:finalize(a.output.resolve())
    elif a.worker:worker(a.original,a.augmentation,a.auc_baseline,a.output,a.snapshot)
    else:run(a.original,a.augmentation,a.auc_baseline,a.output)
