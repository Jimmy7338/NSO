#!/usr/bin/env python3
"""Recheck saved legacy results; never run a policy, sensor, or mapper."""
import argparse, json, csv, hashlib, math, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sources={}; checks={}; cohorts={}
def read(p):
 p=str(p);data=(ROOT/p).read_bytes();sources[p]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)};return json.loads(data)
def close(a,b): assert math.isclose(a,b,rel_tol=0,abs_tol=2e-14),(a,b)
def mean(a):return statistics.mean(a)
sets=[('development','v35_online_development_20260918',8),('confirmation','v36_online_confirmation_20260918',8),('external_cpu','v39_external_cpu_20260920',16),('tare_transfer','v39_tare_adapter_r1_20260920',4)]
raw={};controllers={}
for name,d,count in sets:
 batch=read(f'audit_results/{d}/result.json');rows=[]
 for i in range(count):
  base=f'audit_results/{d}/case{i:02d}';p=f'{base}/result.json';x=read(p);raw[(name,i)]=x;seal=read(f'{base}/main_seal.json');rp=read(f'{base}/replay/result.json');assert rp['passed'];assert rp['main_result_sha256']==sources[p]['sha256'];assert seal['result.json']==sources[p]['sha256']
  c=read(f'{base}/controller.json');controllers[(name,i)]=c;assert seal['controller.json']==sources[f'{base}/controller.json']['sha256']
  identity=x['physical_case'];m=x['stages']['final']['measurement'];m0=x['stages']['prefix']['measurement'];assert m['eligible'] and m['returned'] and m['collisions']==0
  vals={}
  for t in ['02cm','05cm','10cm']:
   s=m[t];f=2*s['precision']*s['recall']/(s['precision']+s['recall']);close(f,s['f1']);close(m['C_map']*s['f1'],s['joint']); vals[t]={'precision':s['precision'],'recall':s['recall'],'F1':s['f1'],'J':s['joint']}
  rows.append({'case_index':i,'parent':identity['parent'],'hypothesis':identity['hypothesis'],'method':identity.get('method',identity.get('mode')),'C_map':m['C_map'],'prefix_C_map':m0['C_map'],'metrics':vals,'paid_actions':x['paid_actions'],'returned':x['returned'],'collisions':x['collisions'],'eligible':m['eligible'],'source':p,'main_seal':f'{base}/main_seal.json','replay':f'{base}/replay/result.json','controller_source':f'{base}/controller.json','policy_reexecuted':rp.get('policy_reexecuted',rp.get('all_controller_decisions_and_receipts_equal'))})
 methods={r['method'] for r in rows};means={}
 for method in sorted(methods):
  rr=[r for r in rows if r['method']==method];means[method]={'episodes':len(rr),'C_map':mean([r['C_map'] for r in rr]),'F1_5':mean([r['metrics']['05cm']['F1'] for r in rr]),'J5':mean([r['metrics']['05cm']['J'] for r in rr]),'paid_actions':mean([r['paid_actions'] for r in rr])}
  if 'means' in batch:
   for key,target in [('C_map','C_map'),('F1_5','Q5'),('J5','J5'),('paid_actions','paid_actions')]:close(means[method][key],batch['means'][method][target])
 pairs=[]
 for par,h in sorted({(r['parent'],r['hypothesis']) for r in rows}):
  mm={r['method']:r for r in rows if r['parent']==par and r['hypothesis']==h}
  if 'G' in mm and 'S' in mm:pairs.append({'parent':par,'hypothesis':h,'G_J5':mm['G']['metrics']['05cm']['J'],'S_J5':mm['S']['metrics']['05cm']['J'],'S_minus_G_J5':mm['S']['metrics']['05cm']['J']-mm['G']['metrics']['05cm']['J'],'C_equal':mm['G']['C_map']==mm['S']['C_map']})
 cohorts[name]={'batch_source':f'audit_results/{d}/result.json','main_episodes':count,'successful_replays':count,'replay_adds_independent_samples':False,'parent_layout_count':len({r['parent'] for r in rows}),'rows':rows,'means':means,'S_vs_G_pairs':pairs}
 if pairs:
  cohorts[name]['S_vs_G_effect']={'absolute_J5':mean([p['S_minus_G_J5'] for p in pairs]),'relative_percent':100*(means['S']['J5']/means['G']['J5']-1),'positive_conditions':sum(p['S_minus_G_J5']>0 for p in pairs),'zero_conditions':sum(p['S_minus_G_J5']==0 for p in pairs),'negative_conditions':sum(p['S_minus_G_J5']<0 for p in pairs)}
checks['sealed_main_results_and_controller_json_verified']=36*2
checks['replay_binding_verified']=36
checks['F1_and_J_arithmetic_checks']=36*3*2
# Check published CSV against actual results, never use a figure-only number as primary source.
for name,p in [('external_cpu','docs/thesis/figures/v39_external/measurements.csv'),('tare_transfer','docs/thesis/figures/v39_tare/measurements.csv')]:
 data=(ROOT/p).read_bytes();sources[p]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)};rr=list(csv.DictReader(data.decode().splitlines()));assert len(rr)==len(cohorts[name]['rows'])
 for a,b in zip(rr,cohorts[name]['rows']):
  assert a['main_source']==b['source'];close(float(a['C_map']),b['C_map']);close(float(a['F5']),b['metrics']['05cm']['F1']);close(float(a['J5']),b['metrics']['05cm']['J']);assert int(a['paid_actions'])==b['paid_actions']
checks['published_CSV_rows_rechecked']=20
# V36 / V39 G,S scores, coverage and executed actions identical; this is repeatability, not new contexts.
parity=[]
for r in cohorts['confirmation']['rows']:
 q=next(q for q in cohorts['external_cpu']['rows'] if all(q[k]==r[k] for k in ['parent','hypothesis','method']))
 for k in ['C_map','metrics','paid_actions']:assert r[k]==q[k]
 c=controllers[('confirmation',r['case_index'])];e=controllers[('external_cpu',q['case_index'])];assert c['actions']==e['actions'];parity.append({'parent':r['parent'],'hypothesis':r['hypothesis'],'method':r['method'],'same_C_P_R_F1_J_all_thresholds_and_actions':True})
checks['V36_V39_parity']=parity
# External semantic activation from saved candidate receipts.
activ=[]
for r in cohorts['external_cpu']['rows']:
 if r['method'] not in ['SWAP','VISTA']:continue
 plans=[x['planning'] for x in controllers[('external_cpu',r['case_index'])]['plans'] if x['planning'].get('phase')=='external_cpu_mechanism_planning']
 activ.append({'parent':r['parent'],'hypothesis':r['hypothesis'],'method':r['method'],'autonomous_decisions':len(plans),'positive_semantic_decisions':sum(x['semantic_term_nonzero'] for x in plans),'changed_first_action_decisions':sum(x['semantic_changed_first_action'] for x in plans),'inspection_decisions':sum(x['mechanism_phase']=='inspection' for x in plans),'geometric_fallback_decisions':sum(x['mechanism_phase']=='geometry_exploration' for x in plans),'source':r['controller_source']})
cohorts['external_cpu']['semantic_activation']=activ
# Publication uses fallback phase spelling; compare to actual list before freezing.
for method in ['SWAP','VISTA']:
 a=[r for r in activ if r['method']==method];cohorts['external_cpu'].setdefault('activation_totals',{})[method]={k:sum(x[k] for x in a) for k in ['autonomous_decisions','positive_semantic_decisions','changed_first_action_decisions','inspection_decisions','geometric_fallback_decisions']}
p='docs/thesis/figures/v39_external/semantic_activation.csv';data=(ROOT/p).read_bytes();sources[p]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
for a,b in zip(csv.DictReader(data.decode().splitlines()),activ):
 for k in ['autonomous_decisions','positive_semantic_decisions','changed_first_action_decisions','inspection_decisions','geometric_fallback_decisions']:assert int(a[k])==b[k],(a,b,k)
# Direct wrong-semantic and feedback contrasts on development cases.
for treatment,baseline in [('S','swapped'),('swapped','swapped_no_feedback')]:
 arr=[]
 for h in [0,1]:
  a=next(r for r in cohorts['development']['rows'] if r['method']==treatment and r['hypothesis']==h);b=next(r for r in cohorts['development']['rows'] if r['method']==baseline and r['hypothesis']==h)
  arr.append({'hypothesis':h,'J2_difference':a['metrics']['02cm']['J']-b['metrics']['02cm']['J'],'J5_difference':a['metrics']['05cm']['J']-b['metrics']['05cm']['J'],'J10_difference':a['metrics']['10cm']['J']-b['metrics']['10cm']['J']})
 means=cohorts['development']['means'];cohorts['development'].setdefault('ablation_contrasts',[]).append({'treatment':treatment,'baseline':baseline,'condition_effects':arr,'mean_J5_difference':means[treatment]['J5']-means[baseline]['J5'],'relative_mean_J5_percent':100*(means[treatment]['J5']/means[baseline]['J5']-1)})
# Confirm saved causality receipts and their batch source bindings.
for name,p in [('development','audit_results/v35_semantic_chain_review_20260918/result.json'),('confirmation','audit_results/v36_online_confirmation_review_20260918/result.json')]:
 x=read(p);assert x['source_batch_result_sha256']==sources[cohorts[name]['batch_source']]['sha256'];cohorts[name]['causality_review']={'source':p,'status':x['status'],'new_physical_calls_this_reaudit':0,'existing_review_counts':x['counts'],'first_action_divergences':x.get('first_semantic_action_divergences',x.get('first_action_divergences'))}
 if name=='development':cohorts[name]['causality_review']['first_correction_same_state_interventions']=[{k:r[k] for k in ['hypothesis','first_correction_paid_step','geometry_feedback_in_both_branches','actual_action_from_saved_online_plan','counterfactual_action','action_changed']} for r in x['first_correction_same_state_interventions']]
em=cohorts['external_cpu']['means'];cohorts['external_cpu']['S_vs_external_relative_J5_percent']={m:100*(em['S']['J5']/em[m]['J5']-1) for m in ['SWAP','VISTA']}
# Read saved system evidence boundaries, without model/world/map work.
v43=read('audit_results/v43_runtime_pipeline_20260921/result.json');v44=read('audit_results/v44_evidence_pipeline_20260921/result.json');
runtime_protocol=read('configs/virtual3d/v43_runtime_protocol_20260921.json')
# TARE action categories from saved source; terminal stop plans are not paid actions.
tare_rows=list(csv.DictReader((ROOT/'docs/thesis/figures/v39_tare/measurements.csv').read_text().splitlines()))
for r,a in zip(cohorts['tare_transfer']['rows'],tare_rows):
 c=controllers[('tare_transfer',r['case_index'])];plans=[p for p in c['plans'] if p['step']>=18 and p['step']<r['paid_actions']];reasons=[p['planning'].get('decision_reason') for p in plans];r['native_target_actions']=reasons.count('native_waypoint');r['adapter_wait_actions']=sum(p['planning'].get('phase')=='adapter_paid_native_cadence_wait' for p in plans);r['adapter_return_actions']=len(plans)-r['native_target_actions']-r['adapter_wait_actions'];assert r['adapter_wait_actions']==int(a['adapter_wait_actions']);assert r['adapter_return_actions']==int(a['adapter_return_actions']);assert r['native_target_actions']==int(a['native_target_actions']);assert 18+len(plans)==r['paid_actions'];assert 18+r['native_target_actions']+r['adapter_wait_actions']+r['adapter_return_actions']==r['paid_actions'];r['native_visibility_degrees']=raw[('tare_transfer',r['case_index'])]['native_visibility_degrees'];r['camera_visibility_model_adapted']=raw[('tare_transfer',r['case_index'])]['camera_visibility_model_adapted'];r['native_node_sha256']=raw[('tare_transfer',r['case_index'])]['native_node_sha256']
tm=read('docs/thesis/figures/v39_tare/manifest.json');cohorts['tare_transfer']['attempt_accounting']={k:tm[k] for k in ['actual_main_experiments','fixed_trajectory_replay_attempts','retained_original_serialization_failures','successful_fixed_trajectory_replays','imported_main_zero_new_worlds','policy_reexecuted']}
old_tare=read('audit_results/v39_tare_adapter_20260920/case00/result.json');assert sources['audit_results/v39_tare_adapter_20260920/case00/result.json']['sha256']==sources[cohorts['tare_transfer']['rows'][0]['source']]['sha256'];checks['TARE_case00_copy_not_new_main_episode']=True
claims=[
 {'id':'C1','level':'controlled_mechanism_supported','claim':'已知设施族中，类别条件结构先验在共同非语义前缀、预算和返航约束下可提高二维覆盖与三维表面完整度的联合得分。','cohorts':['confirmation'],'effect':cohorts['confirmation']['S_vs_G_effect'],'independent_unit':'2 个已见 parent 布局；每布局 2 个结构条件，不能把 8 条运行或重放当作 8 个独立场景。','not_supported':['自然开放词汇识别','未知场景泛化','完整原神经 ANS 优势','SLAM 定位精度']},
 {'id':'C2','level':'development_causal_ablation_supported','claim':'错误语义会改变路线并降低结果；实际几何反馈能纠正先验并在 5 cm 主指标上部分恢复。','cohorts':['development'],'independent_unit':'1 个已见 parent 的 2 个结构条件；开发消融，非独立确认。','not_supported':['所有阈值或条件均改善','四模块各自独立增益','反馈收益已在 V36 重新确认']},
 {'id':'C3','level':'external_mechanism_descriptive_only','claim':'在共同 CPU 设施任务与适配规则下 S 的均值高于 SWAP-I 和 VISTA-I。','cohorts':['external_cpu'],'independent_unit':'2 个已见 parent × 2 条件 × 4 方法；16 主轨迹，16 策略/传感/建图复现。','not_supported':['超过原作者完整 SWAP/VISTA 系统','外部差值全部来自语义','统计显著优越']},
 {'id':'C4','level':'native_system_transfer_feasibility','claim':'官方完整 TARE 原生节点在披露的 RGB-D/地面图适配器下完成 4 个预算、安全、返航合格任务。','cohorts':['tare_transfer'],'independent_unit':'2 个已见 parent × 2 条件；4 主轨迹，5 次固定轨迹复核尝试（1 比较器失败、4 通过）。','not_supported':['公平完整 TARE 性能排名','回放重新运行异步 ROS 策略','相机视场适配已完成']},
 {'id':'C5','level':'engineering_contract_only','claim':'V43/V44 已建立 CPU 四接口、存储门控、观测回放与离线评价通路。','sources':['audit_results/v43_runtime_pipeline_20260921/result.json','audit_results/v44_evidence_pipeline_20260921/result.json'],'actual_development_worlds':v43['actual_development_worlds']+v44['actual_development_worlds'],'actual_V40_autonomous_trajectory_performance_added':False,'runtime_controller_configuration':runtime_protocol['controller'],'runtime_protocol_source':'configs/virtual3d/v43_runtime_protocol_20260921.json','not_supported':['V40 多场景完整闭环已验证','多实例语义性能提升','当前 RPN-UQ 不确定性规划收益']}
]
out={'schema':'v45.claim_evidence_matrix.v1','date':'2026-09-21','scope':'read-only arithmetic and saved receipt audit; existing V35/V36/V39 empirical evidence and V43/V44 implementation boundary','metric':{'main':'J5 = C_map × F1@5cm','C_map':'observed map coverage fraction, full reachable evaluation raster denominator','Q':'surface precision/recall F1 on fixed complete external vertical facility reference, public union-ROI prediction scope','normalization':'dimensionless terminal joint score; not area m² gain, localization error, or percent final coverage improvement','precision_exclusion':'outside padded public ROI not penalized in V35/V36/V39; newer V40 full-domain accounting is a different protocol','thresholds_m':[0.02,0.05,0.10],'relative_effect':'(equal-weight treatment mean / baseline mean - 1) × 100; not mean individual percentage effects'},'cohorts':cohorts,'claims':claims,'validation':checks,'execution_accounting':{'new_worlds':0,'new_sensor_queries':0,'new_main_episodes':0,'new_TSDF_fusions':0,'new_surface_distance_evaluations':0,'new_policy_or_counterfactual_calls':0,'existing_main_result_rows_rechecked':36,'new_independent_samples':0,'statistical_tests_performed':0,'confidence_intervals_fabricated':False,'private_formal_seed_or_escrow_read':False},'sources':sources,'limits':['All performance cohorts use artificial RGB categories and exact simulated odometry.','V35 is one known parent; V36 confirms a frozen policy under held-out noise, not unseen geometry; V39 reuses those conditions.','V39 G/S identical scores/actions repeat V36 and cannot be pooled as independent confirmation.','No full four-module factorial ablation or jointly validated original neural ANS exists in these cohorts.','Natural-scene and robot tests deferred; V40 cross-scene performance remains pending, not implied by static assets.','This audit recomputes F1/J arithmetic from saved P/R only; it does not recompute surface distances or refit policies.']}
p=ROOT/'docs/research/V45_CLAIM_EVIDENCE_MATRIX_20260921.json'
assert json.loads(p.read_text()) == out, 'recomputed evidence matrix differs from saved publication values'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
record={'status':'passed','scope':'legacy arithmetic and recorded provenance only; no new performance evidence',
 'matrix_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'source_files_checked':len(sources),'checks':checks,'new_worlds':0,'new_TSDF_integrations':0,
 'new_surface_distance_evaluations':0,'new_main_episodes':0,'matrix_exactly_reproduced':True}
with args.output.open('x') as stream:
 json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
print(json.dumps(record,ensure_ascii=False))
