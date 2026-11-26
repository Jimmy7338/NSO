import importlib.util,json
from pathlib import Path
import numpy as np
E=Path('/home/ouc/nso_robot/analyses/sensor_repair_20261108/nov06_timing_trial/laser_coverage_evaluator.py');s=importlib.util.spec_from_file_location('original',E);e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
B=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261110');names=['recovery_S40_20261110_01','expand_history_S96_20261110_01'];cfg=B/names[0]/'joint_configuration.json';c=json.loads(cfg.read_text())['configuration'];T=e.rigid(c['planning_from_odom']);center=e.cell_center(c['return_anchor'][:2],c['shape']);coverage=e.Coverage(c['shape'],center);seen=set();last=-1;stages=[];baseline=None
for phase,name in enumerate(names):
 report=B/name;rpc=[json.loads(x) for x in (report/'joint/decisions.jsonl').read_text().splitlines()];obs=[x for x in rpc if x['request']['method']=='observe' and x['response']['result'].get('accepted')];n=0
 for i,row in enumerate(obs):
  scan=e.load_scan(Path(row['request']['params']['path']),T);assert scan['stamp_ns'] not in seen and scan['stamp_ns']>last;seen.add(scan['stamp_ns']);last=scan['stamp_ns'];coverage.add(scan);n+=1
  if phase==1 and i==0:stages.append(dict(stage='expansion_start',accepted_scans=len(seen),**coverage.measures()))
 if phase==0:baseline=coverage.observed.copy()
 stages.append(dict(stage='S40_end' if phase==0 else 'expansion_end',accepted_scans=len(seen),**coverage.measures()))
result=dict(profile=e.PROFILE,evaluator_sha256=e.digest(E),common_config=str(cfg),common_config_sha256=e.digest(cfg),planning_from_odom=T.tolist(),ROI=dict(center=center.tolist(),radius_m=12.5,resolution_m=.2,area_m2=int(coverage.roi.sum())*.04),stages=stages,accepted_unique_scans=len(seen),additional_observed_cells_beyond_S40=int((coverage.observed&~baseline).sum()),additional_observed_m2_beyond_S40=int((coverage.observed&~baseline).sum())*.04,paid_actions=101,distance_m=15.875868883848179,elapsed_s=1330.906964688,limitations=['Scene acquisition development sequence, not independent repeats or efficacy comparison.','Sampled actual laser evidence union, not ground-truth environment coverage or reconstruction accuracy.','Common fixed S40 frame; same continuous CSM odom epoch established externally.','One accepted scan per observation; not all live scans.','ROI omissions retained in stage metrics; native range cap8m and supercover unchanged.'],robot_or_ROS_actions=False)
Path(__file__).with_name('result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
