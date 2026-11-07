import json,csv,statistics,hashlib,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--reports-root',type=Path,default=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261104'));parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent);args=parser.parse_args();D=args.output_dir;D.mkdir(parents=True,exist_ok=True);R=args.reports_root;rows=[];summary={}
for name in ('default_S40_02','history_S40_01'):
 p=R/name;events=[json.loads(line) for line in (p/'events.jsonl').read_text().splitlines()];obs={x['action_id']:x for x in events if x.get('event')=='NSO_joint_observation_accepted'};actions={x['action_id']:x for x in events if x.get('event')=='NSO_joint_action_result'};goals={x['id']:x for x in events if x.get('event')=='goal'};ends={x['id']:x for x in events if x.get('event')=='goal_result'};n=max(actions)
 semantic={}
 for e in events:
  if e.get('event')=='NSO_joint_rpc' and e.get('method')=='observe':
   params=e.get('params',{});a=params.get('action_id');receipt=e.get('response',{}).get('summary',{}).get('semantic_diagnostics',{}).get('receipt') or {};semantic[a]=receipt.get('prediction_wall_seconds')
 for a in range(1,n+1):
  item=actions[a];gid=item['goal']['id'];goal=goals[gid];end=ends[gid];prior=obs[a-1];curr=obs[a];arrival=item['arrival'];start=goal['receipt_monotonic'];done=end['receipt_monotonic'];before=curr['actual_odom_before']['receipt_monotonic'];after=curr['actual_odom_after']['receipt_monotonic'];preafter=prior['actual_odom_after']['receipt_monotonic']
  record=dict(episode=name,action_id=a,action=item['action'],goal_id=gid,navigation_s=done-start,result_to_capture_before_s=before-done,capture_and_packaging_bracket_s=after-before,prior_capture_after_to_goal_s=start-preafter,frontend_inference_s=semantic.get(a-1),arrival_pose_after_minus_result_s=arrival['after']['receipt_monotonic']-done,navigation_result_status=end['status'])
  rows.append(record)
 vals=[r for r in rows if r['episode']==name];summary[name]={'actions':n,'source_events_sha256':hashlib.sha256((p/'events.jsonl').read_bytes()).hexdigest()}
 for field in ('navigation_s','result_to_capture_before_s','capture_and_packaging_bracket_s','prior_capture_after_to_goal_s','frontend_inference_s','arrival_pose_after_minus_result_s'):
  seq=[x[field] for x in vals if x[field] is not None];summary[name][field]=dict(count=len(seq),sum_s=sum(seq),median_s=statistics.median(seq),min_s=min(seq),max_s=max(seq)) if seq else None
 summary[name]['navigation_by_action']={a:dict(count=len(v),sum_s=sum(v),median_s=statistics.median(v)) for a in sorted({x['action'] for x in vals}) if (v:=[x['navigation_s'] for x in vals if x['action']==a])}
with (D/'cycles.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
(D/'summary.json').write_text(json.dumps(dict(scope='recorded event timestamp brackets; no ROS connection or motion. Navigation, capture bracket and prior-capture-after-to-goal brackets are timing proxies; frontend inference belongs inside prior-capture-after-to-goal, and other worker/pose/transport/planning costs are not separately timestamped. No direct inference of CPU bottleneck from brackets. Bracket totals are not full mission elapsed time: startup, final observe and other intervals are not fully covered.',episodes=summary),indent=2));print(json.dumps(summary,indent=2))
