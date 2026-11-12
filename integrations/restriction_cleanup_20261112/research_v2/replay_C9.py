"""Forced recorded observation-prefix replay, not an executed v2 trajectory."""
import json,hashlib
from pathlib import Path
from nso_bridge.joint_engine import JointSession
ROOT=Path('/home/ouc/nso_robot/analyses/ros_checked_exploration_v2_20261112')
REPORT=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261112/comparison_C40_20261112_01')
c=json.loads((REPORT/'joint_configuration.json').read_text());c['configuration']['policy_variant']='ros_checked_exploration_v2';c['execution_allowed']=False;c['semantic_output']=str(ROOT/'semantics')
s=JointSession(**c);n=0
for line in (REPORT/'joint/decisions.jsonl').open():
 row=json.loads(line);req=row.get('request',{});resp=row.get('response',{})
 if req.get('method')!='observe' or not resp.get('ok') or not resp.get('result',{}).get('accepted'):continue
 params=req['params'];aid=params['action_id']
 if aid>9:break
 if aid:
  check=s.backend.assess_action(0,params['action'])
  assert check['allowed'],check
  s.runtime.states[0]['pending']=check
 s.observe(**params);n+=1
before=s.mapper.belief.copy();result=s.next_action();assert (before==s.mapper.belief).all()
(ROOT/'C9_next_action.json').write_text(json.dumps(result,indent=2))
summary=dict(scope='OFFLINE forced actual C prefix observations0..9, then new-profile next_action; not a v2 rollout or actual execution',execution_allowed=False,observations=n,measured_belief_unchanged_by_planning=True,belief_sha256=hashlib.sha256(before.tobytes()).hexdigest(),identity=s.identity(),action=result['action'],pending=result.get('pending'),candidate_audit=(result.get('selection') or {}).get('candidate_audit'),original_report=str(REPORT))
(ROOT/'C9_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(observations=n,action=result['action'],pending=result.get('pending'),candidate_count=(result.get('selection') or {}).get('candidate_audit',{}).get('candidate_count'))))
