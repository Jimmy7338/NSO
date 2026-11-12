import json,hashlib
from pathlib import Path
from nso_bridge.joint_engine import JointSession
out=Path('/home/ouc/nso_robot/analyses/resume_20261112/semantic_audit/preview')
config_path=Path('/home/ouc/nso_robot/analyses/resume_20261112/configuration_S96.json')
c=json.loads(config_path.read_text());c['execution_allowed']=False;c['semantic_output']=str(out/'semantics')
source=json.loads(Path('/home/ouc/nso_robot/analyses/resume_20261112/fresh_sdk50/source.json').read_text())
s=JointSession(**c)
(out/'identity.json').write_text(json.dumps(s.identity(),indent=2))
obs=s.observe(source['joint_input'],0)
(out/'observe.json').write_text(json.dumps(obs,indent=2))
r=s.next_action()
(out/'next_action.json').write_text(json.dumps(r,indent=2))
a=r.get('selection') or {}; ca=a.get('candidate_audit') or {}
summary=dict(scope='OFFLINE stopped-input original policy preview; no ROS, actuator or executed action',execution_allowed=False,configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),input=source['joint_input'],actual_paid_actions=r['summary']['actual_paid_actions'],budget=c['configuration']['total_budget'],action_proposal=r['action'],scores=a.get('scores'),score_audit=a.get('score_audit'),candidate_audit=ca,semantic_diagnostics=r['summary'].get('semantic_diagnostics'))
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(action=r['action'],scores=a.get('scores'),candidate_audit=ca,semantic_diagnostics=summary['semantic_diagnostics']),ensure_ascii=False))
