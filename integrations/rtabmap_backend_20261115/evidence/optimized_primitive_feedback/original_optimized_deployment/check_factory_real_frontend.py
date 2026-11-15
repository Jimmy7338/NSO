"""Staged factory real model/bootstrap; no actuator imports or ROS goal calls."""
import sys,json,types,io,contextlib,time,hashlib
from pathlib import Path
O=Path(__file__).parent;W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');C=O/'candidate';D=O/'factory_real_frontend';D.mkdir(exist_ok=True);sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(W/'runtime'))
import nso_bridge
nso_bridge.__path__=[str(C)]+list(nso_bridge.__path__)
name='nso_bridge.joint_engine';m=types.ModuleType(name);m.__file__=str(W/'nso_bridge/nso_bridge/joint_engine.py');sys.modules[name]=m
exec(compile((C/'joint_engine.py').read_text(),m.__file__,'exec'),m.__dict__)
R=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_primitive_feedback_fresh_S_20261115_01');lines=[json.loads(l)for l in (R/'joint/decisions.jsonl').read_text().splitlines()];settings=lines[0]['request']['params'];settings['calibration']['primitive_geometry_profile']='optimized_rebase_after_feedback_v1';settings['execution_allowed']=True
boot=next(r['request']['params']for r in lines if r['request']['method']=='observe_primitive_result')
requests=[dict(id=1,method='init',params=settings),dict(id=2,method='observe_primitive_result',params=boot),dict(id=3,method='summary',params={}),dict(id=4,method='shutdown',params={})]
saved=sys.stdin;stdout=io.StringIO();started=time.monotonic()
try:
 sys.stdin=io.StringIO(''.join(json.dumps(r,allow_nan=False)+'\n'for r in requests))
 with contextlib.redirect_stdout(stdout):m.main(['--output',str(D/'worker')])
finally:sys.stdin=saved
(D/'stdout.jsonl').write_text(stdout.getvalue());responses=[]
for line in stdout.getvalue().splitlines():
 try:
  r=json.loads(line)
  if isinstance(r,dict)and'id'in r:responses.append(r)
 except json.JSONDecodeError:pass
ok=len(responses)==4 and all(r.get('ok')is True for r in responses)
if ok:
 from joint_ros_coordinator import validate_joint_identity
 identity=validate_joint_identity(responses[0]['result']);assert identity['receipt']['geometry_pilot']['profile']=='optimized_rebase_after_feedback_v1';assert identity['receipt']['geometry_pilot']['full_optimized_history_replay_implemented']is True
receipt=dict(success=ok,elapsed_s=time.monotonic()-started,all_rpc_ok=[r.get('ok')for r in responses],actual_real_model=True,execution_allowed_identity=True,validate_joint_identity_passed=ok,profile='optimized_rebase_after_feedback_v1',new_ROS_goals=False,native_actuator_loaded=False,new_physical_trial=False,source_actual_report=str(R),factory_source_sha256=hashlib.sha256((C/'joint_engine.py').read_bytes()).hexdigest())
(D/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if not ok:
 print(json.dumps([r.get('error')for r in responses if not r.get('ok')],indent=2));raise SystemExit(1)
