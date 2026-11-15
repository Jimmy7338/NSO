"""Real full frontend + actual fresh04 bootstrap; no cached masks, ROS or actuator."""
import json,time,hashlib,subprocess,os
from pathlib import Path
O=Path(__file__).parent/'actual_bootstrap04_parallel';O.mkdir(exist_ok=True)
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');source=Path('/home/ouc/nso_robot/analyses/resume_20261115/NSO_rtabmap_integrated_20261115_04/configuration.json')
settings=json.loads(source.read_text());settings['execution_allowed']=False
settings['calibration']['compute_profile']='exact_candidate_batch_parallel_v1'
packet=Path(json.loads((source.parent/'fresh_capture.json').read_text())['joint'])
requests=[dict(id=1,method='init',params=settings),dict(id=2,method='observe_viewpoint_result',params=dict(path=str(packet),attempt_id=0,ros_status=None,reason='actual_fresh04_no_actuator_import_order_regression')),dict(id=3,method='summary',params={}),dict(id=4,method='shutdown',params={})]
stdin=''.join(json.dumps(x,allow_nan=False)+'\n'for x in requests)
env=dict(os.environ,PYTHONPATH=str(W/'nso_bridge'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
command=['/home/ouc/nso_robot/envs/science312/bin/python','-s','-B','-m','nso_bridge.joint_engine','--output',str(O/'real_worker')]
started=time.monotonic();p=subprocess.run(command,input=stdin,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=120)
(O/'stdout.jsonl').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
responses=[]
for line in p.stdout.splitlines():
 try:
  r=json.loads(line)
  if isinstance(r,dict)and'id'in r:responses.append(r)
 except json.JSONDecodeError:pass
ok=p.returncode==0 and len(responses)==4 and all(x.get('ok')is True for x in responses)
semantic=[]
for path in (O/'real_worker/semantics').glob('*.json'):
 r=json.loads(path.read_text());semantic.append(dict(frame_id=r.get('frame_id'),model_execution_performed=r.get('model_execution_performed'),canonical_labelled_pixels=r.get('canonical_labelled_pixels'),source_file_sha256=r.get('source_file_sha256'),receipt=str(path)))
r=dict(schema='real_frontend_deferred_parallel_bootstrap_regression/1',success=ok,elapsed_s=time.monotonic()-started,returncode=p.returncode,request_methods=[x['method']for x in requests],all_rpc_ok=[x.get('ok')for x in responses],actual_real_frontend_requested=True,actual_model_execution_receipts=semantic,precomputed_semantic_cache_override=False,native_actuation=False,ROS_goal_sent=False,observations_fused=1 if ok else None,full_selection_run=False,compute_profile=settings['calibration']['compute_profile'],source_configuration=str(source),source_configuration_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_packet=str(packet),source_packet_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),joint_engine_sha256=hashlib.sha256((W/'nso_bridge/nso_bridge/joint_engine.py').read_bytes()).hexdigest(),stderr_path=str(O/'stderr.log'))
if len(responses)>1:r['bootstrap_accepted']=responses[1].get('result',{}).get('accepted')
(O/'receipt.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2))
if not ok:raise SystemExit(1)
