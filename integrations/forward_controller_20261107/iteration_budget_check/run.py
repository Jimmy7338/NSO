import os
os.environ['ROS_MASTER_URI']='http://127.0.0.1:11404'
os.environ['ROS_HOSTNAME']='127.0.0.1'
os.environ['ROS_IP']='127.0.0.1'
import sys,json,math,time,socket,subprocess,signal,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parent;SOURCE=D/'probes/control_can_offset_06';OUT=Path(os.environ['ITERATION_OUTPUT_DIR']).resolve();SEQ=Path(os.environ['SEQUENCE_OUTPUT_DIR']).resolve()
if OUT==P or P in OUT.parents:raise ValueError('output must be separate')
OUT.mkdir(parents=True,exist_ok=True)
def worker():
 import rospy,numpy as np
 from geometry_msgs.msg import PoseStamped
 rospy.init_node('iteration_budget_fixture_setup',disable_signals=True)
 z=np.load(SOURCE/'failure_scan_ring.npz');meta=json.loads((SOURCE/'failure_scan_ring.json').read_text());scans=[x for x in meta['messages'] if x['topic']=='/scan'];poses={x['source_stamp_ns']:x for x in meta['messages'] if x['topic']=='/nso_csm/pose_stamped'};inp=OUT/'inputs';inp.mkdir(exist_ok=True)
 def scan(i):
  p=inp/f'scan_{i:03}.bin';p.write_bytes(z[scans[i]['key']].tobytes());return p
 def pose(i):
  p=inp/f'pose_{i:03}.bin';p.write_bytes(z[poses[scans[i]['source_stamp_ns']]['key']].tobytes());return p
 fixed=OUT/'fixed_jobs.txt';fixed.write_text(' '.join(map(str,[96,scan(96),scan(99),pose(96),pose(98),pose(99)]))+'\n')
 jobs=[f'{scan(96)} {pose(96)}']+[' '.join(map(str,[i,scan(i),0,0,0])) for i in range(97,105)];recursive=OUT/'recursive_jobs.txt';recursive.write_text('\n'.join(jobs)+'\n')
 original=dict(json.loads((D/'instrumented_csm_candidate/state_fixture_results.json').read_text())['parameters'],kf_dist_linear=.1,kf_dist_angular=math.pi/18);results=[]
 for limit in [10,50]:
  params=dict(original,max_iterations=limit);rospy.set_param('/fixture/matcher',params);rospy.set_param('/sequence/A/matcher',params)
  for case,binary,job in [('fixed',OUT/'state_fixture',fixed),('recursive',SEQ/'native_sequence',recursive)]:
   cmd=[str(binary),str(job)]+(['A'] if case=='recursive' else []);r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);(OUT/f'{case}_{limit}.stdout.log').write_text(r.stdout);(OUT/f'{case}_{limit}.stderr.log').write_text(r.stderr);assert r.returncode==0
   rows=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')]
   for x in rows:
    x['configured_max_iterations']=limit;x['at_or_above_iteration_budget']=x.get('iterations',0)>=limit
    if case=='recursive' and x['reference_index']==0:x['reference_index']=96
   results.append(dict(case=case,limit=limit,parameters=params,results=rows))
 out=dict(schema='nso.CSM_iteration_budget_check.v1',master='http://127.0.0.1:11404',fixed_state='actual ref096/pose096,actual prev098/pose098,target099 exact original scan/first_guess',recursive_state='actual ref096/pose096 seed then own ICP outputs only through104, native default keyframe policy; no clipping even when guard would stop',results=results,input_npz_sha256=hashlib.sha256((SOURCE/'failure_scan_ring.npz').read_bytes()).hexdigest(),live_modified=False,accuracy_limit='Residual/convergence and consistency are not independent pose accuracy; no external truth.')
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));rospy.signal_shutdown('done')
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:
  s=socket.socket();assert s.connect_ex(('127.0.0.1',11404))!=0;s.close()
  with (OUT/'master.log').open('w') as log:
   m=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p','11404'],stdout=log,stderr=log)
   try:time.sleep(2);subprocess.run([sys.executable,__file__,'worker'],check=True,timeout=50,stdout=(OUT/'console.json').open('w'))
   finally:m.send_signal(signal.SIGINT);m.wait(timeout=10)
