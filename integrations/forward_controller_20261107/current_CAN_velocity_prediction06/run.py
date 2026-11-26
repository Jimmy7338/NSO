import os
os.environ['ROS_MASTER_URI']='http://127.0.0.1:11405';os.environ['ROS_HOSTNAME']='127.0.0.1';os.environ['ROS_IP']='127.0.0.1'
import sys,json,math,time,socket,subprocess,signal
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parent;OUT=Path(os.environ['PREDICTION_OUTPUT_DIR']).resolve();ITER=Path(os.environ['ITERATION_OUTPUT_DIR']).resolve()
if OUT==P or P in OUT.parents:raise ValueError('output must be separate')
OUT.mkdir(parents=True,exist_ok=True)

def worker():
 import rospy
 rospy.init_node('observed_constant_velocity_fixture_setup',disable_signals=True)
 observed=json.loads((P/'observed_CAN_provenance.json').read_text());q=observed['can_pair'];base=dict(json.loads((D/'instrumented_csm_candidate/state_fixture_results.json').read_text())['parameters'],kf_dist_linear=.1,kf_dist_angular=math.pi/18,max_iterations=10)
 results=[]
 for enabled in [False,True]:
  params=dict(base,use_vel=enabled);rospy.set_param('/fixture/matcher',params);label='observed_current_velocity' if enabled else 'identity_increment'
  cmd=[str(OUT/'state_fixture'),str(ITER/'fixed_jobs.txt'),str(q['vx']),str(q['w'])];r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);(OUT/(label+'.stdout.log')).write_text(r.stdout);(OUT/(label+'.stderr.log')).write_text(r.stderr);assert r.returncode==0
  data=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')];assert len(data)==1;results.append(dict(profile=label,parameters=params,result=data[0]))
 a,b=[x['result'] for x in results]
 for key in ['reference_stamp_ns','previous_stamp_ns','target_stamp_ns','actual_keyframe_pose','actual_previous_pose']:assert a[key]==b[key]
 assert a['actual_target_xy_error_m']<1e-12
 dt=(a['target_stamp_ns']-a['previous_stamp_ns'])/1e9
 for x,y in zip(b['motion_increment'],[dt*q['vx'],0,dt*q['w']]):assert abs(x-y)<1e-14
 out=dict(schema='nso.observed_current_constant_velocity_prediction06.v1',source_provenance=observed,prediction_model='Exact installed getPrediction use_vel=true: dt*latest_vx,dt*latest_vy,dt*latest_w; current observed CAN sample held constant across this one step by explicit hypothesis. Not ZOH sequence integration, not actual measured interval motion.',dt_s=dt,expected_increment=[dt*q['vx'],0,dt*q['w']],expected_angular_increment_degrees=math.degrees(dt*q['w']),results=results,source_sample_age_at_scan_ns=q['scan_minus_can_ns'],source_sample_age_at_original_selection_ns=q['current_minus_can_ns'],source_receipt_age_at_original_selection_ns=q['receipt_age_ns'],frozen_ref_prev_target_verified=True,all_other_parameters_equal=True,live_modified=False)
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));rospy.signal_shutdown('done')
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
 else:
  s=socket.socket();assert s.connect_ex(('127.0.0.1',11405))!=0;s.close()
  with (OUT/'master.log').open('w') as log:
   m=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p','11405'],stdout=log,stderr=log)
   try:time.sleep(2);subprocess.run([sys.executable,__file__,'worker'],check=True,timeout=35,stdout=(OUT/'console.json').open('w'))
   finally:m.send_signal(signal.SIGINT);m.wait(timeout=10)
