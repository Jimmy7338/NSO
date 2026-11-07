#!/usr/bin/python3
"""Isolated real CSM replay. Never connects outside localhost:11398."""
import os
os.environ['ROS_MASTER_URI']='http://127.0.0.1:11398'
os.environ['ROS_HOSTNAME']='127.0.0.1'
os.environ['ROS_IP']='127.0.0.1'
import sys,json,time,subprocess,signal,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAP=ROOT.parent/'probes/control_can_offset_06'
OUT=Path(os.environ['REPLAY_OUTPUT_DIR']).resolve()
OUT.mkdir(parents=True,exist_ok=True)
MATCHER=os.environ.get('CSM_MATCHER','/opt/ros/noetic/lib/laser_scan_matcher/laser_scan_matcher_node')

def worker(label,disable,pace):
 import rospy,numpy as np,tf2_ros
 from sensor_msgs.msg import LaserScan
 from geometry_msgs.msg import PoseStamped,TransformStamped
 rospy.init_node('isolated_replay',disable_signals=True)
 params_before=rospy.get_param('/nso_csm/matcher',{})
 receipt=json.loads((ROOT/'inputs/tf_receipt.json').read_text())
 receipt['CSM']=json.loads((CAP/'configuration.json').read_text())['matcher_parameters']['matcher']
 for k,v in receipt['CSM'].items(): rospy.set_param('/nso_csm/matcher/'+k,v)
 if disable: rospy.set_param('/nso_csm/matcher/use_tf',False)
 elif rospy.has_param('/nso_csm/matcher/use_tf'): rospy.delete_param('/nso_csm/matcher/use_tf')
 static=tf2_ros.StaticTransformBroadcaster(); tr=TransformStamped();tr.header.frame_id='base_link';tr.child_frame_id='laser_link'
 t=receipt['transforms']['base_link<-laser_link']
 for k,v in zip('xyz',t['translation']): setattr(tr.transform.translation,k,v)
 for k,v in zip('xyzw',t['quaternion']): setattr(tr.transform.rotation,k,v)
 static.sendTransform(tr)
 params_before_launch=rospy.get_param('/nso_csm/matcher',{})
 assert 'kf_dist_linear' not in params_before and 'kf_dist_angular' not in params_before
 assert 'kf_dist_linear' not in params_before_launch and 'kf_dist_angular' not in params_before_launch
 (OUT/(label+'_parameter_audit.json')).write_text(json.dumps(dict(master=rospy.get_param('/roslaunch/uris',{}),params_before=params_before,params_before_launch=params_before_launch,static_transform=dict(translation=t['translation'],quaternion=t['quaternion'])),indent=2))
 outputs=[]; sent={}
 def callback(m):
  q=m.pose.orientation;p=m.pose.position
  outputs.append(dict(stamp_ns=m.header.stamp.to_nsec(),seq=m.header.seq,x=p.x,y=p.y,yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z),receipt=time.monotonic()))
 sub=rospy.Subscriber('/nso_csm/pose_stamped',PoseStamped,callback,queue_size=200)
 pub=rospy.Publisher('/scan',LaserScan,queue_size=200)
 log=open(OUT/(label+'.log'),'w')
 matcher=subprocess.Popen([MATCHER,'scan:=/scan','__ns:=/nso_csm','__name:=matcher'],stdout=log,stderr=subprocess.STDOUT)
 bridge=None
 try:
  end=time.monotonic()+8
  while pub.get_num_connections()<1 and time.monotonic()<end: time.sleep(.05)
  if not pub.get_num_connections(): raise RuntimeError('matcher failed to subscribe')
  time.sleep(1)
  z=np.load(CAP/'failure_scan_ring.npz'); keys=sorted(k for k in z.files if k.startswith('scan_'))
  selected=os.environ.get('REPLAY_SCAN_INDICES')
  if selected:
   indices=[int(x) for x in selected.split(',')]
   keys=[keys[i] for i in indices]
  msgs=[]
  for key in keys:
   m=LaserScan();m.deserialize(z[key].tobytes());msgs.append(m)
  origin=msgs[0].header.stamp.to_nsec(); shifted=rospy.Time.now().to_nsec();start=time.monotonic();mapping=[]
  for i,m in enumerate(msgs):
   source=m.header.stamp.to_nsec();dt=(source-origin)*1e-9
   deadline=start+dt*pace
   while time.monotonic()<deadline:time.sleep(min(.01,deadline-time.monotonic()))
   stamp=shifted+source-origin;m.header.stamp=rospy.Time(stamp//10**9,stamp%10**9)
   sent[stamp]=time.monotonic();mapping.append(dict(index=i,source_stamp_ns=source,replay_stamp_ns=stamp,source_seq=m.header.seq,publish_monotonic=sent[stamp]));pub.publish(m)
  last_n=-1;stable=time.monotonic(); deadline=time.monotonic()+8
  while time.monotonic()<deadline:
   if len(outputs)!=last_n:last_n=len(outputs);stable=time.monotonic()
   if time.monotonic()-stable>1.5:break
   time.sleep(.05)
  result=dict(label=label,use_tf_explicit_false=disable,pace=pace,input_count=len(msgs),output_count=len(outputs),elapsed_s=time.monotonic()-start,mapping=mapping,outputs=outputs,source_sha256=hashlib.sha256((CAP/'failure_scan_ring.npz').read_bytes()).hexdigest())
  (OUT/(label+'.json')).write_text(json.dumps(result,indent=2))
 finally:
  for proc in [matcher]:proc.send_signal(signal.SIGINT)
  for proc in [matcher]:
   try:proc.wait(timeout=3)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close();rospy.signal_shutdown('done')

if __name__=='__main__':
 if len(sys.argv)!=5 or sys.argv[1]!='worker':
  raise SystemExit('Use repeat_clean_setup.py with an explicit --output directory.')
 worker(sys.argv[2],sys.argv[3]=='false',float(sys.argv[4]))
