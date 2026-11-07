#!/usr/bin/python3
"""Isolated real CSM replay. Never connects outside localhost:11398."""
import os
os.environ['ROS_MASTER_URI']='http://127.0.0.1:11398'
os.environ['ROS_HOSTNAME']='127.0.0.1'
import sys,json,time,subprocess,signal,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAP=ROOT/'inputs'
BRIDGE=str(ROOT.parents[1]/'docs/research/onboard_20261104/deployment/hybrid/runtime/csm_odometry_bridge.py')
import roslib.packages
MATCHER=roslib.packages.find_node('laser_scan_matcher','laser_scan_matcher_node')[0]

def worker(label,disable,pace):
 import rospy,numpy as np,tf2_ros
 from sensor_msgs.msg import LaserScan
 from geometry_msgs.msg import PoseStamped,TransformStamped
 rospy.init_node('isolated_replay',disable_signals=True)
 receipt=json.loads((CAP/'receipt.json').read_text())
 for k,v in receipt['CSM'].items(): rospy.set_param('/nso_csm/matcher/'+k,v)
 if disable: rospy.set_param('/nso_csm/matcher/use_tf',False)
 elif rospy.has_param('/nso_csm/matcher/use_tf'): rospy.delete_param('/nso_csm/matcher/use_tf')
 static=tf2_ros.StaticTransformBroadcaster(); tr=TransformStamped();tr.header.frame_id='base_link';tr.child_frame_id='laser_link'
 t=receipt['transforms']['base_link<-laser_link']
 for k,v in zip('xyz',t['translation']): setattr(tr.transform.translation,k,v)
 for k,v in zip('xyzw',t['quaternion']): setattr(tr.transform.rotation,k,v)
 static.sendTransform(tr)
 outputs=[]; sent={}
 def callback(m):
  q=m.pose.orientation;p=m.pose.position
  outputs.append(dict(stamp_ns=m.header.stamp.to_nsec(),seq=m.header.seq,x=p.x,y=p.y,yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z),receipt=time.monotonic()))
 sub=rospy.Subscriber('/nso_csm/pose_stamped',PoseStamped,callback,queue_size=200)
 pub=rospy.Publisher('/scan',LaserScan,queue_size=200)
 log=open(ROOT/(label+'.log'),'w')
 matcher=subprocess.Popen([MATCHER,'scan:=/scan','__ns:=/nso_csm','__name:=matcher'],stdout=log,stderr=subprocess.STDOUT)
 bridge=subprocess.Popen(['/usr/bin/python3',BRIDGE],stdout=log,stderr=subprocess.STDOUT)
 try:
  end=time.monotonic()+8
  while pub.get_num_connections()<1 and time.monotonic()<end: time.sleep(.05)
  if not pub.get_num_connections(): raise RuntimeError('matcher failed to subscribe')
  time.sleep(1)
  z=np.load(CAP/'scans.npz'); keys=sorted(k for k in z.files if k.startswith('scan_'))
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
  result=dict(label=label,use_tf_explicit_false=disable,pace=pace,input_count=len(msgs),output_count=len(outputs),elapsed_s=time.monotonic()-start,mapping=mapping,outputs=outputs,source_sha256=hashlib.sha256((CAP/'scans.npz').read_bytes()).hexdigest())
  (ROOT/(label+'.json')).write_text(json.dumps(result,indent=2))
 finally:
  for proc in [matcher,bridge]:proc.send_signal(signal.SIGINT)
  for proc in [matcher,bridge]:
   try:proc.wait(timeout=3)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close();rospy.signal_shutdown('done')

def summarize():
 import numpy as np
 records={}
 for path in [ROOT/(prefix+'_'+variant+'.json') for prefix in ['ten_hz','paced'] for variant in ['default','false']]:
  if path.name=='summary.json':continue
  r=json.loads(path.read_text());o=r['outputs'];a=np.array([[v['x'],v['y'],v['yaw']] for v in o]);mapping={v['replay_stamp_ns']:v for v in r['mapping']}
  rsum=dict(input_count=r['input_count'],output_count=len(o),elapsed_s=r['elapsed_s'],warning_count=(ROOT/(r['label']+'.log')).read_text().count('Could not get base to fixed frame transform'),xy_span_m=np.ptp(a[:,:2],axis=0).tolist(),xy_std_m=np.std(a[:,:2],axis=0).tolist(),max_xy_step_m=float(np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1).max()),yaw_span_rad=float(np.ptp(a[:,2])),latency_median_s=float(np.median([v['receipt']-mapping[v['stamp_ns']]['publish_monotonic'] for v in o])))
  records[r['label']]=rsum
 for prefix in ['ten_hz','paced']:
  paths=[ROOT/(prefix+'_'+v+'.json') for v in ['default','false']]
  if not all(p.exists() for p in paths):continue
  rs=[json.loads(p.read_text()) for p in paths]; tables=[]
  for r in rs:
   mp={v['replay_stamp_ns']:v['index'] for v in r['mapping']};tables.append({mp[v['stamp_ns']]:v for v in r['outputs']})
  common=sorted(set(tables[0])&set(tables[1]));diff=np.array([[tables[0][i][k]-tables[1][i][k] for k in ['x','y','yaw']] for i in common])
  records[prefix+'_comparison']=dict(common_source_indices=common,max_xy_difference_m=float(np.linalg.norm(diff[:,:2],axis=1).max()),max_yaw_difference_rad=float(np.abs(diff[:,2]).max()))
 (ROOT/'summary.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3]=='false',float(sys.argv[4]))
 else:
  import socket
  s=socket.socket();assert s.connect_ex(('127.0.0.1',11398))!=0,'11398 already occupied';s.close()
  masterlog=open(ROOT/'master.log','w');master=subprocess.Popen(['roscore','-p','11398'],stdout=masterlog,stderr=subprocess.STDOUT)
  try:
   time.sleep(3)
   for prefix,pace in [('ten_hz',1),('paced',2)]:
    for variant in ['default','false']:subprocess.run([sys.executable,__file__,'worker',prefix+'_'+variant,variant,str(pace)],check=True)
   summarize()
  finally:master.send_signal(signal.SIGINT);master.wait(timeout=10);masterlog.close()
