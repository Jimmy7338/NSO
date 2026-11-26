#!/usr/bin/env python3
import time,json,collections,threading
from pathlib import Path
import rospy,tf2_ros
from nav_msgs.msg import Odometry
rospy.init_node('nso_tf_actual_stamp_wait_probe',anonymous=True,disable_signals=True)
b=tf2_ros.Buffer();listener=tf2_ros.TransformListener(b)
time.sleep(1.)
rows=[];lock=threading.Lock()
def sample(m):
 if not lock.acquire(False):return
 try:
  if len(rows)>=60:return
  row=dict(source_stamp_ns=m.header.stamp.to_nsec(),source_age_s=rospy.Time.now().to_sec()-m.header.stamp.to_sec())
  for name,seconds in [('immediate',0.),('wait_actual_stamp_0p2s',.2)]:
   started=time.monotonic()
   try:
    t=b.lookup_transform('nso_pair_map','odom',m.header.stamp,rospy.Duration(seconds))
    row[name]=dict(ok=True,lookup_time_s=time.monotonic()-started,returned_stamp_ns=t.header.stamp.to_nsec())
   except Exception as e:row[name]=dict(ok=False,lookup_time_s=time.monotonic()-started,error=str(e))
  rows.append(row)
 finally:lock.release()
s=rospy.Subscriber('/nso_csm/odom',Odometry,sample,queue_size=1)
deadline=time.monotonic()+15
while len(rows)<60 and time.monotonic()<deadline:time.sleep(.05)
s.unregister()
summary={name:dict(success=sum(r[name]['ok'] for r in rows),failure=sum(not r[name]['ok'] for r in rows),max_wait_s=max((r[name]['lookup_time_s'] for r in rows),default=0.)) for name in ['immediate','wait_actual_stamp_0p2s']}
result=dict(schema='actual_TF_same_stamp_wait_probe/1',sample_count=len(rows),summary=summary,rows=rows,actuation=False,TF_publishing=False,source='actual /nso_csm/odom and actual /tf only; no timestamp alteration',scope='software transport readiness comparison; not navigation or algorithm efficacy evidence')
p=Path(__file__).resolve().parent/'actual_tf_wait_probe.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(sample_count=len(rows),summary=summary),indent=2))
