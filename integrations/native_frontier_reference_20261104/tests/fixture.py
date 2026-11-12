#!/usr/bin/env python3
"""Synthetic ROS graph only. Publishes a map and TF; fake move_base never actuates."""
import json, math, os, threading, time
import rospy, actionlib, tf2_ros
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
from geometry_msgs.msg import TransformStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseResult

rospy.init_node('fixture', anonymous=False)
log_path = os.environ['FIXTURE_LOG']
lock=threading.Lock(); goals=[]
def emit(kind, **extra):
 with lock:
  with open(log_path,'a') as f: f.write(json.dumps(dict(event=kind,wall_time=time.time(),**extra))+'\n')
pub=rospy.Publisher('/fixture/costmap',OccupancyGrid,queue_size=1,latch=True)
update=rospy.Publisher('/fixture/costmap_updates',OccupancyGridUpdate,queue_size=1)
tfpub=tf2_ros.TransformBroadcaster()
def action(goal):
 idx=len(goals)+1; goals.append(goal);p=goal.target_pose.pose.position;emit('goal',index=idx,x=p.x,y=p.y,frame=goal.target_pose.header.frame_id)
 if idx <= 2:
  rospy.sleep(.7)
  if server.is_preempt_requested(): server.set_preempted();emit('preempted',index=idx);return
  if idx==1:
   for y in range(20,60):
    for x in range(15,20):data[y*80+x]=0
   grid.data=data;grid.header.stamp=rospy.Time.now();pub.publish(grid);rospy.sleep(.4)
   server.set_succeeded(MoveBaseResult());emit('succeeded',index=idx)
  else: server.set_aborted(MoveBaseResult(), 'synthetic abort');emit('aborted',index=idx)
 else:
  while not rospy.is_shutdown() and not server.is_preempt_requested(): rospy.sleep(.05)
  if server.is_preempt_requested():server.set_preempted();emit('preempted',index=idx)
server=actionlib.SimpleActionServer('/fixture/move_base',MoveBaseAction,execute_cb=action,auto_start=False);server.start()
grid=OccupancyGrid();grid.header.frame_id='fixture_map';grid.info.resolution=.1;grid.info.width=80;grid.info.height=80;grid.info.origin.orientation.w=1.;grid.info.origin.position.x=-4.;grid.info.origin.position.y=-4.
# Circular unknown boundary around a known free square. Robot is at its center.
data=[-1]*6400
for y in range(20,60):
 for x in range(20,60):data[y*80+x]=0
grid.data=data
emit('ready')
rate=rospy.Rate(10);n=0
while not rospy.is_shutdown():
 now=rospy.Time.now();grid.header.stamp=now;pub.publish(grid)
 tr=TransformStamped();tr.header.stamp=now;tr.header.frame_id='fixture_map';tr.child_frame_id='fixture_base';tr.transform.rotation.w=1.;tfpub.sendTransform(tr)
 if n%10==0:
  u=OccupancyGridUpdate();u.header.frame_id='fixture_map';u.header.stamp=now;u.x=39;u.y=39;u.width=1;u.height=1;u.data=[0];update.publish(u)
 n+=1;rate.sleep()
