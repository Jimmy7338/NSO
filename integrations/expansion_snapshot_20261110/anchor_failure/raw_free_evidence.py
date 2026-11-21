"""Receive-only synchronized raw-depth audit. No services, settings or motion."""
import json,time,threading
from pathlib import Path
import numpy as np,rospy,tf2_ros,message_filters
from sensor_msgs.msg import Image,CameraInfo
from tf.transformations import quaternion_matrix
O=Path(__file__).parent;P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261110/expand_history_S96_20261110_01');T=np.array(json.loads((P/'joint_configuration.json').read_text())['configuration']['planning_from_odom']);points=np.array(json.loads((O/'old_points.json').read_text()));old=np.load(P/'joint_inputs/joint-1794292130668983305.npz');oldT=T@old['world_from_camera']
rospy.init_node('nso_raw_free_evidence_receiveonly',disable_signals=True,disable_rosout=True);buf=tf2_ros.Buffer(cache_time=rospy.Duration(30));listener=tf2_ros.TransformListener(buf);pending=[];lock=threading.Lock()
def receive(*a):
 with lock:
  if len(pending)<15:pending.append(a)
def decode(m):
 assert m.encoding=='32FC1'
 return np.ndarray((m.height,m.width),dtype=np.dtype('>f4' if m.is_bigendian else '<f4'),buffer=m.data,strides=(m.step,4))
subs=[message_filters.Subscriber('/zed2i/zed_node/'+s,t) for s,t in [('depth/depth_registered',Image),('confidence/confidence_map',Image),('depth/camera_info',CameraInfo)]];sync=message_filters.TimeSynchronizer(subs,10);sync.registerCallback(receive);rows=[];end=time.monotonic()+25
try:
 while len(rows)<10 and time.monotonic()<end:
  with lock:a=pending.pop(0) if pending else None
  if a is None:time.sleep(.03);continue
  d,c,k=a
  try:t=buf.lookup_transform('odom',d.header.frame_id,d.header.stamp,rospy.Duration(1))
  except Exception:continue
  q=t.transform.rotation;A=quaternion_matrix([q.x,q.y,q.z,q.w]);p=t.transform.translation;A[:3,3]=[p.x,p.y,p.z];A=T@A;pc=(points-A[:3,3])@A[:3,:3];K=np.array(k.K).reshape(3,3);proj=pc@K.T;uv=np.rint(proj[:,:2]/proj[:,2,None]).astype(int);raw=decode(d);conf=decode(c);items=[];counts={}
  for point,xyz,(u,v) in zip(points,pc,uv):
   valid=xyz[2]>0 and 0<=u<d.width and 0<=v<d.height;value=float(raw[v,u]) if valid else float('nan');cv=float(conf[v,u]) if valid else float('nan');label='outside' if not valid else 'invalid' if not np.isfinite(value) or value<=0 else 'farther_free' if value>xyz[2]+.12 else 'occluded' if value<xyz[2]-.12 else 'within';counts[label]=counts.get(label,0)+1;items.append(dict(uv=[int(u),int(v)],old_z=float(xyz[2]),raw_depth=str(value) if not np.isfinite(value) else value,confidence=str(cv) if not np.isfinite(cv) else cv,classification=label,raw_gt4=bool(np.isfinite(value) and value>4)))
  rows.append(dict(stamp_ns=d.header.stamp.to_nsec(),frame=d.header.frame_id,publisher=d._connection_header.get('callerid'),K=K.tolist(),planning_from_camera=A.tolist(),translation_delta_m=float(np.linalg.norm(A[:3,3]-oldT[:3,3])),counts=counts,raw_gt4_count=sum(x['raw_gt4'] for x in items),samples=items))
finally:
 (O/'raw_free_evidence.json').write_text(json.dumps(dict(frames=rows,expected_frames=10,source='synchronized receive-only raw32FC1 depth/confidence and exact historical TF'),indent=2));[s.unregister() for s in subs];rospy.signal_shutdown('audit complete')
print(json.dumps([dict(stamp_ns=r['stamp_ns'],counts=r['counts'],raw_gt4=r['raw_gt4_count'],translation_delta_m=r['translation_delta_m']) for r in rows]))
