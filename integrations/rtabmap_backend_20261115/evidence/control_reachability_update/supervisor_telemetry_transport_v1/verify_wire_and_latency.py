#!/usr/bin/python3
"""Offline ROS bytes + mirror-guard equivalence; no node/master/device use."""
import hashlib,importlib.util,io,json,os,sys,time
from pathlib import Path
import numpy as np
from genpy import Time
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg
b=Path(__file__).resolve().parent
sys.path.insert(0,'/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime')
mods=[]
for name,p in [('old',b/'mission.before.py'),('new',Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime/mission.py'))]:
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);mods.append(m)
numpy_grid=numpy_msg(OccupancyGrid)
rng=np.random.RandomState(51215);wire_cases=0
for i in range(40):
 h,w=int(rng.randint(1,140)),int(rng.randint(1,180));a=rng.randint(-128,128,(h,w)).astype(np.int8)
 grid=OccupancyGrid();grid.header.seq=i;grid.header.stamp=Time(12345,i);grid.header.frame_id='nso_pair_map'
 grid.info.height,grid.info.width,grid.info.resolution=h,w,.05;grid.info.origin.orientation.w=1.;grid.info.origin.position.x=-7.25;grid.info.origin.position.y=3.75;grid.data=a.ravel().tolist()
 stream=io.BytesIO();grid.serialize(stream);raw=stream.getvalue();old=OccupancyGrid().deserialize(raw);new=numpy_grid().deserialize(raw)
 assert old.header==new.header and old.info==new.info
 assert np.array_equal(np.asarray(old.data,dtype=np.int8),new.data)
 assert new.data.dtype==np.int8 and new.data.shape==(h*w,)
 assert old._type==new._type and old._md5sum==new._md5sum
 for m in [old,new]:
  output=io.BytesIO();m.serialize(output);assert output.getvalue()==raw
 wire_cases+=1
row=dict(session_id='s',sampled_monotonic=1.,sampled_unix_s=1001.,fault=None,ever_armed=True,can_distance_m=.3,can_absolute_yaw_rad=.2,can_zero_since_monotonic=.1,feedback_accounting_complete=True,hardware_reason=None,can_status=dict(source=1.,receipt=1.,state=0,mode=1,error=0),can_velocity=dict(source=1.,receipt=1.,v=0.,w=0.))
guard_cases=0
for delay in [0.,.1,.4,.5]:
 for status_src in [.7,1.,1.2]:
  for velocity_src in [.7,1.,1.2]:
   rr=dict(row,can_status=dict(row['can_status'],source=status_src),can_velocity=dict(row['can_velocity'],source=velocity_src))
   models=[m.ActuatorTelemetry('s') for m in mods]
   for m in models:m.update(rr,'/nso_radius_guard',1.+delay)
   for now in [1.+delay-1e-9,1.+delay,1.5,1.500001,1.9,2.]:
    assert models[0].hardware_reason(now)==models[1].hardware_reason(now)
    guard_cases+=1
# Serialize one realistic896²map once, then alternate the actual ROS decoders.
grid=OccupancyGrid();grid.header.frame_id='nso_pair_map';grid.info.width=896;grid.info.height=896;grid.info.resolution=.05;grid.info.origin.orientation.w=1.;grid.data=rng.choice(np.array([-1,0,100],dtype=np.int8),896*896).tolist();stream=io.BytesIO();grid.serialize(stream);raw=stream.getvalue()
timings=[[],[]]
for i in range(40):
 for idx in ([0,1] if i%2==0 else [1,0]):
  cls=[OccupancyGrid,numpy_grid][idx];start=time.perf_counter_ns();msg=cls().deserialize(raw);timings[idx].append((time.perf_counter_ns()-start)/1e6)
def stats(v):
 v=sorted(v);return dict(n=len(v),median_ms=float(np.median(v)),p95_ms=v[int(.95*(len(v)-1))],max_ms=v[-1],mean_ms=float(np.mean(v)))
out=dict(schema='supervisor_telemetry_transport_verification/1',ROS_wire_cases=wire_cases,unchanged_mirror_guard_cases=guard_cases,realistic_packet=dict(width=896,height=896,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),deserialization_benchmark=dict(original_tuple=stats(timings[0]),numpy_msg=stats(timings[1])),affinity=sorted(os.sched_getaffinity(0)),source_bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime/mission.py'),Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/runtime/visual_velocity_adapter.py')]},ROS_initialized=False,robot_actuation=False,limits='Measured local syntheticwire decoder cost; actual04telemetry transportdelay and originalCANstatusgap were not recorded, so no retrospectivecause proof; all originalmirrored-sourceTTLguards retained')
(b/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
