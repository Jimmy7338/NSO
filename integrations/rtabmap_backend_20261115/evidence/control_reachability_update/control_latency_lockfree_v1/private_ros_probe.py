#!/usr/bin/python3
"""Real receive-only sources + unarmed actual adapter loop on private topics.
No arm service/native driver/mission/navigation command publisher is created.
"""
import ast, hashlib, importlib.util, json, math, os, statistics, sys, threading, time, types
from pathlib import Path
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
sys.path.insert(0,str(W/'runtime'))
spec=importlib.util.spec_from_file_location('actual_adapter',W/'runtime/visual_velocity_adapter.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
import rospy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header,String
rospy.init_node('nso_latency_private_probe',anonymous=True)
base=Path(__file__).resolve().parent
cfg=a.configuration(json.loads(Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_rtabmap_integrated_20261115_01/adapter-config.json').read_text()))
gate=a.Gate(cfg.get('duration_s',600.),**{key:cfg[key] for key in ['bridge_publisher','proposal_publisher','odom_frame','base_frame','depth_frame','navigation_source','navigation_allow_reverse','ordinary_reverse','mission_mode','nso_correction_reverse','correction_supervisor','duration_profile','maximum_linear_m_s','maximum_feedback_linear_m_s','sensor_loss_profile','odometry_jump_profile']})
gate.session_id='private_unarmed_probe'
lock=threading.RLock()
class NullJournal:
 error=None
 def __init__(self):self.count=0
 def add(self,row):self.count+=1
journal=NullJournal();feedback=a.FeedbackReader(gate,lock,journal,cfg['can_interface'])
private='/nso_latency_private_probe'
publisher=rospy.Publisher(private+'/zero',TwistStamped,queue_size=1,tcp_nodelay=True)
state_pub=rospy.Publisher(private+'/state',String,queue_size=1,tcp_nodelay=True)
zero_received=[]
loopback=rospy.Subscriber(private+'/zero',TwistStamped,lambda m:zero_received.append((m.twist.linear.x,m.twist.angular.z)),queue_size=1)
state_loopback=rospy.Subscriber(private+'/state',String,lambda m:None,queue_size=1)
start=time.monotonic();deadline=start+120.
proxy=types.SimpleNamespace(is_shutdown=lambda:rospy.is_shutdown() or time.monotonic()>=deadline,Time=rospy.Time)
ctx=dict(a.__dict__)
ctx.update(rospy=proxy,gate=gate,lock=lock,journal=journal,cfg=cfg,publisher=publisher,
 session=gate.session_id,pending_deadline=start+3600.,last_stamp=0,sequence=0,next_state_monotonic=0.,
 control_timing=a.ControlTimingRing(capacity=10000),TwistStamped=TwistStamped,
 publish_state=lambda state:state_pub.publish(String(data=json.dumps(state,allow_nan=False))))
tree=ast.parse((W/'runtime/visual_velocity_adapter.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
callbacks=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in ['odom','scan','health','laser_health']]
mod=ast.Module(body=callbacks,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,'actual_callbacks','exec'),ctx)
subscribers=[rospy.Subscriber(cfg['odom_topic'],Odometry,ctx['odom'],queue_size=1,tcp_nodelay=True),
 rospy.Subscriber(cfg['scan_topic'],LaserScan,ctx['scan'],queue_size=1,tcp_nodelay=True),
 rospy.Subscriber(cfg['depth_health_topic'],Header,lambda m:ctx['health']('depth_health',m),queue_size=1,tcp_nodelay=True),
 rospy.Subscriber(cfg['odom_health_topic'],Header,lambda m:ctx['health']('odom_health',m),queue_size=1,tcp_nodelay=True),
 rospy.Subscriber(cfg['laser_health_topic'],String,ctx['laser_health'],queue_size=1,tcp_nodelay=True)]
loop=next(n for n in ast.walk(main) if isinstance(n,ast.While) and isinstance(n.test,ast.UnaryOp) and isinstance(n.test.operand,ast.Call) and isinstance(n.test.operand.func,ast.Attribute) and n.test.operand.func.attr=='is_shutdown')
mod=ast.Module(body=[loop],type_ignores=[]);ast.fix_missing_locations(mod)
# Exercise the documented COW-list read while its management lock is held;
# does not block publish() and never touches any robot topic.
held=[]
def lock_contention():
 for i in range(6):
  time.sleep(5.)
  t=time.monotonic()
  with publisher.impl.c_lock:time.sleep(.100)
  held.append(time.monotonic()-t)
contention=threading.Thread(target=lock_contention,daemon=True);contention.start()
try:exec(compile(mod,'actual_main_loop','exec'),ctx)
finally:feedback.close()
rows=list(ctx['control_timing'].rows)
def stats(values):
 values=sorted(values)
 return {'n':len(values),'min_ms':values[0]*1e3,'p95_ms':values[int(.95*(len(values)-1))]*1e3,'p99_ms':values[int(.99*(len(values)-1))]*1e3,'max_ms':values[-1]*1e3} if values else None
out=dict(schema='private_adapter_real_ROS_timing/1',started_monotonic=start,elapsed_s=time.monotonic()-start,
 adapter_sha256=hashlib.sha256((W/'runtime/visual_velocity_adapter.py').read_bytes()).hexdigest(),affinity=sorted(os.sched_getaffinity(0)),
 stages={k:stats([r[k]*1e-9 for r in rows if k in r]) for k in sorted({k for r in rows for k in r if k.endswith('_ns') and k!='cycle_enter_ns'})},
 tick_intervals=stats([b['gate_now_monotonic']-a['gate_now_monotonic'] for a,b in zip(rows,rows[1:])]),
 cycles=len(rows),fault=gate.fault,fault_evidence=gate.fault_evidence,ever_armed=gate.ever_armed,
 command_nonzero_count=sum(v!=0 or w!=0 for v,w in zero_received),private_zero_received=len(zero_received),
 actual_receive_event_count=journal.count,held_management_lock_s=held,
 scope='Actual adapter loop/callback code with real sensor+CAN inputs, private zero/state publishers and loopback; unarmed, no native or actuator commands; not driving deadline guarantee')
(base/'private_ros_probe_result.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
if gate.fault:ctx['control_timing'].save_fault_once(base/'private_ros_probe_fault.json',gate.fault,gate.fault_evidence)
assert not gate.ever_armed and out['command_nonzero_count']==0
