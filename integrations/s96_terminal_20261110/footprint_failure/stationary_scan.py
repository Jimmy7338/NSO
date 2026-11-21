import rospy,time,json,math
from sensor_msgs.msg import LaserScan
from pathlib import Path
O=Path(__file__).parent;rows=[]
def cb(m):
 rows.append(dict(stamp_ns=m.header.stamp.to_nsec(),seq=m.header.seq,frame=m.header.frame_id,caller=m._connection_header.get('callerid'),receive_monotonic=time.monotonic(),angle_min=m.angle_min,angle_increment=m.angle_increment,range_min=m.range_min,range_max=m.range_max,ranges=[float(x) if math.isfinite(x) else str(x) for x in m.ranges],intensities=list(m.intensities)))
rospy.init_node('nso_s96_readonly_scan_persistence',anonymous=True,disable_signals=True);s=rospy.Subscriber('/scan',LaserScan,cb,queue_size=50);start=time.monotonic()
while time.monotonic()-start<3:time.sleep(.02)
s.unregister();(O/'stationary_scans_3s.json').write_text(json.dumps(rows));stats=[]
for i in range(576,595):
 vals=[r['ranges'][i] for r in rows if len(r['ranges'])>i];finite=[v for v in vals if isinstance(v,float)];stats.append(dict(beam=i,total=len(vals),finite=len(finite),short_under_0p9m=sum(0<v<.9 for v in finite),ranges=finite))
r=dict(scope='Later stationary read-only source scan; cannot replace failed-frame fact or establish object identity',duration_s=time.monotonic()-start,count=len(rows),callers=sorted(set(x['caller'] for x in rows)),stamp_first=rows[0]['stamp_ns'] if rows else None,stamp_last=rows[-1]['stamp_ns'] if rows else None,beams=stats);(O/'stationary_scan_persistence.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
