#!/usr/bin/env python3
"""Read existing canonical neighbors; optional bounded passive5s scan summary."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
P=Path('/home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/publication/S');D=Path(__file__).resolve().parent

def historical():
    m=json.loads((P/'frames_manifest.json').read_text())['frames'];rows=[];target=None
    for aid in [31,32]:
        p=P/m[aid]['file']
        with np.load(p,allow_pickle=False) as z:
            a=float(z['angle_min']);inc=float(z['angle_increment']);target=a+396*inc
            rows.append(dict(action_id=aid,file=str(p),file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                scan_timestamp_s=float(z['scan_timestamp_s']),rgbd_timestamp_s=float(z['rgbd_timestamp_s']),
                rgbd_minus_scan_s=float(z['rgbd_timestamp_s'])-float(z['scan_timestamp_s']),
                raw_range_max_m=float(z['range_max']),angle_min_rad=a,angle_increment_rad=inc,
                base_odom_pose=z['base_odom_pose'].tolist(),
                beams=[dict(index=i,angle_rad=a+i*inc,raw_range_m=float(z['scan_ranges_raw'][i]),
                    supplied_range_m=float(z['ranges'][i]),raw_intensity=float(z['scan_intensities_raw'][i])) for i in range(393,400)]))
    result=dict(schema='nso.existing_scan_endpoint_neighbors.v1',target_index=396,target_angle_rad=target,records=rows,
        attribution='Actualaction32 beam396 inserts cell[195199] under exact native mapper; no filtering or correction performed.',
        limits=['Adjacent beams and prior scan are context, not independent truth or proof of outlier/self-hit/object identity.','Two scans have different measured base yaw; same ray index does not imply the same world direction.'])
    (D/'scan_endpoint_neighbors.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return target

def passive(target):
    import rospy
    from sensor_msgs.msg import LaserScan
    rows=[];begin=time.monotonic();startwall=time.time()
    def callback(msg):
        now=time.monotonic();r=np.asarray(msg.ranges,float);theta=msg.angle_min+np.arange(len(r))*msg.angle_increment
        valid=np.isfinite(r)&(r>0);near=valid&(r>=.4)&(r<=.6)
        delta=(theta-target+math.pi)%(2*math.pi)-math.pi;window=np.abs(delta)<=.02
        selected=np.flatnonzero(near&window);allids=np.flatnonzero(near)
        stamp=int(msg.header.stamp.secs)*1000000000+int(msg.header.stamp.nsecs)
        rows.append(dict(receipt_elapsed_s=now-begin,stamp_ns=stamp,stamp_age_s=time.time()-stamp/1e9,
            seq=int(msg.header.seq),frame_id=msg.header.frame_id,caller=getattr(msg,'_connection_header',{}).get('callerid'),
            ranges=len(r),angle_min=float(msg.angle_min),angle_increment=float(msg.angle_increment),
            scan_time=float(msg.scan_time),time_increment=float(msg.time_increment),
            all_point4_to_point6_hit_count=len(allids),target_window_hit_count=len(selected),
            target_window_min_range_m=None if not np.any(window&valid) else float(np.min(r[window&valid])),
            near_target_hits=[dict(index=int(i),range_m=float(r[i]),angle_rad=float(theta[i])) for i in selected[:8]]))
    rospy.init_node('nso_readonly_scan_persistence',anonymous=True,disable_signals=True)
    sub=rospy.Subscriber('/scan',LaserScan,callback,queue_size=10)
    begin=time.monotonic();startwall=time.time()
    while time.monotonic()-begin<5.0 and not rospy.is_shutdown():time.sleep(.02)
    sub.unregister();elapsed=time.monotonic()-begin;rospy.signal_shutdown('bounded read-only capture complete')
    unique=sorted(set(x['stamp_ns'] for x in rows));withhits=sum(x['target_window_hit_count']>0 for x in rows)
    result=dict(schema='nso.later_stationary_passive_scan_persistence.v1',duration_requested_s=5.,elapsed_s=elapsed,
        started_wall_unix_s=startwall,scan_count=len(rows),unique_scan_stamps=len(unique),
        first_source_stamp_ns=unique[0] if unique else None,last_source_stamp_ns=unique[-1] if unique else None,
        target_sensor_angle_rad=target,target_window_halfwidth_rad=.02,near_range_inclusive_m=[.4,.6],
        scans_with_near_hit_in_target_window=withhits,near_hit_count_target_window=sum(x['target_window_hit_count'] for x in rows),
        observed_scan_intervals_s=[(b-a)/1e9 for a,b in zip(unique,unique[1:])],scans=rows,
        scope='Later passive5s sensor-direction persistence observation only; no motion, map clearing, filter changes or raw scan archive.',
        limits=['Later stationary sample, not synchronized to failure; intervening pose/scene/history equivalence is not independently verified. No later hit does not invalidate the recorded historical endpoint.','No per-ray timestamp inferred from zero time_increment.'])
    (D/'later_scan_persistence_5s.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['scans','observed_scan_intervals_s']},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--passive-scan5s',action='store_true');a=ap.parse_args();target=historical()
    if a.passive_scan5s:passive(target)
