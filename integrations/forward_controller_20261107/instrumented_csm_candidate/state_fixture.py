"""Derive bounded references from actual poses, inject state on isolated 11402."""
import os
os.environ['ROS_MASTER_URI']='http://127.0.0.1:11402'
os.environ['ROS_HOSTNAME']='127.0.0.1'
os.environ['ROS_IP']='127.0.0.1'
import hashlib,json,math,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAP=ROOT.parent/'probes/control_can_offset_06'
OUT=Path(os.environ['FIXTURE_OUTPUT_DIR']).resolve()
if OUT==ROOT or ROOT in OUT.parents:raise ValueError('Use an output directory outside the fixture source')
OUT.mkdir(parents=True,exist_ok=True)

def prepare():
    import numpy as np
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import LaserScan
    z=np.load(CAP/'failure_scan_ring.npz');meta=json.loads((CAP/'failure_scan_ring.json').read_text())
    pose_rows={r['source_stamp_ns']:r for r in meta['messages'] if r['topic']=='/nso_csm/pose_stamped'}
    scans=[r for r in meta['messages'] if r['topic']=='/scan'];poses=[]
    for scan in scans:
        m=PoseStamped();m.deserialize(z[pose_rows[scan['source_stamp_ns']]['key']].tobytes());q=m.pose.orientation;p=m.pose.position
        poses.append([p.x,p.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)])
    target=next(i for i,r in enumerate(scans) if r['source_stamp_ns']==1794053131681172132);previous=target-1
    candidates=[];exclusions=[]
    for j in range(target):
        violating=[k for k in range(j+1,target) if math.hypot(poses[k][0]-poses[j][0],poses[k][1]-poses[j][1])>.1 or abs(math.atan2(math.sin(poses[k][2]-poses[j][2]),math.cos(poses[k][2]-poses[j][2])))>math.pi/18]
        if not violating:candidates.append(j)
        else:exclusions.append(dict(reference_index=j,first_observed_threshold_crossing_index=violating[0]))
    directory=OUT/'state_fixture_inputs';directory.mkdir(exist_ok=True)
    def save_scan(i):
        f=directory/f'scan_{i:03d}.bin';f.write_bytes(z[scans[i]['key']].tobytes());return f
    def save_pose(i):
        f=directory/f'pose_{i:03d}.bin';f.write_bytes(z[pose_rows[scans[i]['source_stamp_ns']]['key']].tobytes());return f
    target_scan=save_scan(target);target_pose=save_pose(target);prev_pose=save_pose(previous)
    jobs=[]
    for j in candidates:jobs.append(' '.join(map(str,[j,save_scan(j),target_scan,save_pose(j),prev_pose,target_pose])))
    (OUT/'state_fixture_jobs.txt').write_text('\n'.join(jobs)+'\n')
    proof=dict(target_index=target,previous_index=previous,candidate_reference_indices=candidates,exclusions=exclusions,default_keyframe_distance_m=.1,default_keyframe_angle_rad=math.pi/18,ring_start_to_previous_yaw_span_rad=abs(poses[previous][2]-poses[0][2]),pre_ring_reference_exclusion='start-to-prev yaw difference exceeds twice10deg, so a single older reference cannot remain within10deg of both observed poses',raw_scan_npz_sha256=hashlib.sha256((CAP/'failure_scan_ring.npz').read_bytes()).hexdigest(),parameters_swept=False)
    (OUT/'state_fixture_candidate_audit.json').write_text(json.dumps(proof,indent=2))

def worker():
    import rospy
    rospy.init_node('state_fixture_parameter_setup',disable_signals=True)
    params=json.loads((ROOT/'state_fixture_results.json').read_text())['parameters']
    rospy.set_param('/fixture/matcher',params)
    result=subprocess.run([str(OUT/'state_fixture'),str(OUT/'state_fixture_jobs.txt')],text=True,capture_output=True,timeout=15)
    (OUT/'state_fixture.stdout.log').write_text(result.stdout);(OUT/'state_fixture.stderr.log').write_text(result.stderr)
    rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    summary=dict(master='http://127.0.0.1:11402',exit_code=result.returncode,parameters=params,results=rows,matched_actual_target_candidates=[r['reference_index'] for r in rows if r.get('actual_target_xy_error_m',1)<1e-10 and r.get('actual_target_yaw_error_rad',1)<1e-10],actual_live_planar_jump_m=.05010596647245115,new_initialization_adjacent_pair_jump_m=.01017,scope='bounded candidate reference-state injection, not a parameter sweep; original historical reference unknown until a match is demonstrated',live_runtime_modified=False)
    (OUT/'state_fixture_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    rospy.signal_shutdown('fixture done')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
    else:
        prepare();s=socket.socket();assert s.connect_ex(('127.0.0.1',11402))!=0,'11402 occupied';s.close()
        with (OUT/'state_fixture_master.log').open('w') as log:
            master=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p','11402'],stdout=log,stderr=subprocess.STDOUT)
            try:time.sleep(2);subprocess.run([sys.executable,__file__,'worker'],check=True,timeout=20)
            finally:master.send_signal(signal.SIGINT);master.wait(timeout=10)
