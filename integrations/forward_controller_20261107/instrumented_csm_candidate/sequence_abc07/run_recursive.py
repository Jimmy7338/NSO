"""Bounded fully recursive native A/B/C, original scans and optional full CAN."""
import os
os.environ['ROS_MASTER_URI']='http://127.0.0.1:11403'
os.environ['ROS_HOSTNAME']='127.0.0.1'
os.environ['ROS_IP']='127.0.0.1'
import hashlib,json,math,signal,socket,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[1]
OUT=Path(os.environ['SEQUENCE_OUTPUT_DIR']).resolve()
if OUT==ROOT or ROOT in OUT.parents:raise ValueError('output must be separate')
OUT.mkdir(parents=True,exist_ok=True)

def worker(case):
    import numpy as np,rospy
    from geometry_msgs.msg import PoseStamped
    rospy.init_node('recursive_fixture_setup',disable_signals=True)
    modes='ABC' if case=='07' else 'AB'
    source=DATA/'probes'/('control_can_offset_'+case)
    z=np.load(source/'failure_scan_ring.npz');meta=json.loads((source/'failure_scan_ring.json').read_text())
    scans=[v for v in meta['messages'] if v['topic']=='/scan'];poses={v['source_stamp_ns']:v for v in meta['messages'] if v['topic']=='/nso_csm/pose_stamped'}
    start,end=(0,len(scans)-1) if case=='07' else (96,104)
    priors={}
    if case=='07':
        priors={v['target_index']:v for v in json.loads((DATA/'can_source_audit07/intervals.json').read_text())}
        for k in range(start+1,end+1):
            v=priors[k]
            assert v['source_interval_ns']==[scans[k-1]['source_stamp_ns'],scans[k]['source_stamp_ns']]
            assert v['eligible_C_source_interval'] and v['full_retained_interval_covered'] and v['all_receipts_before_scan_capture']
            assert v['max_held_source_age_ns']<=150_000_000
    params=json.loads((ROOT.parent/'state_fixture_results.json').read_text())['parameters']
    configs={}
    for mode in modes:
        config=dict(params)
        if mode=='B':config.update(kf_dist_linear=0.,kf_dist_angular=0.)
        configs[mode]=config;rospy.set_param('/sequence/'+mode+'/matcher',config)
    with tempfile.TemporaryDirectory(prefix='finite_inputs_'+case+'_',dir=OUT) as directory:
        directory=Path(directory)
        def scan(i):
            p=directory/f'scan_{i:03d}.bin';p.write_bytes(z[scans[i]['key']].tobytes());return p
        anchor=directory/'anchor.bin';anchor.write_bytes(z[poses[scans[start]['source_stamp_ns']]['key']].tobytes())
        jobs=[f'{scan(start)} {anchor}']
        for k in range(start+1,end+1):
            delta=priors[k]['body_delta_ZOH'] if priors else [0.,0.,0.]
            jobs.append(' '.join(map(str,[k,scan(k),*delta])))
        job_file=directory/'jobs.txt';job_file.write_text('\n'.join(jobs)+'\n')
        result=subprocess.run([str(OUT/'native_sequence'),str(job_file),modes],text=True,capture_output=True,timeout=25)
    (OUT/(case+'_native.stdout.log')).write_text(result.stdout);(OUT/(case+'_native.stderr.log')).write_text(result.stderr)
    rows=[json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
    # Correct the seed reference index label; binary seed=0 denotes seed slot,
    # while scan06's actual seed is index96. All later replacements use real indices.
    if start:
        for row in rows:
            if row['reference_index']==0:row['reference_index']=start
    anchor_msg=PoseStamped();anchor_msg.deserialize(z[poses[scans[start]['source_stamp_ns']]['key']].tobytes());q=anchor_msg.pose.orientation;p=anchor_msg.pose.position
    anchor_vector=[p.x,p.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)]
    summary={}
    for mode in modes:
        sequence=[r for r in rows if r['mode']==mode]
        previous=anchor_vector;recursive_ok=True
        for row in sequence:
            recursive_ok=recursive_ok and max(abs(x-y) for x,y in zip(row['previous_pose'],previous))<1e-12
            previous=row['result_pose']
        durations=[r['icp_microseconds'] for r in sequence]
        summary[mode]=dict(n=len(sequence),invalid_count=sum(not r['valid'] for r in sequence),guard_violation_indices=[r['scan_index'] for r in sequence if r['exceeds_existing_jump_bound']],maximum_jump_m=max(r['planar_jump_m'] for r in sequence),cumulative_xy_path_m=sum(r['planar_jump_m'] for r in sequence),reference_replacement_count=sum(r['reference_replaced'] for r in sequence),final_pose=sequence[-1]['result_pose'],recursion_previous_equals_own_previous_result=recursive_ok,icp_microseconds_median=float(np.median(durations)),icp_microseconds_p95=float(np.percentile(durations,95)))
    output=dict(case=case,scope='fully recursive native fixture, shared initial scan/reference and actual pose anchor only; subsequent poses exclusively from own ICP outputs',seed_scan_index=start,seed_source_stamp_ns=scans[start]['source_stamp_ns'],seed_pose=anchor_vector,configs=configs,results=rows,summary=summary,source_scan_npz_sha256=hashlib.sha256((source/'failure_scan_ring.npz').read_bytes()).hexdigest(),CAN_prior_intervals=priors if priors else None,CAN_prior_semantics='original kernel samples, causal body-frame piecewise constant velocity integration; sampling model not independent motion truth' if priors else 'C not run:06 fault prior has missing retained samples and stale gap',B_reference_policy='existing native newKeyframeNeeded with kf_dist_linear=0,kf_dist_angular=0; no forced update; exactly zero measured motion may retain prior ref',A_C_reference_policy='existing native defaults .10m/10deg; same unmodified newKeyframeNeeded',initialization_limit='07 uses controlled fresh scan0 reference anchored to its recorded actual pose; not claimed to restore unknown historical live keyframe.06 begins at previously exactly recovered ref096/pose096.',guard_semantics='unchanged guard bounds computed only for offline audit; no ICP output rejected or clipped; replay continues after a would-stop violation',live_runtime_modified=False)
    (OUT/(case+'_results.json')).write_text(json.dumps(output,indent=2));print(json.dumps(dict(case=case,summary=summary),indent=2))
    rospy.signal_shutdown('done')

if __name__=='__main__':
    if len(sys.argv)>2 and sys.argv[1]=='worker':worker(sys.argv[2])
    else:
        case=sys.argv[1] if len(sys.argv)>1 else '07';assert case in ['06','07']
        s=socket.socket();assert s.connect_ex(('127.0.0.1',11403))!=0,'11403 occupied';s.close()
        with (OUT/(case+'_master.log')).open('w') as log:
            p=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p','11403'],stdout=log,stderr=subprocess.STDOUT)
            try:time.sleep(2);subprocess.run([sys.executable,__file__,'worker',case],check=True,timeout=35)
            finally:p.send_signal(signal.SIGINT);p.wait(timeout=10)
