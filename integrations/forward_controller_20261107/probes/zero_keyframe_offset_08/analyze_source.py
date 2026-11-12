"""Read-only parallel-estimator and long-wait disagreement audit."""
import json,math
from pathlib import Path
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

ROOT=Path(__file__).resolve().parent
CASE=ROOT/'zero_keyframe_offset_08'
DEPLOY=ROOT/'zero_keyframe_candidate'
REPORT=ROOT.parents[1]/'reports/nso-hybrid-20261107/controller_zero_keyframe_offset_08'

def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def apply(t,p):
    x,y,a=t;px,py,pa=p
    return [x+math.cos(a)*px-math.sin(a)*py,y+math.sin(a)*px+math.cos(a)*py,wrap(a+pa)]
def sample(p):
    q=p.orientation
    return [p.position.x,p.position.y,math.atan2(2*q.w*q.z,1-2*q.z*q.z)]
def stats(a):
    a=np.asarray(a,dtype=float)
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),max=float(a.max()))
def twist(a,b,dt):
    da=wrap(b[2]-a[2]);dx,dy=b[0]-a[0],b[1]-a[1];c,s=math.cos(a[2]),math.sin(a[2]);bx,by=c*dx+s*dy,-s*dx+c*dy
    if abs(da)<1e-8:return [bx/dt,by/dt,da/dt]
    s,c=math.sin(da)/da,(1-math.cos(da))/da;det=s*s+c*c
    return [(s*bx+c*by)/det/dt,(-c*bx+s*by)/det/dt,da/dt]

initial=json.loads((DEPLOY/'live_shadow/initial_alignment.json').read_text())
deployment=json.loads((DEPLOY/'live_deployment/alignment.json').read_text())
ia,da=initial['anchor'],deployment['anchor']
mapped=apply(initial['odom_from_source'],da['source_pose'])
relative=dict(elapsed_source_seconds=(da['source_stamp_ns']-ia['source_stamp_ns'])/1e9,
    candidate_source_xy_change_m=math.hypot(da['source_pose'][0]-ia['source_pose'][0],da['source_pose'][1]-ia['source_pose'][1]),
    old_primary_xy_change_m=math.hypot(da['odom_pose'][0]-ia['odom_pose'][0],da['odom_pose'][1]-ia['odom_pose'][1]),
    fixed_initial_alignment_relative_xy_disagreement_m=math.hypot(mapped[0]-da['odom_pose'][0],mapped[1]-da['odom_pose'][1]),
    fixed_initial_alignment_relative_yaw_disagreement_rad=wrap(mapped[2]-da['odom_pose'][2]),
    initial_anchor=ia,deployment_anchor=da,
    interpretation='relative estimator disagreement over416s; sampled zero CAN and no autonomous mission do not establish continuous independent stationary ground truth; re-alignment does not erase accumulated disagreement')

npz=np.load(CASE/'failure_scan_ring.npz');meta=json.loads((CASE/'failure_scan_ring.json').read_text());groups={};raw_twists={};frames={};callers={}
for row in meta['messages']:
    topic=row['topic']
    if topic=='/scan':continue
    msg=Odometry() if topic=='/nso_csm/odom' else PoseStamped();msg.deserialize(npz[row['key']].tobytes());p=msg.pose.pose if topic=='/nso_csm/odom' else msg.pose
    stamp=msg.header.stamp.to_nsec();groups.setdefault(topic,{})[stamp]=sample(p);frames.setdefault(topic,set()).add(msg.header.frame_id);callers.setdefault(topic,set()).add(row['caller'])
    if topic=='/nso_csm/odom':raw_twists[stamp]=[msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z]
old=groups['/nso_csm/pose_stamped'];source=groups['/nso_csm_zero_keyframe/pose_stamped'];raw=groups['/nso_csm/odom'];aligned={k:apply(deployment['odom_from_source'],v) for k,v in source.items()};common=sorted(set(old)&set(source)&set(raw))
def trajectory(values):
    steps=[];angular=[];violations=[]
    for a,b in zip(common,common[1:]):
        distance=math.hypot(values[b][0]-values[a][0],values[b][1]-values[a][1]);angle=abs(wrap(values[b][2]-values[a][2]));dt=(b-a)/1e9
        steps.append(distance);angular.append(angle)
        if distance>.02+.20*dt or angle>.04+.60*dt:violations.append(b)
    return dict(samples=len(common),xy_path_m=sum(steps),absolute_yaw_path_rad=sum(angular),max_xy_step_m=max(steps),max_yaw_step_rad=max(angular),original_jump_bound_violations=violations,endpoint_displacement_m=math.hypot(values[common[-1]][0]-values[common[0]][0],values[common[-1]][1]-values[common[0]][1]))
events=[json.loads(x) for x in (REPORT/'events.jsonl').read_text().splitlines()];adapter=[json.loads(x) for x in (REPORT/'adapter/events.jsonl').read_text().splitlines()];goal=next(e for e in events if e['event']=='goal');terminal=next(e for e in events if e['event']=='goal_result');arrival=next(e for e in events if e['event']=='arrival_evidence');finish=next(e for e in adapter if e['event']=='finished')
fault_command=next(e for e in adapter if e['event']=='command' and e['reason']=='native_driver_fault:8')
tele=json.loads((CASE/'telemetry.json').read_text());provenance={x['control_provenance']['csm_source_stamp_ns']:x['control_provenance'] for x in tele if x.get('control_provenance')};active=[p for p in provenance.values() if goal['receipt_monotonic']<=p['can_pair']['selected_monotonic_ns']/1e9<=terminal['receipt_monotonic']]
can_ring=json.loads((CASE/'failure_can_ring.json').read_text())
matching_diffs=[]
for a,b in zip(common,common[1:]):
    computed=twist(source[a],source[b],(b-a)/1e9);matching_diffs.append(max(abs(x-y) for x,y in zip(computed,raw_twists[b])))
at=arrival['after']['source_stamp_ns'];goal_xy=[goal['x'],goal['y']]
config=json.loads((CASE/'configuration.json').read_text())
result=dict(scope='parallel same-scan estimator disagreement and one B-controlled physical diagnostic; not a matched navigation efficacy or independent accuracy comparison',
    long_wait_relative_drift=relative,
    physical_result=json.loads((REPORT/'summary.json').read_text()),
    arrival=dict(verified=arrival['verified'],target_error_m=arrival['target_error_m'],target_yaw_error_rad=arrival['target_yaw_error_rad'],DWA_xy_tolerance_m=config['applied_DWA']['xy_goal_tolerance'],DWA_yaw_tolerance_rad=config['applied_DWA']['yaw_goal_tolerance'],supervisor_xy_tolerance_m=arrival['xy_tolerance_m'],supervisor_yaw_tolerance_rad=arrival['yaw_tolerance_rad'],source_stamp_ns=at),
    parallel_same_stamp=dict(common_samples=len(common),first_source_stamp_ns=common[0],last_source_stamp_ns=common[-1],frames={k:sorted(v) for k,v in frames.items()},callers={k:sorted(v) for k,v in callers.items()},old_primary=trajectory(old),candidate_aligned=trajectory(aligned),aligned_raw=trajectory(raw),candidate_old_xy_disagreement_m=stats([math.hypot(aligned[s][0]-old[s][0],aligned[s][1]-old[s][1]) for s in common]),final_candidate_old_xy_disagreement_m=math.hypot(aligned[common[-1]][0]-old[common[-1]][0],aligned[common[-1]][1]-old[common[-1]][1]),maximum_expected_alignment_vs_raw_xy_difference_m=max(math.hypot(aligned[s][0]-raw[s][0],aligned[s][1]-raw[s][1]) for s in common),maximum_expected_alignment_vs_raw_yaw_difference_rad=max(abs(wrap(aligned[s][2]-raw[s][2])) for s in common),maximum_body_twist_difference_fixed_SE2_vs_source= max(matching_diffs),same_arrival_stamp_xy_error_old_primary_m=math.hypot(old[at][0]-goal_xy[0],old[at][1]-goal_xy[1]),same_arrival_stamp_xy_error_aligned_candidate_m=math.hypot(raw[at][0]-goal_xy[0],raw[at][1]-goal_xy[1])),
    CAN_pairing=dict(unique_recorded_provenance=len(provenance),active_recorded_provenance=len(active),active_source_age_ms=stats([p['can_pair']['current_minus_can_ns']/1e6 for p in active]),active_scan_minus_CAN_ms=stats([p['can_pair']['scan_minus_can_ns']/1e6 for p in active]),recorded_source_reader_errors=sorted(set(str(p['can_source']['reader_error']) for p in provenance.values() if p['can_source']['reader_error'] is not None)),source_dropped_counter_max=max(p['can_source']['dropped_frames'] for p in provenance.values()),alignment_sha256_values=sorted(set(p['pose_alignment']['sha256'] for p in provenance.values())),all_declared_coordinate_transform=all(p.get('fixed_coordinate_transform_applied') and p.get('raw_pose_stamp_tf_unchanged') is False for p in provenance.values()),full_bounded_CAN_retained_frames=can_ring['retained_frames'],passive_CAN_source=can_ring['source']),
    end_sequence=dict(goal_result=terminal,first_recorded_native_fault8_zero_command_monotonic=fault_command['at_monotonic'],native_fault8_after_goal_ms=(fault_command['at_monotonic']-terminal['receipt_monotonic'])*1000,adapter_finished_reason=finish['reason'],stop_feedback_verifications=[dict(phase=e['phase'],confirmed=e['confirmed'],status=e['status']) for e in events if e['event']=='stop_feedback_verification'],interpretation='ROS arrival precedes native_driver_fault:8 by the recorded delta; preserve code8 without assigning undocumented causality; it did not interrupt the achieved goal'),
    recommendation='Do not retain zero-threshold B as the long-term/default estimator on this evidence. A bounded physical arrival succeeded but the parallel original estimator also had no guard jump on the same scans, and416s relative drift plus stationary negative control remain material. Root owns any explicit coordinate-preserving restoration/handover; no action is taken here.',
    limits=['No continuous independent pose ground truth for the long stationary wait; candidate-old disagreement is not absolute physical error.','Estimator path length includes measurement noise and is not a reconstruction-accuracy metric.','Parallel original estimator saw the same actual B-controlled motion; it was not a second navigation controller trial.','Telemetry is latest asynchronous polling; the serialized ring provides exact source-stamp pairing for150 recorded scans.','08,07,05/06 have different start conditions; preserve successes and failures without matched efficiency conclusions.'],live_modified=False)
restoration_path=DEPLOY/'live_deployment/restoration_process.json'
if restoration_path.exists():
    result['root_owned_restoration']=dict(process=json.loads(restoration_path.read_text()),
        alignment=json.loads((DEPLOY/'live_deployment/restore_alignment.json').read_text()),
        verification=json.loads((DEPLOY/'live_deployment/restoration_verification.json').read_text()),
        shadow_retired=json.loads((DEPLOY/'live_deployment/shadow_retired.json').read_text()),
        interpretation='Default reference thresholds restored on the uninterrupted original matcher. CAN control-velocity adaptation remains. One fixed same-stamp SE2 alignment joins the current displayed/map/TSDF coordinate epoch; it is representation continuity, not correction against ground truth or proof of improved accuracy. Relative B drift and estimator disagreement are preserved, not erased by alignment.')
(CASE/'parallel_estimator_analysis.json').write_text(json.dumps(result,indent=2))
(ROOT/'zero_keyframe_candidate_review/relative_drift.json').write_text(json.dumps(relative,indent=2))
print(json.dumps(dict(relative_drift_m=relative['fixed_initial_alignment_relative_xy_disagreement_m'],parallel=result['parallel_same_stamp'],arrival=result['arrival'],end_sequence=result['end_sequence']),indent=2))
