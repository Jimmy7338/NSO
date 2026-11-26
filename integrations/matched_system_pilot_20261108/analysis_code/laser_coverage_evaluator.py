#!/usr/bin/env python3
"""Offline saved-scan coverage proxy; no ROS, depth fusion, model or controller.

Frozen evaluation convention: 0.2m native grid, center-in-circle 12.5m ROI,
finite real lidar returns only, native cell-to-cell supercover rasterization.
This measures saved observations, not GT environment coverage/accuracy.
"""
import argparse, bisect, hashlib, json, math, pathlib, struct, sys, tempfile
import numpy as np

WORK=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
sys.path.insert(0,str(WORK/'nso_bridge/joint_vendor'))
from utils.grid_geometry import supercover_line

PROFILE='saved_laser_coverage_v1'
RESOLUTION_M=.2
ROI_RADIUS_M=12.5
RANGE_CAP_M=8.

def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def parse_scan(raw):
    """Decode ROS1 sensor_msgs/LaserScan bytes without importing ROS."""
    data=memoryview(np.asarray(raw,dtype=np.uint8)).cast('B');offset=0
    def read(fmt):
        nonlocal offset
        result=struct.unpack_from('<'+fmt,data,offset);offset+=struct.calcsize('<'+fmt)
        return result
    seq,sec,nsec=read('III');length,=read('I')
    frame=bytes(data[offset:offset+length]).decode('utf8');offset+=length
    values=read('7f');count,=read('I')
    ranges=np.frombuffer(data[offset:offset+count*4],dtype='<f4').copy();offset+=count*4
    intensities,=read('I');offset+=intensities*4
    if offset!=len(data) or nsec>=1000000000:raise ValueError('invalid serialized LaserScan')
    return dict(seq=seq,stamp_ns=sec*1000000000+nsec,frame_id=frame,ranges=ranges,
        angle_min_rad=values[0],angle_max_rad=values[1],angle_increment_rad=values[2],
        time_increment_s=values[3],scan_time_s=values[4],range_min_m=values[5],range_max_m=values[6])

def rigid(value):
    t=np.asarray(value,dtype=float)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1])
        or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-6)
        or not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-6)):
        raise ValueError('rigid measured transform required')
    return t

def grid_cell(point,shape):
    # Same source convention as SensorMapper.grid_cell: world +y -> reversed row.
    return (shape[0]-1-int(np.floor(point[1]/RESOLUTION_M)),
            int(np.floor(point[0]/RESOLUTION_M)))

def cell_center(cell,shape):
    return np.array([(cell[1]+.5)*RESOLUTION_M,(shape[0]-cell[0]-.5)*RESOLUTION_M])

def load_scan(path,planning_from_odom):
    with np.load(path,allow_pickle=False) as z:
        receipt=json.loads(str(z['receipt_json'].item()));h=receipt['headers']['scan']
        geometry=receipt['source_geometry']['scan'];tf=receipt['historical_tf']['odom_from_laser']
        if h['publisher']!='/lslidar_driver_node' or h['frame_id']!='laser_link':
            raise ValueError('real original single-line lidar required; no depth-scan substitution')
        if (tf['target_frame'],tf['source_frame'],tf['query_stamp_ns'])!=('odom','laser_link',h['stamp_ns']):
            raise ValueError('historical scan TF must bind its own exact source timestamp')
        serialized=parse_scan(z['scan_received_ros_serialized'])
        for key in ('seq','stamp_ns','frame_id'):
            if serialized[key]!=h[key]:raise ValueError('serialized scan header mismatch')
        for key in ('angle_min_rad','angle_max_rad','angle_increment_rad','range_min_m','range_max_m','scan_time_s','time_increment_s'):
            if serialized[key]!=geometry[key]:raise ValueError('serialized scan geometry mismatch '+key)
        raw=z['scan_ranges_raw'].copy()
        if len(raw)!=geometry['range_count'] or not np.array_equal(raw,serialized['ranges'],equal_nan=True):
            raise ValueError('serialized scan/range array mismatch')
        odom_from_laser=rigid(tf['matrix'])
        if not np.array_equal(z['world_from_laser'],odom_from_laser):raise ValueError('canonical historical laser TF mismatch')
        if abs(float(z['scan_timestamp_s'])*1e9-h['stamp_ns'])>512:raise ValueError('scan time float binding mismatch')
        if (not all(math.isfinite(geometry[k]) for k in ('angle_min_rad','angle_increment_rad','range_min_m','range_max_m'))
            or geometry['angle_increment_rad']<=0 or not 0<=geometry['range_min_m']<geometry['range_max_m']):
            raise ValueError('invalid angular indexing')
    return dict(path=str(path),stamp_ns=h['stamp_ns'],ranges=raw,geometry=geometry,
        planning_from_laser=rigid(planning_from_odom)@odom_from_laser,
        source='original /scan, /lslidar_driver_node, laser_link',
        raw_ranges_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        serialized_sha256=hashlib.sha256(serialized['ranges'].tobytes()).hexdigest(),
        historical_TF_query_stamp_ns=tf['query_stamp_ns'])

def beam(scan,index):
    raw=float(scan['ranges'][index]);g=scan['geometry']
    if not math.isfinite(raw) or raw<=0 or not g['range_min_m']<=raw<=g['range_max_m']:
        return None
    distance=min(raw,RANGE_CAP_M)
    angle=g['angle_min_rad']+index*g['angle_increment_rad']
    t=scan['planning_from_laser'];end=t@np.array([math.cos(angle)*distance,math.sin(angle)*distance,0.,1.])
    # Original SensorMapper endpoint semantics after sensor_range_matched_v1.
    hit=distance<min(RANGE_CAP_M,g['range_max_m'])-1e-3
    return dict(origin=t[:2,3],endpoint=end[:2],distance=distance,hit=hit,raw=raw)

class Coverage:
    def __init__(self,shape,center):
        self.shape=tuple(shape);self.center=np.asarray(center,dtype=float)
        rows,cols=np.indices(shape);x=(cols+.5)*RESOLUTION_M;y=(shape[0]-rows-.5)*RESOLUTION_M
        self.roi=(x-self.center[0])**2+(y-self.center[1])**2<=ROI_RADIUS_M**2
        self.observed=np.zeros(shape,bool);self.free=np.zeros(shape,bool);self.hit=np.zeros(shape,bool)
        self.outside=set();self.out_grid=set();self.origin_outside=0;self.endpoints_outside=0
    def add(self,scan):
        invalid=clipped=valid=0
        for i in range(len(scan['ranges'])):
            b=beam(scan,i)
            if b is None:invalid+=1;continue
            valid+=1;clipped+=b['raw']>RANGE_CAP_M
            self.endpoints_outside+=int(np.linalg.norm(b['endpoint']-self.center)>ROI_RADIUS_M)
            origin=grid_cell(b['origin'],self.shape);target=grid_cell(b['endpoint'],self.shape)
            ray=supercover_line(target[0]-origin[0],target[1]-origin[1])
            for j,(dr,dc) in enumerate(ray):
                cell=(origin[0]+dr,origin[1]+dc)
                if not (0<=cell[0]<self.shape[0] and 0<=cell[1]<self.shape[1]):
                    self.out_grid.add(cell);continue
                if not self.roi[cell]:self.outside.add(cell);continue
                self.observed[cell]=True
                if j==len(ray)-1 and b['hit']:self.hit[cell]=True
                else:self.free[cell]=True
        if np.linalg.norm(scan['planning_from_laser'][:2,3]-self.center)>ROI_RADIUS_M:self.origin_outside+=1
        return dict(valid_rays=valid,invalid_rays_skipped=invalid,finite_over8m_clipped_free_only=clipped)
    def measures(self):
        return dict(observed_union_m2=int(self.observed.sum())*RESOLUTION_M**2,
            free_evidence_union_m2=int(self.free.sum())*RESOLUTION_M**2,
            hit_endpoint_union_m2=int(self.hit.sum())*RESOLUTION_M**2,
            free_without_ever_hit_m2=int((self.free&~self.hit).sum())*RESOLUTION_M**2,
            free_hit_conflict_cells=int((self.free&self.hit).sum()),
            outside_ROI_unique_raster_cells=len(self.outside),outside_grid_unique_cells=len(self.out_grid),
            laser_origins_outside_ROI=self.origin_outside,beam_endpoints_outside_ROI=self.endpoints_outside)

def can_samples(report):
    result=[]
    for text in (report/'adapter/events.jsonl').read_text().splitlines():
        row=json.loads(text)
        if row.get('event')=='CAN_feedback' and row.get('id')=='0x221':
            result.append((int(round(row['kernel_unix_s']*1e9)),float(row['can_distance_m']),bool(row['feedback_accounting_complete'])))
    result.sort()
    if not result or any(b[1]<a[1]-1e-9 for a,b in zip(result,result[1:])):raise ValueError('missing/nonmonotone cumulative CAN journal')
    return result

def can_bracket(samples,stamp):
    index=bisect.bisect_right([x[0] for x in samples],stamp)
    lower=samples[index-1] if index else None;upper=samples[index] if index<len(samples) else None
    return dict(asof_distance_m=None if lower is None else lower[1],
        asof_source_age_s=None if lower is None else (stamp-lower[0])/1e9,
        next_distance_m=None if upper is None else upper[1],
        next_source_lead_s=None if upper is None else (upper[0]-stamp)/1e9,
        bracket_accounting_complete=bool(lower and upper and lower[2] and upper[2]),
        semantics='Sampled integrated original CAN counters bracketing scan source time; no interpolation or exact cost-at-scan claim. CAN kernel stamps are logged float seconds.')

def evaluate(report,frame_config):
    cfg=json.loads(frame_config.read_text());c=cfg['configuration'];shape=tuple(c['shape']);anchor=tuple(c['return_anchor'][:2]);T=rigid(c['planning_from_odom'])
    center=cell_center(anchor,shape);coverage=Coverage(shape,center);samples=can_samples(report)
    rpc=[json.loads(x) for x in (report/'joint/decisions.jsonl').read_text().splitlines()]
    observations=[x for x in rpc if x['request']['method']=='observe' and x['response']['result'].get('accepted')]
    curve=[];inputs=[];first_stamp=None;baseline=None;seen=set()
    for row in observations:
        params=row['request']['params'];scan=load_scan(pathlib.Path(params['path']),T)
        if scan['stamp_ns'] in seen:raise ValueError('duplicate lidar source scan across accepted observations')
        if first_stamp is None:first_stamp=scan['stamp_ns']
        if seen and scan['stamp_ns']<=curve[-1]['scan_stamp_ns']:raise ValueError('scan source times regressed')
        seen.add(scan['stamp_ns']);rays=coverage.add(scan);metrics=coverage.measures()
        if baseline is None:baseline=dict(metrics)
        curve.append(dict(paid_actions=params['action_id'],scan_stamp_ns=scan['stamp_ns'],
            elapsed_since_first_scan_s=(scan['stamp_ns']-first_stamp)/1e9,**metrics,**rays,
            new_observed_vs_firstscan_m2=metrics['observed_union_m2']-baseline['observed_union_m2'],
            new_free_evidence_vs_firstscan_m2=metrics['free_evidence_union_m2']-baseline['free_evidence_union_m2'],
            CAN_counter=can_bracket(samples,scan['stamp_ns'])))
        inputs.append(dict(path=scan['path'],raw_ranges_sha256=scan['raw_ranges_sha256'],
            receipt_scan_stamp_ns=scan['stamp_ns'],historical_TF_query_stamp_ns=scan['historical_TF_query_stamp_ns'],
            source_packet_sha256_from_logged_receipt=row['response']['result']['summary']['semantic_diagnostics']['receipt']['source_file_sha256']))
    return dict(schema='nso.laser_coverage_evaluation.v1',profile=PROFILE,status='offline_saved_scan_evaluation',
        report=str(report),fixed_frame_config=str(frame_config),frame_config_sha256=digest(frame_config),
        planning_from_odom=T.tolist(),grid_shape=list(shape),resolution_m=RESOLUTION_M,
        ROI=dict(center_planning_xy=center.tolist(),radius_m=ROI_RADIUS_M,cell_rule='cell center inside closed circle',
            raster_area_m2=int(coverage.roi.sum())*RESOLUTION_M**2,purpose='Predefined measurement ROI, not a guarantee enclosing all executed trajectories or known traversable environment.'),
        sensor=dict(source='/scan /lslidar_driver_node laser_link',cap_m=RANGE_CAP_M,
            provenance='All raw ranges, geometry and source stamps checked against saved ROS1 LaserScan bytes; own-source historical odom_from_laser required.',
            invalid='Nonfinite/nonpositive/outside original range limits skipped; no synthesized rays.',
            endpoint='Native matched-range convention: finite returns clipped at8m, hit only below effective_max−1mm; exact/max boundary is free-only.',
            raster='Native grid_cell + supercover_line between endpoint cells; discretization can include portions/centers beyond exact8m metric radius.',
            within_scan_motion='One historical TF per scan source stamp; recorded scan time_increment=0 does not enable ray deskew.'),
        area_semantics='Cumulative observed/free-ray/hit-endpoint evidence unions; free evidence may overlap hit evidence across scans or coarse cells. free_without_ever_hit removes all ever-hit cells, and need not be monotone. Neither quantity is GT free space.',
        observation_sampling='Only one archived real lidar scan per accepted RGBD observation; not every continuously received scan. Sensor/source/phase and capture procedure must be common across pilot arms.',
        initial=curve[0],final=curve[-1],curve=curve,inputs=inputs,
        evidence_sha256={name:digest(report/name) for name in ['joint/decisions.jsonl','adapter/events.jsonl']},
        evaluator_sha256=digest(__file__),limitations=['No ground-truth environment coverage percentage or reconstruction accuracy.','Localization/extrinsic errors affect this independent measurement too.','Fixed common coordinate epoch/registration must be verified across future arms; equal matrices alone cannot prove the same odom epoch.','Sampled CAN cost brackets have finite time uncertainty.','This development sequence is not a predeclared efficacy comparison.'],
        depth_fused_belief_read=False,robot_or_ROS_actions=False)

def validate(report,frame_config,result):
    """Analytic geometric/measurement contracts plus actual-record bindings."""
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(dict(name=name,passed=True))
    def synthetic(ranges,origin=(10.1,10.1),rotation=0.,step=math.pi/2):
        t=np.eye(4);c,s=math.cos(rotation),math.sin(rotation);t[:2,:2]=[[c,-s],[s,c]];t[:2,3]=origin
        return dict(ranges=np.array(ranges),planning_from_laser=t,
            geometry=dict(range_min_m=.1,range_max_m=25.,angle_min_rad=0.,angle_increment_rad=step))
    rotated=synthetic([1.,2.],rotation=math.pi/2)
    check('90deg_laser_TF_maps_local_x_north_and_original_index1_west',
        np.allclose(beam(rotated,0)['endpoint'],[10.1,11.1]) and np.allclose(beam(rotated,1)['endpoint'],[8.1,10.1]))
    check('grid_row_reversal_and_metric_boundary_index',grid_cell([10.1,11.1],(100,100))==(44,50)
        and grid_cell([8.1,10.1],(100,100))==(49,40) and grid_cell([0.,0.],(100,100))==(99,0)
        and grid_cell([-.001,-.001],(100,100))==(100,-1))
    invalid=synthetic([float('nan'),float('inf'),-float('inf'),0.,.05,25.1]);cov=Coverage((100,100),[10.1,10.1]);stats=cov.add(invalid)
    check('invalid_rays_do_not_create_free_space',stats['invalid_rays_skipped']==6 and not cov.observed.any())
    hit=beam(synthetic([7.998]),0);near=beam(synthetic([7.9995]),0);edge=beam(synthetic([8.]),0);far=beam(synthetic([8.001]),0)
    check('native_1mm_max_boundary_and_exact8_free_only',hit['hit'] and not near['hit'] and not edge['hit'] and not far['hit'])
    check('finite_beyond8_endpoint_clipped_geometrically',abs(np.linalg.norm(far['endpoint']-far['origin'])-8.)<1e-12)
    cov=Coverage((100,100),[10.1,10.1]);cov.add(synthetic([25.]));expected={(49,c) for c in range(50,91)}
    check('analytic_axis_8m_ray_has_41_cells_and_free_endpoint',set(map(tuple,np.argwhere(cov.observed)))==expected
        and set(map(tuple,np.argwhere(cov.free)))==expected and not cov.hit.any())
    cov=Coverage((200,200),[20.1,20.1]);cov.add(synthetic([2.],origin=(32.5,20.1)))
    check('ROI_clips_metrics_and_counts_outside_instead_of_hiding',len(cov.outside)>0 and not (cov.observed&~cov.roi).any())
    cov.add(synthetic([1.],origin=(33.1,20.1)));check('out_of_ROI_origin_recorded_not_a_runtime_gate',cov.origin_outside==1)
    curve=result['curve']
    accepted=[r for r in (json.loads(x) for x in (report/'joint/decisions.jsonl').read_text().splitlines())
              if r['request']['method']=='observe' and r['response']['result'].get('accepted')]
    check('all_accepted_saved_scans_bound_and_failed_short_episodes_allowed',len(curve)==len(accepted)
        and curve[-1]['paid_actions']==accepted[-1]['request']['params']['action_id'])
    check('cumulative_observed_and_free_evidence_unions_monotone',all(b['observed_union_m2']>=a['observed_union_m2'] and b['free_evidence_union_m2']>=a['free_evidence_union_m2'] for a,b in zip(curve,curve[1:])))
    check('all_saved_scan_costs_have_complete_CAN_brackets',all(x['CAN_counter']['bracket_accounting_complete'] for x in curve))
    # Reject plausible but wrong depth-scan provenance and wrong historical time.
    first=pathlib.Path(result['inputs'][0]['path']);T=np.array(result['planning_from_odom'])
    with np.load(first,allow_pickle=False) as z:
        subset={k:z[k].copy() for k in ('scan_ranges_raw','scan_received_ros_serialized','world_from_laser','scan_timestamp_s')}
        receipt=json.loads(str(z['receipt_json'].item()))
    actual=load_scan(first,T);laser=np.array(receipt['historical_tf']['odom_from_laser']['matrix'])
    measured=next(beam(actual,i) for i in range(len(actual['ranges'])) if beam(actual,i) is not None)
    check('actual_source_TF_origin_not_camera_or_ideal_anchor',
        np.allclose(measured['origin'],(T@laser)[:2,3])
        and np.linalg.norm(measured['endpoint']-measured['origin'])<=8.+1e-9)
    bracket=can_bracket([(100,1.,True),(200,3.,True)],150)
    check('CAN_cost_keeps_measurement_brackets_without_interpolation',bracket['asof_distance_m']==1.
        and bracket['next_distance_m']==3. and bracket['asof_source_age_s']==50/1e9)
    with tempfile.TemporaryDirectory(prefix='nso_laser_eval_contract_') as td:
        path=pathlib.Path(td)/'bad.npz'
        for name,mutate in [('reject_depth_scan_provenance',lambda r:r['headers']['scan'].update(publisher='/nso_visual_sensor_bridge')),
                            ('reject_TF_from_different_source_stamp',lambda r:r['historical_tf']['odom_from_laser'].update(query_stamp_ns=r['headers']['scan']['stamp_ns']+1))]:
            changed=json.loads(json.dumps(receipt));mutate(changed);np.savez(path,**subset,receipt_json=json.dumps(changed))
            rejected=False
            try:load_scan(path,T)
            except ValueError:rejected=True
            check(name,rejected)
    return dict(schema='nso.laser_coverage_evaluator_validation.v1',status='passed_fixed_development_sequence_not_efficacy',
        checks=checks,evaluator_sha256=digest(__file__),result_content_sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),
        development_sequence=dict(accepted_scans=len(curve),last_paid_action=curve[-1]['paid_actions']),
        reused_source=dict(path=str(WORK/'nso_bridge/joint_vendor/utils/grid_geometry.py'),sha256=digest(WORK/'nso_bridge/joint_vendor/utils/grid_geometry.py')),
        grid_convention_reference=dict(path=str(WORK/'nso_bridge/joint_vendor/nso/mapping3d.py'),sha256=digest(WORK/'nso_bridge/joint_vendor/nso/mapping3d.py'),lines=[19,20,21,34,35,36,37,38]),
        frozen_for_future_pilot=True,
        freeze_scope='Code and fixed ROI/raster/range conventions only. Future experiment protocol, common frame/epoch and all per-arm configuration receipts still require pre-execution freeze.',
        limitations=['Geometry tests and provenance checks do not validate physical calibration or coverage accuracy.','No continuous lidar sequence is archived; this common evaluator uses selected saved scans only.','Phase wall timing/SLAM truth is not evaluated here.'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=pathlib.Path,required=True);p.add_argument('--frame-config',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--validate-output',type=pathlib.Path);args=p.parse_args()
    result=evaluate(args.report,args.frame_config)
    validation=validate(args.report,args.frame_config,result) if args.validate_output else None
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    if validation:
        validation['saved_result_sha256']=digest(args.output)
        with args.validate_output.open('x',encoding='utf8') as f:json.dump(validation,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=str(args.output),observations=len(result['curve']),final=result['final']),ensure_ascii=False))

if __name__=='__main__':main()
