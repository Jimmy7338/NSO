#!/usr/bin/env python3
"""Cross-check recording/3 source metadata and payloads without ROS access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from utils.rgbd_contract import RGBDFrame,PlanarScan
from scripts.audit_ros1_rgbd_sequence import check_sequence
from types import SimpleNamespace as S
from scripts.ros1_recording_v3_contract import convert_depth,convert_ranges,rectified_projection,transform_response


def integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError('invalid integer '+name)
    return value


def finite(value,name,minimum=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('invalid numeric '+name)
    if minimum is not None and value<minimum:raise ValueError('negative '+name)
    return float(value)


def nonempty(value,name):
    if not isinstance(value,str) or not value.strip():raise ValueError('empty '+name)
    return value


def check_metadata_sequence(source):
    source=Path(source);errors=[];records=[];hashes={}
    def read(path):
        raw=path.read_bytes();hashes[str(path.relative_to(source))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    manifest=read(source/'sequence.json')
    if manifest['schema_version']!='ros1_rgbd_recording/3':raise ValueError('recording/3 metadata required')
    count=integer(manifest['frames'],'frames',1)
    slop=finite(manifest['slop_s'],'slop_s',0)
    max_depth=finite(manifest['max_depth_m'],'max_depth_m',0)
    if max_depth==0:raise ValueError('positive max_depth_m required')
    slop_ns=round(slop*1_000_000_000)
    world=nonempty(manifest['world_frame'],'world_frame')
    optical=manifest['optical_frame']
    if not isinstance(optical,str):raise ValueError('invalid optical_frame')
    semantic=bool(manifest['semantic_topic'])
    names={f'{i:05d}' for i in range(count)}
    for sub,suffix in [('frames','.npz'),('scans','.npz'),('raw','.npz'),('metadata','.json')]:
        actual={p.stem for p in (source/sub).glob('*'+suffix)}
        if actual!=names:errors.append(dict(file=sub,reasons=['noncontiguous_missing_or_extra_files'],
                                           missing=sorted(names-actual),extra=sorted(actual-names)))
    previous={};float_allowance=0.
    for stem in sorted(names):
        row=dict(frame=stem,errors=[])
        try:
            meta=read(source/'metadata'/f'{stem}.json')
            if meta['schema_version']!='ros1_rgbd_source_metadata/2':raise ValueError('source metadata schema differs')
            headers=meta['headers'];required={'depth','rgb','scan','camera_info'}|({'semantic'} if semantic else set())
            if set(headers)!=required:raise ValueError('source header set differs from configured subscriptions')
            stamps={}
            for name,h in headers.items():
                secs=integer(h['secs'],'secs');ns=integer(h['nsecs'],'nsecs')
                if ns>=1_000_000_000:raise ValueError('nsecs outside canonical range')
                stamp=integer(h['timestamp_ns'],'timestamp_ns')
                if stamp!=secs*1_000_000_000+ns:raise ValueError('integer timestamp fields disagree')
                integer(h['sequence'],'sequence');nonempty(h['frame_id'],'frame_id')
                stamps[name]=stamp
                if name!='camera_info':
                    if name in previous and stamp<=previous[name]:raise ValueError('nonincreasing source '+name+' timestamp')
                    previous[name]=stamp
            source_times=[v for k,v in stamps.items() if k!='camera_info']
            span=max(source_times)-min(source_times)
            if span>slop_ns:raise ValueError('integer source timestamp span exceeds tolerance')
            if meta['synchronized_message_span_ns']!=span:raise ValueError('stored span differs from source timestamps')
            offsets={k:v-stamps['depth'] for k,v in stamps.items()}
            if meta['depth_relative_offsets_ns']!=offsets:raise ValueError('stored offsets differ from source timestamps')
            expected_payloads={f'frames/{stem}.npz',f'scans/{stem}.npz',f'raw/{stem}.npz'}
            if set(meta['payload_sha256'])!=expected_payloads:raise ValueError('payload hash paths differ')
            for name in expected_payloads:
                actual=hashlib.sha256((source/name).read_bytes()).hexdigest();hashes[name]=actual
                if meta['payload_sha256'][name]!=actual:raise ValueError('payload hash mismatch')
            frame=RGBDFrame.load(source/'frames'/f'{stem}.npz')
            scan=PlanarScan.load(source/'scans'/f'{stem}.npz')
            for name,actual in [('depth',frame.timestamp_s),('scan',scan.timestamp_s)]:
                h=headers[name];expected=float(h['secs'])+float(h['nsecs'])/1e9
                if not math.isfinite(actual) or actual!=expected:raise ValueError('payload/source timestamp differs: '+name)
                float_allowance=max(float_allowance,2*abs(float(np.spacing(expected))))
            enc=meta['image_encodings']
            if enc['depth'] not in ('32FC1','16UC1'):raise ValueError('unsupported declared depth encoding')
            nonempty(enc['rgb'],'RGB source encoding')
            if semantic:nonempty(enc['semantic'],'semantic source encoding')
            elif enc['semantic'] is not None or np.any(frame.semantic):raise ValueError('semantic data present without configured source')
            info=meta['camera_info']
            info_object=S(**{k:v for k,v in info.items() if k!='roi'},roi=S(**info['roi']))
            expected_intrinsic,source_from_rectified,projection=rectified_projection(info_object,frame.depth_m.shape)
            for k,v in projection.items():
                if info[k]!=v:raise ValueError('saved projection derivation differs: '+k)
            matrices={}
            for name,shape in [('K',(3,3)),('P',(3,4)),('R',(3,3))]:
                value=np.asarray(info[name],dtype=float)
                if value.shape!=(int(np.prod(shape)),) or not np.isfinite(value).all():raise ValueError('invalid CameraInfo '+name)
                matrices[name]=value.reshape(shape)
            dist=np.asarray(info['D'],dtype=float)
            if dist.ndim!=1 or not np.isfinite(dist).all():raise ValueError('invalid distortion vector')
            nonempty(info['distortion_model'],'distortion_model')
            if matrices['K'][0,0]<=0 or matrices['K'][1,1]<=0:raise ValueError('uncalibrated CameraInfo K')
            rotation=matrices['R']
            if (not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-5)
                    or abs(np.linalg.det(rotation)-1)>1e-5):raise ValueError('invalid rectification rotation')
            intrinsic=expected_intrinsic
            if not np.array_equal(frame.intrinsic,intrinsic):raise ValueError('stored intrinsic differs from rectified ROI/binning projection')
            if info['association']!='latest received, not timestamp-synchronized':raise ValueError('CameraInfo association differs')
            tf=meta['tf_queries']
            expected_tf=dict(target_frame=world,camera_source_frame=optical or headers['depth']['frame_id'],
                camera_stamp_ns=stamps['depth'],laser_source_frame=headers['scan']['frame_id'],laser_stamp_ns=stamps['scan'])
            if tf!=expected_tf:raise ValueError('TF query does not match configured frames and source timestamps')
            if headers['camera_info']['frame_id']!=expected_tf['camera_source_frame']:
                raise ValueError('CameraInfo optical frame differs from camera TF source')
            transforms={}
            for name,source_frame,query_stamp in [('camera',tf['camera_source_frame'],tf['camera_stamp_ns']),
                                                  ('laser',tf['laser_source_frame'],tf['laser_stamp_ns'])]:
                response=meta['tf_responses'][name]
                xyz=response['translation_xyz_m'];q=response['quaternion_xyzw']
                message=S(header=S(frame_id=response['target_frame'],stamp=S(secs=response['secs'],nsecs=response['nsecs'])),
                    child_frame_id=response['source_frame'],transform=S(translation=S(x=xyz[0],y=xyz[1],z=xyz[2]),
                    rotation=S(x=q[0],y=q[1],z=q[2],w=q[3])))
                matrix,recomputed=transform_response(message,world,source_frame)
                if response!=recomputed:raise ValueError('TF response serialization differs')
                if query_stamp==0 or response['timestamp_ns'] not in (0,query_stamp):
                    raise ValueError('TF response time inconsistent with historical query')
                transforms[name]=matrix
            if not np.array_equal(frame.world_from_camera,transforms['camera']@source_from_rectified):
                raise ValueError('stored camera pose differs from TF and rectification composition')
            if not np.array_equal(scan.world_from_laser,transforms['laser']):raise ValueError('stored laser pose differs from TF response')
            with np.load(source/'raw'/f'{stem}.npz',allow_pickle=False) as raw:
                if set(raw.files)!={'depth_source','ranges_source','intensities_source'}:raise ValueError('raw payload fields differ')
                expected_depth=convert_depth(raw['depth_source'],enc['depth'],max_depth)
                expected_ranges,conversion=convert_ranges(raw['ranges_source'],meta['laser']['range_min_m'],
                    meta['laser']['range_max_m'],manifest['positive_inf_policy'])
                if not np.array_equal(frame.depth_m,expected_depth):raise ValueError('stored depth differs from raw conversion')
                if not np.array_equal(scan.ranges_m,expected_ranges):raise ValueError('stored laser differs from raw conversion')
                if meta['range_conversion']!=conversion:raise ValueError('range conversion audit differs')
                intensities=raw['intensities_source']
                if intensities.ndim!=1 or len(intensities) not in (0,len(expected_ranges)):
                    raise ValueError('raw laser intensities cardinality differs')
            laser=meta['laser'];n=integer(laser['beam_count'],'beam_count',1)
            if n!=len(scan.ranges_m):raise ValueError('laser beam count differs')
            amin=finite(laser['angle_min_rad'],'angle_min');amax=finite(laser['angle_max_rad'],'angle_max')
            inc=finite(laser['angle_increment_rad'],'angle_increment')
            if inc==0:raise ValueError('zero laser angle increment')
            tolerance=8*np.finfo(np.float32).eps*max(1,abs(amin),abs(amax),n*abs(inc))
            if abs(amin+(n-1)*inc-amax)>tolerance:raise ValueError('laser angular endpoint and beam count disagree')
            low=finite(laser['range_min_m'],'range_min',0);high=finite(laser['range_max_m'],'range_max',0)
            if high<=low:raise ValueError('invalid laser range bounds')
            if (scan.angle_min_rad!=amin or scan.angle_increment_rad!=inc or scan.range_max_m!=high):
                raise ValueError('stored laser calibration differs from source metadata')
            if np.any((scan.ranges_m>0)&(scan.ranges_m<low-1e-6)):raise ValueError('positive range below source minimum')
            tinc=finite(laser['time_increment_s'],'time_increment',0)
            period=finite(laser['scan_time_s'],'scan_time',0)
            if laser['motion_compensation_applied'] is not False:raise ValueError('recorder/3 cannot claim deskew')
            if meta['calibration_correctness_verified'] is not False:raise ValueError('recorder/3 cannot certify calibration')
            if meta['raw_rosbag_still_required'] is not True:raise ValueError('raw bag requirement removed')
            row.update(integer_span_ns=span,depth_scan_offset_ns=offsets['scan'],
                camera_info_age_ns=stamps['depth']-stamps['camera_info'],
                camera_info_frame_matches_depth=headers['camera_info']['frame_id']==headers['depth']['frame_id'],
                last_beam_offset_s=(n-1)*tinc,scan_period_s=period,
                per_beam_timing_available=tinc>0,tf_query_declaration_consistent=True)
        except (ValueError,TypeError,KeyError,IndexError,OSError,OverflowError,AttributeError) as exc:
            row['errors'].append(str(exc))
        if row['errors']:errors.append(dict(file=stem,reasons=row['errors']))
        records.append(row)
    # Integer headers above enforce the exact synchronization gate. The legacy
    # array check gets only the representational allowance for float seconds.
    try:
        arrays=check_sequence(source,slop+float_allowance,max_depth)
        if arrays['status']!='passed':errors.append(dict(file='payload_arrays',reasons=arrays['errors']))
    except (ValueError,TypeError,KeyError,OSError) as exc:
        arrays=dict(status='failed',error=str(exc));errors.append(dict(file='payload_arrays',reasons=[str(exc)]))
    return dict(status='failed' if errors else 'passed',schema='ros1_recording_v3_crosscheck/1',
        frames=count,records=records,errors=errors,input_sha256=hashes,payload_array_audit=arrays,
        exact_integer_sync_tolerance_ns=slop_ns,float_seconds_representation_allowance_s=float_allowance,
        source_metadata_internally_consistent=not errors,robot_interface_ready=False,
        motion_output_enabled=False,extrinsic_correctness_verified=False,
        unchecked=['actual TF response correctness and sensor calibration','RGB/depth physical registration',
            'physical interpretation of rectified depth frame and driver infinity convention',
            'per-beam motion compensation','map/odom resets, pose drift and reintegration',
            'live ROS and robot navigation; original bag still required'],
        boundary='Checks raw conversion, saved declarations, projection and TF response composition; does not authenticate calibration, bag or physical alignment.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:result=check_metadata_sequence(a.source)
    except (ValueError,TypeError,KeyError,OSError) as exc:
        result=dict(status='failed',errors=[str(exc)],robot_interface_ready=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','errors','robot_interface_ready')},ensure_ascii=False))
    if result['status']!='passed':raise SystemExit(2)
