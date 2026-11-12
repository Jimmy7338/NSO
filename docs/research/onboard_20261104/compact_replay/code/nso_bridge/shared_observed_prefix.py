"""Audited native-category measured history initialization, never live feedback."""
import hashlib,json,math
from pathlib import Path
import numpy as np

def descriptors(value):
    if value is None:return []
    if not isinstance(value,list):raise ValueError('shared_observed_prefix requires a list of descriptors')
    result=[]
    required={'path','sha256','annotation_receipt','annotation_receipt_sha256','registration'}
    for d in value:
        if not isinstance(d,dict) or set(d)!=required:raise ValueError('prefix descriptor fields must be explicit')
        if d['registration']!={'kind':'same_continuous_odom','externally_verified':True}:raise ValueError('prefix requires externally verified same continuous odom')
        for key,hkey in [('path','sha256'),('annotation_receipt','annotation_receipt_sha256')]:
            if hashlib.sha256(Path(d[key]).read_bytes()).hexdigest()!=d[hkey]:raise ValueError('prefix '+key+' hash mismatch')
        receipt=json.loads(Path(d['annotation_receipt']).read_text())
        if receipt.get('copied_packet_sha256')!=d['sha256']:raise ValueError('annotation receipt does not bind prefix')
        result.append(dict(d))
    return result

def decode(session,items,live):
    from nso_bridge.joint_engine import PhysicalPacket,_rigid
    from utils.rgbd_contract import RGBDFrame,PlanarScan
    from nso_bridge.sensor_range_adapter import adapt_scan
    packets=[];seen={live.frame_id};last=-float('inf')
    for d in descriptors(items):
        raw=Path(d['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=d['sha256']:raise ValueError('prefix changed before bootstrap')
        with np.load(d['path'],allow_pickle=False) as z:
            meta=json.loads(str(z['joint_metadata_json'].item()))
            if meta.get('scope')!='real_observed_joint_input' or meta.get('geometry_world_frame')!='odom':raise ValueError('prefix geometry_world_frame must be odom')
            fid=str(meta['frame_id']);stamp=float(z['rgbd_timestamp_s']);scan_stamp=float(z['scan_timestamp_s'])
            if fid in seen or not last<stamp<live.frame.timestamp_s or not scan_stamp<live.scan.timestamp_s:raise ValueError('prefix must have unique chronological frames before live0')
            seen.add(fid);last=stamp
            labels=z['semantic'].copy();depth=z['depth'].copy()
            if labels.shape!=depth.shape or not np.issubdtype(labels.dtype,np.integer) or not np.isin(labels,[0,2,3]).all() or np.any((labels>0)&(~np.isfinite(depth)|(depth<=0))):raise ValueError('prefix native labels require measured valid depth')
            # Source navigation grid and old planning transform are deliberately never read.
            frame=RGBDFrame(stamp,depth,z['color_rgb'].copy(),z['intrinsic'].copy(),session.planning_from_odom@_rigid(z['world_from_camera'],'prefix camera'),labels)
            scan=PlanarScan(scan_stamp,z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),session.planning_from_odom@_rigid(z['world_from_laser'],'prefix laser'))
            scan,_=adapt_scan(scan,session.sensor_range_profile)
            base=z['base_odom_pose'];point=session.planning_from_odom@[float(base[0]),float(base[1]),0,1];angle=float(base[2])+math.atan2(session.planning_from_odom[1,0],session.planning_from_odom[0,0]);heading=int(math.floor((math.pi/2-angle)/(math.pi/2)+.5))%4
            packet=PhysicalPacket(scene_id=live.scene_id,episode_id=live.episode_id,frame_id=fid,action_id=0,frame=frame,scan=scan,position=session.transform.world_to_cell(point[:2]),heading=heading,sensor_source='audited_native_category_measured_RGBD_scan_history',pose_source='historical_measured_odom_not_current_state',base_pose_planning=(float(point[0]),float(point[1]),angle),observed_grid=np.full(session.shape,-1,np.int8),source_file_sha256=d['sha256'])
            packet.validate(session.transform,session.config);packets.append(packet)
    return packets
