"""Measured ROS occupancy and RGB-D binding for the joint exploration input.

Unknown cells remain unknown. This module does not supply semantic labels,
surface ground truth, candidates, controller decisions, or motion commands.
ROS imports/subscriptions occur only when constructing the collector.
"""
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np

from visual_sensor_bridge import pose_matrix


def _header(message):
    return dict(seq=int(message.header.seq),stamp_ns=int(message.header.stamp.to_nsec()),
                frame_id=message.header.frame_id,
                publisher=getattr(message,'_connection_header',{}).get('callerid'))


def _transform(transform):
    t,q=transform.translation,transform.rotation
    return pose_matrix([t.x,t.y,t.z],[q.x,q.y,q.z,q.w])


def freeze_joint_grid(base_odom_pose,shape=(400,400),floor_odom_z_m=-.233):
    """One fixed frame: the actual initial heading becomes original north/+y."""
    x,y,yaw=np.asarray(base_odom_pose,dtype=float)
    if not np.isfinite([x,y,yaw,floor_odom_z_m]).all():raise ValueError('finite actual initial pose required')
    height,width=shape;row,col=height//2,width//2
    angle=np.pi/2-yaw;c,s=np.cos(angle),np.sin(angle)
    transform=np.eye(4);transform[:2,:2]=[[c,-s],[s,c]]
    transform[:2,3]=np.array([(col+.5)*.2,(height-row-.5)*.2])-transform[:2,:2]@[x,y]
    transform[2,3]=-floor_odom_z_m
    return dict(shape=list(shape),resolution_m=.2,planning_from_odom=transform.tolist(),
                return_anchor=[row,col,0],floor_odom_z_m=floor_odom_z_m,
                planning_domain_is_resource_extent_not_observed_world=True,
                initial_actual_base_odom_pose=[float(x),float(y),float(yaw)])


def canonical_sensor_arrays(source,receipt,*,pixel_stride=4,max_depth_m=4.,self_mask=None):
    """Measured original pixels/rays only; actual asynchronous stamps remain."""
    if pixel_stride not in (1,2,4,8) or not np.isfinite(max_depth_m) or max_depth_m<=0:
        raise ValueError('explicit valid joint sensor sampling configuration required')
    raw_depth=np.asarray(source['depth_m']);rgb=np.asarray(source['rgb_raw'])
    if raw_depth.ndim!=2 or rgb.shape!=raw_depth.shape+(3,) or rgb.dtype!=np.uint8:
        raise ValueError('aligned original measured RGB-D required')
    depth=np.array(raw_depth,dtype=np.float32,copy=True)
    with np.errstate(invalid='ignore'):
        invalid=~np.isfinite(depth)|(depth<=0)|(depth>max_depth_m)
    if self_mask is not None:
        if self_mask.shape!=depth.shape:raise ValueError('approved self-mask shape mismatch')
        invalid|=self_mask
    depth[invalid]=0.
    K=np.asarray(source['intrinsic'],dtype=float).copy();K[:2]/=pixel_stride
    scan=receipt['source_geometry']['scan'];ranges=np.array(source['scan_ranges_raw'],dtype=np.float32,copy=True)
    with np.errstate(invalid='ignore'):
        invalid_scan=~np.isfinite(ranges)|(ranges<scan['range_min_m'])|(ranges>scan['range_max_m'])
    ranges[invalid_scan]=np.nan
    return dict(depth=np.ascontiguousarray(depth[::pixel_stride,::pixel_stride]),
        color_rgb=np.ascontiguousarray(rgb[::pixel_stride,::pixel_stride]),intrinsic=K,
        intrinsic_raw=np.asarray(source['intrinsic']).copy(),ranges=ranges,
        rgbd_timestamp_s=receipt['headers']['depth']['stamp_ns']/1e9,
        scan_timestamp_s=receipt['headers']['scan']['stamp_ns']/1e9,
        world_from_laser=np.asarray(receipt['historical_tf']['odom_from_laser']['matrix'],dtype=float),
        angle_min=scan['angle_min_rad'],angle_increment=scan['angle_increment_rad'],range_max=scan['range_max_m'])


def occupancy_arrays(message):
    info=message.info
    data=np.asarray(message.data)
    if (info.width<1 or info.height<1 or data.size!=info.width*info.height or
            not np.issubdtype(data.dtype,np.integer) or np.any(data < -1) or np.any(data>100) or
            not np.isfinite(info.resolution) or info.resolution<=0):
        raise ValueError('invalid original OccupancyGrid geometry or values')
    p,q=info.origin.position,info.origin.orientation
    origin=pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
    return dict(occupancy=data.astype(np.int8).reshape(info.height,info.width),
                resolution_m=np.asarray(info.resolution),map_from_grid=origin,
                origin_xyz=np.asarray([p.x,p.y,p.z]),origin_xyzw=np.asarray([q.x,q.y,q.z,q.w]))


def conservative_belief(occupancy, raw_resolution_m, planning_from_grid, shape,
                        planning_resolution_m=.2):
    """Exact positive-area square overlap, with no unknown-to-free conversion.

    ROS source rows run along +y; destination rows are reversed, as in the
    original joint planner. Any overlapping occupied cell -> occupied. Free
    requires every overlapping source cell to be in-bounds and exactly zero.
    Unknown, out-of-map and intermediate probabilities otherwise stay unknown.
    No nearest-neighbour interpolation or morphological safety inflation.
    """
    source=np.asarray(occupancy);T=np.asarray(planning_from_grid,dtype=float)
    if (source.ndim!=2 or not np.issubdtype(source.dtype,np.integer) or
            np.any(source < -1) or np.any(source>100)):
        raise ValueError('raw ROS occupancy values required')
    if (len(shape)!=2 or any(type(n) is not int or n<1 for n in shape) or
            not np.isfinite(raw_resolution_m) or raw_resolution_m<=0 or
            not np.isclose(planning_resolution_m,.2,rtol=0,atol=1e-12)):
        raise ValueError('declared fixed shape and original 0.2 m joint grid required')
    if (T.shape!=(4,4) or not np.isfinite(T).all() or
            not np.allclose(T[3],[0,0,0,1]) or
            not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-5) or
            not np.allclose(T[2,:2],0.,atol=1e-6) or not np.isclose(T[2,2],1.,atol=1e-6)):
        raise ValueError('actual planar rigid map/planning binding required')
    inv=np.linalg.inv(T);height,width=shape
    row,col=np.indices(shape)
    cx=(col.ravel()+.5)*planning_resolution_m
    cy=(height-row.ravel()-.5)*planning_resolution_m
    x=inv[0,0]*cx+inv[0,1]*cy+inv[0,3]
    y=inv[1,0]*cx+inv[1,1]*cy+inv[1,3]
    u=inv[:2,0];v=inv[:2,1];half=planning_resolution_m*.5;finehalf=raw_resolution_m*.5
    extent=half*(np.abs(u)+np.abs(v))
    bound_u=half+finehalf*np.abs(u).sum();bound_v=half+finehalf*np.abs(v).sum()
    ix=np.floor(x/raw_resolution_m).astype(np.int64)
    iy=np.floor(y/raw_resolution_m).astype(np.int64)
    free=np.ones(x.shape,bool);occupied=np.zeros(x.shape,bool);covered=np.zeros(x.shape,bool)
    radius=np.ceil((extent+finehalf)/raw_resolution_m).astype(int)+1
    epsilon=1e-10
    for dy in range(-int(radius[1]),int(radius[1])+1):
        sy=iy+dy;delta_y=(sy+.5)*raw_resolution_m-y
        for dx in range(-int(radius[0]),int(radius[0])+1):
            sx=ix+dx;delta_x=(sx+.5)*raw_resolution_m-x
            overlap=((np.abs(delta_x)<extent[0]+finehalf-epsilon)&
                     (np.abs(delta_y)<extent[1]+finehalf-epsilon)&
                     (np.abs(delta_x*u[0]+delta_y*u[1])<bound_u-epsilon)&
                     (np.abs(delta_x*v[0]+delta_y*v[1])<bound_v-epsilon))
            inside=(sx>=0)&(sx<source.shape[1])&(sy>=0)&(sy<source.shape[0])
            values=source[np.clip(sy,0,source.shape[0]-1),np.clip(sx,0,source.shape[1]-1)]
            free &= ~(overlap&(~inside|(values!=0)))
            occupied |= overlap&inside&(values==100)
            covered |= overlap
    belief=np.full(x.shape,-1,np.int8);belief[covered&free]=0;belief[occupied]=1
    return belief.reshape(shape)


class JointObservationCollector:
    def __init__(self, tfbuf, joint_grid, topic='/nso_visual/map', publisher='/slam_gmapping',
                 *,pixel_stride=4,max_depth_m=4.,apply_approved_self_mask=True):
        import rospy
        from nav_msgs.msg import OccupancyGrid
        self.tfbuf=tfbuf;self.joint_grid=json.loads(json.dumps(joint_grid))
        self.topic=topic;self.publisher=publisher;self.lock=threading.Lock();self.latest=None
        self.pixel_stride=pixel_stride;self.max_depth_m=max_depth_m;self.self_mask=None
        self.self_mask_receipt=None
        if apply_approved_self_mask:
            from visual_sensor_bridge import load_approved_mask
            self.self_mask,self.self_mask_receipt=load_approved_mask()
        self.subscriber=rospy.Subscriber(topic,OccupancyGrid,self._receive,queue_size=1)

    def _receive(self,message):
        with self.lock:self.latest=(time.monotonic(),message)

    def close(self):self.subscriber.unregister()

    def capture(self,output_directory,observation_path,heartbeat=lambda:None,*,source_role='current_observation'):
        import rospy
        deadline=time.monotonic()+5.
        while True:
            heartbeat()
            with self.lock:row=self.latest
            if row is not None:break
            if time.monotonic()>=deadline:raise RuntimeError('real OccupancyGrid unavailable')
            time.sleep(.02)
        received,message=row
        header=_header(message)
        if header['publisher']!=self.publisher or header['frame_id']!='map':
            raise ValueError('unexpected original OccupancyGrid source/frame')
        arrays=occupancy_arrays(message)
        # Freeze TF at one real base-odometry sample time; no hypothetical pose.
        odom_base=self.tfbuf.lookup_transform('odom','base_link',rospy.Time(0),rospy.Duration(.3))
        pose_stamp=odom_base.header.stamp
        map_odom=self.tfbuf.lookup_transform('map','odom',pose_stamp,rospy.Duration(.3))
        map_from_odom=_transform(map_odom.transform)
        planning_from_odom=np.asarray(self.joint_grid['planning_from_odom'],dtype=float)
        planning_from_grid=planning_from_odom@np.linalg.inv(map_from_odom)@arrays['map_from_grid']
        belief=conservative_belief(arrays['occupancy'],float(arrays['resolution_m']),
            planning_from_grid,self.joint_grid['shape'],self.joint_grid['resolution_m'])
        raw_path=Path(observation_path).resolve();source_bytes=raw_path.read_bytes()
        with np.load(io.BytesIO(source_bytes),allow_pickle=False) as data:
            source={name:data[name].copy() for name in data.files}
        receipt=json.loads(str(source['receipt_json']))
        historical=receipt['historical_tf'];base=np.asarray(historical['odom_from_base']['matrix'])
        base_pose=np.array([base[0,3],base[1,3],np.arctan2(base[1,0],base[0,0])])
        directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
        identity=str(source['frame_id']);map_path=directory/('map-'+identity+'.npz')
        path=directory/('joint-'+identity+'.npz')
        if map_path.exists() or path.exists():raise FileExistsError('joint observation source already frozen')
        metadata=dict(schema='nso_joint_real_observation/1',scope='real_observed_joint_input',source_role=source_role,
            frame_id=identity,planning_from_odom=planning_from_odom.tolist(),source_topic=self.topic,
            source_header=header,map_load_time_ns=message.info.map_load_time.to_nsec(),
            map_receipt_monotonic=received,decision_capture_monotonic=time.monotonic(),
            map_source_age_s=rospy.Time.now().to_sec()-message.header.stamp.to_sec(),
            occupancy_semantics=dict(unknown=-1,free=0,occupied=100,
                intermediate_probability='retained raw; coarse cell remains unknown unless an occupied cell overlaps'),
            raw_row_order='ROS +y bottom to top; data[row_y,col_x]',
            planning_row_order='original joint top to bottom; y=(height-row-.5)*0.2',
            resampling='positive-area overlapping source squares: occupied if any100; free only if all0 inbounds; else unknown',
            joint_grid=self.joint_grid,planning_from_grid=planning_from_grid.tolist(),
            current_map_from_odom=map_from_odom.tolist(),current_map_from_odom_header=_header(map_odom),
            current_odom_from_base=_transform(odom_base.transform).tolist(),
            current_odom_from_base_header=_header(odom_base),
            rgbd_source_path=str(raw_path),rgbd_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            rgbd_source_stamp_ns=receipt['headers']['depth']['stamp_ns'],
            scan_source_stamp_ns=receipt['headers']['scan']['stamp_ns'],
            map_rgbd_stamp_delta_s=(header['stamp_ns']-receipt['headers']['depth']['stamp_ns'])/1e9,
            source_stamps_are_independent=True,pixel_stride=self.pixel_stride,max_depth_m=self.max_depth_m,
            depth_invalid_policy='nonfinite_nonpositive_or_over_declared_max_to_zero',
            scan_invalid_policy='nonfinite_or_outside_original_sensor_limits_to_nan_not_max_range',
            approved_self_mask_applied=self.self_mask is not None,
            approved_self_mask_receipt=self.self_mask_receipt,
            base_odom_pose_source='original RGBD historical odom_from_base TF; not ideal graph pose',
            geometry_world_frame='odom',semantic_labels_invented=False,truth_map_used=False,
            surface_coverage_not_available_from_occupancy_alone=True)
        serialized=io.BytesIO();message.serialize(serialized)
        np.savez_compressed(map_path,**arrays,
            map_received_ros_serialized=np.frombuffer(serialized.getvalue(),dtype=np.uint8),
            map_from_odom=map_from_odom,receipt_json=json.dumps(metadata,sort_keys=True,allow_nan=False))
        metadata.update(raw_map_path=str(map_path.resolve()),
                        raw_map_sha256=hashlib.sha256(map_path.read_bytes()).hexdigest())
        source.update(canonical_sensor_arrays(source,receipt,pixel_stride=self.pixel_stride,
                                               max_depth_m=self.max_depth_m,self_mask=self.self_mask))
        # The original reference collector's template sampling hash describes
        # its own source file, not this explicitly different joint sampling.
        for key in ('calibration_sha256','calibration_json'):
            if key in source:source['source_reference_'+key]=source.pop(key)
        source.update(base_odom_pose=base_pose,joint_belief_grid=belief,
            joint_metadata_json=json.dumps(metadata,sort_keys=True,allow_nan=False))
        np.savez_compressed(path,**source)
        return str(path.resolve())
