#!/usr/bin/env python3
"""Evaluate untouched heldouts using pre-frozen training profiles."""
import json,pathlib,hashlib,xml.etree.ElementTree as ET,numpy as np
from scipy.spatial.transform import Rotation
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();f=D/'frozen_profiles.json';frozen=json.loads(f.read_text());profiles=frozen['profiles'];legacy_roi=json.loads(pathlib.Path('/home/ouc/nso_robot/analyses/sensor_repair_20261108/validation/protocol.json').read_text());CAP=pathlib.Path('/home/ouc/nso_robot/analyses/evidence_first_20261108/sensor/current_capture');receipt=json.loads((CAP/'receipt.json').read_text());backup=pathlib.Path('/home/ouc/nso_robot/analyses/camera_mount_rollback_20261108/hybrid_camera_neural.calibrated.before.launch');args={x.attrib['name']:float(x.attrib['value']) for x in ET.parse(backup).getroot().iter('arg') if x.attrib.get('name') in ['cam_pos_x','cam_pos_y','cam_pos_z','cam_roll','cam_pitch','cam_yaw']};R=Rotation.from_euler('xyz',[args['cam_roll'],args['cam_pitch'],args['cam_yaw']]).as_matrix();M=np.eye(4);M[:3,:3]=R@np.array([[0,0,1],[-1,0,0],[0,-1,0]]);M[:3,3]=np.array([args['cam_pos_x'],args['cam_pos_y'],args['cam_pos_z']])+R@np.array([-.01,.06,.015]);assert np.allclose(M,profiles['nov06']['base_from_camera'],atol=1e-12,rtol=0)
RPT=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/legacy_mount_S40_20261108_01');rows=[json.loads(x) for x in (RPT/'joint/decisions.jsonl').read_text().splitlines()];entries=[]
for row in receipt['frames'][1:]:entries.append(('stationary_'+str(row['index']),pathlib.Path(row['path']),{'floor':[270,360,825,532]},False))
for aid in (0,18,22):
 x=next(x for x in rows if x['request']['method']=='observe' and x['request']['params']['action_id']==aid);entries.append(('paid_'+str(aid),pathlib.Path(x['request']['params']['path']),legacy_roi['rectangles_preview_xyxy'][str(aid)],True))
output=[]
for name,path,regions,moving in entries:
 with np.load(path,allow_pickle=False) as z:depth=z['depth_m'].copy();K=z['intrinsic_raw' if 'intrinsic_raw' in z.files else 'intrinsic'].copy();meta=json.loads(str(z['receipt_json']));base=np.asarray(meta['historical_tf']['odom_from_base']['matrix'])
 row={'frame':name,'packet_sha256':sha(path),'mounts':{}}
 for mount in ['legacy','nov06']:
  prof=profiles[mount];T=base@np.asarray(prof['base_from_camera']);results={}
  for region,rect in regions.items():
   if not moving: u,v=np.meshgrid(np.arange(360,1100,4),np.arange(480,710,4))
   else:
    x0,y0,x1,y1=[int(round(x*4/3)) for x in rect];u,v=np.meshgrid(np.arange(x0,x1,4),np.arange(y0,y1,4))
   dist=depth[v,u];good=np.isfinite(dist)&(dist>0);uu=u[good];vv=v[good];dist=dist[good];ray=np.linalg.inv(K)@np.stack([uu,vv,np.ones(len(uu))]);pc=(ray*(dist/ray[2])).T;world=pc@T[:3,:3].T+T[:3,3];height=world[:,2]-base[2,3]-prof['floor_relative_base_m'];band=(height>.15)&(height<1.2)
   info={'valid':len(height),'scalar_ground_height_median_m':float(np.median(height)),'scalar_ground_height_p05_p95_m':np.quantile(height,[.05,.95]).tolist(),'original_band_count':int(band.sum())}
   if not moving:
    n=np.asarray(prof['normal_odom']);res=np.abs(world@n+prof['plane_d_odom_m']);info.update(frozen_plane_median_abs_residual_m=float(np.median(res)),frozen_plane_p95_abs_residual_m=float(np.percentile(res,95)),frozen_plane_inlier_1cm_fraction=float(np.mean(res<.01)))
   results[region]=info
  row['mounts'][mount]=results
 output.append(row)
# Forecast sensor heights relative to candidate floor; actual laser z is independent of camera mount.
with np.load(pathlib.Path(receipt['frames'][0]['path']),allow_pickle=False) as z:meta=json.loads(str(z['receipt_json']));base=np.asarray(meta['historical_tf']['odom_from_base']['matrix']);laser=np.asarray(meta['historical_tf']['odom_from_laser']['matrix']);base_laser=np.linalg.inv(base)@laser
forecast={name:{'camera_optical_origin_height_m':float(np.asarray(p['base_from_camera'])[2,3]-p['floor_relative_base_m']),'laser_origin_height_m':float(base_laser[2,3]-p['floor_relative_base_m'])} for name,p in profiles.items()}
out={'schema':'nso.fixed_nov06_mount_ground_heldout.v1','frozen_profile_sha256':sha(f),'source_backup_sha256':sha(backup),'mount_composition_max_abs_error':float(np.max(np.abs(M-np.asarray(profiles['nov06']['base_from_camera'])))),'frames':output,'forecast_height_geometry_only':forecast,'baseline_nominal_camera_m':.536,'baseline_nominal_laser_m':.684,'interpretation':'Potential forecast sensor-height correction is a separate planner input adaptation: camera_pose/aperture/quality prediction may change, requires disclosed counterfactual and validation. Scalar heights alone do not encode camera pitch, translation or full extrinsics.','limits':['Heldout stationary frames share one pose; moving frames one trajectory and one scene.','Nov06 images are reprojected under fixed backup TF; no physical truth or independent depth/extrinsic measurement.','Visible white-face ROIs are sampled, not complete obstacle-preservation or safety proof.','No threshold, filter, scoring, TF or live change.']};(D/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'forecast':forecast,'frames':output}))
