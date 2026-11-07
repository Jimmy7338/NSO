#!/usr/bin/env python3
"""Read-only analysis of recorded RGB-D packets. Does not connect to ROS."""
from pathlib import Path
import hashlib, json, os
import numpy as np

HERE = Path(__file__).resolve().parent
PACKETS = Path(os.environ['NSO_PACKET_DIR'])
EXPECTED = {'joint-1793776658846936340.npz': 'c16a1e474df92557a9d16ded9ad77617e37ac6f479a448440b65b8fc7dc03b40', 'joint-1793776692682644340.npz': '4389c43b1c56354995f6174a9ae390d41962d9ab90aa8894e3142f494fe7efb5', 'joint-1793776703049559340.npz': 'c0ab7f0b06ee1d569d174587ba3e1afe6a8a5588ab083998cb6f90cdf5785442', 'joint-1793776715217130340.npz': 'b4f8b9fab18ef8c20ccc43412819c865a8d5111590087b6497c9a130294803cb', 'joint-1793776727750982340.npz': '25338aaab5396e31723066d13ef4320a41aa51b9679891c72351d76a80558f64', 'joint-1793776739951713340.npz': '2b703aebf162cf08cd53a7e8b4b036a9b3b4753ae097856db8313a9ed811d070', 'joint-1793776752152452340.npz': '9239973ccc82223acecabdb3f3433b9a1173cf3c2eabb1fce3e6da1ccb654d5e', 'joint-1793776764186671340.npz': '69ee592c3f26ae9efeb2d1f87fc52fafe7dc50b3f4795182ec21a1e86ddc05aa'}
ROIS = {3: (610,526,645,550), 4: (608,550,650,575), 5: (580,576,630,607), 6: (535,610,595,643)}
files = sorted(PACKETS.glob('joint-*.npz'))
if len(files) != 8:
    raise RuntimeError(f'Expected eight chronological packets, found {len(files)}')
packets = []
for path in files:
    with path.open('rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    if EXPECTED.get(path.name) != digest: raise RuntimeError(f'Packet name/hash mismatch: {path.name}')
    packets.append({'filename': path.name, 'sha256': digest})
rows = []
poses = {}
for action, roi in ROIS.items():
    with np.load(files[action], allow_pickle=False) as z:
        x1,y1,x2,y2 = roi
        depth = z['depth_m'][y1:y2, x1:x2]
        rgb = z['rgb_raw'][y1:y2, x1:x2]
        yy,xx = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 3) & (rgb.mean(axis=2) < 100)
        yy,xx,ds = yy[valid]+y1,xx[valid]+x1,depth[valid]
        if not len(ds): raise RuntimeError(f'No usable points for action {action}')
        K = z['intrinsic_raw']; T = z['world_from_camera']
        q = np.stack(((xx-K[0,2])*ds/K[0,0],(yy-K[1,2])*ds/K[1,1],ds,np.ones(len(ds))),axis=1)
        world = q @ T.T
        poses[action] = z['base_odom_pose'].copy()
        rows.append({'action':action,'roi_xyxy_exclusive':list(roi),'valid_pixels':len(ds),'depth_median_m':float(np.median(ds)), 'transformed_median_xyz_m':np.median(world[:,:3],axis=0).tolist(),'intrinsic_raw':K.tolist(),'world_from_camera':T.tolist(),'base_odom_pose':poses[action].tolist()})
xy = np.array([r['transformed_median_xyz_m'][:2] for r in rows])
max_span = max(float(np.linalg.norm(a-b)) for a in xy for b in xy)
move = float(np.linalg.norm(poses[6][:2]-poses[3][:2]))
result = {'scope':'offline saved-packet diagnostic; no ROS, motion, or mask changes','packet_directory':str(PACKETS),'packets':packets,'frame_order':'filename timestamps ascending; frame index equals recorded action 0–7','filter':'manual RGB ROI; per-pixel RGB arithmetic mean < 100; finite 0 < depth_m < 3 m','projection':'intrinsic_raw and world_from_camera stored in each same NPZ; median per-pixel transformed XYZ; base_odom_pose used for planar displacement','rows':rows,'base_odom_planar_displacement_action_3_to_6_m':move,'transformed_median_xy_max_pair_distance_m':max_span,'interpretation':'Frames 3–6 are more consistent with a world-fixed object than a camera-fixed object; manual ROI correspondence and depth/pose error are not independently validated. Frame 7 is partially occluded. The single conflicting depth sample cannot reliably be bound to the same surface or rule out depth-edge error; identity and safety remain unconfirmed.'}
(HERE/'result_recomputed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'rows':[{'action':r['action'],'valid_pixels':r['valid_pixels'],'depth_median_m':r['depth_median_m'],'transformed_median_xyz_m':r['transformed_median_xyz_m']} for r in rows], 'base_displacement_m':move,'world_xy_max_span_m':max_span},ensure_ascii=False,indent=2))
