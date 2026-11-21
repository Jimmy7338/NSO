#!/usr/bin/env python3
"""Fit one frozen floor-reference candidate; fixed ROI, no ROS or policy."""
import argparse,hashlib,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--train',required=True,type=pathlib.Path);p.add_argument('--heldout',nargs='+',required=True,type=pathlib.Path);p.add_argument('--epoch',required=True);p.add_argument('--camera-transform-mode',choices=('captured','old_mount_counterfactual'),required=True);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
paths=[a.train.resolve(),*[x.resolve() for x in a.heldout]]
if len(set(paths))!=len(paths):raise ValueError('training and heldout files must be distinct')
if a.output.exists():raise FileExistsError(a.output)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=np.eye(4);old[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];old[:3,3]=[.061,.0591,.303]
def points(path):
 with np.load(path,allow_pickle=False) as z:
  depth=z['depth_m'].copy();K=z['intrinsic_raw' if 'intrinsic_raw' in z.files else 'intrinsic'].copy();T=z['world_from_camera'].copy();receipt=json.loads(str(z['receipt_json']))
  if a.camera_transform_mode=='old_mount_counterfactual':T=np.array(receipt['historical_tf']['odom_from_base']['matrix'])@old
 if depth.shape!=(720,1280):raise ValueError('fixed ROI requires 1280x720 raw aligned depth')
 u,v=np.meshgrid(np.arange(360,1100,4),np.arange(480,710,4));uv=np.column_stack([u.ravel(),v.ravel()]);d=depth[uv[:,1],uv[:,0]];good=np.isfinite(d)&(d>0);uv=uv[good];d=d[good]
 ray=np.linalg.inv(K)@np.column_stack([uv,np.ones(len(uv))]).T;xyz=(ray*(d/ray[2])).T
 base=np.asarray(receipt['historical_tf']['odom_from_base']['matrix'],dtype=float)
 return xyz@T[:3,:3].T+T[:3,3],base
def fit(q):
 rng=np.random.default_rng(0);best=None;count=-1
 for _ in range(120):
  v=q[rng.choice(len(q),3,replace=False)];n=np.cross(v[1]-v[0],v[2]-v[0]);l=np.linalg.norm(n)
  if l<1e-12:continue
  n/=l;mask=np.abs(q@n-n@v[0])<.01
  if mask.sum()>count:count=int(mask.sum());best=mask
 if best is None:raise ValueError('degenerate ROI')
 center=q[best].mean(axis=0);_,_,v=np.linalg.svd(q[best]-center,full_matrices=False);n=v[-1];n=n if n[2]>0 else -n;d=-float(n@center)
 if n[2]<.95:raise ValueError('plane too steep for scalar floor profile')
 return n,d,count
train,base=points(paths[0]);n,d,count=fit(train);floor_at_base=-(n[0]*base[0,3]+n[1]*base[1,3]+d)/n[2];base_relative_floor=float(floor_at_base-base[2,3]);held=[]
for f in paths[1:]:
 q,_=points(f);res=np.abs(q@n+d);held.append(dict(path=str(f),sha256=sha(f),points=len(q),median_abs_residual_m=float(np.median(res)),p95_abs_residual_m=float(np.percentile(res,95)),within_1cm_fraction=float(np.mean(res<.01))))
result=dict(schema='nso.floor_reference_candidate.v1',status='offline_internal_consistency_only_not_physical_truth_not_deployed',epoch=a.epoch,camera_transform_mode=a.camera_transform_mode,source_training=dict(path=str(paths[0]),sha256=sha(paths[0]),points=len(train),ransac_inliers=count),heldout=held,protocol=dict(crop_uv=[360,480,1100,710],stride=4,seed=0,ransac_trials=120,tolerance_m=.01,refinement='SVD over highest count, first tie retained',training_selection='caller predeclared; no anchor/safety criterion',heldout_selection='distinct caller predeclared frames; no fit updates'),normal_odom=n.tolist(),plane_d_odom_m=d,floor_odom_z_at_origin_m=float(-d/n[2]),floor_odom_z_at_training_base_m=float(floor_at_base),floor_relative_to_training_base_m=base_relative_floor,scope='Fixed visually hypothesized floor ROI; scalar horizontal approximation at odom origin; slope and depth/extrinsic errors remain. No claim of independent true ground or current physical mount accuracy.',old_mount_matrix=old.tolist() if a.camera_transform_mode=='old_mount_counterfactual' else None)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'candidate_floor_relative_to_base_m':result['floor_relative_to_training_base_m'],'heldout':held}))
