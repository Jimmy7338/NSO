#!/usr/bin/env python3
"""Freeze plane profiles from one predeclared training packet; no heldout access."""
import json,pathlib,hashlib,numpy as np
D=pathlib.Path(__file__).resolve().parent;CAP=pathlib.Path('/home/ouc/nso_robot/analyses/evidence_first_20261108/sensor/current_capture');rec=json.loads((CAP/'receipt.json').read_text());f=pathlib.Path(rec['frames'][0]['path']);digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert digest(f)==rec['frames'][0]['sha256'];tf=pathlib.Path('/home/ouc/nso_robot/analyses/calibrated_mount_20261108/tf_verification.json');new=np.asarray(json.loads(tf.read_text())['expected_base_from_camera_optical']);old=np.array([[0,0,1,.061],[-1,0,0,.0591],[0,-1,0,.303],[0,0,0,1.]])
with np.load(f,allow_pickle=False) as z:depth=z['depth_m'].copy();K=z['intrinsic'].copy();receipt=json.loads(str(z['receipt_json']));base=np.asarray(receipt['historical_tf']['odom_from_base']['matrix'])
u,v=np.meshgrid(np.arange(360,1100,4),np.arange(480,710,4));uv=np.column_stack([u.ravel(),v.ravel()]);dist=depth[uv[:,1],uv[:,0]];ok=np.isfinite(dist)&(dist>0);uv=uv[ok];dist=dist[ok];ray=np.linalg.inv(K)@np.column_stack([uv,np.ones(len(uv))]).T;pc=(ray*(dist/ray[2])).T
def fit(q):
 rng=np.random.default_rng(0);best=None;count=-1
 for _ in range(120):
  x=q[rng.choice(len(q),3,replace=False)];n=np.cross(x[1]-x[0],x[2]-x[0]);norm=np.linalg.norm(n)
  if norm<1e-12:continue
  n/=norm;mask=np.abs(q@n-n@x[0])<.01
  if mask.sum()>count:best=mask;count=int(mask.sum())
 center=q[best].mean(0);_,_,v=np.linalg.svd(q[best]-center,full_matrices=False);n=v[-1];n=n if n[2]>0 else -n;d=-float(n@center);return n,d,count
out={}
for name,mount in [('legacy',old),('nov06',new)]:
 T=base@mount;q=pc@T[:3,:3].T+T[:3,3];n,d,count=fit(q);zbase=-(n[0]*base[0,3]+n[1]*base[1,3]+d)/n[2];out[name]={'base_from_camera':mount.tolist(),'normal_odom':n.tolist(),'plane_d_odom_m':d,'floor_at_base_odom_m':float(zbase),'floor_relative_base_m':float(zbase-base[2,3]),'slope_deg':float(np.rad2deg(np.arccos(np.clip(n[2],-1,1)))),'training_inliers':count,'training_points':len(q)}
path=D/'frozen_profiles.json';assert not path.exists();path.write_text(json.dumps({'status':'frozen_before_heldout_evaluation_not_deployed','protocol_sha256':digest(D/'protocol.json'),'train_packet_sha256':digest(f),'source_tf_receipt_sha256':digest(tf),'profiles':out,'limits':['Same raw training frame and ROI; Nov06 is fixed-mount reprojection, not an actual Nov06 measurement at this moment.','Not independent physical truth or validated extrinsics.']},indent=2)+'\n');print(json.dumps(out))
