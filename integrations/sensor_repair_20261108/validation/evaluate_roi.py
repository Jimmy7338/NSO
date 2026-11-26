#!/usr/bin/env python3
"""Evaluate predeclared RGB ROIs in heldout raw frames; no ROS or policy."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json,pathlib,hashlib
import numpy as np
D=pathlib.Path(__file__).resolve().parent;P=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/legacy_mount_S40_20261108_01');proto=json.loads((D/'protocol.json').read_text());profile=pathlib.Path('/home/ouc/nso_robot/analyses/evidence_first_20261108/publication_stage/sensor/execution_frozen_profile.json');fp=json.loads(profile.read_text());config=json.loads((P/'joint_configuration.json').read_text());T=np.asarray(config['configuration']['planning_from_odom']);rows=[json.loads(x) for x in (P/'joint/decisions.jsonl').read_text().splitlines()];out=[]
for aid in (0,18,22):
 rec=next(x for x in rows if x['request']['method']=='observe' and x['request']['params']['action_id']==aid);path=pathlib.Path(rec['request']['params']['path']);b=path.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==rec['response']['result']['summary']['semantic_diagnostics']['receipt']['source_file_sha256']
 with np.load(path,allow_pickle=False) as z:depth=z['depth_m'].copy();K=z['intrinsic_raw'].copy();cam=z['world_from_camera'].copy();meta=json.loads(str(z['receipt_json']));base=np.asarray(meta['historical_tf']['odom_from_base']['matrix']);
 delta=-(base[2,3]+fp['floor_relative_to_training_base_m'])-T[2,3];v={}
 for name,rect in proto['rectangles_preview_xyxy'][str(aid)].items():
  x0,y0,x1,y1=[int(round(x*4/3)) for x in rect];u,y=np.meshgrid(np.arange(x0,x1,4),np.arange(y0,y1,4));dep=depth[y,u];good=np.isfinite(dep)&(dep>0);u=u[good];y=y[good];dep=dep[good];rays=np.linalg.inv(K)@np.stack([u,y,np.ones(len(u))]);xyz=(rays*(dep/rays[2])).T;world=xyz@cam[:3,:3].T+cam[:3,3];z=world@T[:3,:3].T+T[:3,3];h=z[:,2];h2=h+delta;inband=lambda q:(q>.15)&(q<1.2)
  v[name]=dict(valid_samples=len(h),baseline_median_z_m=float(np.median(h)),candidate_median_z_m=float(np.median(h2)),candidate_z_p05_p95_m=np.quantile(h2,[.05,.95]).tolist(),baseline_in_band=int(inband(h).sum()),candidate_in_band=int(inband(h2).sum()),baseline_band_to_candidate_out=int((inband(h)&~inband(h2)).sum()))
 out.append(dict(action_id=aid,packet_sha256=sha,delta_z_m=float(delta),regions=v))
(D/'roi_result.json').write_text(json.dumps({'schema':'nso.heldout_visual_regions.v1','frames':out,'protocol_sha256':hashlib.sha256((D/'protocol.json').read_bytes()).hexdigest(),'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),'limits':['Visible face/floor categories are RGB ROI review, not pixel ground truth or independent height reference.','Heldout poses from one trajectory in one scene.','No obstacle preservation or robot safety certification.']},indent=2)+'\n');print(json.dumps(out))
