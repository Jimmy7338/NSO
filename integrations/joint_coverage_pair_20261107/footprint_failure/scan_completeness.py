#!/usr/bin/env python3
"""Two existing raw scans only; no ROS/capture/replay or range filtering."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path('/home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/publication/S');D=Path(__file__).resolve().parent

def runs(mask):
    delta=np.diff(np.r_[False,mask,False].astype(int));return [(int(a),int(b-1)) for a,b in zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1))]
def containing(mask,i):
    return next((dict(start=a,end=b,length=b-a+1) for a,b in runs(mask) if a<=i<=b),None)

manifest=json.loads((P/'frames_manifest.json').read_text())['frames'];records=[]
for aid in [31,32]:
    p=P/manifest[aid]['file']
    with np.load(p,allow_pickle=False) as z:
        r=np.asarray(z['scan_ranges_raw']);inf=np.isinf(r);finite=np.isfinite(r);segments=runs(inf);longest=sorted(segments,key=lambda x:(-(x[1]-x[0]+1),x[0]));n=len(r)
        circular=max([b-a+1 for a,b in segments] or [0])
        if n and inf[0] and inf[-1] and len(segments)>1:circular=max(circular,segments[0][1]+1+n-segments[-1][0])
        left=0
        for i in range(395,-1,-1):
            if not inf[i]:break
            left+=1
        right=0
        for i in range(397,n):
            if not inf[i]:break
            right+=1
        row=dict(action_id=aid,frame_id=manifest[aid]['frame_id'],packet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            scan_timestamp_s=float(z['scan_timestamp_s']),rgbd_timestamp_s=float(z['rgbd_timestamp_s']),
            raw_rays=n,finite_count=int(finite.sum()),finite_positive_count=int((finite&(r>0)).sum()),
            positive_inf_count=int(np.isposinf(r).sum()),negative_inf_count=int(np.isneginf(r).sum()),NaN_count=int(np.isnan(r).sum()),
            finite_range_min_m=float(r[finite].min()),finite_range_max_m=float(r[finite].max()),
            inf_segment_count=len(segments),max_contiguous_inf_linear=max([b-a+1 for a,b in segments] or [0]),
            max_contiguous_inf_circular_if_360_scan=circular,angular_span_rays_times_increment_rad=float(n*float(z['angle_increment'])),
            largest_inf_segments=[dict(start=a,end=b,length=b-a+1) for a,b in longest[:5]],
            target_index=396,target_raw_value_m=float(r[396]),target_is_inf=bool(inf[396]),
            target_finite_run=containing(finite,396),target_inf_run=containing(inf,396),
            immediately_left_inf_run_length=left,immediately_right_inf_run_length=right)
        records.append(row)
result=dict(schema='nso.existing_two_scan_completeness.v1',records=records,
    scope='Original raw range arrays in canonical packets, not a new scan capture or full mapper/policy replay.',
    interpretation='Completeness and neighboring missing-angle structure only; not proof of noise, physical-object identity, packet loss, timestamp race or sweep-publication causality.',
    later_observation='Separate later5s50-scan window did not reproduce0.4–0.6m hits. It is not synchronized independent ground truth and does not invalidate the recorded historical endpoint.')
(D/'scan_completeness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,ensure_ascii=False))
