#!/usr/bin/env python3
"""Read-only offline checks for the shared ROS1/virtual RGB-D sequence format.

Checks stored evidence only. Does not connect to ROS or authorize robot motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from utils.rgbd_contract import RGBDFrame,PlanarScan


def check_sequence(source,max_skew_s=.05,max_depth_m=4.):
    if not np.isfinite([max_skew_s,max_depth_m]).all() or max_skew_s<0 or max_depth_m<=0:
        raise ValueError("finite nonnegative skew and positive maximum depth required")
    source=Path(source)
    frames=sorted((source/"frames").glob("*.npz"))
    if not frames: raise ValueError("no stored RGB-D frames")
    scan_names={p.name for p in (source/"scans").glob("*.npz")}
    previous_frame=previous_scan=-float("inf")
    errors=[]; rows=[]; hashes={}
    for file in frames:
        row=dict(file=file.name,errors=[])
        hashes[str(file.relative_to(source))]=hashlib.sha256(file.read_bytes()).hexdigest()
        try:
            frame=RGBDFrame.load(file)
            if frame.timestamp_s<=previous_frame: row["errors"].append("nonincreasing_depth_timestamp")
            previous_frame=frame.timestamp_s
            if frame.color_rgb.dtype!=np.uint8 or not np.issubdtype(frame.semantic.dtype,np.integer):
                row["errors"].append("wrong_rgb_or_semantic_dtype")
            if (frame.semantic<0).any(): row["errors"].append("negative_semantic_label")
            if float(frame.depth_m.max())>max_depth_m+1e-6: row["errors"].append("depth_exceeds_declared_metres_range")
            row["valid_depth_fraction"]=float(np.mean(frame.depth_m>0))
            if row["valid_depth_fraction"]==0.: row["errors"].append("no_valid_depth")
            scanfile=source/"scans"/file.name
            if not scanfile.is_file():
                row["errors"].append("missing_planar_scan")
            else:
                hashes[str(scanfile.relative_to(source))]=hashlib.sha256(scanfile.read_bytes()).hexdigest()
                scan=PlanarScan.load(scanfile)
                if not np.isfinite([scan.timestamp_s,scan.angle_min_rad,scan.angle_increment_rad,scan.range_max_m]).all():
                    row["errors"].append("nonfinite_laser_calibration")
                if scan.timestamp_s<=previous_scan: row["errors"].append("nonincreasing_scan_timestamp")
                previous_scan=scan.timestamp_s
                if scan.range_max_m<=0 or scan.angle_increment_rad==0: row["errors"].append("invalid_laser_range_or_angle_increment")
                ranges=np.asarray(scan.ranges_m)
                if (ranges.ndim!=1 or not len(ranges) or not np.isfinite(ranges).all()
                        or (ranges<0).any() or (ranges>scan.range_max_m+1e-6).any()):
                    row["errors"].append("invalid_stored_laser_ranges")
                pose=np.asarray(scan.world_from_laser)
                rigid=(pose.shape==(4,4) and np.isfinite(pose).all()
                    and np.allclose(pose[3],[0,0,0,1],rtol=0,atol=1e-8)
                    and np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),rtol=0,atol=1e-5)
                    and abs(np.linalg.det(pose[:3,:3])-1.)<=1e-5)
                if not rigid: row["errors"].append("invalid_world_from_laser")
                skew=abs(frame.timestamp_s-scan.timestamp_s)
                row["depth_scan_skew_s"]=skew if np.isfinite(skew) else None
                if not np.isfinite(skew) or skew>max_skew_s+1e-12:
                    row["errors"].append("depth_scan_time_skew")
        except (ValueError,KeyError,TypeError,IndexError,OSError) as error:
            row["errors"].append("invalid_record: "+str(error))
        if row["errors"]: errors.append(dict(file=file.name,reasons=row["errors"]))
        rows.append(row)
    orphan=sorted(scan_names-{p.name for p in frames})
    if orphan: errors.append(dict(file="scans",reasons=["orphan_scan_files"],names=orphan))
    return dict(status="passed" if not errors else "failed",frames=len(frames),
        maximum_allowed_depth_scan_skew_s=max_skew_s,declared_max_depth_m=max_depth_m,
        errors=errors,records=rows,input_sha256=hashes,
        robot_interface_ready=False,motion_output_enabled=False,
        source_metadata_files_found=len(list((source/"metadata").glob("*.json"))),
        source_metadata_validated=False,
        unchecked=["RGB/semantic original timestamps: absent in legacy data; optional new metadata not checked by this array auditor",
                   "CameraInfo source frame/timestamp and association: optional new metadata not checked by this array auditor",
                   "sensor extrinsic calibration correctness",
                   "laser per-beam motion compensation: not implemented; optional metadata time_increment not checked by this array auditor",
                   "ZED pose drift and map/odom reset events",
                   "live ROS delivery, navigation action server and physical emergency stop"],
        boundary="Pass certifies stored-array and depth/scan timestamp checks only, not robot readiness or reconstruction accuracy.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-skew-s",type=float,default=.05)
    parser.add_argument("--max-depth-m",type=float,default=4.)
    args=parser.parse_args()
    result=check_sequence(args.source,args.max_skew_s,args.max_depth_m)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("status","frames","robot_interface_ready","errors")},ensure_ascii=False))
    if result["status"]!="passed": raise SystemExit(2)
