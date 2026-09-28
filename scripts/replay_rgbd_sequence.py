#!/usr/bin/env python3
"""Reconstruct saved virtual/ROS RGB-D frames with the identical CPU backend."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d import VirtualConfig
from nso.mapping3d import SensorMapper
from utils.rgbd_contract import RGBDFrame,PlanarScan


def replay(source,output,config):
    source=Path(source);output=Path(output)
    files=sorted((source/'frames').glob('*.npz'))
    if not files:raise ValueError('no RGB-D frames')
    output.mkdir(parents=True,exist_ok=False)
    mapper=SensorMapper((30,40),config)
    previous=-float('inf');start=time.perf_counter()
    for path in files:
        frame=RGBDFrame.load(path)
        if frame.timestamp_s<=previous:raise ValueError('timestamps must strictly increase')
        previous=frame.timestamp_s
        scanfile=source/'scans'/path.name
        scan=PlanarScan.load(scanfile) if scanfile.exists() else None
        mapper.update(frame,scan)
    mesh=mapper.mesh();o3d.io.write_triangle_mesh(str(output/'reconstruction.ply'),mesh)
    result=dict(frames=len(files),vertices=len(mesh.vertices),triangles=len(mesh.triangles),
                wall_time_s=time.perf_counter()-start,backend='Open3D CPU ScalableTSDFVolume',
                truth_available=False,quality_claim='no GT accuracy score from real sequence alone',
                map_scope='TSDF uses full metric poses; 2D prototype grid is not a real navigation map')
    if (source/'reconstruction.ply').exists():
        original=o3d.io.read_triangle_mesh(str(source/'reconstruction.ply'))
        result['matches_online_vertices']=bool(np.asarray(mesh.vertices).shape==np.asarray(original.vertices).shape
            and np.allclose(np.asarray(mesh.vertices),np.asarray(original.vertices),atol=1e-8))
        result['matches_online_triangles']=bool(np.array_equal(np.asarray(mesh.triangles),np.asarray(original.triangles)))
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',type=Path,required=True,help='JSON with environment parameters; match recorded depth range and voxel scale')
    a=p.parse_args();replay(a.source,a.output,VirtualConfig(**json.loads(a.config.read_text())['environment']))
