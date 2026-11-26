"""Locate missing positive floor evidence in the exact terminal sweep probes."""

import json

import numpy as np
import shapely
from scipy.spatial import cKDTree

from evidence import AccumulatedEvidence, _swept_xy
from floor_surface import support_faces
from inspect_corridor import LAYOUT, OUT, make_world, read_poses


TARGET = ((52,26),(53,26))
CHECKPOINTS = (17,38,192)


def main():
    world = make_world(LAYOUT)
    p,q = (np.asarray(world.xy(c)) for c in TARGET)
    xy = _swept_xy(p,q,.20)
    source_norms = [.06,.08,.10,.12]
    masks = {key:[] for key in source_norms}
    measured = {}
    rays = AccumulatedEvidence()
    lo,hi = np.minimum(p,q)-.40,np.maximum(p,q)+.40
    poses = {t:(scan,frame) for t,_,_,scan,frame in read_poses(world)}
    for paid in range(max(CHECKPOINTS)+1):
        if paid in poses:
            scan,frame = poses[paid]
            rays.observe(frame,scan)
            triangles,length = support_faces(frame,scan,lo,hi)
            for key in source_norms:
                masks[key].append(shapely.polygons(triangles[length<=key,:,:2]))
        if paid in CHECKPOINTS:
            pool = np.vstack(rays._floor_returns)
            tree = cKDTree(pool)
            nearest,idx = tree.query(xy)
            rejected = np.flatnonzero(nearest>.025)
            unions = {key:shapely.union_all(np.concatenate(masks[key])) for key in source_norms}
            case = []
            for i in rejected:
                point = shapely.Point(xy[i])
                distances,indices = tree.query(xy[i],k=10)
                case.append(dict(probe_xy_m=xy[i].tolist(),nearest_actual_floor_sample_mm=nearest[i]*1000,
                                 nearest_actual_floor_points_xy_m=pool[indices].tolist(),
                                 nearest_actual_floor_distances_mm=(distances*1000).tolist(),
                                 adjacent_pixel_interpolated_support={str(key):dict(covers=union.covers(point),
                                     distance_mm=union.distance(point)*1000) for key,union in unions.items()}))
            measured[str(paid)] = dict(rejected_probe_count=len(case),probes=case,
                                       sensor_only_full_certificate=rays.certify(p,q,.20))
    (OUT/"exit_witness.json").write_text(json.dumps({"transition":TARGET,"evidence":measured},indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
