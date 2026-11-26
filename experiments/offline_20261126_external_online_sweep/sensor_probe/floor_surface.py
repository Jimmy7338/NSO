"""Measure continuous floor support polygons inferred from adjacent positive pixels only."""

import json

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from inspect_corridor import CUTOFFS, EDGES, LAYOUT, OUT, make_world, read_poses


MAX_EDGES_M = (.04, .06, .08, .10)
MAX_VERTEX_HEIGHT_DISAGREEMENT_M = .012


def support_faces(frame, scan, lo, hi):
    h,w = frame.depth_m.shape
    rr,cc = np.indices((h,w))
    depth = np.asarray(frame.depth_m)
    uv = np.stack((cc,rr,np.ones_like(cc)),axis=-1)
    camera = (uv @ np.linalg.inv(frame.intrinsic).T) * depth[...,None]
    xyz = camera @ frame.world_from_camera[:3,:3].T + frame.world_from_camera[:3,3]
    z0 = scan.world_from_laser[2,3]-.25
    valid = ((depth > .15)&(depth < scan.range_max_m-.001)&(np.abs(xyz[...,2]-z0)<=.04)
             & np.all(xyz[...,:2]>=lo,axis=2) & np.all(xyz[...,:2]<=hi,axis=2))
    four = valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    tl,tr,bl,br = (a[four] for a in (xyz[:-1,:-1],xyz[:-1,1:],xyz[1:,:-1],xyz[1:,1:]))
    triangles = np.concatenate((np.stack((tl,tr,br),axis=1),np.stack((tl,bl,br),axis=1)),axis=0)
    triangles = triangles[np.ptp(triangles[:,:,2],axis=1)<=MAX_VERTEX_HEIGHT_DISAGREEMENT_M]
    lengths = np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=2).max(axis=1)
    return triangles, lengths


def main():
    world = make_world(LAYOUT)
    points = {cell:np.asarray(world.xy(cell)) for edge in EDGES for cell in edge}
    xy = np.stack(list(points.values()))
    low,high = xy.min(0)-.45,xy.max(0)+.45
    masks = {limit: [] for limit in MAX_EDGES_M}
    observed = {}
    frames = {t:(scan,frame) for t,_,_,scan,frame in read_poses(world)}
    for t in range(max(CUTOFFS)+1):
        if t in frames:
            scan,frame = frames[t]
            triangles,length = support_faces(frame,scan,low,high)
            for limit in MAX_EDGES_M:
                masks[limit].append(shapely.polygons(triangles[length<=limit,:,:2]))
        if t in CUTOFFS:
            observed[str(t)] = {}
            for lim,polygons in masks.items():
                full = np.concatenate(polygons)
                union = shapely.union_all(full) if len(full) else shapely.GeometryCollection()
                swept_areas = []
                for a,b in EDGES:
                    radius = .20/np.cos(np.pi/128)
                    capsule = LineString((points[a],points[b])).buffer(radius,quad_segs=32)
                    initial = Point(points[a]).buffer(radius,quad_segs=32)
                    area = capsule.difference(initial)
                    unobserved_whole = capsule.difference(union)
                    unobserved_new = area.difference(union)
                    swept_areas.append({"start":a,"end":b,"full_sweep_area_m2":capsule.area,
                                        "full_unsupported_area_m2":unobserved_whole.area,
                                        "new_sweep_area_m2":area.area,"new_unsupported_area_m2":unobserved_new.area})
                observed[str(t)][str(lim)] = {"total_triangles":len(full),"area_by_transition":swept_areas}
    (OUT/"positive_floor_surface.json").write_text(json.dumps(
        {"diagnostic_only":True,"method":"Adjacent positive-return 2x2 image neighborhoods with <=12mm vertical relief",
         "maximum_pixel_triangle_sides_m":MAX_EDGES_M, "known_ground_height_from_mount":True,
         "model_limit":"Interpolated triangles assume continuity between pixels under this fixed size; no range censors were treated as valid.",
         "results":observed},indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
