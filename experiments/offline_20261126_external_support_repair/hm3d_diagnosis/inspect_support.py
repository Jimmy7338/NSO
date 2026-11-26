"""Inspect raw mesh heights and triangulation near the failed HM3D actions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points, triangulate
import trimesh


ROOT = Path(__file__).resolve().parents[2] / "ras_20261126/heterogeneous"
sys.path.insert(0, str(ROOT / "scenes"))
from scene import make_world, source_scene
import replicacad_world as geometry


def bounds(triangles, xy0, xy1):
    mins, maxs = triangles.min(1), triangles.max(1)
    return np.all((maxs[:, :2] >= xy0) & (mins[:, :2] <= xy1), axis=-1)


def triangles_at(world, p, q):
    original = source_scene(world.spec["source_floor"])
    parts, sources = [], []
    for node in original.graph.nodes_geometry:
        transform, name = original.graph[node]
        geo = original.geometry[name]
        tri = trimesh.transform_points(geo.vertices, world.world_from_source @ transform)[geo.faces]
        local = bounds(tri, np.minimum(p, q)-.7, np.maximum(p, q)+.7)
        parts.append(tri[local])
        sources.extend((node, int(i)) for i in np.flatnonzero(local))
    return np.concatenate(parts), sources


def clipped_union(tri, cos=.96, lo=-.09, hi=.09, clip=False):
    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    keep = np.abs(n[:, 2]) >= cos*np.maximum(np.linalg.norm(n, axis=1), 1e-15)
    lo_tri, hi_tri = tri[:, :, 2].min(1), tri[:, :, 2].max(1)
    chosen = keep & ((hi_tri >= lo) & (lo_tri <= hi) if clip else (lo_tri >= lo) & (hi_tri <= hi))
    parts = []
    for raw in tri[chosen]:
        points = geometry._clip_halfspace(geometry._clip_halfspace(raw, lo, True), hi, False) if clip else raw
        if len(points) >= 3:
            poly = Polygon(points[:, :2])
            if poly.area > 1e-12:
                parts.append(poly)
    return shapely.union_all(parts), int(chosen.sum())


def rays(world, xy, z):
    origin = np.column_stack((xy, np.full(len(xy), z), np.ones(len(xy))))
    origin_src = (origin @ world.source_from_world.T)[:, :3]
    ray = np.column_stack((origin_src, np.tile((0, 0, -1), (len(xy), 1)))).astype(np.float32)
    evidence = world._ray.cast_rays(o3d.core.Tensor(ray), nthreads=1)
    return z-evidence["t_hit"].numpy(), evidence["primitive_normals"].numpy()


def area_cropped(rule, capsule):
    return float(capsule.difference(rule).area)


def stats(values):
    match = values[np.isfinite(values)]
    if not len(match):
        return None
    return dict(count=len(match), min=float(match.min()), q05_q50_q95=np.quantile(match, [.05,.5,.95]).tolist(),
                max=float(match.max()))


def inspect_seam(world, hole, floor, capsule):
    interior_radius = shapely.maximum_inscribed_circle(hole, tolerance=.000_000_2).length
    patch = hole.intersection(capsule)
    rng = np.random.default_rng(4)
    pool = []
    for part in triangulate(patch):
        pts = np.asarray(part.exterior.coords)[:3]
        u, v = rng.uniform(size=(128, 2)).T
        reflect = u+v > 1
        u[reflect], v[reflect] = 1-u[reflect], 1-v[reflect]
        trial = pts[0] + u[:,None]*(pts[1]-pts[0]) + v[:,None]*(pts[2]-pts[0])
        pool.extend(trial[shapely.contains_xy(patch,*trial.T)])
    center = np.asarray(patch.representative_point().coords[:])
    interior = np.vstack([np.asarray(pool).reshape(-1,2), center])
    inner_z, inner_n = rays(world, interior, .149)
    banks = hole.buffer(.003, quad_segs=12).boundary
    xy = np.asarray([banks.interpolate(t, normalized=True).coords[0] for t in np.linspace(0,1,256,endpoint=False)])
    xy = xy[shapely.contains_xy(floor,*xy.T)]
    bank_z, bank_n = rays(world,xy,.149)
    return {"area_m2":float(hole.area),"bounds_m":list(hole.bounds),
            "capsule_overlap_area_m2":float(patch.area),"max_disk_diameter_mm":float(interior_radius*2000),
            "bank_offset_m":.003,"bank_samples":len(xy),"bank_elevation_m":stats(bank_z),
            "bank_normal_abs_z":stats(np.abs(bank_n[:,2])),
            "interior_rays_sampled":len(interior),"interior_uppermost_elevation_m":stats(inner_z),
            "interior_no_return":int((~np.isfinite(inner_z)).sum()),
            "interior_below_nominal_9_cm":int((np.isfinite(inner_z) & (inner_z<-.09)).sum()),
            "interior_normals_abs_z":stats(np.abs(inner_n[:,2][np.isfinite(inner_z)]))}


def inspect(case, axis):
    world = make_world(case["runs"][0]["layout"])
    p = np.asarray(world.xy(tuple(case["cell"])))
    q = np.asarray(world.xy(tuple(case["target"])))
    line = LineString([p, q])
    radius = case["capsule_radius_m"]
    capsule = line.buffer(radius, quad_segs=48)
    tri, ids = triangles_at(world, p, q)
    floor, count = clipped_union(tri)
    if not np.isclose(line.distance(floor.boundary), case["floor_clearance_m"], atol=1e-6):
        raise ValueError("Failed to recreate the archived floor-clearance result")
    closest = np.asarray(nearest_points(line, floor.boundary)[1].coords[0])
    ob = shapely.union_all([geometry._project_polygon(clipped) for t in tri
                           if len((clipped := geometry._clip_halfspace(geometry._clip_halfspace(t, .15, True), 1.2, False)))])
    enclosed, same_level_patches = [], []
    group = floor.geoms if floor.geom_type == "MultiPolygon" else [floor]
    for poly in group:
        for ring in poly.interiors:
            hole = Polygon(ring)
            overlap = hole.intersection(capsule).area
            if overlap > 1e-9:
                geometry_receipt = inspect_seam(world,hole,floor,capsule)
                enclosed.append(geometry_receipt)
                elevation = geometry_receipt["bank_elevation_m"]
                if (geometry_receipt["max_disk_diameter_mm"] <= 1.0 and hole.area <= .001
                        and geometry_receipt["bank_samples"] >= 64
                        and elevation["max"]-elevation["min"] <= .012
                        and geometry_receipt["bank_normal_abs_z"]["min"] >= .99
                        and geometry_receipt["interior_no_return"] == 0
                        and geometry_receipt["interior_below_nominal_9_cm"] == 0
                        and geometry_receipt["interior_uppermost_elevation_m"]["min"] >= elevation["min"]-.002
                        and geometry_receipt["interior_uppermost_elevation_m"]["max"] <= elevation["max"]+.002):
                    same_level_patches.append(hole)
    rules = []
    grids = []
    for cos, band, clipping in [(.96,.09,False),(.96,.09,True),(.96,.12,True),(.96,.15,True),(.90,.09,True),(.90,.12,True),(.90,.15,True),(.8,.15,True)]:
        support, size = clipped_union(tri, cos, -band, band, clipping)
        rules.append(dict(cosine=cos, half_band_m=band, clip_to_band=clipping,
                          capsule_uncovered_m2=area_cropped(support,capsule),
                          line_clearance_m=float(line.distance(support.boundary)) if support.covers(line) else 0.,
                          support_triangles=size))
        grids.append(support)
    scale = .0025
    lower, upper = np.minimum(p,q)-.35,np.maximum(p,q)+.35
    x = np.arange(lower[0],upper[0]+scale,scale)
    y = np.arange(lower[1],upper[1]+scale,scale)
    xx, yy = np.meshgrid(x,y)
    xy = np.column_stack((xx.ravel(),yy.ravel()))
    z, n = rays(world,xy,.149)
    cap = shapely.contains_xy(capsule, *xy.T)
    strict_missing = cap & ~shapely.contains_xy(floor, *xy.T)
    measured = np.isfinite(z)
    samples = {}
    classes = (("upper_near_same_level", measured & (np.abs(z)<=.09)),
               ("above_old_band", measured & (z>.09)),
               ("lower_9_to_15_cm", measured & (z<-.09) & (z>=-.15)),
               ("lower_15_to_25_cm", measured & (z<-.15) & (z>=-.25)),
               ("lower_over_25_cm", measured & (z<-.25)),
               ("no_lower_hit", ~measured))
    for key, select in (("capsule",cap),("strict_missing_capsule",strict_missing)):
        samples[key] = {"sample_count":int(select.sum()),"z_first_hit_stats_m":stats(z[select]),
                        "counts_by_height":{name:int((mask&select).sum()) for name,mask in classes},
                        "near_same_level_normals_z_abs":stats(np.abs(n[:,2][select & measured & (np.abs(z)<=.09)]))}
    middle = (p+q)/2
    direction = (closest-middle)/np.linalg.norm(closest-middle)
    profile_xy = middle[None,:]+np.arange(-.30,.321,.01)[:,None]*direction
    profile_z, profile_n = rays(world,profile_xy,.149)
    profile = [dict(xy_m=a.tolist(),z_m=float(z1) if np.isfinite(z1) else None,cosine_z=float(n1)) 
               for a,z1,n1 in zip(profile_xy, profile_z, profile_n[:,2], strict=True)]
    plot = axis.pcolormesh(xx,yy,np.where(np.isfinite(z),z,np.nan).reshape(xx.shape),cmap="coolwarm",vmin=-.16,vmax=.16,shading="auto")
    axis.plot(*capsule.exterior.xy,c="k",linewidth=1.5,label="20 cm swept capsule")
    axis.plot(*line.xy,c="k",linewidth=2)
    for group in (floor.geoms if floor.geom_type=="MultiPolygon" else [floor]):
        axis.plot(*group.exterior.xy,c="#33a02c",linewidth=.7,alpha=.65)
        for ring in group.interiors:
            axis.plot(*ring.xy,c="#33a02c",linewidth=.7,alpha=.65)
    axis.scatter(*closest,s=12,c="yellow")
    axis.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),aspect="equal",xlabel="World x (m)",ylabel="World y (m)",
             title=f'Cell {case["cell"]} → {case["target"]}')
    patched = shapely.union_all([floor, *same_level_patches])
    repaired_clearance = float(line.distance(patched.boundary)) if patched.covers(line) else 0.
    return {"cell":case["cell"],"target":case["target"],"heading":case["heading"],"archives":case["runs"],
            "world_p":p.tolist(),"world_q":q.tolist(),"closest_old_floor_boundary_xy":closest.tolist(),
            "min_body_distance_m":float(ob.distance(line)),"strict_floor_parts":count,
            "strict_floor_enclosed_holes":enclosed,"local_triangles":len(tri),
            "support_model_comparison":rules,"vertical_rays":samples,"cross_boundary_profile":profile,
            "local_same_level_seam_closure_check":{"eligible_polygon_count":len(same_level_patches),
                  "diameter_limit_mm":1.0,"area_limit_m2":.001,"max_bank_elevation_span_m":.012,
                  "min_bank_normal_abs_z":.99,"inside_height_outside_bank_tolerance_m":.002,
                  "capsule_uncovered_m2":area_cropped(patched,capsule),
                  "new_support_clearance_m":repaired_clearance,"body_collision_margin_unchanged":True,
                  "passes_frozen_20cm_sweep":bool(repaired_clearance >= radius and ob.distance(line)>radius)},
            "nearby_native_triangles":[{"node":n,"face_index":i,"z_m":v[:,2].tolist()}
                                      for (n,i),v in zip(ids,tri,strict=True) if np.linalg.norm(v[:,:2]-closest,axis=1).min()<.05]},plot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    audit = json.loads((ROOT / "FORMAL_COLLISION_AUDIT.json").read_text())
    cases = [record for record in audit["failures"] if record["source_floor"]=="hm3d_00770"]
    figure, axes = plt.subplots(1,len(cases),figsize=(9,4),layout="constrained")
    rows=[]
    for axis, case in zip(axes,cases,strict=True):
        row,plot=inspect(case,axis)
        rows.append(row)
    figure.colorbar(plot,ax=axes,label="Uppermost raw hit elevation below z=0.149 m (m)")
    figure.savefig(args.out/"raw_elevation.png",dpi=190)
    result={"schema":"nso_hm3d_failed_edge_raw_height_v1","source":"HM3D 00770 raw triangle mesh",
            "vertical_surface_rays_launch_z_m":.149,"ray_grid_step_m":.0025,"cases":rows}
    (args.out/"raw_elevation.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
