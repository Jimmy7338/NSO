"""Classify the archived, out-of-matrix HM3D 00337 swept edge."""

import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
import trimesh

from inspect_support import ROOT, clipped_union, make_world, rays, source_scene, stats, triangles_at
from audit_formal_collisions import local_original_triangles


LAYOUT = "b26_hm3d_00337_s1"
CELL = (31, 47)
TARGET = (32, 47)
HEADING = 2


def main():
    archived_steps = ROOT / "development_smoke/run/steps.json"
    last = json.loads(archived_steps.read_text())[-1]
    assert (last["step"], last["action"], last["position"], last["heading"], last["collision"]) == (
        9, "forward", list(CELL), HEADING, True,
    )
    world = make_world(LAYOUT)
    p, q = np.asarray(world.xy(CELL)), np.asarray(world.xy(TARGET))
    line = LineString([p, q])
    radius = world.config.robot_radius_m
    capsule = line.buffer(radius / np.cos(np.pi / 128), quad_segs=32)
    tri, _ = triangles_at(world, p, q)
    legacy, _ = clipped_union(tri)
    clipped, _ = clipped_union(tri, clip=True)
    raw, _ = clipped_union(tri, cos=0.0, clip=True)
    scan = source_scene("hm3d_00337")
    _, _, closest = local_original_triangles(scan, world.world_from_source, p, q)
    global_triangles = []
    for node in scan.graph.nodes_geometry:
        transform, name = scan.graph[node]
        geo = scan.geometry[name]
        global_triangles.append(trimesh.transform_points(geo.vertices, world.world_from_source @ transform)[geo.faces])
    all_clipped, _ = clipped_union(np.concatenate(global_triangles), clip=True)
    outer = shapely.union_all([Polygon(poly.exterior) for poly in shapely.get_parts(all_clipped)
                               if poly.geom_type == "Polygon"])
    start, stop = np.asarray(capsule.bounds).reshape(2, 2)
    sample_step = .0025
    sizes = np.ceil((stop-start)/sample_step).astype(int) + 1
    xx, yy = np.meshgrid(*[np.linspace(lo, hi, size) for lo, hi, size in zip(start, stop, sizes, strict=True)])
    grid = shapely.contains_xy(capsule, xx, yy)
    xy = np.column_stack([xx[grid], yy[grid]])
    z, normals = rays(world, xy, .149)
    finite = np.isfinite(z)
    missing = ~shapely.contains_xy(clipped, *xy.T)
    plane = np.full(grid.shape, np.nan)
    plane[grid] = np.where(finite, z, np.nan)
    jump = []
    for axis in (0, 1):
        differences = np.abs(np.diff(plane, axis=axis))
        jump.append(float(np.nanmax(differences)))
    sections = {}
    for name, mask in [("whole_footprint", np.ones(len(z), dtype=bool)), ("clipped_support_missing", missing)]:
        supported = mask & finite & (z >= -.09) & (z <= .09)
        sections[name] = dict(samples=int(mask.sum()), no_lower_return=int((mask & ~finite).sum()),
                              hits_below_band=int((mask & finite & (z < -.09)).sum()),
                              hits_above_band=int((mask & finite & (z > .09)).sum()),
                              absolute_height_m=stats(z[mask]), relief_m=float(np.ptp(z[mask & finite])),
                              in_band_normals_abs_z=stats(np.abs(normals[supported, 2])))
    result = {
        "scope": "archived development case only; no policy execution",
        "source_floor": "hm3d_00337", "layout": LAYOUT, "step": 9, "cell": CELL,
        "target": TARGET, "heading": HEADING, "position_xy_m": [p.tolist(), q.tolist()],
        "capsule_radius_m": radius, "body_height_band_m": [.15, 1.2],
        "nearest_body_face": closest, "body_band_intersects_footprint": closest["distance_to_centerline_m"] <= radius + 1e-9,
        "legacy_uncovered_footprint_m2": float(capsule.difference(legacy).area),
        "clipped_floor_uncovered_footprint_m2": float(capsule.difference(clipped).area),
        "raw_in_band_uncovered_footprint_m2": float(capsule.difference(raw).area),
        "global_clipped_floor_outer_boundary_uncovered_m2": float(capsule.difference(outer).area),
        "global_clipped_floor_outer_boundary_covers_capsule": bool(outer.covers(capsule)),
        "legacy_centerline_inside_floor": bool(legacy.covers(line)),
        "clipped_centerline_inside_floor": bool(clipped.covers(line)),
        "legacy_centerline_support_clearance_m": float(line.distance(legacy.boundary)) if legacy.covers(line) else 0.0,
        "clipped_centerline_support_clearance_m": float(line.distance(clipped.boundary)) if clipped.covers(line) else 0.0,
        "ray_spacing_at_most_m":sample_step, "ray_launch_z_m": .149, "ray_sets": sections,
        "max_sampled_neighbor_height_jump_m": max(jump),
    }
    result["interpretation"] = {
        "raw_scan": "Continuous shallow in-band support and no body-band intersection for this sweep.",
        "legacy_rejection": "A small original floor patch falls below the per-triangle 0.96 normal cosine threshold.",
        "physical_limit": "These measurements classify the reconstructed mesh and do not validate actual physical traversability.",
    }
    if not result["global_clipped_floor_outer_boundary_covers_capsule"]:
        result["interpretation"]["retained_rejection"] = "The repair only recovers omissions enclosed by the strict floor exterior. This patch reaches the strict floor's exterior boundary."
    frozen_cache = Path(__file__).resolve().parents[1] / "scenes/precomputed_assets/hm3d_00337.npz"
    with np.load(frozen_cache, allow_pickle=False) as cached:
        result["frozen_repaired_cache_allows_edge"] = bool(cached["edge_valid"][CELL+(HEADING,)])
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
