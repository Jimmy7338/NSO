"""Inspect actual projected and vertical support, with the official Detour mesh as a second reference."""

from pathlib import Path
import hashlib
import json
import math
import struct
import sys

import numpy as np
import open3d as o3d
import shapely
from shapely.geometry import LineString, Polygon, box
import trimesh

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1] / "ras_20261126/heterogeneous"
sys.path.insert(0, str(STUDY / "scenes"))
from scene import source_scene

FLOOR = "replica_apartment1"
NAV = ROOT / "apartment_1.navmesh"
URL = "https://huggingface.co/datasets/ai-habitat/habitat_test_scenes/resolve/910c783fb954da8497ea5f811b843a76590ddddc/apartment_1.navmesh"
RES = .005


def load_official_navmesh(transform):
    """Read the pinned test scene's documented Detour MeshSet v1 / Mesh v7."""
    raw = NAV.read_bytes()
    magic, version, tiles = struct.unpack_from("<3I", raw)
    assert (magic, version, tiles) == (0x4d534554, 1, 1)
    offset = 40
    reference, size = struct.unpack_from("<2I", raw, offset)
    assert reference and offset + 8 + size == len(raw)
    off = offset + 8
    head = struct.unpack_from("<15i10f", raw, off)
    assert head[:2] == (0x444e4156, 7)
    count, positions = head[6:8]
    offset = off + struct.calcsize("<15i10f")
    native = np.frombuffer(raw, dtype="<f4", offset=offset, count=positions * 3).reshape(-1, 3)
    # Habitat uses Y-up. This stage uses the Replica GLB's native Z-up.
    world = native[:, [0, 2, 1]].copy()
    world[:, 1] *= -1
    world = trimesh.transform_points(world, transform)
    assert -.06 < np.quantile(world[:, 2], .1) < .10, np.quantile(world[:, 2], [.1, .5, .9])
    offset += positions * 12
    floor = []
    for index in range(count):
        fields = struct.unpack_from("<I13H2B", raw, offset + index * 32)
        vertices = world[np.asarray(fields[1:1+fields[-2]])]
        if fields[-1] >> 6 == 0 and np.all(np.abs(vertices[:, 2]) < .15):
            polygon = Polygon(vertices[:, :2])
            if polygon.is_valid and polygon.area > 0:
                floor.append(polygon)
    return shapely.union_all(floor), dict(url=URL, sha256=hashlib.sha256(raw).hexdigest(),
                                          polygons_on_matched_floor=len(floor),
                                          walkable_height_m=head[15], radius_m=head[16],
                                          max_climb_m=head[17],
                                          source_note="Published navigation mesh for this scan. Derived from scene data, not an independent physical survey. Settings are weaker than the study's 0.20 m footprint.")


def union_triangles(faces, mask):
    paths = shapely.polygons(faces[mask, :, :2])
    paths = paths[shapely.area(paths) > 1e-12]
    return shapely.union_all(paths)


def bounds(points):
    return [float(v) for v in np.quantile(points, [0, .05, .5, .95, 1])] if len(points) else None


def sample_missing(missing):
    clouds = []
    for polygon in shapely.get_parts(missing):
        if polygon.geom_type != "Polygon" or polygon.area <= 1e-12:
            continue
        x0, y0, x1, y1 = polygon.bounds
        if polygon.area < .0001:
            step = min(RES, max(.0001, math.sqrt(polygon.area / 90)))
        else:
            step = RES
        xx, yy = np.meshgrid(np.arange(x0+step/3, x1, step), np.arange(y0+step/3, y1, step))
        inside = shapely.contains_xy(polygon, xx, yy)
        cloud = np.column_stack((xx[inside], yy[inside])) if inside.any() else np.empty((0,2))
        clouds.append(np.vstack((cloud, np.asarray(polygon.representative_point().coords[0]))))
    return np.vstack(clouds)


def trace(scene, xy):
    xyz = np.column_stack((xy, np.full(len(xy), .149)))
    directions = np.tile([0., 0., -1.], (len(xyz), 1))
    cast = scene.cast_rays(o3d.core.Tensor(np.column_stack((xyz, directions)), dtype=o3d.core.Dtype.Float32),
                           nthreads=1)
    first = cast["t_hit"].numpy()
    return xyz, xyz[:, 2] - first, abs(cast["primitive_normals"].numpy()[:, 2]), np.isfinite(first)


def inspect(edge, faces, scene, nav):
    layout = STARTS[edge["runs"][0]["layout"]]
    world = lambda cell: np.asarray([(cell[1]+.5)*.2, (layout["shape"][0]-cell[0]-.5)*.2])
    p, q = map(world, (edge["cell"], edge["target"]))
    line = LineString((p, q))
    capsule = line.buffer(.2, quad_segs=64)
    xmin, ymin, xmax, ymax = capsule.bounds
    local = faces[np.all(faces.max(axis=1)[:, :2] >= [xmin-.35, ymin-.35], axis=1) &
                  np.all(faces.min(axis=1)[:, :2] <= [xmax+.35, ymax+.35], axis=1)]
    normal = np.cross(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
    cosine = abs(normal[:, 2]) / np.maximum(np.linalg.norm(normal, axis=1), 1e-15)
    zmin,zmax = local[:, :, 2].min(1), local[:, :, 2].max(1)
    legacy = union_triangles(local, (zmin >= -.09) & (zmax <= .09) & (cosine >= .96))
    missing = capsule.difference(legacy)
    xyz, z, n, valid = trace(scene, sample_missing(missing))
    near = valid & (np.abs(z) <= .09)
    upper = valid & (z > .09)
    lower = valid & (z < -.09)
    xx, yy = np.meshgrid(np.arange(xmin, xmax+RES/2, RES), np.arange(ymin, ymax+RES/2, RES))
    portal = shapely.contains_xy(capsule, xx, yy)
    _, cap_z, _, cap_covered = trace(scene, np.column_stack((xx[portal], yy[portal])))
    component_spans = [[v.bounds[2]-v.bounds[0], v.bounds[3]-v.bounds[1]]
                       for v in shapely.get_parts(missing) if v.geom_type == "Polygon" and v.area > 1e-12]
    clipping_only = union_triangles(local, (zmin >= -.09) & (zmax <= .09) & (cosine >= .90))
    broad_normal = union_triangles(local, (zmin >= -.09) & (zmax <= .09) & (cosine >= .84))
    existing_same_band = union_triangles(local, (zmin >= -.09) & (zmax <= .09))
    detail = dict(
        layout=edge["runs"][0]["layout"], edge=[edge["cell"], edge["target"]],
        p_world_xy_m=p.tolist(), q_world_xy_m=q.tolist(), full_capsule_m2=float(capsule.area),
        old_support_gap_m2=float(missing.area), old_support_gap_pct=100*float(missing.area/capsule.area),
        gap_components_bbox_spans_m=component_spans,
        gap_fraction_after_lowering_normal_threshold_to_090=float(capsule.difference(clipping_only).area/capsule.area),
        gap_fraction_after_lowering_normal_threshold_to_084=float(capsule.difference(broad_normal).area/capsule.area),
        raw_original_same_band_union_covers_entire_capsule=bool(existing_same_band.covers(capsule)),
        uncovered_capsule_m2_in_raw_original_same_band_union=float(capsule.difference(existing_same_band).area),
        full_capsule_raw_height_quantiles_m=bounds(cap_z[cap_covered]),
        full_capsule_raw_below_negative_009_count=int(np.count_nonzero(cap_z[cap_covered] < -.09)),
        full_capsule_raw_no_downward_intersection_count=int((~cap_covered).sum()),
        full_capsule_samples=len(cap_z),
        raw_support_points_sampled=len(xyz),
        within_original_vertical_band_any_normal=int(near.sum()),
        within_original_vertical_band_cosine_ge_096=int((near & (n >= .96)).sum()),
        above_original_vertical_band=int(upper.sum()), below_original_vertical_band=int(lower.sum()),
        below_original_floor_height_quantile_m=bounds(z[lower]),
        distance_no_downward_intersection=int((~valid).sum()),
        raw_support_z_quantiles_m=bounds(z[valid]), near_band_cosine_quantiles=bounds(n[near]),
        navmesh_contains_centerline=bool(nav.covers(line)),
        navmesh_contains_endpoints=[bool(v) for v in shapely.contains_xy(nav, [p[0],q[0]], [p[1],q[1]])],
        navmesh_centerline_01m_residual_erosion=bool(nav.buffer(-.10).covers(line)),
        navmesh_overlap_with_missing_fraction=float(nav.intersection(missing).area / missing.area),
    )
    return detail, dict(line=line, capsule=capsule, legacy=legacy, nav=nav, missing=missing, points=xyz,
                        finite_height=z, finite_normal=n, valid=valid)


def plot(cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def outline(ax, geometry, c, width, alpha):
        for geom in shapely.get_parts(geometry):
            if geom.geom_type == "Polygon":
                ax.plot(*geom.exterior.xy, c=c, lw=width, alpha=alpha, zorder=2)
                for ring in geom.interiors:
                    ax.plot(*ring.xy, c=c, lw=width, alpha=alpha, zorder=2)

    fig, axs = plt.subplots(2,2,figsize=(9,8.25))
    for (d,x),ax in zip(cases, axs.ravel(), strict=True):
        a,b,c,o = x["capsule"].bounds
        clip = box(a-.20,b-.20,c+.20,o+.20)
        outline(ax,x["legacy"].intersection(clip),"#768b9a",1,.9)
        outline(ax,x["nav"].intersection(clip),"#56a85c",1,.6)
        outline(ax,x["capsule"],"#202631",1,.9)
        height = x["finite_height"].copy()
        height[~x["valid"]] = np.nan
        im=ax.scatter(*x["points"][x["valid"],:2].T,c=height[x["valid"]],s=3,
                      vmin=-.18,vmax=.15,cmap="coolwarm",zorder=4)
        ax.scatter(*x["points"][~x["valid"],:2].T,c="#ffb000",marker="x",s=6,zorder=4)
        ax.plot(*x["line"].xy,c="#181820",lw=2,zorder=6)
        ax.set_title(f'{list(d["edge"][0])} → {list(d["edge"][1])} · {d["old_support_gap_pct"]:.1f}% missing')
        ax.set(xlim=(a-.2,c+.2), ylim=(b-.2,o+.2), aspect="equal", xlabel="x (m)", ylabel="y (m)")
    handles=[Line2D([0],[0],c=c,label=label,marker=marker) for c,label,marker in
             (("#768b9a","Old support outline",None),("#56a85c","Official navmesh outline",None),
              ("#202631","0.20 m swept footprint",None),("#ffb000","No downward mesh hit","x"))]
    fig.legend(handles=handles,loc="lower center",ncol=2)
    fig.colorbar(im,ax=axs.ravel().tolist(),label="Raw downward-hit height relative to floor (m)",shrink=.66)
    fig.suptitle("Replica apartment_1 · four scan-floor support rejections")
    fig.subplots_adjust(bottom=.12,right=.84,hspace=.34)
    fig.savefig(ROOT/"replica_support_profiles.png",dpi=180)


if __name__ == "__main__":
    STARTS = json.loads((STUDY / "scenes/STARTS.json").read_text())["layouts"]
    transform = np.asarray(STARTS["b26_replica_apartment1_s1"]["world_from_source"])
    source = source_scene(FLOOR)
    transformed, shapes = [], []
    for node in source.graph.nodes_geometry:
        local, geom = source.graph[node]
        mesh = source.geometry[geom]
        verts = trimesh.transform_points(mesh.vertices, transform @ local)
        transformed.append(verts[mesh.faces])
        shapes.append(o3d.t.geometry.TriangleMesh(vertex_positions=o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32),
                                                  triangle_indices=o3d.core.Tensor(mesh.faces, dtype=o3d.core.Dtype.Int64)))
    ray = o3d.t.geometry.RaycastingScene(nthreads=1)
    for part in shapes:
        ray.add_triangles(part)
    faces = np.concatenate(transformed)
    nav, nav_meta = load_official_navmesh(transform)
    original = json.loads((STUDY / "FORMAL_COLLISION_AUDIT.json").read_text())["failures"]
    cases = [inspect(e, faces, ray, nav) for e in original if e["source_floor"] == FLOOR]
    report = {"official_navmesh":nav_meta,"discretization_m":RES,
              "scope":"Vertical ray profiles below z=0.149 m within those parts of each 0.20 m capsule that the original floor filter excludes.",
              "cases":[d for d,x in cases]}
    (ROOT/"replica_support_profiles.json").write_text(json.dumps(report,indent=2)+"\n")
    plot(cases)
    print(json.dumps(report,indent=2))
