"""Use official Detour polygons as secondary evidence for two HM3D edges."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
import trimesh


HERE = Path(__file__).resolve().parent
SCENE = HERE.parents[1] / "ras_20261126/heterogeneous"
NAV_FILE = HERE / "NBg5UqG3di3.basis.navmesh"
FLOOR_ID = "hm3d_00770"


def read_detour(file, world_from_source):
    raw = file.read_bytes()
    mesh_magic, mesh_version, tile_count = struct.unpack_from("<3I", raw)
    if (mesh_magic, mesh_version, tile_count) != (0x4D534554, 1, 1):
        raise ValueError("Expected single tile from Habitat Detour meshset v1")
    reference, tile_size = struct.unpack_from("<2I", raw, 40)
    if not reference or 48 + tile_size != len(raw):
        raise ValueError("Invalid native tile length")
    header = struct.unpack_from("<15i10f", raw, 48)
    if header[:2] != (0x444E4156, 7):
        raise ValueError("Expected Detour geometry tile v7")
    poly_count, vertex_count = header[6:8]
    offset = 48 + struct.calcsize("<15i10f")
    native = np.frombuffer(raw, dtype="<f4", offset=offset, count=vertex_count * 3).reshape(-1, 3)
    source = native[:, [0, 2, 1]].copy()
    source[:, 1] *= -1
    world = trimesh.transform_points(source, world_from_source)
    offset += vertex_count * 12
    accepted = []
    for i in range(poly_count):
        fields = struct.unpack_from("<I13H2B", raw, offset + i * 32)
        vertices = world[np.asarray(fields[1:1 + fields[-2]])]
        if fields[-1] >> 6 == 0 and np.all(np.abs(vertices[:, 2]) < .15):
            polygon = Polygon(vertices[:, :2])
            if polygon.is_valid and polygon.area > 0.0:
                accepted.append(polygon)
    report = {"walkable_height_m": float(header[15]), "walkable_radius_m": float(header[16]),
              "encoded_max_climb_m": float(header[17]), "matching_floor_polygons": len(accepted),
              "navmesh_height_quantiles_m": np.quantile(world[:, 2], [0, .5, 1]).tolist(),
              "navmesh_bound_xy_m": [world[:, :2].min(0).tolist(), world[:, :2].max(0).tolist()],
              "limit": "The official navmesh is derived from processed scan geometry. Its encoded radius and climb tolerance differ from the rover's physical and study assumptions."}
    return shapely.union_all(accepted), report


def cell_position(cell, shape):
    return (cell[1] + .5) * .2, (shape[0] - cell[0] - .5) * .2


def main():
    starts = json.loads((SCENE / "scenes/STARTS.json").read_text())["layouts"]
    previous = json.loads((SCENE / "FORMAL_COLLISION_AUDIT.json").read_text())
    reference = json.loads((HERE / "hm3d_navmesh_source_receipt.json").read_text())
    if hashlib.sha256(NAV_FILE.read_bytes()).hexdigest() != reference["matches"][0]["sha256"]:
        raise ValueError("Native navmesh differs from the byte-range receipt")
    cases = [case for case in previous["failures"] if case["source_floor"] == FLOOR_ID]
    layout = starts[cases[0]["runs"][0]["layout"]]
    floor, metadata = read_detour(NAV_FILE, np.asarray(layout["world_from_source"]))
    if floor.is_empty or not all(floor.covers(Point(row["xy_world_m"])) for row in starts.values()
                                 if row["source_floor"] == FLOOR_ID):
        raise ValueError("Converted official navigation polygons do not admit both frozen start centers")
    observed = []
    for case in cases:
        p, q = (cell_position(case[k], layout["shape"]) for k in ("cell", "target"))
        line = LineString((p, q))
        observed.append(dict(cell=case["cell"], target=case["target"], xy_centerline_m=[p, q],
                             supported_by_source_navmesh=bool(floor.covers(line)),
                             unadmitted_centerline_m=float(line.difference(floor).length),
                             current_center_clearance_m=float(Point(p).distance(floor.boundary)),
                             target_center_clearance_m=float(Point(q).distance(floor.boundary)),
                             raw_support_clearance_m=case["floor_clearance_m"]))
    report = {"source": reference["source"], "source_navmesh": reference["matches"], "official_archive":
              reference["official_archive"], "registered_starts_contained": True,
              "settings": metadata, "legacy_failed_edges": observed}
    with (HERE / "HM3D_NAV_REFERENCE.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(observed))


if __name__ == "__main__":
    main()
