"""Run synthetic geometry probes through the simulator's support adapter.

The fixtures have analytic ground truth and use no held-out floor geometry,
observations, policy output, or chosen path. A geometry factory takes a
trimesh.Scene and a world transform and returns the common .valid interface.
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from math import inf
from pathlib import Path
import sys

import numpy as np
import trimesh


ANALYSES = Path(__file__).resolve().parents[2]
ORIGINAL = ANALYSES / "semantic_scene_redesign_20261119/scene_adapter"
RADIUS = 0.20
ANCHOR = (-0.1, 0.0)
NEXT = (0.1, 0.0)


def quad(xy, z):
    vertices = [[x, y, z(x, y) if callable(z) else z] for x, y in xy]
    return tris(vertices)


def tris(vertices):
    return trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)


def patch(x0, y0, x1, y1, z=0.0):
    return quad(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), z)


def rim(a, b, bottom):
    return tris([[*a, bottom], [*b, bottom], [*b, 0], [*a, 0]])


def gapped_floor(hole, bottom=None, attach_to_outside=False):
    """A support gap inside the swept circle, with/without measured lower floor."""
    x0, y0, x1, y1 = hole
    floor = [patch(-1, -1, x0, 1), patch(x1, -1, 1, 1), patch(x0, y1, x1, 1)]
    if not attach_to_outside:
        floor.append(patch(x0, -1, x1, y0))
    if bottom is not None:
        floor.append(patch(x0, y0, x1, y1, z=bottom))
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        floor.extend(rim(a, b, bottom) for a, b in zip(corners, corners[1:] + corners[:1]))
    return floor


def measured_ridge(hole, height):
    """Add raw support with a steep, narrow ridge omitted from the nominal union."""
    x0, y0, x1, y1 = hole
    xm = (x0+x1)/2
    return [*gapped_floor(hole),
            patch(x0, y0, xm, y1, lambda x, y: height*(x-x0)/(xm-x0)),
            patch(xm, y0, x1, y1, lambda x, y: height*(x1-x)/(x1-xm))]


def world(parts):
    scene = trimesh.Scene()
    for i, mesh in enumerate(parts):
        scene.add_geometry(mesh, node_name=f"part_{i}", geom_name=f"part_{i}")
    # The production implementation requires at least one obstacle and floor.
    # These independently supported patches and walls are outside every test sweep.
    scene.add_geometry(patch(2.4, -0.5, 2.8, 0.5), node_name="distant_floor")
    box = trimesh.creation.box((0.2, 0.2, 0.5))
    box.apply_translation((2.6, 0, 0.25))
    scene.add_geometry(box, node_name="distant_obstacle")
    return scene


@dataclass(frozen=True)
class Probe:
    name: str
    scene: trimesh.Scene
    start: tuple[float, float]
    end: tuple[float, float]
    expected: bool | None
    basis: str


def controls(allow_same_level_hole_completion):
    hole = (-0.04, 0.11, 0.04, 0.17)  # Small enclosed hole inside the footprint but off its centre line.
    seam = (-0.0004, 0.11, 0.0004, 0.17)
    ridge = (-0.01, 0.11, 0.01, 0.17)
    low = -0.20
    flat = patch(-1, -1, 1, 1)
    blocking = trimesh.creation.box((0.06, 0.24, 0.5))
    blocking.apply_translation((0.06, 0, 0.25))
    return [
        Probe("continuous_level_floor", world([flat]), ANCHOR, NEXT, True, "flat floor is analytically complete"),
        Probe("allow_in_band_portion_of_crossing_ramp", world([patch(-2, -1, 4, 1, z=lambda x, y: 0.04*x)]),
              ANCHOR, NEXT, True, "swept z is within the support band and rise is 4 cm/m"),
        Probe("steep_ramp_rejected", world([patch(-0.3, -0.6, 0.3, 0.6, z=lambda x, y: 0.55*x)]),
              ANCHOR, NEXT, False, "slope exceeds the declared normal threshold"),
        Probe("exterior_edge_rejected", world([patch(-1, -1, 0.18, 1)]),
              ANCHOR, NEXT, False, "endpoint centre stands 0.08 m from unsupported exterior"),
        Probe("open_notch_rejected", world(gapped_floor(hole, attach_to_outside=True)),
              ANCHOR, NEXT, False, "notch connects to unsupported exterior"),
        Probe("measured_enclosed_drop_rejected", world(gapped_floor(hole, bottom=low)),
              ANCHOR, NEXT, False, "original mesh shows a 0.20 m lower surface and vertical sides"),
        Probe("open_step_rejected", world([patch(-1, -1, 0.06, 1), patch(0.06, -1, 1, 1, z=low),
                                         rim((0.06, -1), (0.06, 1), low)]),
              ANCHOR, NEXT, False, "original mesh shows a continuous leading edge with 0.20 m vertical drop"),
        Probe("body_band_obstacle_rejected", world([flat, blocking]),
              ANCHOR, NEXT, False, "recorded 0.50 m fixture intersects the body sweep"),
        Probe("measured_shallow_continuity_recovered", world(measured_ridge(ridge, 0.005)),
              ANCHOR, NEXT, True, "steep facets form a continuous measured bump only 5 mm high"),
        Probe("measured_submillimetre_seam_recovered", world(measured_ridge(seam, 0.001)),
              ANCHOR, NEXT, True, "0.8 mm seam has same-level raw geometry with 1 mm relief"),
        Probe("measured_high_relief_rejected", world(measured_ridge(ridge, 0.06)),
              ANCHOR, NEXT, False, "steep facets supply raw X/Y support but 60 mm relief"),
        Probe("enclosed_absent_faces", world(gapped_floor(hole)), ANCHOR, NEXT,
              True if allow_same_level_hole_completion else None,
              "analytical missing-face scenario has known true flat floor, but archive XY topology alone cannot prove it"),
        Probe("preexisting_within_band_step_behavior", world(gapped_floor(hole, bottom=-0.07)),
              ANCHOR, NEXT, None,
              "diagnostic for an actual 70 mm depression within the inherited support band, outside the collision band"),
    ]


def factory_at(source):
    module_path, name = source.rsplit(":", 1)
    if module_path.endswith(".py"):
        path = Path(module_path).resolve()
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    return getattr(module, name)


def evaluate(factory, probes):
    out = []
    for probe in probes:
        geometry = factory(probe.scene, np.eye(4))
        valid = bool(geometry.valid(probe.start, probe.end, RADIUS))
        reverse = bool(geometry.valid(probe.end, probe.start, RADIUS))
        _, covered, clearance = geometry.clearance(probe.start, probe.end)
        out.append({"fixture": probe.name, "expected_valid": probe.expected, "valid": valid,
                    "reverse_valid": reverse, "matches": (probe.expected == valid
                         if probe.expected is not None else None),
                    "centreline_supported": bool(covered),
                    "geometry_clearance_m": float(clearance) if abs(clearance) < inf else None,
                    "basis": probe.basis})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", help="Supports module:Factory or /absolute/file.py:Factory")
    parser.add_argument("--expect-gap-recovery", action="store_true",
                        help="Also require recovery of analytically flat missing faces")
    args = parser.parse_args()
    sys.path.insert(0, str(ORIGINAL))
    import replicacad_world
    replicacad_world.FLOOR_BAND_M = (-0.09, 0.09)
    replicacad_world.FLOOR_NORMAL_MIN = 0.96
    probes = controls(args.expect_gap_recovery)
    result = {"robot_radius_m": RADIUS, "revision": "blind_swept_support_controls_v2",
              "original": evaluate(replicacad_world.MeshMotionGeometry, probes)}
    if args.adapter:
        result["adapter"] = args.adapter
        module_path = args.adapter.rsplit(":", 1)[0]
        if module_path.endswith(".py"):
            result["adapter_sha256"] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
        result["repair"] = evaluate(factory_at(args.adapter), probes)
        tested = [r for r in result["repair"] if r["matches"] is not None]
        result["all_required_pass"] = all(r["matches"] and r["valid"] == r["reverse_valid"] for r in tested)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if args.adapter and not result["all_required_pass"] else 0


if __name__ == "__main__":
    sys.exit(main())
