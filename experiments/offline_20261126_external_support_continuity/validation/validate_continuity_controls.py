"""Check scan support continuity on blinded, analytically defined fixtures."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh


ANALYSES = Path(__file__).resolve().parents[2]
V1 = ANALYSES / "external_support_repair_20261126/validation"
sys.path.insert(0, str(V1))
from validate_support_controls import (
    ANCHOR, NEXT, ORIGINAL, RADIUS, Probe, controls, evaluate, factory_at,
    gapped_floor, patch, world,
)


def rough_patch(x0, y0, x1, y1, amplitude):
    """An observed piecewise-linear sheet joining the surrounding flat floor."""
    xs, ys = np.linspace(x0, x1, 11), np.linspace(y0, y1, 13)
    nx, ny = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    rows, cols = np.indices(zz[1:-1, 1:-1].shape)
    zz[1:-1, 1:-1] = amplitude * ((rows + cols) % 2 * 2 - 1)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    faces = []
    for r, c in np.ndindex(ny-1, nx-1):
        a = r*nx+c
        faces.extend(((a, a+1, a+nx+1), (a, a+nx+1, a+nx)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def slit_floor(x0, y0, x1, y1, *, right_height=0.0, lower_floor=None, open_edge=False):
    """Return two banks and optional below-slit geometry, retaining the slit in the source mesh."""
    bridging_z = lambda x, y: right_height * (x-x0)/(x1-x0)
    parts = [patch(-1, -1, x0, 1), patch(x1, -1, 1, 1, right_height),
             patch(x0, -1, x1, y0, bridging_z)]
    if not open_edge:
        parts.append(patch(x0, y1, x1, 1, bridging_z))
    if lower_floor is not None:
        parts.append(patch(x0, y0, x1, y1, lower_floor))
    return parts


def continuity_controls():
    patch_off_center = (-0.05, 0.04, 0.05, 0.16)
    patch_crossing_center = (-0.05, -0.06, 0.05, 0.06)
    slit = (-0.00025, 0.08, 0.00025, 0.16)
    crossing_slit = (-0.00025, -0.04, 0.00025, 0.04)
    return [
        Probe("observed_rough_patch_larger_than_legacy_area", world([
            *gapped_floor(patch_off_center), rough_patch(*patch_off_center, 0.0035),
        ]), ANCHOR, NEXT, True,
              "0.012 m² observed surface, 7 mm peak-to-trough continuous roughness"),
        Probe("observed_rough_patch_crosses_centreline", world([
            *gapped_floor(patch_crossing_center), rough_patch(*patch_crossing_center, 0.0035),
        ]), ANCHOR, NEXT, True,
              "same 0.012 m² continuous roughness intersects the centreline"),
        Probe("observed_large_high_relief_patch_rejected", world([
            *gapped_floor(patch_off_center), rough_patch(*patch_off_center, 0.025),
        ]), ANCHOR, NEXT, False,
              "same geometry and complete raw XY support, but 50 mm peak-to-trough relief"),
        Probe("tiny_true_missing_seam_between_flat_banks", world(slit_floor(*slit)), ANCHOR, NEXT, True,
              "0.5 mm enclosed gap has no raw triangles and both banks are measured level"),
        Probe("tiny_true_missing_seam_crosses_centreline", world(slit_floor(*crossing_slit)), ANCHOR, NEXT, True,
              "0.5 mm enclosed same-level gap cuts the centreline"),
        Probe("tiny_slit_with_height_mismatch_rejected", world(slit_floor(*slit, right_height=0.020)),
              ANCHOR, NEXT, False, "banks are each flat but differ in elevation by 20 mm"),
        Probe("tiny_slit_with_lower_surface_rejected", world(slit_floor(*slit, lower_floor=-0.18)),
              ANCHOR, NEXT, False, "banks agree but the raw scan shows a floor 180 mm below them through the hole"),
        Probe("tiny_exterior_connected_slit_rejected", world(slit_floor(*slit[:3], 1.0, open_edge=True)),
              ANCHOR, NEXT, False, "same-level 0.5 mm slit opens to the exterior and lacks an enclosing floor"),
        Probe("enclosed_large_unobserved_hole_rejected", world(gapped_floor((-0.07, 0.05, 0.07, 0.15))),
              ANCHOR, NEXT, False, "unsupported 0.014 m² hole has no measured support despite same-level banks"),
        Probe("too_wide_slit_grazing_sweep_rejected", world(gapped_floor((-0.0008, 0.1997, 0.0008, 0.35))),
              ANCHOR, NEXT, False, "true cross-bank gap is 1.6 mm although only a 0.36 mm strip enters the sweep"),
    ]


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--adapter", required=True, help="Import /absolute/file.py:Factory or module:Factory")
    args = cli.parse_args()
    sys.path.insert(0, str(ORIGINAL))
    import replicacad_world
    replicacad_world.FLOOR_BAND_M = (-0.09, 0.09)
    replicacad_world.FLOOR_NORMAL_MIN = 0.96
    legacy_probes, new_probes = controls(False), continuity_controls()
    factory = factory_at(args.adapter)
    summary = {"schema": "nso_blind_support_continuity_v3", "adapter": args.adapter, "robot_radius_m": RADIUS}
    module_path = args.adapter.rsplit(":", 1)[0]
    if module_path.endswith(".py"):
        summary["adapter_sha256"] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    for key, probes in (("inherited", legacy_probes), ("continuity", new_probes)):
        summary[key] = evaluate(factory, probes)
    tests = [item for key in ("inherited", "continuity") for item in summary[key] if item["expected_valid"] is not None]
    summary["passed"] = sum(row["matches"] and row["valid"] == row["reverse_valid"] for row in tests)
    summary["total"] = len(tests)
    summary["all_required_pass"] = summary["passed"] == summary["total"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if summary["all_required_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
