"""Shared, deterministic grid geometry. No simulator or neural dependencies."""
from functools import lru_cache
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def inflated_obstacles(obstacles, radius_cells):
    """Conservative disk footprint: cell squares bounded by circumscribed disks.

    Border outside the array is blocked. The sqrt(2)/2 margin accounts for
    the finite area of an obstacle cell, rather than a point at its center.
    """
    obstacles = np.asarray(obstacles, dtype=bool)
    if radius_cells < 0:
        raise ValueError('negative robot radius')
    padded = np.pad(~obstacles, 1, constant_values=False)
    distance = distance_transform_edt(padded)[1:-1, 1:-1]
    return distance <= radius_cells + math.sqrt(2) / 2


def supercover_line(row, col):
    """Cells intersected by a ray from (0,0) to an integer cell center.

    At a corner include BOTH adjacent cells before the diagonal; an opaque
    corner cannot let a ray see through two touching obstacle cells.
    """
    nr, nc = abs(row), abs(col)
    sr, sc = int(np.sign(row)), int(np.sign(col))
    r = c = ir = ic = 0
    cells = [(0, 0)]
    while ir < nr or ic < nc:
        dr, dc = (1 + 2 * ir) * nc, (1 + 2 * ic) * nr
        if dr == dc:
            cells.extend([(r + sr, c), (r, c + sc)])
            r += sr; c += sc; ir += 1; ic += 1
        elif dr < dc:
            r += sr; ir += 1
        else:
            c += sc; ic += 1
        cells.append((r, c))
    return cells


@lru_cache(maxsize=32)
def ray_templates(range_cells, fov_degrees, heading):
    if range_cells < 1 or not 0 < fov_degrees <= 360 or heading not in range(4):
        raise ValueError('invalid sensor geometry')
    # Dense angular sampling relative to range; identical for both policies.
    count = max(3, int(math.ceil(math.radians(fov_degrees) * range_cells * 2)) + 1)
    center = heading * math.pi / 2
    angles = np.linspace(center - math.radians(fov_degrees) / 2,
                         center + math.radians(fov_degrees) / 2, count)
    endpoints = sorted({(int(round(-range_cells * math.cos(a))),
                         int(round(range_cells * math.sin(a)))) for a in angles})
    lines = [supercover_line(r, c) for r, c in endpoints]
    width = max(map(len, lines))
    offsets = np.zeros((len(lines), width, 2), dtype=int)
    valid = np.zeros((len(lines), width), dtype=bool)
    for i, line in enumerate(lines):
        offsets[i, :len(line)] = line
        valid[i, :len(line)] = True
    # Quantization may put endpoints slightly past the intended metric radius.
    valid &= np.sum(offsets ** 2, axis=2) <= range_cells ** 2
    offsets.setflags(write=False); valid.setflags(write=False)
    return offsets, valid


def visible_mask(occupied, position, heading, range_cells, fov_degrees):
    occupied = np.asarray(occupied, dtype=bool)
    offsets, ray_valid = ray_templates(range_cells, fov_degrees, heading)
    coords = offsets + np.asarray(position)
    rows, cols = coords[..., 0], coords[..., 1]
    inside = ((rows >= 0) & (rows < occupied.shape[0]) &
              (cols >= 0) & (cols < occupied.shape[1]) & ray_valid)
    rr, cc = np.clip(rows, 0, occupied.shape[0] - 1), np.clip(cols, 0, occupied.shape[1] - 1)
    hits = occupied[rr, cc] | ~inside
    # Include the first obstacle surface, exclude every cell after it.
    visible = inside & ((np.cumsum(hits, axis=1) - hits) == 0)
    result = np.zeros_like(occupied)
    result[rr[visible], cc[visible]] = True
    return result
