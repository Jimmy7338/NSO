"""Seeded synthetic layouts for mechanism validation, not Gibson/MP3D scenes."""
import numpy as np

LAYOUTS = ('single_room', 'multi_room', 'corridor', 'dead_end', 'loop', 'hall')


def generate_layout(kind, size=64, seed=0, door_width=5):
    if kind not in LAYOUTS or size < 40 or not 3 <= door_width < size // 4:
        raise ValueError('invalid layout, size (<40), or door width')
    rng = np.random.default_rng(seed)
    grid = np.ones((size, size), dtype=np.uint8)
    margin = 3
    mid = size // 2
    width = max(7, size // 10)
    def carve(r0, r1, c0, c1):
        grid[max(margin, r0):min(size - margin, r1),
             max(margin, c0):min(size - margin, c1)] = 0
    if kind in ('single_room', 'multi_room', 'hall'):
        carve(margin, size - margin, margin, size - margin)
        if kind == 'multi_room':
            split_r, split_c = mid + int(rng.integers(-3, 4)), mid + int(rng.integers(-3, 4))
            grid[split_r, margin:-margin] = 1
            grid[margin:-margin, split_c] = 1
            for lo, hi in ((margin + 4, split_c - door_width - 3),
                           (split_c + 4, size - margin - door_width - 3)):
                c = int(rng.integers(lo, max(lo + 1, hi)))
                grid[split_r, c:c + door_width] = 0
            for lo, hi in ((margin + 4, split_r - door_width - 3),
                           (split_r + 4, size - margin - door_width - 3)):
                r = int(rng.integers(lo, max(lo + 1, hi)))
                grid[r:r + door_width, split_c] = 0
        elif kind == 'hall':
            for r in range(size // 4, size * 3 // 4 + 1, size // 4):
                for c in range(size // 4, size * 3 // 4 + 1, size // 4):
                    dr, dc = rng.integers(-2, 3, size=2)
                    grid[r + dr:r + dr + 3, c + dc:c + dc + 3] = 1
        else:
            # A single room with one movable geometric pillar, no semantic labels.
            r, c = rng.integers(size // 3, size * 2 // 3, size=2)
            grid[r:r + 3, c:c + 3] = 1
    elif kind == 'corridor':
        bend = mid + int(rng.integers(-3, 4))
        carve(margin, margin + width, margin, size - margin)
        carve(margin, size - margin, size - margin - width, size - margin)
        carve(size - margin - width, size - margin, bend, size - margin)
        carve(mid, size - margin, bend, bend + width)
    elif kind == 'dead_end':
        carve(mid - width // 2, mid + width // 2 + 1, margin, size - margin)
        for i, c in enumerate(range(size // 5, size * 4 // 5, size // 5)):
            tip = int(rng.integers(margin, max(margin + 1, size // 5)))
            if i % 2:
                carve(mid, size - tip, c, c + width)
            else:
                carve(tip, mid + 1, c, c + width)
    elif kind == 'loop':
        inset = margin + int(rng.integers(0, 3))
        carve(inset, size - inset, inset, size - inset)
        grid[inset + width:size - inset - width, inset + width:size - inset - width] = 1
    return grid
