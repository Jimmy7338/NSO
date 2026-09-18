"""V27-A measured support, separate from the unchanged 256-patch planner view.

No labels, owner, world, reference geometry or evaluation score is consumed.
Keep the preceding full compact key index; local membership uses AFTER points.
This is adjacent-frame support, not a whole-option reward accumulator.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from nso.observed_state_v26 import geometry_state_v26


@dataclass(frozen=True)
class FeedbackPatchV27:
    key: tuple
    point: tuple
    bits: int
    best_range: float


@dataclass(frozen=True)
class FeedbackSupportV27:
    action_id: int
    planning_geometry_sha256: str
    patches: tuple
    support_sha256: str


def make_feedback_support_v27(action_id, planning_geometry_sha256, patches):
    """Compact already-whitelisted geometry; also usable by analytic fixtures."""
    if type(action_id) is not int or action_id < 0 or not isinstance(planning_geometry_sha256, str):
        raise ValueError('valid support action and planning identity required')
    rows = []
    for p in patches:
        key, point = tuple(p.key), tuple(float(x) for x in p.point)
        if (len(key) != 3 or any(type(x) is not int for x in key)
                or len(point) != 3 or not all(math.isfinite(x) for x in point)
                or type(p.bits) is not int or not 1 <= p.bits <= 255
                or not math.isfinite(p.best_range) or p.best_range <= 0):
            raise ValueError('invalid measured feedback patch')
        rows.append(FeedbackPatchV27(key, point, p.bits, float(p.best_range)))
    rows.sort(key=lambda p: p.key)
    if len({p.key for p in rows}) != len(rows):
        raise ValueError('duplicate feedback voxel key')
    payload = dict(action_id=action_id, planning_geometry_sha256=planning_geometry_sha256,
                   patches=[vars(p) for p in rows])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    return FeedbackSupportV27(action_id, planning_geometry_sha256, tuple(rows), digest)


def capture_feedback_support_v27(mapper, packet, anchor, remaining_budget, planning_state):
    """Reuse V26 validation, with no second mapper update and no mesh extraction.

    The full temporary state is never passed to plan() or _semantic_gain().
    Only the compact support object escapes this function.
    """
    full = geometry_state_v26(mapper, packet, anchor, remaining_budget, max_patches=2**31-1)
    sampled = (tuple(full.patches[i] for i in np.linspace(0, len(full.patches)-1, 256, dtype=int))
               if len(full.patches) > 256 else full.patches)
    if sampled != planning_state.patches:
        raise ValueError('feedback support does not reproduce the original 256 planning sample')
    for name in full.__dataclass_fields__:
        if name in ('patches', 'geometry_sha256'):
            continue
        equal = (np.array_equal(full.belief, planning_state.belief) if name == 'belief'
                 else getattr(full, name) == getattr(planning_state, name))
        if not equal:
            raise ValueError('feedback and planning observation differ: '+name)
    return make_feedback_support_v27(full.action_id, planning_state.geometry_sha256, full.patches)


def validate_feedback_pair_v27(before, after, support_before, support_after):
    for state, support in ((before, support_before), (after, support_after)):
        if (not isinstance(support, FeedbackSupportV27) or support.action_id != state.action_id
                or support.planning_geometry_sha256 != state.geometry_sha256):
            raise ValueError('feedback support must match its actual planning observation')
    if after.action_id != before.action_id + 1:
        raise ValueError('feedback requires one consecutive paid transition')


def local_support_v27(before, after, center):
    """Common voxel keys; after-point radius; new keys never count as improvement."""
    old = {p.key: p for p in before.patches}
    current = after.patches
    common = [p for p in current if p.key in old]
    local = [p for p in common if math.hypot(p.point[0]-center[0], p.point[1]-center[1]) <= 1.5]
    direction = {p.key for p in local if p.bits & ~old[p.key].bits}
    distance = {p.key for p in local if p.best_range < old[p.key].best_range-.01}
    return dict(comparable_patches=len(local), improved_common_patches=len(direction | distance),
        direction_improved_patches=len(direction), range_improved_patches=len(distance),
        both_improved_patches=len(direction & distance), new_keys=len(current)-len(common))
