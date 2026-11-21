"""OFFLINE ONLY. Radius-preserving, fixed entry/deep standoff translation.

This changes candidate targets and membership, not scoring or circle safety.
It does not modify imported module globals or any production source file.
"""
import math
from types import FunctionType

PROFILE = 'clearance_matched_entries_v1'
REFERENCE_RADIUS_M = .2

def offsets_for_radius(radius_m):
    radius_m = float(radius_m)
    if not math.isfinite(radius_m) or radius_m < 0:
        raise ValueError('radius must be finite and nonnegative')
    shift = max(0., radius_m - REFERENCE_RADIUS_M)
    return (('entry', .4 + shift), ('deep', .8 + shift))

def offline_generator(original_generator, radius_m):
    """Clone the original function with one private constant substitution.

    The unmodified original module is left intact, so actual trace replay
    continues to use the original candidate rules. No source code is patched.
    """
    if original_generator.__globals__['VIEW_OFFSETS_M'] != (('entry', .4), ('deep', .8)):
        raise ValueError('source entry contract differs from frozen reference')
    if original_generator.__globals__['TARGET_ERROR_LIMIT_M'] != .35:
        raise ValueError('source projection tolerance differs from frozen reference')
    namespace = dict(original_generator.__globals__)
    namespace['VIEW_OFFSETS_M'] = offsets_for_radius(radius_m)
    result = FunctionType(original_generator.__code__, namespace,
                          original_generator.__name__, original_generator.__defaults__,
                          original_generator.__closure__)
    result.__kwdefaults__ = dict(original_generator.__kwdefaults__ or {})
    return result
