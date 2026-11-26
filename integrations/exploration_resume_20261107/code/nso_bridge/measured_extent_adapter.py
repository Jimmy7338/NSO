"""Explicit measured-patch input variants; no scoring or vendor changes."""
from dataclasses import replace
from pathlib import Path
import threading

PROFILE = 'measured_extent_3m_v1'
MAX_HORIZONTAL_EXTENT_M = 3.0
_original = None
_lock = threading.Lock()
_DEFAULT = object()


def validate_profile(profile, *, policy_variant):
    if profile is not None and (profile != PROFILE or policy_variant != 'exploration_joint_v1'):
        raise ValueError('measured_extent_3m_v1 requires exploration_joint_v1')


def _dispatch(mapper, config=_DEFAULT):
    # Preserve omitted versus explicitly supplied config on the legacy path.
    extent = getattr(mapper, 'measured_extent_profile', None) == PROFILE
    instances = getattr(mapper, 'asset_grouping_profile', None) == 'measured_instance_support_v1'
    if not extent and not instances:
        return _original(mapper) if config is _DEFAULT else _original(mapper, config)
    from nso.response_features_v7 import ResponseFeatureConfig
    config = ResponseFeatureConfig() if config is _DEFAULT else config
    if extent:
        config = replace(config, max_horizontal_extent_m=MAX_HORIZONTAL_EXTENT_M)
    if instances:
        from nso_bridge.instance_asset_adapter import observed_instance_patches
        return observed_instance_patches(mapper, config, _original)
    return _original(mapper, config)


def install_dispatcher():
    """Bind all from-import aliases once; each mapper owns its profile."""
    global _original
    import nso.response_features_v7 as rf
    import nso.competition_candidates_v8 as v8
    import nso.competition_candidates_v8_1 as v81
    modules = (rf, v8, v81)
    vendor = Path(__file__).resolve().parents[1] / 'joint_vendor'
    with _lock:
        for module in modules:
            if not Path(module.__file__).resolve().is_relative_to(vendor.resolve()):
                raise RuntimeError('measured extent binding requires declared vendor namespace')
        if _original is None:
            refs = [m.observed_patches for m in modules]
            if not all(ref is refs[0] for ref in refs):
                raise RuntimeError('unexpected observed_patches aliases before extent binding')
            _original = refs[0]
        for module in modules:
            if module.observed_patches not in (_original, _dispatch):
                raise RuntimeError('unexpected observed_patches override')
        for module in modules:
            module.observed_patches = _dispatch


def profile_receipt(profile):
    if profile is None:
        return None
    return dict(profile=PROFILE, geometry_change=True, original_method_equivalence=False,
        original_max_horizontal_extent_m=2.2, max_horizontal_extent_m=MAX_HORIZONTAL_EXTENT_M,
        predicate='original measured planning-frame XY AABB 5th/95th span; fixed3m cap',
        rotation_invariant=False, parameter_scope='horizontal applicability cap only',
        point_cropping=False, semantic_dependent=False,
        scoring_weights_and_guards_unchanged=True,
        evidence_scope='geometry applicability variant; no semantic efficacy or physical accuracy claim')
