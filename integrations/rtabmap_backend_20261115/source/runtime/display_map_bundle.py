"""Immutable startup display bundle; no ROS publishers or navigation changes."""
BUNDLES = {
    'original': dict(name='original', topic='/nso_visual/map', frame='map', label='原始建图'),
    'navpair': dict(name='navpair', topic='/nso_pair/map', frame='nso_pair_map', label='导航独立建图'),
}

def map_bundle(name):
    return dict(BUNDLES[name])

def validate_map(metadata, bundle):
    if not metadata or metadata.get('frame') != bundle['frame']:
        raise ValueError('display_map_bundle_frame_mismatch')
    # Original pre-selector upstream remains compatible; pair requires explicit identity.
    source = metadata.get('source_bundle')
    if source != bundle['name'] and not (bundle['name'] == 'original' and source is None):
        raise ValueError('display_map_bundle_source_mismatch')
