"""Shared experiment scene identity and construction, never a planner input."""
from dataclasses import asdict
import hashlib
import json

from env.virtual3d_v2 import VirtualConfigV2, VirtualWorldV2


def scene_configuration(config, entry):
    family = entry.get('world_family', config.get('world_family'))
    if family is None:
        family = 'inspection_v4' if entry.get('layout') == 'inspection' else 'virtual_v2'
    values = config['environment'] | entry.get('environment', {})
    if family == 'virtual_v2':
        if entry.get('layout') not in ('rooms', 'warehouse'):
            raise ValueError('virtual_v2 requires rooms or warehouse layout')
        return family, VirtualConfigV2(**values)
    if family == 'inspection_v4':
        if entry.get('layout', 'inspection') != 'inspection':
            raise ValueError('inspection_v4 requires inspection layout')
        from env.virtual3d_inspection_v4 import InspectionConfigV4
        return family, InspectionConfigV4(**values)
    raise ValueError(f'unknown world_family: {family}')


def scene_key(config, entry, version=3):
    family, c = scene_configuration(config, entry)
    condition = entry.get('semantic_condition', 'aligned')
    if family == 'virtual_v2':
        # Preserve all archived identities, including pre-backboard version 1.
        key = f"{entry['layout']}_{entry['seed']}_{condition}_d{c.depth_sigma_m}_p{c.pose_noise_m}"
        return key + (f'_opaque{int(c.occluded_objects)}' if version >= 2 else '')
    if version < 3:
        raise ValueError('inspection scenes require scene_key_version 3')
    payload = {'family': family, 'environment': asdict(c), 'seed': entry['seed'],
               'semantic_condition': condition}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"inspection_{entry['seed']}_{condition}_{digest}"


def create_scene(config, entry):
    family, c = scene_configuration(config, entry)
    args = {'seed': entry['seed'], 'semantic_condition': entry.get('semantic_condition', 'aligned')}
    if family == 'virtual_v2':
        return VirtualWorldV2(c, layout=entry['layout'], **args)
    from env.virtual3d_inspection_v4 import InspectionWorldV4
    return InspectionWorldV4(c, **args)
