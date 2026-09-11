"""Declared category/structure prior for an artificial asset positive control.

No coefficient is learned, calibrated, or derived from actual branch outcomes.
These scores are geometric proxies, not measured area predictions or F1.
"""
import numpy as np
from env.virtual3d import camera_pose
from nso.competition_candidates_v8 import measured_assets, aperture_support
from nso.response_features_v7 import response_features


def score_routes(mapper, routes, prefix_camera_poses):
    prefix_camera_poses = np.asarray(prefix_camera_poses, dtype=float)
    if (prefix_camera_poses.ndim != 3 or prefix_camera_poses.shape[1:] != (4, 4)
            or not len(prefix_camera_poses) or not np.isfinite(prefix_camera_poses).all()):
        raise ValueError('all actual prefix camera poses are required; keyframe decimation is insufficient')
    assets = measured_assets(mapper)
    features, _ = response_features(mapper, routes)
    q = mapper.quality_evidence()
    npoints = 0 if q is None else len(q['point'])
    # Restore the common descriptor's explicit physical scales. The quality
    # contribution remains a measured-voxel proxy, not unique truth area.
    common = ((features['N'][:, 0] + features['N'][:, 1]) * mapper.config.max_depth_m ** 2
              + features['N'][:, 2] * max(1, npoints) * .15 ** 2)
    prefix_support = np.zeros((len(assets), 25))
    for pose in prefix_camera_poses:
        for ai, asset in enumerate(assets):
            prefix_support[ai] = np.maximum(prefix_support[ai], aperture_support(
                asset, pose, mapper.config))
    scores = {mode: [] for mode in ('N', 'G', 'O', 'S', 'X', 'M')}
    rows = []
    for ri, route in enumerate(routes):
        support = np.zeros((len(assets), 25))
        for state in route['states'][1:]:
            pose = camera_pose(tuple(state[:2]), state[2], mapper.config, mapper.shape[0])
            for ai, asset in enumerate(assets):
                support[ai] = np.maximum(support[ai], aperture_support(asset, pose, mapper.config))
        factors = np.maximum(0., support - prefix_support).mean(axis=1)
        generic = []; semantic = []; swapped = []; marked = []
        for asset in assets:
            w, d, h = (asset['measured_width_m'], asset['measured_depth_m'], asset['measured_height_m'])
            vote = asset['class_vote']  # -1 is raw class2; +1 is raw class3.
            generic.append(w * h + 3 * w * d)
            semantic.append(w * h + 3 * (1 - vote) * w * d)
            swapped.append(w * h + 3 * (1 + vote) * w * d)
            marked.append(float(asset['marked_points'] > 0))
        generic, semantic, swapped, marked = map(np.asarray, (generic, semantic, swapped, marked))
        values = {'N': 0., 'G': factors @ generic, 'O': factors @ (generic * marked),
                  'S': factors @ (semantic * marked + generic * (1 - marked)),
                  'X': factors @ (swapped * marked + generic * (1 - marked)), 'M': factors @ generic}
        for mode, potential in values.items():
            scores[mode].append(float(common[ri] + potential / route['cost']))
        rows.append({'candidate_id': route['candidate_id'], 'common_proxy_per_action': float(common[ri]),
                     'aperture_factors': factors.tolist(), 'generic_area_proxies': generic.tolist(),
                     'semantic_area_proxies': semantic.tolist(), 'swapped_area_proxies': swapped.tolist(),
                     'prefix_aperture_support': prefix_support.mean(axis=1).tolist(),
                     'potential_proxies_before_cost': {k: float(v) for k, v in values.items()}})
    choices = {mode: int(min(range(len(routes)), key=lambda i: (-values[i], routes[i]['cost'],
                                                                routes[i]['candidate_id'])))
               for mode, values in scores.items()}
    return {'scores': scores, 'selected_candidate_ids': {mode: routes[index]['candidate_id']
                                                        for mode, index in choices.items()},
            'audit': rows, 'trained': False, 'calibrated': False,
            'claim_scope': 'hand-declared asset-structure prior with observed dimensions',
            'nominal_prior': 'width*height + 2*n_layers*width*depth; class2=3, class3=0, missing=1.5',
            'truth_or_future_outcomes_used': False, 'aperture_support_certifies_visibility': False}
