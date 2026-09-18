"""Class-blind second-stage choice for an explicitly committed observation option.

This is a selector, not a controller. The caller must execute the returned route
through the existing safety/budget runtime. It does not authorize motion.
"""
import math
from nso.observed_region_evidence_v17 import link_region


def select_continuation(anchor, regions, peers, routes, geometry_scores, score_rows):
    if not len(routes) == len(geometry_scores) == len(score_rows):
        raise ValueError('candidate and score lengths differ')
    if len({r['candidate_id'] for r in routes}) != len(routes):
        raise ValueError('duplicate candidate IDs')
    for route, score, row in zip(routes, geometry_scores, score_rows):
        if row['candidate_id'] != route['candidate_id'] or not math.isfinite(score):
            raise ValueError('invalid aligned geometry scores')
        if not math.isfinite(route['cost']) or route['cost'] < 0:
            raise ValueError('invalid route cost')
        if any(not math.isfinite(v) or v < 0 for v in row['corrected_aperture']):
            raise ValueError('invalid observed aperture support')
    link = link_region(anchor, regions, peers)
    ai = link['matched_asset_index']
    eligible = []
    if link['status'] == 'unique_observed_region':
        for i, (route, row) in enumerate(zip(routes, score_rows)):
            if route.get('asset_index') == ai and route['group'] == f'asset_{ai}_aperture_view':
                if not 0 <= ai < len(row['corrected_aperture']):
                    raise ValueError('asset and aperture columns differ')
                if row['corrected_aperture'][ai] > 0 and geometry_scores[i] > 0:
                    eligible.append(i)
    def best(indices):
        return min(indices, key=lambda i: (-geometry_scores[i], routes[i]['cost'], routes[i]['candidate_id']))
    if eligible:
        selected = best(eligible); status = 'continue_observed_region'; reason = 'unique_region_with_supported_view'
    else:
        candidates = [i for i, value in enumerate(geometry_scores) if value > 0]
        selected = best(candidates) if candidates else None
        status = 'common_geometry_fallback' if candidates else 'request_guarded_return'
        reason = link['status'] if ai is None else 'no_supported_same_region_view'
    return dict(status=status, reason=reason, region_link=link,
        candidate_id=None if selected is None else routes[selected]['candidate_id'],
        candidate_index=selected, semantic_labels_used=False,
        target_specific_credit_allowed=status == 'continue_observed_region',
        motion_authorized=False, actual_target_observation_verified=False)
