"""Prospectively fixed total-potential objective; unchanged V8 rate comparator.

The multiplication is applied to every method with the same paid route cost.
No new coefficient, fitted parameter or outcome-dependent choice is introduced.
"""
import numpy as np

from nso.competition_prior_v8 import score_routes as rate_score_routes


def score_routes(mapper, routes, prefix_camera_poses):
    if not routes:
        raise ValueError('a nonempty predeclared route pool is required')
    costs = np.asarray([r['cost'] for r in routes], dtype=float)
    ids = [r['candidate_id'] for r in routes]
    if (not np.isfinite(costs).all() or np.any(costs <= 0)
            or np.any(costs > 48) or len(ids) != len(set(ids))):
        raise ValueError('unique routes with positive paid costs at most 48 required')
    prediction = rate_score_routes(mapper, routes, prefix_camera_poses)
    rate = prediction['scores']
    total = {mode: (np.asarray(values) * costs).tolist() for mode, values in rate.items()}
    if not all(np.isfinite(values).all() for values in total.values()):
        raise ValueError('nonfinite score')
    selected = {mode: ids[min(range(len(ids)), key=lambda i: (-values[i], costs[i], ids[i]))]
                for mode, values in total.items()}
    return {**prediction, 'scores': total, 'selected_candidate_ids': selected,
            'rate_scores': rate,
            'rate_selected_candidate_ids': prediction['selected_candidate_ids'],
            'objective': 'fixed_total_potential_under_48_paid_action_cap',
            'total_score_formula': 'frozen_rate_score * candidate_planned_paid_cost',
            'unused_budget_policy': 'stop and hold map; no additional free observations',
            'learned_or_calibrated_response_model': False}
