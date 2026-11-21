"""Pure fixed-state B/B2 visibility diagnostics. No installation or actor step."""
from pathlib import Path
import importlib.util
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'method/occlusion_repair.py'
spec = importlib.util.spec_from_file_location('isolated_occlusion_trace', PATH)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def audit_state(mapper, models, measured_ray, selected_route, paid_action):
    from nso.semantic_completion_v3 import ObjectCompletionModel
    result = dict(paid_action=int(paid_action), candidate_id=selected_route['candidate_id'],
                  pose=selected_route['pose'], retained_keyframes=len(mapper.keyframes),
                  counterfactual_only=True, actor_or_mapper_modified=False, models={})
    for name, model in models.items():
        before = np.zeros(len(model.points)); after = before.copy(); calls = []
        states = list(dict.fromkeys(tuple(map(int, s)) for s in selected_route['outbound_states'][1:]))
        for state in states:
            old, old_audit = repair.trace_gain(model, state[:2], state[2], measured_ray, repair_enabled=False)
            np.testing.assert_array_equal(old, ObjectCompletionModel.gain(model, state[:2], state[2], measured_ray))
            new, new_audit = repair.trace_gain(model, state[:2], state[2], measured_ray, repair_enabled=True)
            assert np.all(new >= old)
            np.maximum(before, old, out=before); np.maximum(after, new, out=after)
            calls.append(new_audit)
        result['models'][name] = dict(original_route_area_m2=float(before.sum()),
             repaired_route_area_m2=float(after.sum()), recovered_union_area_m2=float((after-before).sum()),
             original_gain_elementwise_identical=True, unique_outbound_views=len(states),
             points=len(model.points), per_view=calls,
             history_model_audit={k:v for k,v in model.audit.items() if k != 'objects'})
    return result
