"""Online finite planning with one noisy future information event per lookahead.

Both geometry and semantic policies use this same approximation. Actual belief
updates are external, from paid sensor data; no actual-world identity or stored
witness is accepted here. Template visibility is a potential reward, not Q.
"""
from collections import deque
from functools import lru_cache
from math import isfinite
from time import monotonic

NEG = -1e30


class PlannerLimitV35(RuntimeError):
    pass


class ForecastBeliefPlannerV35:
    def __init__(self, models, informative_nodes, reliability=.99,
                 maximum_states=1500000, maximum_seconds=20.):
        if len(models) != 2:
            raise ValueError('two common public templates required')
        a, b = models
        if (a.poses, a.edges, a.anchor) != (b.poses, b.edges, b.anchor):
            raise ValueError('common legal graph and return anchor required')
        if not .5 <= reliability < 1 or maximum_states < 1 or maximum_seconds <= 0:
            raise ValueError('nonperfect informative channel and positive resource bounds required')
        self.models = tuple(models)
        self.poses, self.edges, self.anchor = a.poses, a.edges, a.anchor
        self.informative = frozenset(informative_nodes)
        if not self.informative.issubset(range(len(self.poses))):
            raise ValueError('information node outside public graph')
        self.reliability = float(reliability)
        self.maximum_states, self.maximum_seconds = maximum_states, maximum_seconds
        reverse = [[] for _ in self.edges]
        for node, links in enumerate(self.edges):
            for _, following in links:
                reverse[following].append(node)
        self.return_distance = {self.anchor: 0}
        queue = deque([self.anchor])
        while queue:
            node = queue.popleft()
            for prior in reverse[node]:
                if prior not in self.return_distance:
                    self.return_distance[prior] = self.return_distance[node]+1
                    queue.append(prior)

    def select(self, node, remaining, masks, probability0, geometry_feedback=True,
               excluded_information_nodes=()):
        if (type(node) is not int or node not in range(len(self.poses))
                or type(remaining) is not int or remaining < 0 or len(masks) != 2
                or not isfinite(probability0) or not 0 <= probability0 <= 1):
            raise ValueError('valid paid state, two template masks and posterior required')
        masks = tuple(int(m) for m in masks)
        if any(m < 0 for m in masks):
            raise ValueError('negative template mask')
        probability0 = float(probability0)
        excluded = frozenset(excluded_information_nodes)
        diagnostic = self.informative - excluded
        start = monotonic()
        states = 0

        def count():
            nonlocal states
            states += 1
            if states > self.maximum_states:
                raise PlannerLimitV35('declared memo-state limit exceeded')
            if states % 1024 == 0 and monotonic()-start > self.maximum_seconds:
                raise PlannerLimitV35('declared planning-time limit exceeded')

        def stop(n, m0, m1, p0):
            if n != self.anchor:
                return NEG
            outcomes = [m.terminal(bits) for m, bits in zip(self.models, (m0, m1))]
            # Retain both-template qualification even when one is unlikely.
            if not all(o['feasible'] for o in outcomes):
                return NEG
            return p0*outcomes[0]['joint']+(1-p0)*outcomes[1]['joint']

        def updated_masks(n, m0, m1):
            return (m0 | self.models[0].observed_masks[n],
                    m1 | self.models[1].observed_masks[n])

        @lru_cache(None)
        def quiet(n, left, m0, m1, p0):
            """No additional information event inside this forecast branch."""
            count()
            if self.return_distance.get(n, 10**9) > left:
                return NEG
            best = stop(n, m0, m1, p0)
            if left:
                for _, following in self.edges[n]:
                    best = max(best, quiet(following, left-1,
                        *updated_masks(following, m0, m1), p0))
            return best

        def continuation(n, left, m0, m1):
            if geometry_feedback and n in diagnostic:
                r = self.reliability
                likelihood0 = probability0*r+(1-probability0)*(1-r)
                likelihood1 = 1-likelihood0
                posterior0 = probability0*r/likelihood0
                posterior1 = probability0*(1-r)/likelihood1
                branches = (quiet(n, left, m0, m1, posterior0),
                            quiet(n, left, m0, m1, posterior1))
                if min(branches) <= NEG/2:
                    return NEG
                return likelihood0*branches[0]+likelihood1*branches[1]
            return value(n, left, m0, m1)

        @lru_cache(None)
        def value(n, left, m0, m1):
            count()
            if self.return_distance.get(n, 10**9) > left:
                return NEG
            best = stop(n, m0, m1, probability0)
            if left:
                for _, following in self.edges[n]:
                    best = max(best, continuation(following, left-1,
                        *updated_masks(following, m0, m1)))
            return best

        try:
            best = value(node, remaining, *masks)
            stop_value = stop(node, *masks, probability0)
            candidates = []
            if remaining:
                for action, following in self.edges[node]:
                    candidate = continuation(following, remaining-1,
                        *updated_masks(following, *masks))
                    candidates.append(dict(action=action, following=following,
                        expected_proxy=candidate,
                        return_feasible=self.return_distance.get(following, 10**9) <= remaining-1,
                        forecast_information=bool(geometry_feedback and following in diagnostic)))
            if best <= NEG/2:
                raise ValueError('no both-template coverage-qualified exact-return plan')
            action = None
            if stop_value < best-1e-12:
                action = next(c['action'] for c in candidates if c['expected_proxy'] >= best-1e-12)
            if monotonic()-start > self.maximum_seconds:
                raise PlannerLimitV35('declared planning-time limit exceeded')
            return dict(action=action, expected_proxy=best, stop_value=stop_value,
                candidates=candidates, memo_states=states,
                probability0=probability0, remaining=remaining,
                geometry_feedback=bool(geometry_feedback),
                forecast_reliability=self.reliability,
                forecast_scope='one nonperfect information event per receding lookahead; uncalibrated approximation',
                objective_scope='public-template potential area times ideal center coverage, not measured Q',
                both_template_terminal_qualification=True)
        finally:
            quiet.cache_clear()
            value.cache_clear()


def load_public_models_v35(root, parent_id):
    """Load both nominal CAD/exposure priors, never a selected actual template.

    Saved geometry tables are explicitly public task knowledge for both arms.
    Their reference areas are CAD potential, never the executed scene's Q.
    No policy/witness or fixed-route quality file is read.
    """
    import json
    from pathlib import Path
    from nso.pixel_information_v34 import SavedPotentialModelV34
    if parent_id not in ('P00', 'P01'):
        raise ValueError('unknown public parent')
    path = Path(root)/'audit_results/v33_direction_information_r1_20260917'/f'{parent_id}_geometry.json'
    tables = json.loads(path.read_text())['tables']
    models = tuple(SavedPotentialModelV34(t, ['unused']*len(t['poses'])) for t in tables)
    if len(models) != 2 or models[0].prefix_nodes != models[1].prefix_nodes:
        raise ValueError('paired public templates and common prefix required')
    nodes = models[0].prefix_nodes
    actions = []
    for current, following in zip(nodes, nodes[1:]):
        actions.append(next(a for a, n in models[0].edges[current] if n == following))
    if len(actions) != 18 or nodes[0] != models[0].anchor or nodes[-1] != models[0].anchor:
        raise ValueError('declared paid common prefix differs')
    return models, tuple(actions)
