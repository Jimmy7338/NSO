"""Exact bounded two-hypothesis planning on a declared finite observation graph.

Geometry models supply shared poses/edges/anchor, signatures, observed bitsets,
and evaluator-only terminal(mask) with feasible, coverage, surface, and joint.
This is a value-of-information screen, never a physical planner or Q estimate.
"""
from collections import deque
from functools import lru_cache

NEG = -1e30


class SolverLimitV33(RuntimeError):
    pass


class ExactCoverageBeliefSolverV33:
    def __init__(self, models, initial_masks, maximum_states=1500000):
        if len(models) != 2 or len(initial_masks) != 2:
            raise ValueError('exactly two public equal-prior hypotheses required')
        a, b = models
        if a.poses != b.poses or a.edges != b.edges or a.anchor != b.anchor:
            raise ValueError('identical legal action graph and return anchor required')
        if type(maximum_states) is not int or maximum_states < 1:
            raise ValueError('positive memo-state limit required')
        self.models, self.initial = tuple(models), tuple(initial_masks)
        self.edges, self.anchor = a.edges, a.anchor
        self.maximum_states, self.states = maximum_states, 0
        reverse = [[] for _ in self.edges]
        for node, links in enumerate(self.edges):
            for _, other in links:
                reverse[other].append(node)
        self.return_distance = {self.anchor: 0}
        queue = deque([self.anchor])
        while queue:
            node = queue.popleft()
            for before in reverse[node]:
                if before not in self.return_distance:
                    self.return_distance[before] = self.return_distance[node]+1
                    queue.append(before)
        self.known = lru_cache(None)(self._known)
        self.uncertain = lru_cache(None)(self._uncertain)

    def clear(self):
        self.known.cache_clear(); self.uncertain.cache_clear()

    def _count(self):
        self.states += 1
        if self.states > self.maximum_states:
            raise SolverLimitV33('fixed memo-state cap exceeded; no certified optimum')

    def _stop(self, hypothesis, node, mask):
        if node != self.anchor:
            return NEG
        outcome = self.models[hypothesis].terminal(mask)
        return float(outcome['joint']) if outcome['feasible'] else NEG

    def _known(self, hypothesis, node, remaining, mask):
        self._count()
        if self.return_distance.get(node, 10**9) > remaining:
            return NEG
        best = self._stop(hypothesis, node, mask)
        if remaining:
            for _, following in self.edges[node]:
                best = max(best, self.known(hypothesis, following, remaining-1,
                    mask | self.models[hypothesis].observed_masks[following]))
        return best

    def _uncertain(self, node, remaining, mask0, mask1):
        self._count()
        if self.return_distance.get(node, 10**9) > remaining:
            return NEG
        if self.models[0].signatures[node] != self.models[1].signatures[node]:
            branch = [self.known(h, node, remaining, m) for h, m in enumerate((mask0, mask1))]
            return sum(branch)/2 if min(branch) > NEG/2 else NEG
        stop = [self._stop(h, node, m) for h, m in enumerate((mask0, mask1))]
        best = sum(stop)/2 if min(stop) > NEG/2 else NEG
        if remaining:
            for _, following in self.edges[node]:
                masks = [old | model.observed_masks[following]
                         for model, old in zip(self.models, (mask0, mask1))]
                if self.models[0].signatures[following] == self.models[1].signatures[following]:
                    value = self.uncertain(following, remaining-1, *masks)
                else:
                    branch = [self.known(h, following, remaining-1, masks[h]) for h in (0, 1)]
                    value = sum(branch)/2 if min(branch) > NEG/2 else NEG
                best = max(best, value)
        return best

    def _action(self, hypothesis, node, remaining, masks):
        if hypothesis is None:
            value = self.uncertain(node, remaining, *masks)
            stops = [self._stop(h, node, masks[h]) for h in (0, 1)]
            stop = sum(stops)/2 if min(stops) > NEG/2 else NEG
        else:
            value = self.known(hypothesis, node, remaining, masks[hypothesis])
            stop = self._stop(hypothesis, node, masks[hypothesis])
        if value <= NEG/2:
            raise ValueError('no coverage-qualified exact-return policy in declared budget')
        if stop >= value-1e-12:
            return None
        if not remaining:
            raise AssertionError('positive continuation value with no paid budget')
        for action, following in self.edges[node]:
            next_masks = [m | model.observed_masks[following] for model, m in zip(self.models, masks)]
            if hypothesis is not None:
                candidate = self.known(hypothesis, following, remaining-1, next_masks[hypothesis])
            elif self.models[0].signatures[following] == self.models[1].signatures[following]:
                candidate = self.uncertain(following, remaining-1, *next_masks)
            else:
                branches = [self.known(h, following, remaining-1, next_masks[h]) for h in (0, 1)]
                candidate = sum(branches)/2 if min(branches) > NEG/2 else NEG
            if candidate >= value-1e-12:
                return action
        raise AssertionError('optimal value has no legal witness action')

    def witness(self, actual, budget, policy='G', hint=None):
        if type(actual) is not int or actual not in (0, 1):
            raise ValueError('actual hypothesis must be0 or1')
        if type(budget) is not int or budget < 0:
            raise ValueError('nonnegative integer paid budget required')
        if policy not in ('G', 'class_oracle', 'swapped_class_with_correction'):
            raise ValueError('unknown policy')
        if policy == 'swapped_class_with_correction':
            if type(hint) is not int or hint != 1-actual:
                raise ValueError('swapped control requires the opposite initial hypothesis')
        elif hint is not None:
            raise ValueError('G/class oracle must not receive an extra hint')
        node, remaining, masks = self.anchor, budget, list(self.initial)
        hypothesis = actual if policy == 'class_oracle' else hint
        if policy == 'G':
            hypothesis = None
        actions, nodes, sources, history = [], [node], [], []
        first_information = None
        # If the initial observation separates hypotheses, G gets that fact
        # for free too. Paired-prefix gates separately forbid such a scene
        # from being presented as advance semantic information.
        if self.models[0].signatures[node] != self.models[1].signatures[node]:
            hypothesis = actual; first_information = 0
        while True:
            action = self._action(hypothesis, node, remaining, masks)
            if action is None:
                break
            sources.append('geometry_belief' if hypothesis is None else
                           'revealed_geometry' if first_information is not None else 'class_hypothesis')
            node = dict(self.edges[node])[action]; remaining -= 1
            actions.append(action); nodes.append(node)
            masks = [mask | model.observed_masks[node] for model, mask in zip(self.models, masks)]
            revealed = self.models[0].signatures[node] != self.models[1].signatures[node]
            if revealed:
                if first_information is None: first_information = len(actions)
                hypothesis = actual
            history.append(dict(action=action, node=node, remaining=remaining,
                                geometric_hypothesis_revealed=revealed))
        if node != self.anchor or remaining < 0:
            raise AssertionError('witness failed paid exact return')
        outcome = dict(self.models[actual].terminal(masks[actual]))
        if policy in ('G', 'class_oracle') and not outcome['feasible']:
            raise AssertionError('primary witness violates terminal coverage constraint')
        return dict(policy=policy, actual_hypothesis=actual, initial_hint=hint,
            suffix_actions=actions, suffix_nodes=nodes,
            suffix_poses=[list(self.models[actual].poses[n]) for n in nodes],
            suffix_paid_actions=len(actions), returned_exact_pose=True, budget_compliant=True,
            first_information_suffix_action=first_information, decision_sources=sources,
            events=history, terminal=outcome, final_observed_mask_hex=hex(masks[actual]),
            actual_coverage_qualified=bool(outcome['feasible']),
            scope='exact finite-model witness; not an executed physical or ANS policy')
