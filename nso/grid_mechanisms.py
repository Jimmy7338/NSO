"""CPU semantic/region ablations and bounded-controller reachability records."""
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from nso.frontier_policy import FrontierPolicy, trace_path, CROSS
from nso.grid_topology import ObservedTopology
from utils.grid_geometry import DIRECTIONS

MECHANISM_METHODS = ('gain_control', 'gain_semantic', 'gain_structure', 'gain_semantic_structure',
                     'gain_semantic_structure_rpn', 'gain_budget_rule', 'gain_semantic_structure_budget_rule',
                     'gain_semantic_structure_budget_soft')


def action_cost(path, heading, final_heading):
    cost = 0
    for a, b in zip(path, path[1:]):
        target = DIRECTIONS.index((b[0]-a[0], b[1]-a[1]))
        cost += 1 + min((target-heading) % 4, (heading-target) % 4)
        heading = target
    return cost + min((final_heading-heading) % 4, (heading-final_heading) % 4)


class MechanismPolicy(FrontierPolicy):
    def __init__(self, config, method, max_candidates=24, options=None, predictor=None):
        self.variant = method
        self.options = dict(semantic_weight=1., structure_weight=1., goal_budget=32,
                            topology_period=10, max_door_width=7)
        supplied = options or {}
        if set(supplied) - set(self.options):
            raise ValueError('unknown mechanism option')
        self.options.update(supplied)
        if any(not np.isfinite(v) or v < 0 for v in self.options.values()):
            raise ValueError('mechanism options must be finite and nonnegative')
        if any(self.options[k] < 1 or int(self.options[k]) != self.options[k]
               for k in ('goal_budget', 'topology_period', 'max_door_width')):
            raise ValueError('budgets and periods must be positive integers')
        if method not in MECHANISM_METHODS or (method.endswith('_rpn') and predictor is None):
            raise ValueError('invalid mechanism method or missing frozen RPN')
        self.predictor = predictor
        super().__init__(config, 'gain_per_cost', max_candidates)

    def reset(self):
        super().reset()
        self.topology = ObservedTopology(self.options['max_door_width'])
        self.semantic = None
        self.semantic_updates = self.rpn_calls = self.goal_changes = 0
        self.selection_audit = []
        self.attempts = []
        self._active = None
        self._candidate_rows = []
        self._blocked_goals = set()
        self._region_frontier = {}

    def _finish(self, obs, status, success=None):
        if self._active is not None:
            self.attempts.append(dict(self._active, end_step=obs.step, status=status,
                                      controller_success=success,
                                      elapsed_actions=obs.step-self._active['start_step']))
            self._active = None

    def observe_outcome(self, obs, episode_done=False):
        if self._active is None:
            return
        if obs.position == tuple(self._active['goal']) and obs.heading == self._active['goal_heading']:
            self._finish(obs, 'reached', 1)
        elif obs.collision:
            self._finish(obs, 'collision', 0)
            self.goal = None
        elif obs.step-self._active['start_step'] >= self.options['goal_budget']:
            self._blocked_goals.add(tuple(self._active['goal']))
            self._finish(obs, 'budget_exhausted', 0)
            self.goal = None
        elif episode_done:
            self._finish(obs, 'episode_censored')

    def act(self, obs):
        if self.semantic is None:
            self.semantic = np.zeros(obs.belief.shape, np.float32)
        if 'semantic' in self.variant and hasattr(obs, 'semantic'):
            valid = obs.visible & np.isfinite(obs.semantic)
            self.semantic[valid] = obs.semantic[valid]
            self.semantic_updates += 1
        if 'structure' in self.variant and (self.topology.labels is None or obs.step % self.options['topology_period'] == 0):
            self.topology.update(obs.belief)
            frontier = (obs.belief == 0) & binary_dilation(obs.belief == -1, structure=CROSS)
            self._region_frontier = {n: np.count_nonzero(frontier & (self.topology.labels == n))
                                     for n in self.topology.nodes}
        return super().act(obs)

    def _select(self, obs, traversable):
        if self._active is not None:
            self._finish(obs, 'path_invalidated_censored')
        self._candidate_rows = []
        self._clearance = distance_transform_edt(obs.belief != 1)
        self._unknown_near = binary_dilation(obs.belief == -1, structure=CROSS)
        selected = super()._select(obs, traversable)
        for row in self._candidate_rows:
            row['selected'] = selected and tuple(row['goal']) == self.goal and row['goal_heading'] == self.goal_heading
            self.selection_audit.append(row)
        return selected

    def _candidate_score(self, obs, cell, heading, gain, distance, parent, base_score):
        path = [obs.position] + trace_path(parent, obs.position, cell)
        cost = action_cost(path, obs.heading, heading)
        r, c = cell
        patch = self.semantic[max(0,r-5):r+6, max(0,c-5):c+6]
        semantic = float(patch.max()) if 'semantic' in self.variant else 0.
        node, structure = 0, 0.
        if 'structure' in self.variant:
            node = int(self.topology.labels[cell])
            start_node = int(self.topology.labels[obs.position])
            hops = self.topology.hops(start_node).get(node)
            maximum = max(self._region_frontier.values(), default=0)
            if hops is not None and maximum:
                structure = self._region_frontier.get(node, 0) / maximum / (1+hops)
        index = tuple(np.asarray(path).T)
        features = [distance / self.options['goal_budget'], cost / self.options['goal_budget'],
                    float(self._clearance[index].min()) / 10,
                    float(self._unknown_near[index].mean()), gain / 1000]
        probability = 1.
        if self.variant.endswith('_rpn'):
            probability = float(self.predictor.predict(np.asarray([features]))[0])
            self.rpn_calls += 1
        elif self.variant.endswith('_budget_soft'):
            # Smooth deterministic action-cost comparator; fixed four-action
            # transition scale, never masks every candidate or trains a model.
            probability = float(1 / (1 + np.exp(np.clip((cost-self.options['goal_budget']) / 4, -40, 40))))
        score = base_score * (1 + self.options['semantic_weight']*semantic
                              + self.options['structure_weight']*structure) * probability
        # Timed-out goals remain excluded for this episode; empty scores use the
        # base policy's bounded scan/STOP fallback, never an oracle target.
        over_budget = self.variant.endswith('_budget_rule') and cost > self.options['goal_budget']
        excluded = cell in self._blocked_goals or over_budget
        self._candidate_rows.append(dict(step=obs.step, goal=list(cell), goal_heading=heading,
            geometric_connected=True, controller_success=None, features=features,
            estimated_actions=cost, gain_per_cost=base_score, semantic_term=semantic,
            structure_term=float(structure), region_id=node, reach_probability=probability,
            score=score, excluded=excluded,
            exclusion_reason='estimated_over_budget' if over_budget else 'previous_timeout' if excluded else None))
        return -float('inf') if excluded else score

    def _selected(self, obs):
        row = next(r for r in self._candidate_rows if tuple(r['goal']) == self.goal)
        self.goal_changes += 1
        self._active = dict(start_step=obs.step, goal=list(self.goal), goal_heading=self.goal_heading,
                            features=row['features'], geometric_connected=True,
                            budget=self.options['goal_budget'])

    def diagnostics(self):
        labeled = [a for a in self.attempts if a['controller_success'] is not None]
        return dict(semantic_updates=self.semantic_updates, topo_updates=self.topology.updates,
                    rpn_calls=self.rpn_calls, goal_changes=self.goal_changes,
                    topology_nodes=len(self.topology.nodes), topology_edges=len(self.topology.edges),
                    controller_attempts=len(self.attempts), controller_labeled=len(labeled),
                    controller_success_rate=float(np.mean([a['controller_success'] for a in labeled])) if labeled else None,
                    controller_censored=len(self.attempts)-len(labeled))
