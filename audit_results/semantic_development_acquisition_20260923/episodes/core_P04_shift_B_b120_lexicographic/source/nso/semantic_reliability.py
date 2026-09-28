"""Finite shared-class reliability and one-observation value-of-information.

This is a standard hierarchical Bayes construction, not a claim of a new
inference algorithm. Geometry scores supplied by the current CPU residual
adapter are *generalized evidence*, not calibrated sensor log likelihoods.
The two fixed domain models use either the geometry prior (lambda=0) or the
class prior (lambda=1), with equal initial weights. A class shares its domain
model across independently associated, already observed instances.

No simulator, renderer, evaluator, filesystem or future sensor is accessed.
Caller responsibilities include association identity, evidence provenance,
correlation control, and budget feasibility of the terminal action set.
"""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np


def _probabilities(value, count=None):
    array = np.asarray(value, dtype=float)
    if (array.ndim != 1 or len(array) < 2 or
            (count is not None and array.shape != (count,)) or
            not np.isfinite(array).all() or np.any(array <= 0)):
        raise ValueError('strictly positive finite structure weights required')
    array = array / array.max()
    return array / array.sum()


def _logsumexp(values):
    maximum = float(np.max(values))
    if np.isneginf(maximum):
        return maximum
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _normalize_logs(values):
    normalizer = _logsumexp(values)
    if not np.isfinite(normalizer):
        raise ValueError('observation outcome has no positive-probability structure')
    result = np.exp(values - normalizer)
    return result / result.sum()


class SemanticReliabilityBelief:
    """Replace cumulative instance messages; exclude self from prior transfer.

    For instance i with observed class c, q_k(h|c) is its geometry or class
    prior. Each peer contributes m_j(k)=sum_h q_k(h|c) exp(L_j(h)). The prior
    for i uses w^-i(k) proportional to w0(k) times peer messages. Its own
    L_i is then applied exactly once. In the nonsharing control w^-i=w0.

    An individual instance therefore has exactly the same marginal posterior
    as ordinary Bayes using the fixed initial mixture prior. New predictions
    are possible only through observed evidence from other same-class objects.
    """

    def __init__(self, *, geometry_prior, class_structure_priors,
                 share_across_instances=True):
        if type(share_across_instances) is not bool:
            raise ValueError('share_across_instances must be bool')
        if not isinstance(class_structure_priors, Mapping):
            raise TypeError('explicit class to structure prior mapping required')
        self._geometry_prior = _probabilities(geometry_prior)
        self._count = len(self._geometry_prior)
        self._class_priors = {}
        for label, prior in class_structure_priors.items():
            if not isinstance(label, str) or not label:
                raise ValueError('nonempty observed class string required')
            self._class_priors[label] = _probabilities(prior, self._count)
        self.share_across_instances = share_across_instances
        self._instances = {}
        self._is_forecast_branch = False

    @property
    def structure_count(self):
        return self._count

    @property
    def instance_ids(self):
        return tuple(sorted(self._instances))

    def _validate_class(self, class_label):
        if class_label is not None and class_label not in self._class_priors:
            raise ValueError('class must be absent or belong to the public prior table')

    def register(self, instance_id, class_label=None):
        if not isinstance(instance_id, str) or not instance_id or instance_id in self._instances:
            raise ValueError('unique nonempty observed instance ID required')
        self._validate_class(class_label)
        self._instances[instance_id] = dict(class_label=class_label,
                                           log_evidence=np.zeros(self._count))

    def set_class(self, instance_id, class_label):
        """Set current qualified observed class, or None for uncertain/conflict.

        This preserves geometry. No observations of a formerly qualified
        class keep contributing to that old class after qualification changes.
        """
        self._validate_class(class_label)
        self._instances[instance_id]['class_label'] = class_label

    def replace_log_evidence(self, instance_id, cumulative_log_evidence):
        """Replace a whole cumulative paid-geometry message, never append it.

        The caller already suppresses correlated or repeated measurements.
        Common additive offsets do not carry evidence and are removed. No
        calibration or independence is inferred from these numeric values.
        """
        state = self._instances[instance_id]
        evidence = np.asarray(cumulative_log_evidence, dtype=float)
        if evidence.shape != (self._count,) or not np.isfinite(evidence).all():
            raise ValueError('one finite cumulative geometry score per structure required')
        shifted = evidence - evidence.max()
        if not np.isfinite(shifted).all():
            raise ValueError('geometry score differences exceed numeric range')
        state['log_evidence'] = shifted.copy()

    def _priors(self, class_label):
        semantic = (self._geometry_prior if class_label is None
                    else self._class_priors[class_label])
        return np.stack((self._geometry_prior, semantic))

    def posterior(self, instance_id):
        state = self._instances[instance_id]
        label = state['class_label']
        priors = self._priors(label)
        peers = ([key for key in self.instance_ids if key != instance_id and
                  self._instances[key]['class_label'] == label]
                 if self.share_across_instances and label is not None else [])
        log_weights = np.log(np.array([.5, .5]))
        for key in peers:
            evidence = self._instances[key]['log_evidence']
            log_weights += np.array([_logsumexp(np.log(prior) + evidence)
                                     for prior in priors])
        weights = _normalize_logs(log_weights)
        prior = weights @ priors
        probabilities = _normalize_logs(np.log(prior) + state['log_evidence'])
        return dict(instance_id=instance_id, observed_class=label,
                    structure_probabilities=probabilities.tolist(),
                    active_structure_prior=prior.tolist(),
                    reliability_weights_leave_one_out=weights.tolist(),
                    rho=None if label is None else float(weights[1]),
                    peer_instance_ids=peers,
                    geometry_log_evidence=[None if np.isneginf(v) else float(v)
                                           for v in state['log_evidence']],
                    semantic_conditioning_used=label is not None,
                    score_is_calibrated=False,
                    forecast_branch=self._is_forecast_branch)

    def snapshot(self):
        return dict(schema='semantic_reliability.generalized_bayes.v1',
                    share_across_instances=self.share_across_instances,
                    lambda_values=[0., 1.], initial_reliability_weights=[.5, .5],
                    geometry_prior=self._geometry_prior.tolist(),
                    class_structure_priors={key: value.tolist()
                                            for key, value in self._class_priors.items()},
                    score_is_calibrated=False,
                    inference_status='standard hierarchical Bayes using generalized geometry evidence',
                    instances=[self.posterior(key) for key in self.instance_ids])

    def observation_branch(self, instance_id, likelihood_by_structure):
        """Return an independent hypothetical observation branch.

        ``likelihood_by_structure[h]`` is a caller's predictive channel for a
        single outcome, including possible zeros. It is NOT a measured packet.
        Branches have no mutation effect on the paid-evidence state.
        """
        evidence = self._instances[instance_id]['log_evidence']
        likelihood = np.asarray(likelihood_by_structure, dtype=float)
        if (likelihood.shape != (self._count,) or not np.isfinite(likelihood).all()
                or np.any(likelihood < 0) or np.any(likelihood > 1)):
            raise ValueError('one predictive outcome probability in [0,1] per structure required')
        probability = float(np.asarray(self.posterior(instance_id)['structure_probabilities']) @ likelihood)
        if probability <= 0:
            raise ValueError('cannot branch on a zero-probability outcome')
        with np.errstate(divide='ignore'):
            updated = evidence + np.log(likelihood)
        branch = deepcopy(self)
        branch._instances[instance_id]['log_evidence'] = updated - updated.max()
        branch._is_forecast_branch = True
        return branch


def discrete_observation_evi(belief, target_instance_id, observation_channel,
                             terminal_action_utilities):
    """One predictive observation followed by one common finite terminal action.

    Channel shape is H x Y, with stochastic rows. Each already observed
    instance's utility matrix is A x H; all matrices describe the SAME A
    budget-feasible action choices. Utilities are summed over instances before
    maximizing over actions. This does not grant independently executable
    terminal actions to every object. The action set must be identical before
    and after observation; travel/observation cost is not modeled here.

    This is a one-event finite approximation, not a full POMDP solution and
    not calibrated information value or measured TSDF quality. G, fixed-prior
    Bayes and shared-reliability controllers can use the same function.
    """
    if not isinstance(belief, SemanticReliabilityBelief):
        raise TypeError('SemanticReliabilityBelief required')
    target = belief.posterior(target_instance_id)
    channel = np.asarray(observation_channel, dtype=float)
    if (channel.ndim != 2 or channel.shape[0] != belief.structure_count or
            channel.shape[1] < 1 or not np.isfinite(channel).all() or
            np.any(channel < 0) or np.any(channel > 1) or
            not np.allclose(channel.sum(axis=1), 1., rtol=0., atol=1e-12)):
        raise ValueError('finite H x Y stochastic observation channel required')
    if not isinstance(terminal_action_utilities, Mapping) or not terminal_action_utilities:
        raise ValueError('nonempty observed-instance action utility mapping required')
    utilities, action_count = {}, None
    for key, value in terminal_action_utilities.items():
        if key not in belief.instance_ids:
            raise ValueError('terminal utility cannot reference an undiscovered instance')
        array = np.asarray(value, dtype=float)
        if (array.ndim != 2 or array.shape[1] != belief.structure_count or
                array.shape[0] < 1 or not np.isfinite(array).all()):
            raise ValueError('finite A x H terminal utility matrix required')
        if action_count is not None and array.shape[0] != action_count:
            raise ValueError('all instances must share the same terminal action set')
        action_count = array.shape[0]
        utilities[key] = array.copy()

    def best_action(branch):
        posteriors = {key: branch.posterior(key) for key in sorted(utilities)}
        expected = np.zeros(action_count)
        for key, matrix in utilities.items():
            expected += matrix @ np.asarray(posteriors[key]['structure_probabilities'])
        if not np.isfinite(expected).all():
            raise ValueError('aggregate utilities exceed numeric range')
        action = int(np.argmax(expected))
        return action, float(expected[action]), expected.tolist(), posteriors

    prior_action, prior_best, prior_utilities, _ = best_action(belief)
    outcomes = np.asarray(target['structure_probabilities']) @ channel
    branches, after_best = [], 0.
    for outcome, probability in enumerate(outcomes):
        if probability <= 0:
            continue
        branch = belief.observation_branch(target_instance_id, channel[:, outcome])
        action, value, expected, posteriors = best_action(branch)
        after_best += float(probability) * value
        branches.append(dict(outcome=outcome, probability=float(probability),
                             best_action=action, best_expected_utility=value,
                             expected_action_utilities=expected,
                             terminal_instance_posteriors=posteriors))
    difference = after_best - prior_best
    tolerance = 1e-11 * max(1., abs(after_best), abs(prior_best))
    if difference < -tolerance:
        raise ArithmeticError('negative value of information violates shared action-set contract')
    return dict(schema='semantic_reliability.finite_evi.v1',
                target_instance_id=target_instance_id,
                evi=0. if abs(difference) <= tolerance else float(difference),
                best_action_before_observation=prior_action,
                best_utility_before_observation=prior_best,
                expected_action_utilities_before_observation=prior_utilities,
                expected_best_utility_after_observation=float(after_best),
                branches=branches, score_is_calibrated=False,
                observation_channel_is_caller_forecast=True,
                actual_future_observation_used=False,
                travel_and_observation_cost_included=False,
                scope='one forecast observation then one common finite terminal action; not measured TSDF quality')
