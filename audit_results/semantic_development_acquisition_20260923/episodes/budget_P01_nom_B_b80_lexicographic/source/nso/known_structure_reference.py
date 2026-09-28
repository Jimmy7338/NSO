"""Declared hidden-structure reference for already observed instances only.

The trusted matcher sees private marker centres to identify the paid detection;
it discloses only a structure name. The planner keeps measured poses/support,
the common candidate pool and all movement, sensing and reconstruction costs.
This approximate planner is not an optimal reconstruction upper bound.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from nso.controller_semantic_mechanism import SemanticMechanismController
from nso.controller_v43 import _digest
from nso.semantic_reliability import SemanticReliabilityBelief


class KnownStructureBelief(SemanticReliabilityBelief):
    """Exact structure probabilities for disclosed instances, geometry otherwise."""

    def __init__(self, **kwargs):
        super().__init__(share_across_instances=False, **kwargs)
        self._known = {}

    def disclose(self, instance_id, structure_index):
        if instance_id not in self.instance_ids:
            raise ValueError('only an already registered measured instance may receive truth')
        if structure_index is None:
            self._known.pop(instance_id, None)
        elif type(structure_index) is not int or not 0 <= structure_index < self.structure_count:
            raise ValueError('valid structure index required')
        else:
            self._known[instance_id] = structure_index

    def posterior(self, instance_id):
        result = super().posterior(instance_id)
        if instance_id in self._known:
            p = np.eye(self.structure_count)[self._known[instance_id]].tolist()
            result.update(structure_probabilities=p, active_structure_prior=p,
                          known_structure_disclosed=True,
                          semantic_conditioning_used=False)
        else:
            result['known_structure_disclosed'] = False
        return result


class PaidInstanceStructureMatcher:
    """Trusted identity matching; no categories, dimensions or poses disclosed.

    The 0.35 m match radius covers the nominal label half-diagonal plus three
    1%-depth standard deviations at 4 m. Multiple matches and repeated identity
    claims remain unresolved. A reference with unresolved instances is partial.
    """

    def __init__(self, records, *, structure_names, match_radius_m=.35):
        if float(match_radius_m) != .35:
            raise ValueError('the declared 0.35 m identity radius is fixed')
        self._names = tuple(structure_names)
        self._records = []
        for record in records:
            if set(record) != {'marker_center_world_m', 'structure'}:
                raise ValueError('only marker centre and structure are permitted in trusted matcher')
            centre = np.asarray(record['marker_center_world_m'], dtype=float)
            if centre.shape != (3,) or not np.isfinite(centre).all() or record['structure'] not in self._names:
                raise ValueError('finite marker centre and supported structure required')
            self._records.append((centre.copy(), self._names.index(record['structure'])))
        self._claims = {}

    @classmethod
    def from_asset(cls, asset_root, *, structure_names):
        root = Path(asset_root)
        markers = json.loads((root/'renderer_private/markers.json').read_text())['marker_patches']
        instances = json.loads((root/'evaluation_private/instances.json').read_text())['private_instances']
        structures = {x['instance_id']: x['structure'] for x in instances}
        return cls([dict(marker_center_world_m=m['center_world_m'],
                         structure=structures[m['instance_id']]) for m in markers],
                   structure_names=structure_names)

    def resolve(self, observed_instances):
        output = {}
        for instance in observed_instances:
            key = instance['instance_id']
            if instance['association_uncertain']:
                output[key] = None
                continue
            anchor = np.asarray(instance['marker_anchor_world_m'], dtype=float)
            if anchor.shape != (3,) or not np.isfinite(anchor).all():
                raise ValueError('measured marker anchor required')
            matches = [i for i, (centre, _) in enumerate(self._records)
                       if np.linalg.norm(anchor-centre) <= .35]
            if len(matches) != 1:
                output[key] = None
                continue
            match = matches[0]
            if match in self._claims and self._claims[match] != key:
                output[key] = None
                continue
            self._claims[match] = key
            output[key] = self._records[match][1]
        return output


class KnownStructureReferenceController(SemanticMechanismController):
    """Same hierarchical executor, with declared structure-only disclosures."""

    def __init__(self, graph, *, structure_matcher, **kwargs):
        if type(structure_matcher) is not PaidInstanceStructureMatcher or 'method' in kwargs:
            raise ValueError('explicit trusted structure matcher required')
        super().__init__(graph, method='G', **kwargs)
        if structure_matcher._names != tuple(self._ledger.names):
            raise ValueError('trusted matcher and public structure bank must use the same order')
        self._structure_matcher = structure_matcher
        self._belief = KnownStructureBelief(geometry_prior=self._ledger.geometry_prior,
                                            class_structure_priors=self._ledger.class_priors)
        self.method = 'known_structure_reference'
        self._configuration.update(method=self.method,
            extra_information='true structure name of uniquely matched already observed instance',
            private_pose_used_by_planner=False,
            trusted_matcher_uses_private_marker_centres=True,
            unmatched_instances='retain geometry-only belief; report unresolved reference',
            exact_one_hot_structure_weights=True, reconstruction_optimal_upper_bound=False)
        self._configuration_sha256 = _digest(self._configuration)

    def accept(self, observation, mapper, *, execution_outcome='success'):
        evidence = super().accept(observation, mapper, execution_outcome=execution_outcome)
        try:
            known = self._structure_matcher.resolve(self._ledger.snapshot()['instances'])
            for key, index in known.items():
                self._belief.disclose(key, index)
            evidence.update(structure_belief=self._belief.snapshot(),
                reference_structure_disclosures={key:None if index is None else
                    self._ledger.names[index] for key,index in known.items()},
                reference_all_observed_instances_resolved=all(v is not None for v in known.values()),
                reference_is_optimal_quality_upper_bound=False)
            self._last_accept = deepcopy(evidence)
            return evidence
        except Exception:
            self._poisoned = True
            raise

    def _planning_instances(self):
        # The shared V42 area generator validates finite positive measured
        # posteriors. Give it the original measured geometry belief: per-shape
        # area vectors do not depend on its probabilities. Global action values
        # use the exact known-structure belief in _select_global. This retains
        # the same area computation without epsilon smoothing of reference h.
        return self._ledger.snapshot()['instances']

    def choose(self):
        result = super().choose()
        result.update(ground_truth_scene_input=True, ground_truth_structure_input=True,
                      ground_truth_pose_used_in_planning=False,
                      forecast_area_vectors_use_common_observed_geometry=True,
                      forecast_weighted_summaries_are_geometry_only=True,
                      action_values_use_declared_structure_reference=True,
                      reconstruction_optimal_upper_bound=False)
        self._last_choose = deepcopy(result)
        return result
