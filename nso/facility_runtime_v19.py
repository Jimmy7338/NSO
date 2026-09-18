"""Shared observation-quality geometric continuation for the V19 pilot.

No learned or category-conditioned score is deployed. Semantic information is
tested by choosing among the same physically executable initial options.
"""
from copy import deepcopy
import numpy as np
from nso.facility_runtime_v18 import FacilityBackendV18, FacilityRuntimeV18
from nso.cpu_four_modules_v16 import axis_components_v16
from nso.observed_precision_response_v19 import ObservedPrecisionResponseV19


class FacilityBackendV19(FacilityBackendV18):
    def select_target(self, scene_idx):
        state = self.scenes[scene_idx]
        pending = state.get('region_continuation')
        if pending is not None and state['packet'].action_id <= pending['armed_action_id']:
            # A denied first action has produced no new observation. Cancel
            # its commitment before the common safety recovery replans.
            state['v19_cancelled_continuation'] = dict(reason='no_paid_first_stage_observation',
                action_id=state['packet'].action_id, first_candidate_id=pending['first_candidate_id'])
            state['region_continuation'] = None
            self._record(scene_idx, 'STGHP', 'cancel_unobserved_continuation',
                dict(armed_action_id=pending['armed_action_id']),
                state['v19_cancelled_continuation'])
        return super().select_target(scene_idx)

    def _scores_v10_1(self, state, routes, attempted):
        scores, rows = super()._scores_v10_1(state, routes, attempted)
        mapper = state['mapper']; config = mapper.config
        task_count = int(getattr(self.args, 'cpu_task_asset_count', 6))
        if task_count < 1: raise ValueError('positive common task inventory count required')
        response = ObservedPrecisionResponseV19(mapper, state['assets'],
            intrinsic=state['packet'].frame.intrinsic, max_points_per_asset=128,
            occlusion_tolerance_m=.12)
        marked = [i for i, a in enumerate(state['assets']) if a['marked_points'] > 0]
        for index, (route, row) in enumerate(zip(routes, rows)):
            quality = response.score_route(route)
            additions = quality['per_asset_expected_precision_change']
            aperture = np.asarray(row['corrected_aperture'], dtype=float)
            coverage = row['v18_task_proxy']['expected_new_coverage_fraction']
            observation = float(sum(aperture[i] for i in marked))/task_count
            precision = float(sum(additions[i] for i in marked))/task_count
            value = float(coverage + observation + precision)
            row['v19_task_proxy'] = dict(expected_coverage_fraction=coverage,
                observed_direction_term=observation, observed_precision_term=precision,
                precision_response=quality, task_asset_count=task_count,
                weights=[1., 1., 1.], class_used=False, evaluation_truth_used=False,
                calibrated_future_tsdf_quality=False)
            for mode in scores: scores[mode][index] = float(coverage) if mode == 'N' else value
        return scores, rows


class FacilityRuntimeV19(FacilityRuntimeV18):
    def install_pilot_option(self, option_name):
        selection, receipt = super().install_pilot_option(option_name)
        receipt = deepcopy(receipt); receipt['event'] = 'v19_declared_option'
        self.audit[-1] = deepcopy(receipt)
        return selection, receipt


def facility_components_v19(args, shape):
    comp = axis_components_v16(args, shape)
    comp._cpu_backend = FacilityBackendV19(args, 1, shape)
    comp.capabilities = dict(comp._cpu_backend.capabilities)
    comp.capabilities['v19_role'] = 'common geometry and measured-depth-quality continuation; no fitted semantic head'
    return comp
