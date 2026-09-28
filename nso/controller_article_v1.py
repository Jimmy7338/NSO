"""Article revision: common measured-evidence repairs and explicit CPU baselines.

NBV is a myopic geometry-only nominal-area mechanism, not a reproduction of a
named external planner. G/B/S retain the original paid diagnostic lookahead;
B and S differ in cross-instance class-reliability sharing. No GT is accepted.
"""
from copy import deepcopy
import numpy as np

from nso.controller_semantic_mechanism import SemanticMechanismController, select_unique_target
from nso.controller_v43 import _digest, _state_receipt
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.observed_residual_v41 import ObservedResidualV41, pixel_mask_sha256_v41
from nso.prototype_observation_channel import _PLANE_KEYS
from nso.public_navigation_v43 import select_candidate_states_v43
from nso.view_quality_v42 import ViewQualityPredictorV42


METHOD_MAP = {'NBV': 'G', 'G': 'G', 'B': 'bayes_semantic', 'S': 'shared_semantic',
              'S_no_feedback': 'shared_no_feedback', 'S_no_future': 'shared_no_future',
              'S_no_cross_future': 'shared_no_cross_future'}


def _without_article_audit(rows):
    # Strip exactly the new diagnostic field, not arbitrary unknown inputs.
    return [{k:v for k,v in row.items() if k != 'article_feedback_eligibility'} for row in rows]


class ArticleObservedResidualV1(ObservedResidualV41):
    def __init__(self, *, multi_view_planes=False):
        super().__init__()
        self.article_planes = None
        if multi_view_planes:
            from nso.observed_marker_planes_article_v1 import ObservedMarkerPlanesArticleV1
            self.article_planes = ObservedMarkerPlanesArticleV1()
        self.last_article_plane_receipt = None

    def observe(self, observation, accepted_associations):
        result = super().observe(observation, accepted_associations)
        if self.article_planes is None:
            return result
        # The base call has already validated exact packet, pixels and support.
        receipt = self.article_planes.observe(observation, _without_article_audit(accepted_associations))
        self.last_article_plane_receipt = deepcopy(receipt)
        result['article_plane_evidence'] = deepcopy(receipt)
        associations = {row['instance_id']:row for row in accepted_associations}
        for row in result['results']:
            key = row['instance_id']
            if self.article_planes.receipt(key)['conflict']:
                self._planes.pop(key,None)
                row.update(accepted=False,reason='conflicting_paid_marker_planes')
                continue
            if row['accepted'] or row.get('reason') != 'no_reliable_observed_label_plane':
                continue
            plane = self.article_planes.plane(key)
            if plane is None:
                continue
            # Full multi-frame provenance stays in the audit receipt. The old
            # nominal predictor/channel receive only their public geometry keys.
            compatible = {k:deepcopy(v) for k,v in plane.items() if k in _PLANE_KEYS}
            self._planes[key] = deepcopy(compatible)
            pixels = np.asarray(associations[key]['pixel_indices'], dtype=np.int64)
            depth = observation.depth_m.ravel()[pixels]
            pixels = pixels[(depth >= .1) & (depth <= 4.)]
            if len(pixels) < 16:
                continue
            if len(pixels) > self.maximum_sample_pixels:
                pixels = pixels[np.linspace(0,len(pixels)-1,self.maximum_sample_pixels,dtype=int)]
            per_structure, losses, scores = self._score(observation, pixels, compatible)
            row.update(accepted=True, reason='current_depth_residual_with_paid_multiview_plane',
                plane_fit=deepcopy(compatible), per_structure=per_structure, losses_m=losses,
                valid_sample_counts=[len(pixels)]*4, log_likelihoods=scores,
                informative=bool(np.ptp(scores)>1e-6), associated_pixel_count=len(associations[key]['pixel_indices']),
                evaluated_pixel_mask_sha256=pixel_mask_sha256_v41(pixels,observation.depth_m.shape),
                missing_depth_is_empty_space_evidence=False, loss='mean_min_absolute_axial_residual',
                truncation_m=self.truncation_m, evidence_scale_m=self.evidence_scale_m,
                nuisance_parameters_profiled=True, candidate_count_per_structure=45,
                multi_frame_pose_used=True, sample_rule='valid paid pixels; deterministic uniform subsample')
        return result


class ArticleViewPredictorV1(ViewQualityPredictorV42):
    def __init__(self, *, residual, **kwargs):
        super().__init__(**kwargs)
        if type(residual) is not ArticleObservedResidualV1:
            raise TypeError('owned measured residual adapter required')
        self.article_residual = residual

    def observe(self, observation, accepted_associations):
        receipt = super().observe(observation, _without_article_audit(accepted_associations))
        supplemented = []
        if self.article_residual.article_planes is not None:
            for key,state in self._instances.items():
                if self.article_residual.article_planes.receipt(key)['conflict']:
                    state['plane_conflict'] = True
                    continue
                if state['plane'] is not None or state['plane_conflict']:
                    continue
                plane = self.article_residual.article_planes.plane(key)
                if plane is not None:
                    state['plane'] = {k:deepcopy(v) for k,v in plane.items() if k in _PLANE_KEYS}
                    supplemented.append(key)
            if supplemented:
                self._forecast_cache.clear()
        receipt.update(article_multiview_supplemented_instances=supplemented,
                       evidence_source='same paid marker pixels; no private scene fields')
        return receipt


class ArticleControllerV1(SemanticMechanismController):
    def __init__(self, graph, *, method='S', cache_feedback_repair=True,
                 multi_view_planes=False, **kwargs):
        if method not in METHOD_MAP:
            raise ValueError('declared article method required')
        if type(cache_feedback_repair) is not bool or type(multi_view_planes) is not bool:
            raise ValueError('repair switches must be explicit bools')
        super().__init__(graph, method=METHOD_MAP[method], **kwargs)
        if cache_feedback_repair:
            old = self._ledger
            self._ledger = ObservedInstancesArticleV1(palette=old.palette,
                structure_names=old.names, class_structure_prior=old.class_priors,
                mode=old.mode, geometry_prior=old.geometry_prior,
                maximum_instances=self._configuration['maximum_instances'])
        self._residual = ArticleObservedResidualV1(multi_view_planes=multi_view_planes)
        self._predictor = ArticleViewPredictorV1(residual=self._residual,
            maximum_instances=self._configuration['maximum_instances'])
        self.article_method = method
        self._configuration.update(article_version='article_v1', article_method=method,
            common_cache_feedback_repair=cache_feedback_repair,
            common_multiview_plane_estimation=multi_view_planes,
            nbv_scope='myopic geometry-only shared nominal surface gain; CPU mechanism baseline',
            independent_external_implementation_reproduction=False)
        self._configuration_sha256 = _digest(self._configuration)

    def _select_global(self):
        if self.article_method != 'NBV':
            return super()._select_global()
        current,home = self._router.state,self._router.home
        remaining = self._router.budget-self._router.step
        instances = self._planning_instances()
        protected = {}
        if self._acquisition is None:
            states,pool = select_candidate_states_v43(self._graph,current,
                paid_camera_states=tuple(self._paid_states),limit=self._configuration['maximum_candidates'])
        else:
            states,pool,protected,initialization_rows = self._acquisition.candidates(
                self._graph,current,home,remaining,tuple(self._paid_states),self._camera,
                instances,self._predictor.observed_plane_receipts(),limit=self._configuration['maximum_candidates'])
            target = self._select_initialization(initialization_rows,instances,pool,remaining)
            if target is not None:
                return target
        views = [self._candidate_view(state) for state in states]
        discovery_raw = self._safety.discovery_utilities(states,radius_m=self._configuration['discovery_radius_m'])
        gains = {state:0. if state.node in self._visited_nodes else
            self._configuration['discovery_weight']*float(discovery_raw[state]) for state in states}
        forecasts,allocations = [],[]
        for instance in instances:
            key = instance['instance_id']
            ordered = self._instance_candidate_order(instance,states,views)
            reserved = protected.get(key,[])
            selected = ([r for s in reserved for r in ordered if r[0]==s]
                + [r for r in ordered if r[0] not in reserved])[:self._configuration['maximum_views_per_instance']]
            allocations.append(dict(instance_id=key,candidates=[_state_receipt(s) for s,_ in selected]))
            if selected:
                forecast = self._predictor.forecast(instance,[v for _,v in selected])
                forecasts.append(forecast)
                posterior = np.asarray(self._belief.posterior(key)['structure_probabilities'])
                for (state,_),row in zip(selected,forecast['candidates']):
                    gains[state] += self._configuration['inspection_weight']*float(
                        np.asarray(row['structure_new_surface_area_m2'])@posterior)
        rows,options = [],[]
        for state in states:
            out,back = self._graph.route(current,state),self._graph.route(state,home)
            if out is None or back is None or out.cost+1+back.cost>remaining:
                continue
            cost = out.cost+1+back.cost
            row = dict(kind='direct',target=_state_receipt(state),expected_gain=gains[state],
                       total_cost=cost,score=gains[state]/cost)
            rows.append(row); options.append((row['score'],state,row))
        selected = select_unique_target(options,tie_rule=self.tie_rule,rng=self._tie_rng)
        self._last_global_selection = dict(paid_step=self._router.step,remaining=remaining,
            candidate_pool=pool,instance_candidate_allocations=allocations,direct_options=rows,
            diagnostic_options=[],selected=None if selected is None else deepcopy(selected[2]),
            forecasts=forecasts,predicted_channel_only=True,future_sensor_rendered=False,
            score_is_calibrated=False,myopic_nbv=True)
        return None if selected is None else selected[1]

    def choose(self):
        result = super().choose()
        result['article_method'] = self.article_method
        self._last_choose = deepcopy(result)
        return result

    def snapshot(self):
        result = super().snapshot()
        result['article_method'] = self.article_method
        return result
