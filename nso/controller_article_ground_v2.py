"""ArticleV1 planning with one common, measured-ground association change.

The original RGB-D packet, mapper, priors, view predictor and paid macro planner
are inherited unchanged. Ground removal acts only on association proposals.
"""
from nso.controller_article_v1 import ArticleControllerV1
from nso.controller_v43 import _digest
from nso.observed_instances_article_ground_v2 import (
    GROUND_POLICY, ObservedInstancesArticleGroundV2,
)


class ArticleControllerGroundV2(ArticleControllerV1):
    def __init__(self, graph, *, ground_association=True, **kwargs):
        if type(ground_association) is not bool:
            raise ValueError('ground_association must be an explicit bool')
        if ground_association and kwargs.get('cache_feedback_repair', True) is not True:
            raise ValueError('ground frontend requires the unchanged ArticleV1 feedback repair')
        super().__init__(graph, **kwargs)
        if ground_association:
            old = self._ledger
            self._ledger = ObservedInstancesArticleGroundV2(
                palette=old.palette, structure_names=old.names,
                class_structure_prior=old.class_priors, mode=old.mode,
                geometry_prior=old.geometry_prior,
                maximum_instances=self._configuration['maximum_instances'])
        self._configuration.update(article_version='article_ground_v2',
            common_ground_association=ground_association,
            ground_association_policy=dict(GROUND_POLICY),
            ground_removal_scope='association proposals only; complete paid RGB-D fusion unchanged')
        self._configuration_sha256 = _digest(self._configuration)
