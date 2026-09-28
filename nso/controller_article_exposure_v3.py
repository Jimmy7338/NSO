"""Common same-center paid-exposure eligibility over the frozen Ground V2 stack.

The residual, association ledger, posterior, macro planner and mapper interface
are unchanged. Nominal attempted exposure never becomes measured TSDF support.
"""
from nso.controller_article_ground_v2 import ArticleControllerGroundV2
from nso.controller_v43 import _digest
from nso.view_quality_article_exposure_v3 import ArticleExposureViewPredictorV3


class ArticleControllerExposureV3(ArticleControllerGroundV2):
    def __init__(self, graph, *, same_center_exposure_dedup=True, **kwargs):
        if type(same_center_exposure_dedup) is not bool:
            raise ValueError('same_center_exposure_dedup must be an explicit bool')
        super().__init__(graph, **kwargs)
        if same_center_exposure_dedup:
            self._predictor = ArticleExposureViewPredictorV3(
                residual=self._residual,
                maximum_instances=self._configuration['maximum_instances'])
        self._configuration.update(
            same_center_exposure_dedup=same_center_exposure_dedup,
            article_exposure_version='article.same_center_paid_exposure.v3',
            exposure_predictor_configuration_sha256=self._predictor.configuration_sha256)
        self._configuration_sha256 = _digest(self._configuration)
