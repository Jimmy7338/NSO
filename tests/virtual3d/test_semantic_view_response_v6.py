import unittest
import numpy as np
from nso.semantic_view_response_v6 import RelativeResponseRidge
from nso.semantic_view_response_v6 import response_features
from nso.counterfactual_view_scoring import CounterfactualScoreConfig,score_routes
from tests.virtual3d.test_counterfactual_view_scoring import MeasuredMapper
from env.grid_exploration import GridConfig
import copy


class RelativeRankingTests(unittest.TestCase):
    def test_history_offsets_do_not_train_a_scene_intercept(self):
        x=np.array([[0,0],[1,2],[2,1],[10,8],[12,9],[11,10]],float)
        y=np.array([[0,2],[1,1],[2,0],[4,4],[6,2],[5,3]],float)
        ids=np.array([0,0,0,1,1,1])
        a=RelativeResponseRidge.fit(x,y,ids)
        shifted=y.copy();shifted[ids==0]+=100;shifted[ids==1]-=200
        b=RelativeResponseRidge.fit(x,shifted,ids)
        np.testing.assert_allclose(a.predict(x[:3]),b.predict(x[:3]),atol=1e-12)
        np.testing.assert_allclose(a.predict(x[:3]),a.predict(x[:3]+500),atol=1e-12)

    def test_constant_features_do_not_create_nonfinite_scores(self):
        x=np.ones((4,12));y=np.arange(8).reshape(4,2)
        model=RelativeResponseRidge.fit(x,y,[0,0,1,1])
        np.testing.assert_array_equal(model.predict(x),np.zeros((4,2)))

    def test_prediction_has_no_test_target_argument(self):
        x=np.array([[0],[1],[2],[3]],float);y=np.array([[0],[1],[2],[3]],float)
        model=RelativeResponseRidge.fit(x,y,[0,0,1,1])
        self.assertGreater(model.predict(x)[3,0],model.predict(x)[0,0])
        with self.assertRaises(TypeError):model.predict(x,y)

    def test_label_intervention_changes_conditional_features_only(self):
        mappers={name:MeasuredMapper(name) for name in ('aligned','shuffled','absent')}
        virtual=mappers['aligned'].config
        config=CounterfactualScoreConfig(GridConfig(resolution_m=.2,robot_radius_m=.2,
            sensor_range_m=4.,sensor_fov_deg=360.),virtual)
        states=[(15,10,h) for h in (1,2,3,0,1)]
        routes=[{'candidate_id':0,'cost':4,'states':states}]
        predicted=score_routes(mappers,routes,config)
        matrices,_=response_features(mappers['aligned'],routes,predicted)
        np.testing.assert_array_equal(matrices['S'][:,:8],matrices['X'][:,:8])
        self.assertGreater(np.max(abs(matrices['S'][:,8:]-matrices['X'][:,8:])),0.)
        np.testing.assert_array_equal(matrices['G'],matrices['M'])
        bad=copy.deepcopy(predicted);bad['objects']['M'][0]['shelf_probability']+=.1
        with self.assertRaises(AssertionError):response_features(mappers['aligned'],routes,bad)


if __name__=='__main__':unittest.main()
