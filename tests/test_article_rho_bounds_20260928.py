"""Pure algebra, synthetic saved states; no World or private scene information."""
from copy import deepcopy
import math
import random
import unittest

from scripts import analyze_article_rho_bounds_20260928 as audit


def belief(peer=True, semantic=True, evidence=(0., 0., 0., 0.)):
    g=[.25]*4;q=[.4,.3,.2,.1]
    instances=[]
    for i in range(2 if peer else 1):
        row=dict(instance_id=str(i),observed_class='cabinet' if semantic else None,
            geometry_log_evidence=list(evidence),rho=.5 if semantic else None,
            active_structure_prior=[(a+b)/2 for a,b in zip(g,q)] if semantic else g)
        row['structure_probabilities']=audit.posterior(g,q if semantic else g,
            [math.exp(x-max(evidence)) for x in evidence],.5)
        instances.append(row)
    return dict(geometry_prior=g,class_structure_priors={'cabinet':q},instances=instances)


def candidate_data(models,offset=0.):
    vectors=([4.,0.,0.,0.],[0.,4.,4.,4.])
    rows=[]
    for i,vector in enumerate(vectors):
        areas={k:vector if k=='0' else [0.]*4 for k in models}
        score=sum(audit.saved.dot(a,models[k]['current_posterior']) for k,a in areas.items())
        rows.append(dict(target=('a:0','b:0')[i],cost=1.,score=score+(offset if i else 0.),structure_areas=areas))
    return dict(candidate_snapshot=rows)


class RhoBoundTests(unittest.TestCase):
    def test_signed_fractional_expectation_endpoint_bounds_dense_interior(self):
        rng=random.Random(20)
        for _ in range(50):
            g=[.25]*4;q=[.4,.3,.2,.1]
            likelihood=[math.exp(rng.uniform(-40,0)) for _ in range(4)]
            utility=[rng.uniform(-10,10) for _ in range(4)]
            low,high=audit.endpoint_range(g,q,likelihood,utility,.28,.62)
            for j in range(101):
                v=audit.saved.dot(utility,audit.posterior(g,q,likelihood,.28+.34*j/100))
                self.assertGreaterEqual(v,low-1e-12)
                self.assertLessEqual(v,high+1e-12)

    def test_peer_eligibility_derived_from_labels_not_method_message_fields(self):
        b=belief();m=audit.models_from_belief(b)
        self.assertEqual(m['0']['qualified_peer_count'],1)
        self.assertAlmostEqual(m['0']['rho_interval'][0],2/7)
        self.assertAlmostEqual(m['0']['rho_interval'][1],8/13)
        one=belief(peer=False)
        self.assertEqual(audit.models_from_belief(one)['0']['rho_interval'],[.5,.5])
        self.assertNotEqual(audit.models_from_belief(one,optimistic_missing_peer=True)['0']['rho_interval'],[.5,.5])
        geometry=audit.models_from_belief(belief(semantic=False),optimistic_missing_peer=True)
        self.assertEqual(geometry['0']['rho_interval'],[.5,.5])

    def test_posterior_reconstruction_and_more_than_one_peer_rejected(self):
        b=belief();b['instances'][0]['structure_probabilities']=[.25]*4
        with self.assertRaisesRegex(ValueError,'reconstruct'):audit.models_from_belief(b)
        b=belief();b['instances'].append(deepcopy(b['instances'][0]));b['instances'][-1]['instance_id']='2'
        with self.assertRaisesRegex(ValueError,'at-most-one'):audit.models_from_belief(b)

    def test_rank_invariance_and_possible_crossing(self):
        models=audit.models_from_belief(belief())
        data=candidate_data(models)
        bound=audit.bound_candidates(data,models,1.)
        self.assertTrue(bound['all_fixed_B_direct_changes_excluded'])
        # Bring the score boundary inside the cabinet one-peer interval.
        data['candidate_snapshot'][0]['score']+=1.30
        bound=audit.bound_candidates(data,models,1.)
        self.assertFalse(bound['all_fixed_B_direct_changes_excluded'])
        self.assertTrue(bound['individual_endpoint_top1_change'])
        self.assertFalse(bound['independent_endpoint_combinations_are_executable'])

    def test_pairwise_conservative_envelope_covers_all_box_points(self):
        models=audit.models_from_belief(belief(evidence=(0.,-.2,-.4,-.6)))
        data=candidate_data(models)
        # Two terms share a class and need not reach endpoints jointly;
        # the larger independent box still must contain every possible point.
        data['candidate_snapshot'][0]['structure_areas']['1']=[0.,2.,0.,1.]
        data['candidate_snapshot'][1]['structure_areas']['1']=[1.,0.,3.,0.]
        for row in data['candidate_snapshot']:
            row['score']=sum(audit.saved.dot(a,models[k]['current_posterior']) for k,a in row['structure_areas'].items())
        bound=audit.bound_candidates(data,models,1.)
        pair=bound['pairwise_score_envelopes'][0]
        rows={r['target']:r for r in data['candidate_snapshot']}
        for j in range(11):
            for k in range(11):
                pp={}
                for key,fraction in zip(models,(j/10,k/10)):
                    m=models[key];lo,hi=m['rho_interval']
                    pp[key]=audit.posterior(m['g'],m['q'],m['likelihood'],lo+(hi-lo)*fraction)
                scores={t:sum(audit.saved.dot(a,pp[key]) for key,a in r['structure_areas'].items()) for t,r in rows.items()}
                value=scores[pair['challenger']]-scores[pair['reference']]
                self.assertGreaterEqual(value,pair['difference_lower']-1e-12)
                self.assertLessEqual(value,pair['difference_upper']+1e-12)

    def test_zero_scores_none_and_fixed_lex_tie(self):
        models=audit.models_from_belief(belief(semantic=False))
        data=candidate_data(models)
        for row in data['candidate_snapshot']:
            row['score']=0.;row['structure_areas']={k:[0.]*4 for k in models}
        bound=audit.bound_candidates(data,models,1.)
        self.assertIsNone(bound['recorded_direct_winner'])
        self.assertTrue(bound['all_fixed_B_direct_changes_excluded'])
        for row in data['candidate_snapshot']:row['score']=1.
        bound=audit.bound_candidates(data,models,1.)
        self.assertEqual(bound['recorded_direct_winner'],'a:0')
        self.assertTrue(bound['all_fixed_B_direct_changes_excluded'])


if __name__=='__main__':unittest.main()
