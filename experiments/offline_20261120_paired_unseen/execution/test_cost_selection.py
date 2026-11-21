import unittest
from short_branches import cost_band,final_three
class TestCostSelection(unittest.TestCase):
 def routes(self,costs):return [dict(candidate_id=i,outbound_actions=['forward']*n)for i,n in enumerate(costs)]
 def test_largest_then_lower_band(self):
  p,a=cost_band(self.routes([10,11,13,20,21,23]));self.assertEqual(a['selected_pool_ids'],[0,1,2]);self.assertEqual(a['lower_actions'],10)
 def test_no_widening(self):
  p,a=cost_band(self.routes([1,10,20]));self.assertEqual(a['status'],'insufficient_cost_matched_routes');self.assertEqual(len(p),1)
 def test_both_winners_retained(self):
  p={'visibility_S':[dict(candidate_id=i,cost=10,score=s)for i,s in enumerate([1,2,3,4])],'acquisition_S':[dict(candidate_id=i,cost=10,score=s)for i,s in enumerate([4,3,2,1])]}
  self.assertEqual(final_three(p),[0,1,3])
 def test_same_winner_fills_by_id(self):
  rows=[dict(candidate_id=i,cost=10,score=s)for i,s in enumerate([1,2,3,4])];self.assertEqual(final_three({'visibility_S':rows,'acquisition_S':rows}),[0,1,3])
if __name__=='__main__':unittest.main()
