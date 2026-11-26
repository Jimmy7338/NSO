"""Post-hoc geometric prior multiplier; all other joint policy behavior inherited."""
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules
class GWeight2(ExplorationJointCPUFourModules):
 policy_variant='exploration_joint_v1_G_weight2_posthoc'
 def _scores_v10_1(self,s,routes,attempted):
  scores,audit=super()._scores_v10_1(s,routes,attempted)
  for i,row in enumerate(audit):
   prior=row['prior_potentials']['G'];common=row['common_total_proxy']
   assert abs(scores['G'][i]-(common+prior))<1e-9
   scores['G'][i]=common+2.0*prior
   row['posthoc_G_weight2']=dict(common=common,prior_G=prior,multiplier=2.0,total=scores['G'][i])
  return scores,audit
