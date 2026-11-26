import json,hashlib,math
from fractions import Fraction as F
from pathlib import Path
D=Path(__file__).parent;R=Path('/home/ouc/nso_robot/analyses/reproduction_20261104');T=Path('/home/ouc/nso_robot/analyses/independent_layout_20261104/frozen_execution/two_room_joint');S=R/'simulation_publication_stage/experiments/unknown_exploration_20261104/source';f=lambda x:F.from_float(float(x));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class Interval:
 def __init__(self):self.low=F(0);self.li=True;self.high=None;self.hi=False;self.lwhy='lambda>=0';self.hwhy=None;self.impossible=None
 def add(self,a,b,strict,why):
  # a+b*lambda >= 0, or > 0 when strict
  if b==0:
   if a<0 or (a==0 and strict):self.impossible=why
   return
  v=-a/b
  if b>0:
   if v>self.low:self.low=v;self.li=not strict;self.lwhy=why
   elif v==self.low:self.li=self.li and not strict
  else:
   if self.high is None or v<self.high:self.high=v;self.hi=not strict;self.hwhy=why
   elif v==self.high:self.hi=self.hi and not strict
 def merge(self,other,why):
  self.add(-other.low,F(1),not other.li,why+':'+str(other.lwhy))
  if other.high is not None:self.add(other.high,F(-1),not other.hi,why+':'+str(other.hwhy))
  if other.impossible:self.impossible=why+':'+str(other.impossible)
 def empty(self):return bool(self.impossible is not None or self.high is not None and (self.low>self.high or self.low==self.high and not(self.li and self.hi)))
 def out(self):return dict(empty=self.empty(),lower=float(self.low),lower_fraction=str(self.low),lower_inclusive=self.li,upper=None if self.high is None else float(self.high),upper_fraction=None if self.high is None else str(self.high),upper_inclusive=self.hi,lower_witness=self.lwhy,upper_witness=self.hwhy,impossible=self.impossible)
 def contains(self,v):v=F(v);return not self.empty() and (v>self.low or v==self.low and self.li) and (self.high is None or v<self.high or v==self.high and self.hi)
files=[]
for fam in ('fresh_matrix','deployed_policy_matrix'):
 for mode in 'GS':
  for p in sorted((R/fam).glob(f'*_{mode}_choices.json')):files.append((fam,mode,p))
for mode in 'GS':files.append(('two_room',mode,T/f'two_room_shelf_west_96_{mode}_choices.json'))
records=[];trajectories=[];overall=Interval();all_inputs={};skipped=[]
for family,trajectory_mode,path in files:
 all_inputs[str(path)]=sha(path);calls=json.loads(path.read_text());inter=Interval();valid=0
 for j,c in enumerate(calls):
  o=c['outputs'];routes=o.get('candidates');audit=o.get('score_audit');scores=o.get('scores');state=dict(family=family,trajectory_mode=trajectory_mode,file=path.name,call_1based=j+1,completed_actions=96-c['inputs']['remaining_budget'] if family=='two_room' else int(path.name.split('_')[-3])-c['inputs']['remaining_budget'])
  if routes is None or audit is None or scores is None or len(routes)!=len(audit) or len(scores.get('S',[]))!=len(routes):skipped.append({**state,'reason':'missing complete candidate or scores'});continue
  if any(a.get('candidate_id')!=route.get('candidate_id') or 'common_total_proxy' not in a or 'G' not in a.get('prior_potentials',{}) for a,route in zip(audit,routes)):skipped.append({**state,'reason':'missing score components or candidate mismatch'});continue
  if any(not math.isclose(a['common_total_proxy']+a['prior_potentials']['G'],scores['G'][i],abs_tol=1e-9,rel_tol=1e-12) for i,a in enumerate(audit)):skipped.append({**state,'reason':'component score mismatch'});continue
  def winner(vals):
   if not routes:return None
   idx=min(range(len(routes)),key=lambda i:(-vals[i],routes[i]['cost'],routes[i]['candidate_id']))
   return None if vals[idx]<=0 else idx
  idx=winner(scores['S']);actual=winner(scores[trajectory_mode]);selected=o.get('selected');assert (None if actual is None else routes[actual]['candidate_id'])==(None if selected is None else selected['candidate_id'])
  z=Interval();terms=[(f(a['common_total_proxy']),f(a['prior_potentials']['G'])) for a in audit]
  if idx is None:
   for k,(a,b) in enumerate(terms):z.add(-a,-b,False,f'nonpositive candidate {routes[k]["candidate_id"]}')
  else:
   ai,bi=terms[idx];z.add(ai,bi,True,'selected score > 0');selected_key=(routes[idx]['cost'],routes[idx]['candidate_id'],idx)
   for k,(a,b) in enumerate(terms):
    if k==idx:continue
    other_key=(routes[k]['cost'],routes[k]['candidate_id'],k);strict=selected_key>other_key
    z.add(ai-a,bi-b,strict,f'candidate {routes[k]["candidate_id"]}'+(' wins tie' if strict else ' loses tie'))
  valid+=1;state.update(selected_S_candidate=None if idx is None else routes[idx]['candidate_id'],selected_S_pose=None if idx is None else routes[idx]['pose'],target_kind='actual_S_selection' if trajectory_mode=='S' else 'counterfactual_S_score_on_G_visited_state',candidate_count=len(routes),interval=z.out());records.append(state)
  if trajectory_mode=='S':inter.merge(z,f'{path.name} call {j+1}');overall.merge(z,f'{path.name} call {j+1}')
 trajectories.append(dict(family=family,trajectory_mode=trajectory_mode,file=path.name,total_calls=len(calls),audited_calls=valid,all_states_intersection=inter.out() if trajectory_mode=='S' else None,lambda_2_in_intersection=inter.contains(2) if trajectory_mode=='S' else None))
family_inter={}
for fam in ('fresh_matrix','deployed_policy_matrix','two_room'):
 acc=Interval()
 for r in records:
  if r['family']==fam and r['trajectory_mode']=='S':
   v=r['interval'];q=Interval();q.low=F(v['lower_fraction']);q.li=v['lower_inclusive'];q.high=None if v['upper_fraction'] is None else F(v['upper_fraction']);q.hi=v['upper_inclusive'];q.lwhy=v['lower_witness'];q.hwhy=v['upper_witness'];q.impossible=v['impossible'];acc.merge(q,r['file']+' call '+str(r['call_1based']))
 family_inter[fam]=acc.out()
res=dict(scope='Observed states only. Actual S target on S trajectories; counterfactual S-score winner on G visited states. No new rollout. Intersections include only actual S trajectories.',mathematics='Exact rational affine constraints on binary-float input coefficients; tie lexicographic cost then candidate_id then first index. Actual floating addition can round differently exactly at a mathematical boundary; exclude boundary for robust execution.',input_hashes=all_inputs,selection_source_hashes={str(p):sha(p) for p in (S/'nso/cpu_four_modules_v10.py',S/'nso_bridge/exploration_joint_policy.py')},counts=dict(files=len(files),recorded_calls=sum(x['total_calls'] for x in trajectories),audited_states=len(records),actual_S_states=sum(r['trajectory_mode']=='S' for r in records),counterfactual_S_on_G_states=sum(r['trajectory_mode']=='G' for r in records),skipped=len(skipped),unreproducible_states=sum(r['interval']['empty'] for r in records)),skipped=skipped,trajectories=trajectories,actual_S_family_intersections=family_inter,actual_S_cross_condition_intersection=overall.out(),states=records)
(D/'audit.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k not in ('states','input_hashes')},indent=2))
