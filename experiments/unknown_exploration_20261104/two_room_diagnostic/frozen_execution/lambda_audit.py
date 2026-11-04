import json,hashlib,math
from pathlib import Path
D=Path(__file__).parent;P=D/'two_room_joint';path=P/'two_room_shelf_west_96_G_choices.json';calls=json.loads(path.read_text());o=calls[1]['outputs'];audit=o['score_audit'];scores=o['scores'];assert len(audit)==9 and calls[1]['inputs']['remaining_budget']==77
rows=[]
for a,c in zip(audit,o['candidates']):
 i=a['candidate_id'];assert i==c['candidate_id'];base=a['common_total_proxy'];pg=a['prior_potentials']['G'];ps=a['prior_potentials']['S'];assert math.isclose(base+pg,scores['G'][i],abs_tol=1e-10) and math.isclose(base+ps,scores['S'][i],abs_tol=1e-10);rows.append(dict(id=i,pose=c['pose'],common=base,prior_G=pg,prior_S=ps,S_over_G=(ps/pg if pg!=0 else None)))
target=next(x for x in rows if x['id']==7);lo=0.;hi=math.inf;lower_from=[];upper_from=[];constraints=[]
for r in rows:
 if r['id']==7:continue
 # target minus competitor = intercept + lambda*slope > 0
 intercept=target['common']-r['common'];slope=target['prior_G']-r['prior_G'];boundary=None if slope==0 else -intercept/slope
 if slope>0:lo=max(lo,boundary);lower_from.append((boundary,r['id']))
 elif slope<0:hi=min(hi,boundary);upper_from.append((boundary,r['id']))
 elif intercept<=0:raise ValueError('strict dominance impossible for all lambda')
 constraints.append(dict(competitor_id=r['id'],intercept=intercept,slope=slope,boundary=boundary,relation=('lambda >' if slope>0 else 'lambda <' if slope<0 else 'always')))
assert lo<hi
refs={}
for lam in (0,1,2,3,4):
 values=[x['common']+lam*x['prior_G'] for x in rows];mx=max(values);winners=[i for i,v in enumerate(values) if abs(v-mx)<1e-12];refs[str(lam)]=dict(winners=winners,winning_score=mx,target_score=values[7])
ratios=[r['S_over_G'] for r in rows if r['S_over_G'] is not None];res=dict(scope='retrospective algebra at recorded same state; no new policy or intervention; fitted lambda is not a predeclared baseline',input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),completed_actions=19,target_id=7,strict_interval=dict(lower=lo,lower_inclusive=False,upper=None if math.isinf(hi) else hi,upper_inclusive=False,lower_limiting_competitor=max(lower_from)[1],upper_limiting_competitor=None if not upper_from else min(upper_from)[1]),lambda_references=refs,ratio_nonzero_prior_min=min(ratios),ratio_nonzero_prior_max=max(ratios),ratios_constant=all(math.isclose(x,ratios[0],rel_tol=1e-10,abs_tol=1e-10) for x in ratios),candidates=rows,constraints=constraints)
(D/'lambda_audit.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
