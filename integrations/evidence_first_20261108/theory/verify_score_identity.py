"""Pure arithmetic and archived-score audit; no planner, simulation or ROS."""
from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent
# Coefficient-level constructive example, not a certified generated candidate set.
g=[4.,4.];s=[7.,1.];x=[1.,7.];f=[[.5,0.],[0.,.5]];common=[1.,2.]
score=lambda weights:[a+sum(u*v for u,v in zip(row,weights)) for a,row in zip(common,f)]
G,S,X=score(g),score(s),score(x)
assert G==[3.,4.] and S==[4.5,2.5] and X==[1.5,5.5]
assert all(a+b==2*c for a,b,c in zip(S,X,G))
# For every lambda>=0, B-A = (2+2lambda)-(1+2lambda)=1>0.
proof=dict(width_depth_height_each=[1.,1.,1.],marked=[1,1],class_votes=[-1.,1.],generic=g,semantic=s,swapped=x,factors=f,common=common,G=G,S=S,X=X,uniform_weight_G_scores=['1+2*lambda','2+2*lambda'],uniform_weight_B_minus_A='1 for every lambda>=0',S_selected='A',G_lambda_selected='B for every lambda>=0',scope='Original score algebra witness only; no assertion native candidate geometry realizes these exact factors or that unknown coverage improves')
q1=Path('/home/ouc/nso_robot/analyses/rapid_evidence_20261108/semantic/execution');residual=0.;states=0
for p in sorted(q1.glob('*_choices.json')):
 for row in json.load(open(p)):
  scores=row['outputs']['scores'];states+=1
  for a,b,c in zip(scores['S'],scores['X'],scores['G']):residual=max(residual,abs(a+b-2*c))
assert residual<1e-10
out=dict(algebraic_witness=proof,archived_Q1_midpoint_check=dict(states=states,maximum_abs_S_plus_X_minus_2G=residual,scope='Saved score arrays only; no new execution'),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(D/'arithmetic_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
