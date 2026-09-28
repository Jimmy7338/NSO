from pathlib import Path
import sys
sys.path.insert(0,'/root/NSO')
import numpy as np
from nso.controller_v43 import ANSControllerV43
from nso.analytic_fixture_v42 import analytic_forward_sequence_v42,ANALYTIC_MARKER_COLOR_V42
from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41,PrimitiveStateV41
from nso.observed_mapper_v42 import ObservedMapperV42
spec=dict(schema_version='v41.public_navigation.v1',source_kind='provided_navigation_prior',nodes={'home':[.75,.75],'advance':[1.,.75]},edges=[['home','advance']])
cs={name:ANSControllerV43(PublicPrimitiveGraphV41(spec),home=PrimitiveStateV41('home',0),budget=32,palette={'cabinet':ANALYTIC_MARKER_COLOR_V42},structure_names=['planar','recessed','louvered','open_frame'],class_structure_prior={'cabinet':[.4,.3,.2,.1]},mode=name) for name in ('G','S')}
m=ObservedMapperV42(shape=(30,30),origin_xy_m=(-.5,-.5))
# Pure predeclared packets already generated before any policy call; no World.
packets=analytic_forward_sequence_v42()
for p in packets:
 m.update(p)
 for name,c in cs.items():
  c.accept(p,m); d=c.choose(); print(p.paid_step,name,d['action'],d.get('routing',{}).get('target'),len(d.get('candidate_utilities',[])))
