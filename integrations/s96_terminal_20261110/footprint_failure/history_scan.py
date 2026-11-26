import numpy as np,json
from pathlib import Path
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261109/coverage_S96_20261109_01');O=Path(__file__).parent;T=np.array(json.loads((P/'joint_configuration.json').read_text())['configuration']['planning_from_odom']);rows=[];targets={(196,207),(196,208),(197,208)}
for aid,f in enumerate(sorted((P/'joint_inputs').glob('joint-*.npz'))):
 z=np.load(f);A=T@z['world_from_laser'];d=z['ranges'];a=float(z['angle_min'])+np.arange(len(d))*float(z['angle_increment']);p=np.stack([np.cos(a)*d,np.sin(a)*d,np.zeros_like(d)],axis=1)@A[:3,:3].T+A[:3,3];valid=np.isfinite(d)&(d>0)&(d<min(4,float(z['range_max']))-.001);hits=[]
 for i in np.flatnonzero(valid):
  cell=(399-int(np.floor(p[i,1]/.2)),int(np.floor(p[i,0]/.2)))
  if cell in targets:hits.append(dict(beam=int(i),range_m=float(d[i]),cell=cell))
 if hits:rows.append(dict(action=aid,frame=f.stem[6:],hits=hits))
(O/'historical_scan_hits.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
