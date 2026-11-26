import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261109/coverage_S96_20261109_01');O=Path(__file__).parent;T=np.array(json.loads((P/'joint_configuration.json').read_text())['configuration']['planning_from_odom']);rows=[]
for aid,f in [(86,'1794224925706198868'),(87,'1794224938740404868')]:
 z=np.load(P/'joint_inputs'/('joint-'+f+'.npz'));A=T@z['world_from_laser'];C=T@z['world_from_camera'];K=z['intrinsic_raw'];rgb=z['rgb_raw'];im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);items=[]
 for i in range(576,595):
  d=float(z['ranges'][i]);
  if not np.isfinite(d):
   items.append(dict(beam=i,raw_range_m=str(d)));continue
  a=float(z['angle_min'])+i*float(z['angle_increment']);p=A@np.array([np.cos(a)*d,np.sin(a)*d,0,1]);pc=np.linalg.inv(C)@p;uv=(K@pc[:3]);uv=uv[:2]/uv[2];cell=[399-int(np.floor(p[1]/.2)),int(np.floor(p[0]/.2))];item=dict(beam=i,raw_range_m=float(z['scan_ranges_raw'][i]),range_m=d,angle_rad=a,planning_point=p[:3].tolist(),cell=cell,camera_point=pc[:3].tolist(),raw_rgb_uv=uv.tolist());items.append(item)
  if i in [583,584,585,586] and pc[2]>0 and np.isfinite(uv).all():
   u,v=map(float,uv);draw.ellipse((u-5,v-5,u+5,v+5),outline='red',width=2);draw.text((u+8,v),str(i),fill='red')
 im.save(O/('rgb_projected_scan_'+str(aid)+'.png'));Image.fromarray(rgb).save(O/('rgb_raw_'+str(aid)+'.png'));rows.append(dict(action=aid,frame=f,beam_neighborhood=items))
(O/'scan_rgb_evidence.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
