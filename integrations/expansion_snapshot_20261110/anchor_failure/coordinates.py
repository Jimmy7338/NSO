import json,numpy as np
from pathlib import Path
O=Path(__file__).parent;P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261110/expand_history_S96_20261110_01');cfg=json.loads((P/'joint_configuration.json').read_text())['configuration'];T=np.array(cfg['planning_from_odom']);rows=json.loads((O/'bounded.json').read_text())
for row in rows:
 z=np.load(P/'joint_inputs'/('joint-'+row['frame']+'.npz'));b=z['base_odom_pose'];row['base_odom_pose']=b.tolist()
 for item in row['bad']:
  p=np.c_[item['depth_points'],np.ones(len(item['depth_points']))];odom=(np.linalg.inv(T)@p.T).T;cam=(np.linalg.inv(T@z['world_from_camera'])@p.T).T
  item['odom_points']=odom[:,:3].tolist();item['camera_optical_points']=cam[:,:3].tolist();item['base_relative_xy']=((odom[:,:2]-b[:2])@np.array([[np.cos(b[2]),-np.sin(b[2])],[np.sin(b[2]),np.cos(b[2])]])).tolist()
(O/'coordinates.json').write_text(json.dumps(rows,indent=2))
