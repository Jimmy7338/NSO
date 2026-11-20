"""Evaluator-only geometry figure, never an actor map or trial outcome."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent;SCENE=HERE.parent;sys.path.insert(0,str(SCENE))
from unseen_world import SPECS,LAYOUTS
plt.rcParams.update({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(8.4,7.8))
for ax,layout in zip(axes.ravel(),LAYOUTS):
 q=SPECS[layout];data=np.load(SCENE/'prepared'/layout/'evaluator_truth.npz',allow_pickle=False);W,H=q['width_m'],q['height_m'];ax.imshow(data['occupancy'],origin='upper',extent=[0,W,0,H],cmap=ListedColormap(['#fafafa','#263746']),vmin=0,vmax=1,interpolation='nearest')
 for wx,wy,wz,ww,wh,wheight in q['wall_boxes_xyz_size']:ax.add_patch(Rectangle((wx,wy),ww,wh,facecolor='#263746',edgecolor='none'))
 x,y=1.1,1.1;heading=0;xy=[[x,y]];d=[(0,.2),(.2,0),(0,-.2),(-.2,0)]
 for a in q['sealed_prefixes'][1]['actions']:
  if a=='forward':x+=d[heading][0];y+=d[heading][1]
  elif a=='right':heading=(heading+1)%4
  else:heading=(heading-1)%4
  xy.append([x,y])
 xy=np.array(xy);ax.plot(xy[:,0],xy[:,1],color='#009E73',lw=1.5,label='Fixed acquisition prefix');ax.scatter([1.1],[1.1],s=24,c='#0072B2',zorder=5);ax.text(1.1,.65,'Start',ha='center',color='#0072B2')
 for p in q['sealed_prefixes']:
  x,y=p['expected_xy'];ax.scatter([x],[y],s=30,marker='D',c='#D55E00',zorder=6);ax.text(x,y-.5,f"{p['prefix_id']} ({p['paid_actions']} actions)",ha='center',fontsize=7,color='#9a4300')
 for name,xx,yy in [('A',q['split_x']/2,q['split_y']/2),('B',(q['split_x']+W)/2,q['split_y']/2),('C',(q['split_x']+W)/2,(q['split_y']+H)/2),('D',q['split_x']/2,(q['split_y']+H)/2)]:ax.text(xx,yy+.35,name,color='#576674',fontsize=10,ha='center')
 ax.set_title(f"{layout.replace('_',' ').title()} · {W:g} × {H:g} m",fontsize=10);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_xlim(0,W);ax.set_ylim(0,H);ax.set_aspect('equal')
fig.tight_layout(pad=1.2)
for ext in ['pdf','svg','png']:fig.savefig(HERE/f'four_predeclared_layouts.{ext}',dpi=200)
caption='Four predeclared controlled layouts with reused authored furniture and two room-connection topologies. The chain layouts have edges A–B–C–D (cycle rank 0); the cycle layouts add D–A (cycle rank 1). Walls, room dimensions, door locations and furniture positions/orientations differ across layouts. Dark shapes show exact frozen architecture rectangles and sampled furniture occupancy from evaluator-only geometry, not an input map supplied to the actor. Wall rectangles are drawn directly because radius-zero grid points exactly on triangle boundaries may be omitted by the inherited floating-point spatial query; motion and reachable topology use the unchanged swept 0.2 m footprint. Green lines show fixed scripted acquisition paths; marked p1/p2 are predeclared decision states with all prefix actions charged. These are controlled initial conditions, not autonomous exploration outcomes. All eight paths passed the retained swept-footprint collision check; no gain or future surface measurement selected any path or layout.'
(HERE/'CAPTION.md').write_text(caption+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(HERE/'receipt.json').write_text(json.dumps(dict(scene_freeze_sha256=sha(SCENE/'freeze.json'),all_four_layouts_included=True,source_geometry_is_evaluator_only=True,outcome_data_read=False,files={p.name:sha(p)for p in HERE.iterdir()if p.is_file()and p.name!='receipt.json'}),indent=2)+'\n')
