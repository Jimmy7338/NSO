"""Predeclared L24/R24 acquired-point and hypothesis-envelope comparison.

The plotting input contains only already acquired points and frozen predictions.
No source meshes, ground-truth surfaces or hand-adjusted geometry are loaded.
"""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent

def corners(envelope):
    return np.array(list(itertools.product((-.5,.5),repeat=3)))*np.array(envelope['dims'])@np.array(envelope['rotation']).T+np.array(envelope['center'])

def plot(data):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.labelpad':1,
        'axes.spines.top':False,'axes.spines.right':False})
    figure=plt.figure(figsize=(9.0,7.5),layout='constrained')
    colors={'object_support':'#D55E00','object_fit':'#0072B2'}
    styles={'object_support':(0,(4,2)),'object_fit':'-'}
    letters='abcd'
    for number,panel in enumerate(data['panels']):
        ax=figure.add_subplot(2,2,number+1,projection='3d',proj_type='ortho')
        pts=np.array(panel['points']);origin=np.r_[panel['origin_xy'],0.]
        pts=pts-origin;extra=np.array(panel['outside_old_crop'])
        ax.scatter(*pts[~extra].T,s=14,color='#333333',alpha=.9,depthshade=False)
        ax.scatter(*pts[extra].T,s=22,marker='^',color='#CC79A7',alpha=1,depthshade=False)
        extent=[pts]
        for stage,envelope in panel['envelopes'].items():
            vertices=corners(envelope)-origin;extent.append(vertices)
            for i,j in itertools.combinations(range(8),2):
                if bin(i^j).count('1')!=1:continue
                edge=vertices[[i,j]]
                ax.plot(*edge.T,color=colors[stage],ls=styles[stage],lw=1.4,alpha=.9)
        allp=np.concatenate(extent)
        # A single equal metric scale across panels preserves visible size
        # differences. XY origins are measured-point centroids, never a fit.
        ax.set_xlim(-1.15,1.15);ax.set_ylim(-1.15,1.15);ax.set_zlim(0,2.2)
        ax.set_box_aspect((2.3,2.3,2.2));ax.view_init(elev=23,azim=-56)
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([0,1,2])
        ax.set_xlabel(r'$\Delta x$ (m)');ax.set_ylabel(r'$\Delta y$ (m)');ax.set_zlabel('$z$ (m)')
        layout='Left layout' if 'left' in panel['state'] else 'Right layout'
        ax.set_title(f'({letters[number]}) {layout}, track {panel["track"]}',loc='left',pad=1,fontweight='medium')
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False;axis.pane.set_edgecolor('#FFFFFF')
            axis._axinfo['grid'].update(color='#E4E4E4',linewidth=.5)
        ax.tick_params(labelsize=8,pad=0)
    handles=[Line2D([],[],marker='o',ls='',ms=4,color='#333333',label='Acquired points within old crop'),
        Line2D([],[],marker='^',ls='',ms=5,color='#CC79A7',label='Acquired points outside old crop'),
        Line2D([],[],color=colors['object_support'],ls=styles['object_support'],lw=1.5,label='O: inherited envelope'),
        Line2D([],[],color=colors['object_fit'],lw=1.5,label='OF: fitted envelope')]
    figure.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    for extension in ('svg','pdf','png'):
        directory=HERE if extension=='svg' else HERE.parent/'figures'
        directory.mkdir(parents=True,exist_ok=True)
        figure.savefig(directory/f'observed_points_template_envelopes.{extension}',dpi=220)
    plt.close(figure)

if __name__=='__main__':
    data_path=HERE/'observed_points_template_envelopes_data.json'
    if not data_path.exists():
        raise FileNotFoundError('Published measured plotting input is required; no geometry is reconstructed or inferred by this renderer.')
    plot(json.loads(data_path.read_text()))
