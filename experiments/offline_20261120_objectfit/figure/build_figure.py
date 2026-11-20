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
ROOT=HERE.parents[1]
STATES=('shelf_left_24','shelf_right_24')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def extract():
    result={'selection':'Predeclared left/right layout prefix 24; both observed object tracks included',
        'evidence_type':'Acquired simulated quality points and predicted shared template envelopes, not reconstructed or true complete surfaces',
        'sources':{},'panels':[]}
    for state in STATES:
        support=ROOT/'replay/observed_prefixes'/state/'acquired_points.npz'
        data=np.load(support)
        files={s:ROOT/'replay'/f'{state}_attempt_001'/f'{s}_predictions.json' for s in ('object_support','object_fit')}
        predictions={s:json.loads(p.read_text()) for s,p in files.items()}
        result['sources'][str(support.relative_to(ROOT))]=sha(support)
        result['sources'].update({str(p.relative_to(ROOT)):sha(p) for p in files.values()})
        objects={s:{int(o['support_track']):o for o in d['model_audits']['G']['objects']} for s,d in predictions.items()}
        assert set(objects['object_support'])==set(objects['object_fit'])
        for track in sorted(objects['object_support']):
            mask=data['all_support_track']==track
            points=data['all_point'][mask]
            panel={'state':state,'track':track,'points':points.tolist(),
                'outside_old_crop':((points[:,2]<=.12)|(points[:,2]>=1.8)).tolist(),
                'origin_xy':points[:,:2].mean(0).tolist(),'envelopes':{}}
            for stage,items in objects.items():
                o=items[track]
                panel['envelopes'][stage]={k:o[k] for k in ('center','dims','rotation','observed_points','fit_input_points_sha256')}
            result['panels'].append(panel)
    (HERE/'figure_data.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

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
        figure.savefig(HERE/f'observed_points_template_envelopes.{extension}',dpi=220)
    plt.close(figure)

if __name__=='__main__':
    data_path=HERE/'figure_data.json'
    plot(json.loads(data_path.read_text()) if data_path.exists() else extract())
