#!/usr/bin/env python3
"""Two saved development mechanisms; all paid poses, no new simulation/evaluation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('MPLCONFIGDIR','/tmp/nso-mechanism-cases-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'audit_results/article_stage_20260928'
COLORS={'B':'#C86E18','G':'#246A9A','NBV':'#218776'}
INK='#253640';MUTED='#62717B'
PINS={}


def sha(b): return hashlib.sha256(b).hexdigest()


def read(path,pin=None):
    b=path.read_bytes();h=sha(b)
    if pin is not None and h!=pin: raise ValueError('Input changed: '+str(path))
    PINS[str(path.relative_to(ROOT))]=dict(sha256=h,bytes=len(b))
    return b


def obj(path,pin=None): return json.loads(read(path,pin))


def sealed(folder):
    m=obj(folder/'manifest.json')
    for name,pin in m['files'].items(): read(folder/name,pin['sha256'])
    return m


def gather():
    aisle_root=STAGE/'analysis_v1/exposure_AISLE_mechanism_final'
    am=sealed(aisle_root);aisle=obj(aisle_root/'summary.json')
    cell_root=STAGE/'analysis_v1/exposure_cell_g_nbv_lookahead'
    sealed(cell_root);cell=obj(cell_root/'report.json')
    assert cell['actual_first_diagnostic_information']['geometry_feedback_applied'] is False
    assert cell['actual_first_diagnostic_information']['posterior_before_after_equal'] is True
    assets=STAGE/'scene_assets_v1'
    asset_manifest=obj(assets/'manifest.json','a0319f7e5683a91a56effdbbab28c90e226ce5b0ac7178c2b2408cb8954ae14f')
    cases={}
    for family,methods,commit,first,completions in (
        ('AISLE',('B','G'),27,32,{'B':41,'G':44}),
        ('CELL',('G','NBV'),17,20,{'G':20,'NBV':29})):
        data=dict(methods=methods,commit=commit,first_action=first,completions=completions,paths={},metrics={})
        name=f'ART1_{family}_DEV/evaluation_private/instances.json'
        data['offline_background']=obj(assets/name,asset_manifest['artifact_sha256'][name])
        for method in methods:
            run=f'exposure_{family}_{method}_b160_n92801';folder=STAGE/'exposure_ablation_v3/episodes'/run
            if family=='AISLE':
                expected=am['inputs'][str((folder/'artifact_manifest.json').relative_to(ROOT))]['sha256']
                metrics=aisle['methods'][method]['metrics']
            else:
                expected=cell['sides'][method]['artifact_manifest_sha256'];metrics=cell['terminal_metrics'][method]
            manifest=obj(folder/'artifact_manifest.json',expected)
            def original(name): return obj(folder/name,manifest['files'][name]['sha256'])
            result=original('result.json');evaluation=original('evaluation.json')
            for key in ('C_nav','J_nav'): assert abs(metrics[key]-evaluation['metrics'][key])<1e-12
            path=[]
            for step in range(result['acquired_and_saved_packets']):
                packet=original(f'packets/{step:03d}_receipt.json')['execution']
                assert packet['paid_step']==step
                path.append(dict(paid_step=step,pose_xyyaw_rad=packet['pose_xyyaw_rad'],action=packet['action']))
            assert path[completions[method]]['action']=='observe'
            assert len(path)==len(result['actions'])+1
            assert max(abs(x-y) for x,y in zip(path[0]['pose_xyyaw_rad'],path[-1]['pose_xyyaw_rad']))<1e-10
            data['paths'][method]=path;data['metrics'][method]={k:metrics[k] for k in ('C_nav','F1','J_nav')}
        data['delta_J']=data['metrics'][methods[0]]['J_nav']-data['metrics'][methods[1]]['J_nav']
        assert abs(data['delta_J']-(-.0487163074094115 if family=='AISLE' else .27104362257830894))<1e-12
        cases[family]=data
    return cases


def background(ax,metadata):
    ax.set_facecolor('#FCFDFE')
    for i,box in enumerate(metadata['background_boxes'][1:]):
        x0,x1,y0,y1,_,_=box
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor='#CAD1D5' if i<4 else '#858F97',
            edgecolor='#AEB8BE' if i<4 else '#66727B',lw=.45,hatch=None if i<4 else '////',zorder=1))
    for instance in metadata['private_instances']:
        xy=np.array(instance['conservative_footprint_xy_m'])
        ax.add_patch(Polygon(xy,closed=True,facecolor='#E3E9ED',edgecolor='#A4B2BC',lw=.7,zorder=2))
    ax.set(xlim=(-.1,8.1),ylim=(-.1,8.1),aspect='equal',xticks=(0,2,4,6,8),yticks=(0,2,4,6,8))
    ax.tick_params(labelsize=8,colors=MUTED,length=2.5,width=.5)
    ax.set_xlabel('$x$ (m)',fontsize=8,labelpad=1)
    ax.set_ylabel('$y$ (m)',fontsize=8,labelpad=1)
    ax.spines[['top','right']].set_visible(False)
    for side in ('left','bottom'):ax.spines[side].set(color='#ADBBC4',linewidth=.6)


def direction(ax,pose,color,length=.66):
    x,y,yaw=pose
    ax.annotate('',xy=(x+length*math.cos(yaw),y+length*math.sin(yaw)),xytext=(x,y),
        arrowprops=dict(arrowstyle='-|>',mutation_scale=9,lw=1.2,color=color),zorder=8)


def callout(ax,text,xy,xytext,color=INK):
    ax.annotate(text,xy=xy,xytext=xytext,color=color,fontsize=8,ha='left',va='center',
        bbox=dict(boxstyle='round,pad=.15',fc='white',ec='none',alpha=.91),
        arrowprops=dict(arrowstyle='-',color=color,lw=.65,shrinkA=2,shrinkB=3),zorder=12)


def panel(ax,family,data):
    background(ax,data['offline_background'])
    methods=data['methods']
    for index,method in enumerate(methods):
        path=data['paths'][method];points=np.array([r['pose_xyyaw_rad'][:2] for r in path])
        # Every saved position is retained. Dashed second trace exposes overlap
        # without jittering world coordinates or cropping later route segments.
        ax.plot(points[:,0],points[:,1],color=COLORS[method],lw=1.75 if index==0 else 1.35,
            ls='-' if index==0 else (0,(3,2)),alpha=.86,zorder=3+index)
        complete=data['completions'][method];pose=path[complete]['pose_xyyaw_rad']
        ax.scatter(*pose[:2],s=32,marker='o',facecolors='white',edgecolors=COLORS[method],linewidths=1.3,zorder=9)
        direction(ax,pose,COLORS[method])
        direction(ax,path[data['first_action']]['pose_xyyaw_rad'],COLORS[method],.55)
    first_path=data['paths'][methods[0]]
    prefix=np.array([r['pose_xyyaw_rad'][:2] for r in first_path[:data['first_action']]])
    ax.plot(prefix[:,0],prefix[:,1],color=INK,lw=1,ls=(0,(1,2)),zorder=6)
    decision=first_path[data['commit']]['pose_xyyaw_rad'][:2]
    first=first_path[data['first_action']]['pose_xyyaw_rad'][:2]
    ax.scatter(*decision,s=27,marker='s',facecolors='white',edgecolors=INK,linewidths=1,zorder=10)
    ax.scatter(*first,s=25,marker='D',facecolors='white',edgecolors=INK,linewidths=1,zorder=10)
    home=first_path[0]['pose_xyyaw_rad'][:2]
    ax.scatter(*home,s=60,marker='*',facecolors='#253640',edgecolors='white',lw=.5,zorder=11)
    ax.text(.24,.23,'H',fontsize=8,weight='bold',color=INK,zorder=12)
    if family=='AISLE':
        callout(ax,'D27',decision,(2.35,1.30))
        callout(ax,'F32',first,(.30,4.20))
        callout(ax,'B: O41',first_path[41]['pose_xyyaw_rad'][:2],(1.12,3.78),COLORS['B'])
        callout(ax,'G: O44',data['paths']['G'][44]['pose_xyyaw_rad'][:2],(1.78,.43),COLORS['G'])
    else:
        callout(ax,'D17',decision,(2.08,.44))
        callout(ax,'F20 / G: O20',first,(.34,3.48),COLORS['G'])
        callout(ax,'NBV: O29',data['paths']['NBV'][29]['pose_xyyaw_rad'][:2],(5.12,.54),COLORS['NBV'])
    handles=[Line2D([],[],color=COLORS[m],lw=1.7,ls='-' if i==0 else '--',label=m) for i,m in enumerate(methods)]
    ax.legend(handles=handles,loc='upper right',fontsize=8,ncol=2,frameon=True,facecolor='white',
        edgecolor='none',framealpha=.90,handlelength=1.8,columnspacing=.9,borderpad=.25)


def main(output):
    if output.exists(): raise ValueError('Exclusive new figure output required')
    cases=gather()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7,5.9),facecolor='white')
    for left,family,title in ((.067,'AISLE','(a) AISLE: category-conditioned planning'),
                              (.557,'CELL','(b) CELL: geometric look-ahead')):
        ax=fig.add_axes([left,.370,.405,.481]);panel(ax,family,cases[family])
        fig.text(left,.900,title,fontsize=9,weight='bold',color=INK)
    fig.text(.067,.958,'Saved decisions and complete paid trajectories',fontsize=11,weight='bold',color=INK)
    fig.text(.067,.927,'Exposure v3 · development layouts · budget 160 · noise seed 92801',fontsize=8,color=MUTED)
    fig.text(.067,.300,r'$\Delta J_{\mathrm{B-G}}=-0.048716$',fontsize=10,color=COLORS['B'],weight='bold')
    fig.text(.557,.300,r'$\Delta J_{\mathrm{G-NBV}}=+0.271044$',fontsize=10,color=COLORS['G'],weight='bold')
    left_lines=['D27: category prior changes the selected target.',
                'F32: B turns right; G turns left.',
                'Paid target views: B at 41; G at 44.',
                'Changed actions did not improve endpoint J.']
    right_lines=['D17: a diagnostic option beats direct NBV.',
                 'F20: G observes; NBV turns right.',
                 'O20: no new geometric feedback or posterior change.',
                 'NBV completes its direct target at O29.']
    for left,lines in ((.067,left_lines),(.557,right_lines)):
        for i,line in enumerate(lines):fig.text(left,.265-i*.025,line,fontsize=8,color=INK)
    keys=[Line2D([],[],marker='*',color=INK,lw=0,ms=7,label='H: start / finish'),
          Line2D([],[],marker='s',mfc='white',mec=INK,lw=0,ms=4,label='D: target decision'),
          Line2D([],[],marker='D',mfc='white',mec=INK,lw=0,ms=4,label='F: first action split'),
          Line2D([],[],marker='o',mfc='white',mec=INK,lw=0,ms=4,label='O: paid target view')]
    fig.legend(handles=keys,loc='lower center',bbox_to_anchor=(.52,.130),ncol=4,fontsize=8,
               frameon=False,handlelength=.9,columnspacing=1.)
    fig.text(.067,.108,'Both maps use the same metric scale. Full paths include return; arrows show camera heading.',fontsize=8,color=MUTED)
    fig.text(.067,.082,'Private geometry is an offline background only. Dashed / solid lines do not shift coordinates.',fontsize=8,color=MUTED)
    fig.text(.067,.056,'Post hoc development examples; endpoint differences include all subsequent actions.',fontsize=8,color=MUTED)
    fig.text(.067,.030,'Neither panel establishes cross-instance S/B sharing benefit or per-view reconstruction gain.',fontsize=8,color=MUTED)
    output.mkdir(parents=True)
    for ext in ('pdf','svg','png'):
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
        fig.savefig(output/f'mechanism_cases.{ext}',dpi=350,metadata=metadata)
    plt.close(fig)
    (output/'plotted_data.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n')
    captions='''# Figure caption / 图注

Two observed mechanisms with opposite endpoint effects on development layouts. Every saved paid pose, including the full return path, is displayed at the same metric scale; no trajectory is shortened, smoothed or displaced. D marks target selection, F the first executed action difference, O completion of the selected paid observation, and H the shared start/final pose. (a) AISLE: B and G use the same observations through frame31; category conditioning changes the direct target at27, followed by opposite turns at32 and completed observations at41/44. J(B)−J(G)=−0.048716. (b) CELL: G and NBV share observations through19; the diagnostic option selected at17 produces observe/turn-right at20, and target completion occurs at20/29. J(G)−J(NBV)=+0.271044. G's diagnostic completion at20 produces no fresh geometric feedback or posterior change. The common-version J values are taken from sealed independent reviews. These are explanatory post hoc development cases; later trajectories differ and the endpoint difference is not an isolated effect of the first view. Both CELL methods are geometry-only. Neither panel establishes cross-instance semantic-sharing benefit. Private equipment and occluder geometry appears only as an offline explanatory background, never as planner input.

两个开发布局中的已执行决策机制与方向相反的终点效应。两图统一米制尺度，保留所有付费位置与完整返航路径，不裁切、平滑或挪动轨迹。D为目标选择，F为首次实际动作分歧，O为该目标付费观察完成，H为共同起终点。AISLE中B/G在帧0–31物理输入相同，步27类别条件化改变direct目标，步32实际反向转动，步41/44分别完成观察，终点J差为−0.048716。CELL中G/NBV在帧0–19输入相同，步17诊断选项胜出，步20产生观察/右转分歧，步20/29分别完成目标，终点J差为+0.271044；G在诊断完成当帧没有新增几何反馈或后验变化。指标取自封存独立复核，两例均为事后开发机制解释，不能把整个后续路径终点差归于单个观察，也不能视为S/B跨实例语义共享收益。私有设备和遮挡几何仅用作离线背景。
'''
    (output/'captions.md').write_text(captions)
    source=Path(__file__).resolve();read(source);(output/'source.py').write_bytes(source.read_bytes())
    manifest=dict(schema='article.saved_mechanism_figure.v1',print_width_inches=7,minimum_font_pt=8,
        source_inputs=PINS,new_worlds=0,new_tsdf=0,new_evaluations=0,raw_trajectory_points={
            f+'_'+m:len(d['paths'][m]) for f,d in cases.items() for m in d['methods']},
        files={f.name:dict(sha256=sha(f.read_bytes()),bytes=f.stat().st_size) for f in sorted(output.iterdir())})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    size=sum(f.stat().st_size for f in output.iterdir());assert size<2*1024**2
    print(json.dumps(dict(output=str(output),bytes=size,trajectory_points=manifest['raw_trajectory_points'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output.resolve())
